from __future__ import annotations

import json
import logging
from typing import TypeDict, Literal

from langchain_core.documents import Document

from src.retrieval.retriever import Retriever
from src.llm import LLMClient

logger = logging.getLogger(__name__)

class RAGState(TypeDict):
  query: str 
  strategy: str
  strategy_reason: str

  documents: list[Document]
  context: str
  answer: str
  reflection: str
  reflection_reason: str
  retry_count: int
  final_answer: str


class RAGNodes:
  def __init__(self, retriever: Retriever, llm: LLMClient):
    self.retriever = retriever
    self.llm = llm


  def query_analyzer(self, state: RAGState) -> dict:
    """
    Analyze the query to choose suitable retrieval
    
    Strategy:
    - dense: define, cause, reason question
    - hybrid: specific keyword
    - graph:  relationship between entities, e.g.,
    """
    query = state["query"]
    logger.info(f"[query_analyzer] Analyzing query: {query[:180]}...")

    prompt = f"""Bạn là chuyên gia phân tích câu hỏi pháp lý.
Phân tích câu hỏi sau và chọn chiến lược tìm kiếm phù hợp nhất:
CÂU HỎI: {query}
CÁC CHIẾN LƯỢC:
- "dense": Tìm kiếm theo ngữ nghĩa. Dùng khi câu hỏi hỏi về ý nghĩa, khái niệm, điều kiện, quy định chung.
  Ví dụ: "Điều kiện để được cấp phép kinh doanh là gì?"
  
- "hybrid": Tìm kiếm kết hợp ngữ nghĩa + từ khóa. Dùng khi câu hỏi có tên văn bản cụ thể, số hiệu luật, điều khoản cụ thể.
  Ví dụ: "Điều 17 Luật Đất đai 2024 quy định gì?"
  
- "graph": Tìm kiếm theo mạng lưới quan hệ văn bản. Dùng khi câu hỏi về mối quan hệ giữa nhiều văn bản, hiệu lực, sửa đổi, thay thế.
  Ví dụ: "Những văn bản nào sửa đổi bổ sung Luật Đầu tư 2020?"
Trả về JSON với format sau (KHÔNG thêm gì khác):
{{
  "strategy": "dense" | "hybrid" | "graph",
  "reason": "giải thích ngắn gọn tại sao chọn strategy này"
}}"""
    
    try:
      raw = self.llm.invoke_json(prompt)
      parsed = json.loads(raw)
      strategy = parsed.get("strategy", "hybrid")
      reason = parsed.get("reason", "")

      # Validate strategy value
      if strategy not in ["dense", "hybrid", "graph"]:
        logger.warning(f"[query_analyzer] Invalid strategy value: {strategy}. Defaulting to 'hybrid'.")
        strategy = "hybrid"
        reason = "Invalid strategy value returned by LLM. Defaulting to 'hybrid'."
    except (json.JSONDecodeError, Exception) as e:
      logger.error(f"[query_analyzer] Failed to parse LLM response: {e}. Defaulting to 'hybrid'.")
      strategy = "hybrid"
      reason = f"Failed to parse LLM response: {str(e)}. Defaulting to 'hybrid'."
    
    logger.info(f"[query_analyzer] Chosen strategy: {strategy}, reason: {reason}")

    return {
      "strategy": strategy,
      "strategy_reason": reason
    }
  

  def retriever_node(self, state: RAGState) -> dict:
    """
    Retrieval base on strategy
    When retry, if strategy is dense, we can switch to hybrid
    If strategy is hybrid, we can switch to graph
    If strategy is graph -> not change
    """

    query = state["query"]
    strategy = state["strategy"]
    retry_count = state.get("retry_count", 0)

    if retry_count > 0:
      escalation_map ={
        "dense": "hybrid",
        "hybrid": "graph",
        "graph": "graph"
      }
      new_strategy = escalation_map.get(strategy, "hybrid")
      if new_strategy != strategy:
        logger.info(
            f"[retriever_node] Retry {retry_count}: "
            f"escalating {strategy} -> {new_strategy}"
          )
        strategy = new_strategy
    logger.info(f"[retriever_node] Using strategy: {strategy}")

    documents = self.retriever.retrieve(
      query=query,
      strategy=strategy,
      k=5
    )

    logger.info(f"[retriever_node] Retrieved {len(documents)} documents")
    return {
      "strategy": strategy,
      "documents": documents
    }
  
  def answer_node(self, state: RAGState) -> dict:
    query = state["query"]
    documents = state["documents"]

    # Format documents into context string
    context_parts = []
    for doc in documents:
      doc_id = doc.metadata.get("doc_id", "unknown")
      title = doc.metadata.get("title", "")
      chunk_idx = doc.metadata.get("chunk_index", -1)

      context_parts.append(
        f"[van ban: {doc_id} | {title} | chunk {chunk_idx}]\n"
        f"{doc.page_content}"
      )
    
    context = "\n\n---\n\n".join(context_parts)

    if not context.strip():
      return {
        "context": "",
        "answer": "Không tìm thấy tài liệu liên quan để trả lời câu hỏi này.",
      }
    prompt = f"""Bạn là trợ lý pháp lý chuyên về văn bản pháp luật Việt Nam.
              NGUYÊN TẮC:
              - Chỉ trả lời dựa trên context được cung cấp bên dưới
              - Trích dẫn số hiệu văn bản, điều khoản cụ thể khi có thể
              - Nếu context không đủ thông tin, nói rõ "Không tìm thấy đủ căn cứ"
              - Không bịa đặt thông tin
              CONTEXT:
              {context}
              CÂU HỎI:
              {query}
              TRẢ LỜI:"""
    response = self.llm.invoke(prompt)
    answer = response.content

    logger.info(f"[answer_node] Generated answer (length {len(answer)}) chars")

    return {
      "context": context,
      "answer": answer
    }
  
  def reflection_node(self, state: RAGState) -> dict:
    query = state["query"]
    answer = state["answer"]
    context = state["context"]

    if not context.strip():
      return {
        "reflection": "insufficient",
        "reflection_reason": "Không tìm thấy tài liệu liên quan"
      }

    prompt =f"""Bạn là người kiểm tra chất lượng câu trả lời pháp lý.
              Đánh giá xem câu trả lời có đầy đủ căn cứ và thực sự trả lời được câu hỏi không.
              CÂU HỎI: {query}
              CÂU TRẢ LỜI: {answer}
              Tiêu chí đánh giá:
              - "sufficient": Câu trả lời có căn cứ rõ ràng, trả lời trực tiếp câu hỏi
              - "insufficient": Câu trả lời mơ hồ, thiếu căn cứ, hoặc nói "không tìm thấy"
              Trả về JSON (KHÔNG thêm gì khác):
              {{
                "verdict": "sufficient" | "insufficient",
                "reason": "giải thích ngắn gọn"
              }}"""
    try:
      raw = self.llm.invoke_json(prompt)
      parsed = json.loads(raw)
      verdict = parsed.get("verdict", "insufficient")
      reason = parsed.get("reason", "")

      if verdict not in ["sufficient", "insufficient"]:
        verdict = "insufficient"
        reason = "Invalid verdict from LLM"
    
    except Exception as e:
      logger.warning(f"reflection_node failed: {e}. Marking sufficient to avoid loop.")

      # If LLM fails, we can mark sufficient to avoid infinite loop
      verdict = "sufficient"
      reason = f"Reflection error: {str(e)}"

    logger.info(f"[reflection_node] Reflection verdict: {verdict}, reason: {reason}")

    return {
      "reflection": verdict,
      "reflection_reason": reason
    }



  def finalize_node(self, state: RAGState) -> dict:
    return {
      "final_answer": state["answer"]
    }
  
  # CONDITIONAL EDGE: should_retry
  def should_retry(
    self, 
    state: RAGState) -> Literal["retriever_node", "finalize_node"]:
    """
    Return: node name to go next
    """

    reflection = state.get("reflection", "sufficient")
    retry_count = state.get("retry_count", 0)
    max_retries = 2

    if reflection == "insufficient" and retry_count < max_retries:
      logger.info(
        f"[should_retry] Retrying..."
        f")attempt {retry_count + 1}/{max_retries}"
      )
      state["retry_count"] = retry_count + 1
      return "retriever_node"
    else:
      if reflection == "insufficient":
        logger.info("[should_retry] Max retries reached. Finalizing answer.")
      else:
        logger.info("[should_retry] Answer sufficient. Finalizing answer.")

      return "finalize_node"
