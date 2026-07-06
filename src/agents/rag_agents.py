from __future__ import annotations

import json
import logging
from typing import TypedDict, Literal

from langchain_core.documents import Document

from src.retrieval.retriever import Retriever
from src.llm import LLMClient

logger = logging.getLogger(__name__)

class RAGState(TypedDict):
  query: str 
  history: list[dict]
  search_query: str
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
    query = state["query"]
    history = state.get("history", [])

    # Gom lich su thanh 1 doan text de LLM de doc
    history_text = "Không có lịch sử"
    if history:
      history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    logger.info(f"[query_analyzer] Analyzing query: {query[:180]}...")

    prompt = f"""<role>
Bạn là chuyên gia phân tích câu hỏi pháp lý.
</role>

<context>
Dưới đây là lịch sử hội thoại gần nhất:
{history_text}

CÂU HỎI MỚI: {query}
</context>

<instruction>
NHIỆM VỤ 1 (Query Rewriting):
Nếu câu hỏi mới có chứa đại từ nhân xưng (ví dụ: luật đó, mức tiền này, văn bản ấy) hoặc đang hỏi nối tiếp nội dung trước đó, hãy dựa vào Lịch sử hội thoại để VIẾT LẠI câu hỏi thành một câu hoàn chỉnh, rõ nghĩa và độc lập. Nếu câu hỏi đã đầy đủ, hãy giữ nguyên.

NHIỆM VỤ 2 (Strategy Selection):
Dựa vào câu hỏi đã viết lại, chọn chiến lược:
- "dense": hỏi về khái niệm, quy định chung.
- "hybrid": hỏi đích danh tên văn bản, số hiệu (VD: Quyết định 105).
- "graph": hỏi về mối quan hệ (sửa đổi, thay thế, hướng dẫn).

Trả về JSON với format sau (KHÔNG thêm gì khác):
{{
  "search_query": "câu hỏi đã viết lại ở NV1",
  "strategy": "dense" | "hybrid" | "graph",
  "reason": "lý do chọn chiến lược"
}}
</instruction>
    """
    
    try:
      raw = self.llm.invoke_json(prompt)
      parsed = json.loads(raw)
      search_query = parsed.get("search_query", query)
      strategy = parsed.get("strategy", "hybrid")
      reason = parsed.get("reason", "")

      # Validate strategy value
      if strategy not in ["dense", "hybrid", "graph"]:
        logger.warning(f"[query_analyzer] Invalid strategy value: {strategy}. Defaulting to 'hybrid'.")
        strategy = "hybrid"
        reason = "Invalid strategy value returned by LLM. Defaulting to 'hybrid'."
    except (json.JSONDecodeError, Exception) as e:
      logger.error(f"[query_analyzer] Failed to parse LLM response: {e}. Defaulting to 'hybrid'.")
      search_query = query
      strategy = "hybrid"
      reason = f"Failed to parse LLM response: {str(e)}. Defaulting to 'hybrid'."
    
    logger.info(f"[query_analyzer] Chosen strategy: {strategy}, reason: {reason}")

    return {
      "search_query": search_query,
      "strategy": strategy,
      "strategy_reason": reason
    }
  

  def retriever_node(self, state: RAGState) -> dict:
    query = state.get("search_query", state["query"])
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
    query = state.get("search_query", state["query"])
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

  def increment_retry_node(self, state: RAGState) -> dict:
    """
    Node trung gian: tăng retry_count lên 1 trước khi quay lại retriever.

    Tại sao cần node riêng:
    LangGraph không cho phép mutate state trong conditional edge function.
    State chỉ được update thông qua giá trị return của NODE.
    Nếu viết state["retry_count"] = ... trong edge -> không có tác dụng,
    retry_count luôn = 0 -> vòng lặp vô hạn.
    """
    current = state.get("retry_count", 0)
    logger.info(f"[increment_retry] retry_count: {current} -> {current + 1}")
    return {"retry_count": current + 1}

  # CONDITIONAL EDGE: should_retry
  def should_retry(
    self,
    state: RAGState) -> Literal["increment_retry_node", "finalize_node"]:
    """
    Conditional edge: quyết định retry hay kết thúc.

    Trả về tên node tiếp theo.
    Không được mutate state ở đây!
    """
    reflection = state.get("reflection", "sufficient")
    retry_count = state.get("retry_count", 0)
    max_retries = 2

    if reflection == "insufficient" and retry_count < max_retries:
      logger.info(
        f"[should_retry] Retrying "
        f"(attempt {retry_count + 1}/{max_retries})"
      )
      return "increment_retry_node"
    else:
      if reflection == "insufficient":
        logger.info("[should_retry] Max retries reached. Finalizing answer.")
      else:
        logger.info("[should_retry] Answer sufficient. Finalizing answer.")
      return "finalize_node"
