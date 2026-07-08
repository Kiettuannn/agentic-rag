from __future__ import annotations

import json
import logging
from typing import TypedDict, Literal

from langchain_core.documents import Document

from src.retrieval.retriever import Retriever
from src.llm import LLMClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

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
  messages: list  # Lưu hội thoại giữa Agent và Tool (AIMessage, ToolMessage...)

class RAGNodes:
  def __init__(self, retriever: Retriever, llm: LLMClient, langchain_llm, tools: list):
    self.retriever = retriever
    self.llm = llm
    self.agent_llm = langchain_llm.bind_tools(tools)
    self.tools_map = {t.name: t for t in tools}


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
Dựa vào câu hỏi đã viết lại, chọn chiến lược tra cứu theo các quy tắc nghiêm ngặt sau:

1. "bm25": LÀ CÔNG CỤ TRA CỨU ĐÍCH DANH. Dùng KHI VÀ CHỈ KHI user muốn tìm hiểu tổng quan, xem hoặc tải một văn bản cụ thể dựa trên số hiệu/tên riêng. 
   -> Ví dụ: "Quyết định 52/2009 quy định về cái gì?", "Cho tôi xem Nghị định 100", "Luật Doanh nghiệp 2020".

2. "graph": LÀ CÔNG CỤ TÌM QUAN HỆ. Dùng khi hỏi về sự liên kết giữa các văn bản.
   -> Ví dụ: "Văn bản nào sửa đổi Quyết định 52?", "Nghị định này thay thế cho luật nào?".

3. "hybrid": LÀ CÔNG CỤ TRA CỨU TÌNH HUỐNG/KHÁI NIỆM. Dùng cho TẤT CẢ các trường hợp còn lại. Đặc biệt: Kể cả khi user có nhắc đến số hiệu văn bản, nhưng lại hỏi kèm MỘT TÌNH HUỐNG hoặc VẤN ĐỀ CỤ THỂ, thì BẮT BUỘC dùng hybrid.
   -> Ví dụ: "Thế nào là tài sản công?" (Hỏi khái niệm -> hybrid)
   -> Ví dụ: "Theo Nghị định 100, vượt đèn đỏ phạt bao nhiêu?" (Có nhắc NĐ 100 nhưng hỏi tình huống cụ thể "vượt đèn đỏ" -> Bắt buộc dùng hybrid để bắt được ngữ nghĩa).
Trả về JSON với format sau (KHÔNG thêm gì khác):
{{
  "search_query": "câu hỏi đã viết lại ở NV1",
  "strategy": "hybrid" | "bm25" | "graph",
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
      if strategy not in ["bm25", "hybrid", "graph"]:
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
  

  # def retriever_node(self, state: RAGState) -> dict:
  #   query = state.get("search_query", state["query"])
  #   strategy = state["strategy"]
  #   retry_count = state.get("retry_count", 0)
  #
  #   if retry_count > 0:
  #     escalation_map ={
  #       "dense": "hybrid",
  #       "hybrid": "graph",
  #       "graph": "graph"
  #     }
  #     new_strategy = escalation_map.get(strategy, "hybrid")
  #     if new_strategy != strategy:
  #       logger.info(
  #           f"[retriever_node] Retry {retry_count}: "
  #           f"escalating {strategy} -> {new_strategy}"
  #         )
  #       strategy = new_strategy
  #   logger.info(f"[retriever_node] Using strategy: {strategy}")
  #
  #   documents = self.retriever.retrieve(
  #     query=query,
  #     strategy=strategy,
  #     k=5
  #   )
  #
  #   logger.info(f"[retriever_node] Retrieved {len(documents)} documents")
  #   return {
  #     "strategy": strategy,
  #     "documents": documents
  #   }
  def agent_node(self, state: RAGState) -> dict:
    messages = state.get("messages", [])
    search_query = state.get("search_query", state["query"])
    suggested_strategy = state.get("strategy", "hybrid")

    if not messages:
      system_prompt = f"""Bạn là trợ lý pháp lý. Bạn có các công cụ tra cứu:
      - aggregate_search (Hybrid - MẶC ĐỊNH, dùng cho khái niệm, quy định chung)
      - keyword_search (BM25 - chỉ dùng khi có số hiệu chính xác)
      - related_document_search (Graph - cho quan hệ sửa đổi, thay thế)
      Hệ thống phân tích trước đó GỢI Ý bạn nên dùng chiến lược: '{suggested_strategy}'.
      (Lưu ý: 'hybrid' ~ tìm tổng hợp, 'bm25' ~ tìm chính xác, 'graph' ~ tìm liên quan).
      Hãy tôn trọng gợi ý này để chọn tool phù hợp nhất."""

      messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=search_query),
      ]
    logger.info(f"[agent_node] Invoking Tool-calling LLM...")
    response = self.agent_llm.invoke(messages)
    return {"messages": messages + [response]}

  def tool_node(self, state: RAGState) -> dict:
    messages = state["messages"]
    last_message = messages[-1]
    new_messages = []
    for tool_call in last_message.tool_calls:
      tool_name = tool_call["name"]
      tool_args = tool_call["args"]
      tool_id = tool_call["id"]
      logger.info(f"[tool_node] Executing: {tool_name}({tool_args})")
      tool_fn = self.tools_map.get(tool_name)

      if tool_fn:
        try:
          result_str = tool_fn.invoke(tool_args)
        except Exception as e:
          result_str = f"Lỗi thực thi: {str(e)}"
      else:
        result_str = "Lỗi: Không tìm thấy tool."

      new_messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
    return {"messages": messages + new_messages}

  def route_after_agent(self, state: RAGState) -> Literal["tool_node", "answer_node"]:
    """Điều hướng: Nếu LLM chọn tool thì sang tool_node, không thì sang answer_node."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    
    # Đếm số lần đã nhận kết quả từ tool
    tool_calls_count = sum(1 for m in messages if isinstance(m, ToolMessage))
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
      # Giới hạn tối đa 3 lần gọi tool để tránh lặp vô hạn
      if tool_calls_count >= 3:
        logger.warning(f"[route_after_agent] LLM đang bị kẹt vòng lặp (đã thử {tool_calls_count} lần). Ép dừng!")
        return "answer_node"
      return "tool_node"
    return "answer_node"


  
  def answer_node(self, state: RAGState) -> dict:
    query = state.get("search_query", state["query"])

    messages = state.get("messages", [])

    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    context = "\n\n---\n\n".join([m.content for m in tool_results])

    if not context.strip():
      return {
        "context": "",
        "answer": "Không tìm thấy tài liệu liên quan để trả lời.",
        "documents": []
      }

    prompt = f"""Bạn là trợ lý pháp lý chuyên về văn bản pháp luật Việt Nam.
    NGUYÊN TẮC:
    - Chỉ trả lời dựa trên context được cung cấp
    - Trích dẫn số hiệu cụ thể
    CONTEXT:
    {context}
    CÂU HỎI:
    {query}
    TRẢ LỜI:"""

    response = self.llm.invoke(prompt)
    answer = response.content
    try:
      # [FIX] Lấy chính xác query mà LLM đã bóc tách đưa vào Tool để retrieve lại
      # Nếu không, truyền cả câu dài vào BM25 sẽ làm nhiễu kết quả Citations.
      retrieve_query = query
      for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
          last_args = msg.tool_calls[-1].get("args", {})
          retrieve_query = last_args.get("keyword") or last_args.get("question") or query
          
      docs_for_citations = self.retriever.retrieve(query=retrieve_query, strategy=state["strategy"], k=5)
    except:
      docs_for_citations = []

    return {
      "context": context,
      "answer": answer,
      "documents": docs_for_citations
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
