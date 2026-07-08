from __future__ import annotations

import logging
from langgraph.graph import StateGraph, END

from src.agents.rag_agents import RAGState, RAGNodes
from src.retrieval.retriever import Retriever
from src.llm import LLMClient

logger = logging.getLogger(__name__)

def build_rag_graph(
  retriever: Retriever, llm: LLMClient, langchain_llm, tools: list
):
  nodes = RAGNodes(retriever=retriever, llm=llm, langchain_llm=langchain_llm, tools=tools)

  graph = StateGraph(RAGState)

  # Register nodes
  graph.add_node("query_analyzer", nodes.query_analyzer)
  graph.add_node("agent_node", nodes.agent_node)
  graph.add_node("tool_node", nodes.tool_node)
  graph.add_node("answer_node", nodes.answer_node)
  graph.add_node("reflection_node", nodes.reflection_node)
  graph.add_node("increment_retry_node", nodes.increment_retry_node)
  graph.add_node("finalize_node", nodes.finalize_node)


  # Entry point
  graph.set_entry_point("query_analyzer")
  graph.add_edge("query_analyzer", "agent_node")

  graph.add_conditional_edges(
    "agent_node",
    nodes.route_after_agent,
    {"tool_node": "tool_node", "answer_node": "answer_node"}
  )
  graph.add_edge("tool_node", "agent_node")

  graph.add_edge("answer_node", "reflection_node")
  graph.add_conditional_edges(
      "reflection_node",
      nodes.should_retry,
      {"increment_retry_node": "increment_retry_node", "finalize_node": "finalize_node"}
  )
  # Retry quay lại agent_node để dùng tool khác.
  graph.add_edge("increment_retry_node", "agent_node")
  graph.add_edge("finalize_node", END)
  return graph.compile()



class RAGOrchestrator:
  def __init__(self, retriever: Retriever, llm: LLMClient, langchain_llm, tools: list):
    self.graph = build_rag_graph(retriever, llm, langchain_llm, tools)
  
  def run(self, query: str, history: list = None) -> dict:
    history = history or []
    initial_state: RAGState ={
      "query": query,
      "history": history,
      "search_query": query,  # default search query, will be updated by query_analyzer
      "strategy": "hybrid", # default strategy, will be updated by query_analyzer
      "strategy_reason": "",
      "documents": [],
      "context": "",
      "answer": "",
      "reflection": "",
      "reflection_reason": "",
      "retry_count": 0,
      "final_answer": "",
      "messages": [],
    }
    logger.info(f"Starting RAG orchestration for query: {query[:80]}")

    final_state = self.graph.invoke(initial_state)

    return {
      "final_answer": final_state.get("final_answer", ""),
      "strategy": final_state.get("strategy", ""),
      "strategy_reason": final_state.get("strategy_reason", ""),
      "documents": final_state.get("documents", []),
      "reflection": final_state.get("reflection", ""),
      "reflection_reason": final_state.get("reflection_reason", ""),
      "retry_count": final_state.get("retry_count", 0),
      "search_query": final_state.get("search_query", query),
      "messages": final_state.get("messages", [])
    }
  

