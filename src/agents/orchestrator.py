from __future__ import annotations

import logging
from langgraph.graph import StateGraph, END

from src.agents.rag_agents import RAGState, RAGNodes
from src.retrieval.retriever import Retriever
from src.llm import LLMClient

logger = logging.getLogger(__name__)

def build_rag_graph(
  retriever: Retriever,
  llm: LLMClient
):
  nodes = RAGNodes(retriever=retriever, llm=llm)

  graph = StateGraph(RAGState)

  # Register nodes
  graph.add_node("query_analyzer", nodes.query_analyzer)
  graph.add_node("retriever_node", nodes.retriever_node)
  graph.add_node("answer_node", nodes.answer_node)
  graph.add_node("reflection_node", nodes.reflection_node)
  graph.add_node("final_answer_node", nodes.finalize_node)


  # Define edges

  # Entry point
  graph.set_entry_point("query_analyzer")

  # Linear edges
  graph.add_edge("query_analyzer", "retriever_node")
  graph.add_edge("retriever_node", "answer_node")
  graph.add_edge("answer_node", "reflection_node")

  # Conditional edge: retry or finalize
  graph.add_conditional_edge(
    source="reflection_node",
    path=nodes.should_retry,
    path_map={
      "retriever_node": "retriever_node",
      "finalize_node": "finalize_node"
    }
  )

  # Terminal edge
  graph.add_edge("finalize_node", END)

  # Compile graph
  compiled = graph.compile()

  logger.info("RAG graph built and compiled successfully.")
  return compiled

class RAGOrchestrator:
  def __init__(self, retriever: Retriever, llm: LLMClient):
    self.graph = build_rag_graph(retriever, llm)
  
  def run(self, query: str) -> dict:
    initial_state: RAGState ={
      "query": query,
      "strategy": "hybrid", # default strategy, will be updated by query_analyzer
      "strategy_reason": "",
      "documents": [],
      "context": "",
      "answer": "",
      "reflection": "",
      "reflection_reason": "",
      "retry_count": 0,
      "final_answer": ""
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
      "retry_count": final_state.get("retry_count", 0)
    }
  

