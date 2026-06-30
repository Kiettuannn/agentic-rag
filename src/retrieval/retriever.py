from __future__ import annotations

# Document type để type hint kết quả trả về
from langchain_core.documents import Document

# Dense retrieval
from src.retrieval.dense import dense_search

# Hybrid retrieval
from src.retrieval.hybrid import hybrid_search

# Graph retrieval
from src.retrieval.graph import graph_search

# BM25 index type
from src.indexing.bm25_index import BM25Index

# Chroma store type
from src.indexing.chroma_store import ChromaStore

# NetworkX graph type
import networkx as nx


class Retriever:
  def __init__(
    self,
    store: ChromaStore,
    bm25_index: BM25Index,
    graph: nx.Graph,
  ):
    self.store = store
    self.bm25_index = bm25_index
    self.graph = graph
  
  def retrieve(
      self,
      query: str,
      strategy: str = "dense",
      k: int = 5,
  ) -> list[Document]:
    
    if strategy == "dense":
      return dense_search(store=self.store, query=query, k=k)
    
    elif strategy == "hybrid":
      return hybrid_search(store=self.store, bm25_index=self.bm25_index, query=query, k=k)
    
    elif strategy == "graph":
      return graph_search(
        store=self.store,
        graph=self.graph,
        query=query,
        k=k,
        bm25_index=self.bm25_index
      )
    
    raise ValueError(f"Invalid retrieval strategy: {strategy}. Must be one of ['dense', 'hybrid', 'graph'].")
    