from __future__ import annotations

from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.indexing.embeddings import Embedder

class ChromaStore:
  def __init__(
    self,
    persist_directory: str = "/data/chroma_store",
  ):
    
    self.persist_directory = persist_directory

    self.embedder = Embedder()

    self.store = None

  def build(
    self,
    documents: list[Document],
  ):
    self.store = Chroma.from_documents(
      documents=documents,
      embedding=self.embedder.langchain_embedding,
      persist_directory=self.persist_directory,
    )
    return self.store
  
  def add_documents(
    self,
    documents: list[Document],
  ):
    if self.store is None:
      raise ValueError("Store is not built yet. Call build() first.")
    
    self.store.add_documents(documents)

  def load(self):
    self.store = Chroma(
      persist_directory=self.persist_directory,
      embedding_function=self.langchain_embedding,
    )
    return self.store
  
  def similarity_search(
    self,
    query: str,
    k: int = 4,
    filter: dict | None = None,
  ):
    if self.store is None:
      raise ValueError("Store is not built yet. Call build() or load() first.")
    
    return self.store.similarity_search(
      query=query,
      k=k,
      filter=filter,
    )
  
