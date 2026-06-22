from __future__ import annotations

# pickle de save/load bm25 object xuong disk
import pickle

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

class BM25Index:
  def __init__(
    self,
    persist_path: str = "/data/bm25_index.pkl",
  ):
    self.persist_path = persist_path
    self.bm25 = None
    self.documents = []
  
  def tokenize(self, text: str) -> list[str]:
    # Simple tokenizer: lowercase + split by space
    # Vi du:
    """ Điều 1 quy định chung ->  ["điều", "1", "quy", "định", "chung"] """
    return text.lower().split()
  
  def build(
      self,
      documents: list[Document],
  ):
    # luu full documents de mapping ket qua sau search
    self.documents = documents

    # tokenize tung document
    tokenized_docs = [
      self.tokenize(doc.page_content) for doc in documents
    ]

    # build bm25 index
    self.bm25 = BM25Okapi(tokenized_docs)

    return self.bm25
  
  def save(self):
    if self.bm25 is None:
      raise ValueError("BM25 index is not built yet. Call build() first.")
    
    with open(self.persist_path, "wb") as f:
      pickle.dump(
        {
          "bm25": self.bm25,
          "documents": self.documents,
        }, f)
  
  def load(self):
    with open(self.persist_path, "rb") as f:
      data = pickle.load(f)
      self.bm25 = data["bm25"]
      self.documents = data["documents"]
  
  def search(
    self,
    query: str,
    k: int = 5,
  ) -> list[Document]:
    if self.bm25 is None:
      raise ValueError("BM25 index is not built yet. Call build() or load() first.")
    
    tokenized_query = self.tokenize(query)
    scores = self.bm25.get_scores(tokenized_query)

    # Sort theo score giam dan
    ranked_indices = sorted(
      range(len(scores)),
      key=lambda i: scores[i],
      reverse=True,
    )

    # Lay top k documents
    top_docs = [
      self.documents[i] for i in ranked_indices[:k]
    ]
    return top_docs

  
    
