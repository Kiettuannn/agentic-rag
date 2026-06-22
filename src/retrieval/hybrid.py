from __future__ import annotations

from langchain_core.documents import Document
from src.retrieval.dense import dense_search

from src.indexing.bm25_index import BM25Index

from src.indexing.chroma_store import ChromaStore

def hybrid_search(
  store: ChromaStore,
  bm25_index: BM25Index,
  query: str,
  k: int = 5,
) -> list[Document]:
  """
  Hybrid search
  Dense + BM25
  Logic:
  - get top-k from dense
  - get top-k from BM25
  - merge
  - dedupe
  """

  # Step 1: Dense retrieval
  dense_docs = dense_search(store=store, query=query, k=k)

  # Step 2: BM25 retrieval
  bm25_docs = bm25_index.search(query=query, k=k)

  # Step 3: Merge results
  all_docs = dense_docs + bm25_docs

  # Step 4: Deduplicate
  seen = set()
  unique_docs = []

  for doc in all_docs:
    chunk_key = (
      doc.metadata.get("doc_id", ""),
      doc.metadata.get("chunk_index", 0)
    )

    # Neu chunk chua thay thi giu lai
    if chunk_key not in seen:
      seen.add(chunk_key)
      unique_docs.append(doc)

  return unique_docs[:k]






