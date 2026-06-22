from typing import Optional
from langchain_core.documents import Document

def dense_search(
    store,
    query: str,
    k: int = 5,
    metadata_filter: Optional[dict] = None,
) -> list[Document]:
  """
  Basic dense retrieval wrapper
  """
  return store.similarity_search(
    query,
    k=k,
    filter=metadata_filter
  )