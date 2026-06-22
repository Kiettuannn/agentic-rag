from typing import Optional

def dense_search(
    store,
    query: str,
    k: int = 5,
    metadata_filter: Optional[dict] = None,
):
  """
  Basic dense retrieval wrapper
  """
  return store.similarity_search(
    query,
    k=k,
    filter=metadata_filter
  )