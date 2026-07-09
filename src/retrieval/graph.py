from __future__ import annotations
from langchain_core.documents import Document

import networkx as nx
from typing import Iterable

from .dense import dense_search


REL_MAP = {
    # 1. Nhóm Căn cứ / Tham chiếu (Base & Reference)
    "Văn bản căn cứ": ("bases_on", False),         # A lấy B làm gốc (A -> bases_on -> B)
    "Văn bản dẫn chiếu": ("cites", False),         # A nhắc đến B (A -> cites -> B)

    # 2. Nhóm Sửa đổi / Bổ sung (Modify)
    "Văn bản sửa đổi": ("amends", False),          # A sửa B
    "Văn bản được sửa đổi": ("amends", True),      # B sửa A (Đúng: Văn bản bị sửa đổi)
    
    "Văn bản bổ sung": ("supplements", False),     # A thêm vào B
    "Văn bản được bổ sung": ("supplements", True), # B thêm vào A

    # 3. Nhóm Chấm dứt hiệu lực (Terminate - ĐÃ SỬA LOGIC)
    "Văn bản hết hiệu lực": ("invalidates", False),              # A làm B chết
    "Văn bản quy định hết hiệu lực": ("invalidates", True),      # B làm A chết (Phải là True)

    "Văn bản bị hết hiệu lực 1 phần": ("partially_invalidates", False), # A cắt 1 phần B (False)
    "Văn bản quy định hết hiệu lực 1 phần": ("partially_invalidates", True), # B cắt 1 phần A (True)

    # (Bổ sung dự phòng 2 nhãn rất hay gặp trên VBPL)
    "Văn bản thay thế": ("replaces", False),
    "Văn bản bị thay thế": ("replaces", True),
    "Văn bản bãi bỏ": ("repeals", False),
    "Văn bản bị bãi bỏ": ("repeals", True),

    # 4. Nhóm Hướng dẫn / Thi hành (Guide/Detail)
    "Văn bản HD, QĐ chi tiết": ("guides", False),       # A hướng dẫn B (Dùng 'guides' chuẩn hơn 'details')
    "Văn bản được HD, QĐ chi tiết": ("guides", True),   # B hướng dẫn A

    # 5. Nhóm Đình chỉ (Suspend - ĐÃ SỬA LOGIC)
    "Văn bản đình chỉ": ("suspends", False),            # A tạm dừng B
    "Văn bản bị đình chỉ": ("suspends", True),          # B tạm dừng A
    
    "Văn bản đình chỉ 1 phần": ("partially_suspends", False),   # A dừng 1 phần B
    "Văn bản bị đình chỉ 1 phần": ("partially_suspends", True), # B dừng 1 phần A

    # 6. Nhóm Khác
    "Văn bản liên quan khác": ("related_to", False),
}

def normalize_relation(rel: str) -> tuple[str, bool]:
  """Normalize relation name to a standard form."""
  if not rel:
    return ("unknown", False)
  
  rel = rel.strip()
  return REL_MAP.get(rel, ("unknown", False))


def expand_neighbors(driver, seed_ids: list[str], max_hops: int = 2) -> set[str]:
    if not seed_ids:
        return set()
    # Cyber query
    query = f"""
        MATCH (seed:Document)
        WHERE seed.doc_id IN $seed_ids
        MATCH (seed)-[*1..{max_hops}]-(neighbor:Document)
        RETURN DISTINCT neighbor.doc_id AS id
        """
    with driver.session() as session:
        result = session.run(query, seed_ids=list(seed_ids))
        expanded = {record["id"] for record in result}

    expanded.update(seed_ids)
    return expanded

def graph_search(
  store,
  neo4j_driver,
  query: str,
  k: int = 5,
  initial_k: int = 3,
  max_hops: int = 2,
  bm25_index=None,
) -> list[Document]:
  """
  Hybrid search:
  dense/hybrid retrieval -> graph expansion -> document aggregation
  """

  # Step 1: Tìm seed docs (ưu tiên hybrid để bắt keyword chính xác như "93493")
  if bm25_index:
    from .hybrid import hybrid_search
    seed_docs = hybrid_search(store, bm25_index, query, k=initial_k)
  else:
    seed_docs = dense_search(store, query, k=initial_k)

  # Step 2: Extract seed doc_ids
  seed_doc_ids = {
    doc.metadata["doc_id"] for doc in seed_docs if "doc_id" in doc.metadata
  }

  # Step 3: Expand graph neighbors
  expanded_doc_ids = expand_neighbors(neo4j_driver, seed_doc_ids, max_hops=max_hops)

  # Step 4: Retrieve related chunks
  all_docs = []

  for doc_id in expanded_doc_ids:
    docs = dense_search(
      store,
      query,
      k=k,
      metadata_filter={"doc_id": doc_id}
    )
    all_docs.extend(docs)

  # Step 5: Deduplicate chunks
  seen = set()
  unique_docs = []

  for doc in all_docs:
    chunk_key = (
      doc.metadata.get("doc_id", ""),
      doc.metadata.get("chunk_index", 0)
    )
    if chunk_key not in seen:
      seen.add(chunk_key)
      unique_docs.append(doc)
  return unique_docs[:k]



