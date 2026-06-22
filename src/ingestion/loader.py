# from typing import List, Optional
# from datasets import load_dataset
# from langchain_core.documents import Document


# def build_metadata_lookup(metadata_rows):
#   """
#   Convert metadata rows into 0(1) lookup dictionary
#   """
#   lookup = {}

#   for row in metadata_rows:
#     doc_id = row.get("id")
#     if doc_id:
#       lookup[doc_id] = row

#   return lookup

# def load_documents(
#     config,
#     sample_size: Optional[int] = None,
# ) -> List[Document]:
#   """"
#   Load legal documents from HF dataset

#   Responsibilities:
#   - Load content split
#   - Load metadata split
#   - Join by doc_id
#   - Convert into LangChain Document
#   """

#   dataset_name = config["dataset"]["name"]

#   # Load content split
#   content_ds = load_dataset(dataset_name,"content")["data"]

#   # Load metadata split
#   metadata_ds = load_dataset(dataset_name, "metadata")["data"]

#   if sample_size:
#     content_ds = content_ds.select(range(sample_size))

#   metadata_lookup = build_metadata_lookup(metadata_ds)

#   docs = []
#   skipped = 0

#   for row in content_ds:
#     doc_id = row.get("id")

#     if doc_id is None:
#       skipped += 1
#       continue

#     content_html = row.get("content_html", "")

#     if not content_html:
#       skipped += 1
#       continue

#     meta = metadata_lookup.get(doc_id, {})

#     docs.append(
#       Document(
#         page_content=content_html,
#         metadata={
#           "doc_id": doc_id,
#           "title": meta.get("title",""),
#           "doc_type": meta.get("loai_van_ban",""),
#           "authority": meta.get("co_quan_ban_hanh",""),
#           "issue_date": meta.get("ngay_ban_hanh",""),
#           "effective_date": meta.get("ngay_co_hieu_luc",""),
#           "status": meta.get("tinh_trang_hieu_luc",""),
#         }
#       )
#     )
#   print(f"Loaded {len(docs)} documents")
#   print(f"Skipped {skipped} invalid rows")

#   return docs


import pandas as pd
from typing import List, Optional
from datasets import load_dataset
from langchain_core.documents import Document

def load_documents(
    config: dict,
    sample_size: Optional[int] = None,
) -> List[Document]:
    """
    Tải tài liệu bằng chiến lược kết hợp: 
    HF Datasets (Metadata) + Pandas (Content Parquet) để né lỗi PyArrow.
    """
    dataset_name = config["dataset"]["name"]

    print("--> 1. Đang tải metadata (Hugging Face Datasets)...")
    metadata_ds = load_dataset(dataset_name, "metadata", split="data")
    
    # Ép doc_id về dạng string để đảm bảo map chính xác
    metadata_lookup = {}
    for row in metadata_ds:
        doc_id = row.get("id")
        if doc_id is not None:
            metadata_lookup[str(doc_id)] = row

    print("--> 2. Đang tải nội dung trực tiếp bằng Pandas...")
    # Đường dẫn file Parquet gốc trên Hugging Face Hub
    parquet_url = f"hf://datasets/{dataset_name}/data/content.parquet"
    
    # Pandas đọc trực tiếp file parquet sẽ không bị lỗi ép kiểu 32-bit
    content_df = pd.read_parquet(parquet_url)

    if sample_size:
        content_df = content_df.sample(n=sample_size, random_state=42)

    docs = []
    skipped = 0

    print("--> 3. Đang xử lý tài liệu và chuyển thành LangChain Documents...")
    for _, row in content_df.iterrows():
        doc_id = row.get("id")
        content_html = row.get("content_html")

        # Kiểm tra None hoặc NaN trong Pandas
        if pd.isna(doc_id) or pd.isna(content_html) or not str(content_html).strip():
            skipped += 1
            continue

        doc_id_str = str(doc_id)
        meta = metadata_lookup.get(doc_id_str, {})

        docs.append(
            Document(
                page_content=str(content_html),
                metadata={
                    "doc_id": doc_id_str,
                    "title": meta.get("title", ""),
                    "doc_type": meta.get("loai_van_ban", ""),
                    "authority": meta.get("co_quan_ban_hanh", ""),
                    "issue_date": meta.get("ngay_ban_hanh", ""),
                    "effective_date": meta.get("ngay_co_hieu_luc", ""),
                    "status": meta.get("tinh_trang_hieu_luc", ""),
                }
            )
        )
        
    print(f"[Thành công] Đã tải {len(docs)} tài liệu.")
    print(f"[Bỏ qua] Đã bỏ qua {skipped} dòng không hợp lệ.")

    return docs



def load_relationships(
    config: dict,
    sample_size: Optional[int] = None,
) -> list[dict]:
    """
    Load quan hệ giữa các văn bản pháp luật.
    """

    dataset_name = config["dataset"]["name"]

    print("--> Đang tải relationships...")

    relationship_ds = load_dataset(
        dataset_name,
        "relationships",
        split="data"
    )

    if sample_size:
        relationship_ds = relationship_ds.select(
            range(sample_size)
        )

    relationships = []

    for row in relationship_ds:
        doc_id = row.get("doc_id")
        other_doc_id = row.get("other_doc_id")
        relationship = row.get("relationship")

        if not doc_id or not other_doc_id:
            continue

        relationships.append({
            "doc_id": str(doc_id),
            "other_doc_id": str(other_doc_id),
            "relationship": relationship
        })

    print(f"[Thành công] Loaded {len(relationships)} relationships")

    return relationships


