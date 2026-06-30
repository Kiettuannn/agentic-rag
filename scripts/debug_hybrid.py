import sys
sys.stdout.reconfigure(encoding='utf-8')
from configs.setting import load_config
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.hybrid import hybrid_search
import logging

# Tắt log rác
logging.getLogger("httpx").setLevel(logging.WARNING)

config = load_config()

print("Loading indexes...")
store = ChromaStore()
store.load()
bm25 = BM25Index()
bm25.load()

# Lấy tất cả doc_ids hiện có trong Chroma
collection = store.store._collection
meta = collection.get(include=["metadatas"])
doc_ids = {m.get("doc_id") for m in meta["metadatas"] if m.get("doc_id")}

print(f"\nTổng số doc_ids trong Chroma: {len(doc_ids)}")
print(f"Quyết định 93493 có trong DB không? {'93493' in doc_ids}")

# Thử một truy vấn Hybrid
query = "Quyết định 93493 có liên quan hay dẫn chiếu đến những văn bản pháp luật nào khác không?"
print(f"\nThử chạy Hybrid Search với câu hỏi: '{query}'")
docs = hybrid_search(store, bm25, query, k=3)

for i, doc in enumerate(docs):
    print(f"\n[Result {i+1}] doc_id={doc.metadata.get('doc_id')}")
    print(f"Content snippet: {doc.page_content[:150]}...")
