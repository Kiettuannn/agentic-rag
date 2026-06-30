import sys
import shutil
import os
sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from configs.setting import load_config
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.graph import build_graph
import networkx as nx

config = load_config()

from src.ingestion.loader import load_documents, load_relationships
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents

# 1. TẠO 3 VĂN BẢN MOCK ĐỂ TEST TRONG ĐÁM NHIỄU
mock_docs = [
    Document(
        page_content="Quyết định 100/QĐ-UBND quy định mức hỗ trợ tiền điện cho sinh viên có hoàn cảnh khó khăn trên địa bàn thành phố. Theo đó, mỗi sinh viên sẽ được hỗ trợ mức cố định là 50.000 đồng/tháng. Nguồn tiền trích từ ngân sách thành phố.",
        metadata={"doc_id": "QD100", "title": "Quyết định 100/QĐ-UBND hỗ trợ tiền điện"}
    ),
    Document(
        page_content="Quyết định 105/QĐ-UBND ban hành nhằm cập nhật mức chi ngân sách. Cụ thể, do vật giá leo thang, mức hỗ trợ tiền điện cho sinh viên sẽ được điều chỉnh tăng lên 70.000 đồng/tháng. Các nội dung khác giữ nguyên.",
        metadata={"doc_id": "QD105", "title": "Quyết định 105/QĐ-UBND tăng mức hỗ trợ"}
    ),
    Document(
        page_content="Thông tư 01/TT-BTC quy định chi tiết quy trình nộp hồ sơ nhận tiền hỗ trợ. Sinh viên muốn nhận hỗ trợ phải nộp hồ sơ gồm: Thẻ sinh viên photo và Giấy xác nhận hộ nghèo do UBND xã/phường cấp. Hồ sơ nộp về Phòng Công tác sinh viên.",
        metadata={"doc_id": "TT01", "title": "Thông tư 01/TT-BTC hướng dẫn hồ sơ"}
    ),
]

mock_rels = [
    {"doc_id": "QD105", "relationship": "Văn bản sửa đổi", "other_doc_id": "QD100"},
    {"doc_id": "TT01", "relationship": "Văn bản HD, QĐ chi tiết", "other_doc_id": "QD100"}
]

def main():
    print("="*50)
    print("INGEST MIXED DATA (100 REAL DOCS + 3 MOCK DOCS)")
    print("="*50)

    # Lấy 100 văn bản từ Dataset thực tế làm "nhiễu"
    print("Đang tải 100 văn bản từ Dataset thực tế...")
    real_docs = load_documents(config=config, sample_size=100)
    cleaned_real_docs = clean_documents(real_docs, workers=2)
    chunked_real_docs = chunk_documents(cleaned_real_docs, config=config)

    # Chuyển đổi 3 mock docs thành các chunk (giả lập 1 doc = 1 chunk luôn cho nhanh)
    for doc in mock_docs:
        doc.metadata["chunk_index"] = 0
    
    # Gộp chung
    final_chunks = chunked_real_docs + mock_docs
    print(f"Tổng số chunks (Real + Mock): {len(final_chunks)}")

    # Xử lý Relationships: Lấy Real Rels kết hợp với Mock Rels
    print("Đang tải relationships từ Dataset thực tế...")
    real_rels = load_relationships(config=config)
    
    # Filter rels của 100 văn bản thực tế
    real_doc_ids = {doc.metadata.get("doc_id") for doc in real_docs}
    filtered_real_rels = [
        r for r in real_rels
        if r.get("doc_id") in real_doc_ids or r.get("other_doc_id") in real_doc_ids
    ]
    
    final_rels = filtered_real_rels + mock_rels

    # LƯU DANH SÁCH VĂN BẢN ĐỂ NGƯỜI DÚNG KIỂM TRA
    os.makedirs("data", exist_ok=True)
    with open("data/mock_dataset_docs.txt", "w", encoding="utf-8") as f:
        f.write("DANH SÁCH 3 VĂN BẢN MOCK:\n")
        for doc in mock_docs:
            f.write(f"- [{doc.metadata.get('doc_id')}] {doc.metadata.get('title')}\n")
        
        f.write("\nDANH SÁCH 100 VĂN BẢN THỰC TẾ (NHIỄU):\n")
        for doc in real_docs:
            f.write(f"- [{doc.metadata.get('doc_id')}] {doc.metadata.get('title', 'No Title')}\n")
    print("\n[INFO] Đã xuất danh sách các văn bản vào file: data/mock_dataset_docs.txt")

    # Reset (Xóa) DB cũ để không bị nhiễu quá lố
    chroma_dir = config.get("chroma", {}).get("persist_directory", "data/chroma_db")
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
        print(f"Đã xóa database cũ tại {chroma_dir} để đảm bảo sạch sẽ.")

    # Build Graph
    graph = build_graph(final_rels)
    print(f"Graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")

    # Build BM25
    bm25 = BM25Index()
    bm25.build(final_chunks)
    bm25.save()
    print("BM25 index built and saved.")

    # Build Chroma
    store = ChromaStore()
    store.build(final_chunks)
    print("Chroma index built and saved.")

    print("\nHoàn tất! Hệ thống hiện tại chứa 100 văn bản thực tế + 3 văn bản Mock.")
    print("Bạn có thể sang file test.py và đặt các câu hỏi sau để kiểm tra:")
    print("1. Dense: 'Điều kiện và hồ sơ để nhận tiền hỗ trợ điện là gì?' (Phải ra TT01)")
    print("2. Hybrid: 'Theo Quyết định 105, mức hỗ trợ là bao nhiêu?' (Phải ra 70k, không phải 50k)")
    print("3. Graph: 'Quyết định 100 bị sửa đổi bởi văn bản nào và văn bản nào hướng dẫn nó?'")

if __name__ == "__main__":
    main()
