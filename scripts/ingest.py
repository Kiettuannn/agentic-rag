
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from configs.setting import load_config
from src.ingestion.loader import load_documents, load_relationships
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.graph import build_graph

import sys
sys.stdout.reconfigure(encoding='utf-8')

SEP = "=" * 60
config = load_config()


def main():

    # ----------------------------------------------------------------
    # STEP 1: Load content docs TRƯỚC
    # ----------------------------------------------------------------
    print(SEP)
    print("STEP 1: LOAD DOCUMENTS")
    print(SEP)

    # sample_size=500: đủ lớn để có diversity, đủ nhỏ để chạy nhanh.
    # Tăng lên nếu muốn coverage rộng hơn.
    all_docs = load_documents(config=config, sample_size=500)
    print(f"Loaded {len(all_docs)} docs")

    content_doc_ids = {doc.metadata.get("doc_id") for doc in all_docs}
    print(f"Unique doc_ids trong content: {len(content_doc_ids)}")

    # ----------------------------------------------------------------
    # STEP 2: Load relationships → filter theo content doc_ids
    # ----------------------------------------------------------------
    print(SEP)
    print("STEP 2: LOAD & FILTER RELATIONSHIPS")
    print(SEP)

    # Load toàn bộ relationships (nhẹ, chỉ là metadata).
    # Filter: chỉ giữ relationships mà ít nhất 1 đầu trong content_doc_ids.
    # Kết quả: graph nhỏ, chỉ chứa docs đã được index → BFS expand đúng mục đích.
    all_relationships = load_relationships(config=config)
    filtered_relationships = [
        r for r in all_relationships
        if r.get("doc_id") in content_doc_ids
        or r.get("other_doc_id") in content_doc_ids
    ]
    print(f"Filtered: {len(filtered_relationships)}/{len(all_relationships)} relationships")

    graph = build_graph(filtered_relationships)
    graph_doc_ids = set(graph.nodes())
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    overlap = content_doc_ids & graph_doc_ids
    print(f"Content-Graph overlap: {len(overlap)} docs")

    final_docs = all_docs


    # ----------------------------------------------------------------
    # STEP 3: Clean
    # ----------------------------------------------------------------
    print(SEP)
    print("STEP 3: CLEAN DOCUMENTS")
    print(SEP)

    cleaned_docs = clean_documents(final_docs, workers=2)
    print(f"Cleaned {len(cleaned_docs)} documents")

    # ----------------------------------------------------------------
    # STEP 4: Chunk
    # ----------------------------------------------------------------
    print(SEP)
    print("STEP 4: CHUNK DOCUMENTS")
    print(SEP)

    chunks = chunk_documents(cleaned_docs, config=config)
    print(f"Created {len(chunks)} chunks")

    # Thống kê doc_ids trong chunks
    chunk_doc_ids = {c.metadata.get("doc_id") for c in chunks}
    overlap = chunk_doc_ids & graph_doc_ids
    print(f"Doc_ids trong chunks: {len(chunk_doc_ids)}")
    print(f"Overlap với graph: {len(overlap)} docs → graph search sẽ expand được!")

    # ----------------------------------------------------------------
    # STEP 5: Build Chroma
    # ----------------------------------------------------------------
    print(SEP)
    print("STEP 5: BUILD CHROMA")
    print(SEP)

    store = ChromaStore()
    store.build(chunks)
    print("Chroma built successfully")

    # ----------------------------------------------------------------
    # STEP 6: Build BM25
    # ----------------------------------------------------------------
    print(SEP)
    print("STEP 6: BUILD BM25")
    print(SEP)

    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save()
    print("BM25 built and saved successfully")

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    print(SEP)
    print("INGESTION COMPLETE")
    print(SEP)
    print(f"  Chunks indexed       : {len(chunks)}")
    print(f"  Unique docs indexed  : {len(chunk_doc_ids)}")
    print(f"  Graph nodes          : {graph.number_of_nodes()}")
    print(f"  Graph-Content overlap: {len(overlap)} docs")
    print()

    if len(overlap) > 0:
        print("[OK] Graph search sẽ hoạt động — có overlap giữa graph và index.")
    else:
        print("[WARN] Graph-Content overlap = 0. Graph search sẽ không expand được.")
        print("       Tăng sample_size hoặc load toàn bộ dataset.")


if __name__ == "__main__":
    main()