"""
Debug script - kiem tra tung tang cua pipeline.
Chay: python -m scripts.debug_retrieval
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np

from configs.setting import load_config
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.graph import build_graph, graph_search
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.ingestion.loader import load_relationships
from src.llm import LLMClient


# Query 1: Phù hợp với data đang có trong index (doc 93493 về khiếu nại)
# Dùng để VERIFY pipeline hoạt động khi data có liên quan
QUERY_RELEVANT = "Quy trình tiếp nhận và giải quyết đơn khiếu nại tố cáo là gì?"

# Query 2: Query gốc về đất đai — sẽ fail vì index không có data này
QUERY_LAND = "Điều kiện chuyển nhượng quyền sử dụng đất là gì?"

# Đổi QUERY để test
QUERY = QUERY_RELEVANT

SEP = "=" * 70

config = load_config()


def section(title: str):
    print(f"\n{SEP}\n{title}\n{SEP}")


def main():

    # ----------------------------------------------------------------
    # TANG 1: Co gi trong Chroma index?
    # Muc dich: Xac dinh data coverage.
    # Neu index chi co van ban tai chinh thi retrieval tra ve rac la dung.
    # ----------------------------------------------------------------
    section("TANG 1: KIEM TRA INDEX — CO GI TRONG CHROMA?")

    store = ChromaStore()
    store.load()

    collection = store.store._collection
    total = collection.count()
    print(f"Tong so chunks trong Chroma: {total}")

    sample = collection.get(limit=15, include=["documents", "metadatas"])
    print(f"\nMau 15 chunks dau trong index:")
    print("-" * 50)
    for i, (doc, meta) in enumerate(
        zip(sample["documents"], sample["metadatas"]), 1
    ):
        title = meta.get("title", "?")[:70]
        doc_id = meta.get("doc_id", "?")
        preview = doc[:100].strip().replace("\n", " ")
        print(f"[{i:02d}] doc_id={doc_id} | title={title}")
        print(f"      preview: {preview}")
        print()

    # ----------------------------------------------------------------
    # TANG 2: Dense retrieval tra ve gi?
    # Muc dich: Score cao (voi cosine distance: thap = tot) thi embedding tot.
    # Score > 1.5 thi query khong khop voi bat ky doc nao trong index.
    # ----------------------------------------------------------------
    section(f"TANG 2: DENSE RETRIEVAL\nQuery: {QUERY}")

    dense_docs = dense_search(store=store, query=QUERY, k=5)
    print(f"So docs tra ve: {len(dense_docs)}")
    for i, doc in enumerate(dense_docs, 1):
        doc_id = doc.metadata.get("doc_id", "?")
        title = doc.metadata.get("title", "?")[:60]
        content = doc.page_content[:200].strip().replace("\n", " ")
        print(f"\n[Dense {i}] doc_id={doc_id} | title={title}")
        print(f"  Content: {content}")

    print("\n--- Cosine distance scores (thap hon = lien quan hon) ---")
    scored = store.store.similarity_search_with_score(QUERY, k=5)
    for doc, score in scored:
        doc_id = doc.metadata.get("doc_id", "?")
        preview = doc.page_content[:80].strip().replace("\n", " ")
        flag = " <-- GOOD" if score < 0.5 else (" <-- OK" if score < 1.0 else " <-- IRRELEVANT")
        print(f"  score={score:.4f}{flag} | {doc_id} | {preview}")

    # ----------------------------------------------------------------
    # TANG 3: BM25 keyword search
    # Muc dich: Keyword "chuyen nhuong", "quyen su dung dat" co trong index?
    # Score = 0 het thi tokenizer qua don gian hoac khong co tu khoa khop.
    # ----------------------------------------------------------------
    section("TANG 3: BM25 RETRIEVAL")

    bm25 = BM25Index()
    bm25.load()

    print(f"Tong chunks trong BM25: {len(bm25.documents)}")

    tokenized = bm25.tokenize(QUERY)
    print(f"Query sau khi tokenize: {tokenized}")

    scores = bm25.bm25.get_scores(tokenized)
    top_indices = np.argsort(scores)[::-1][:5]

    print("\nTop 5 BM25 results:")
    for rank, idx in enumerate(top_indices, 1):
        doc_id = bm25.documents[idx].metadata.get("doc_id", "?")
        score = scores[idx]
        preview = bm25.documents[idx].page_content[:100].strip().replace("\n", " ")
        flag = " <-- ZERO (khong khop tu khoa nao)" if score == 0 else ""
        print(f"  [{rank}] score={score:.4f}{flag} | {doc_id}")
        print(f"       {preview}")

    # ----------------------------------------------------------------
    # TANG 4: Graph retrieval
    # Muc dich: Sau khi expand tu seed docs, co tim them duoc gi?
    # ----------------------------------------------------------------
    section("TANG 4: GRAPH RETRIEVAL")

    relationships = load_relationships(config=config, sample_size=100)
    graph = build_graph(relationships)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Xem seed docs truoc khi expand
    seed_docs = dense_search(store=store, query=QUERY, k=3)
    seed_ids = {d.metadata.get("doc_id") for d in seed_docs if "doc_id" in d.metadata}
    print(f"Seed doc_ids tu dense: {seed_ids}")

    # Xem graph neighbors
    from src.retrieval.graph import expand_neighbors
    expanded = expand_neighbors(graph, seed_ids, max_hops=2)
    new_ids = expanded - seed_ids
    print(f"Expanded them {len(new_ids)} doc_ids qua graph: {new_ids}")

    graph_docs = graph_search(store=store, graph=graph, query=QUERY, k=5)
    print(f"\nGraph tra ve {len(graph_docs)} docs:")
    for i, doc in enumerate(graph_docs, 1):
        doc_id = doc.metadata.get("doc_id", "?")
        title = doc.metadata.get("title", "?")[:60]
        content = doc.page_content[:150].strip().replace("\n", " ")
        print(f"  [{i}] {doc_id} | {title}")
        print(f"       {content}")

    # ----------------------------------------------------------------
    # TANG 5: LLM JSON parsing
    # Muc dich: invoke_json co stable khong?
    # ----------------------------------------------------------------
    section("TANG 5: LLM JSON PARSING")

    llm = LLMClient.from_config(config)

    test_prompt = (
        'Chon strategy cho query: "Dieu kien chuyen nhuong dat la gi?"\n'
        'Tra ve JSON: {"strategy": "dense" hoac "hybrid" hoac "graph", "reason": "ly do"}'
    )

    raw = llm.invoke_json(test_prompt)
    print(f"Raw output tu LLM:\n{raw}\n")

    try:
        parsed = json.loads(raw)
        print(f"Parse OK:")
        print(f"  strategy = {parsed.get('strategy')}")
        print(f"  reason   = {parsed.get('reason', '')[:100]}")
    except Exception as e:
        print(f"Parse FAIL: {e}")

    # ----------------------------------------------------------------
    # CHAN DOAN
    # ----------------------------------------------------------------
    section("CHAN DOAN CUOI CUNG")

    print(f"Tong chunks: {total}")
    print(f"Dense score tot nhat: {scored[0][1]:.4f}" if scored else "N/A")
    print(f"BM25 score tot nhat: {scores[top_indices[0]]:.4f}" if len(top_indices) > 0 else "N/A")
    print(f"Graph mo rong duoc: {len(new_ids)} docs moi")
    print()

    if total < 100:
        print("[!] INDEX QUA NHO — Chi co", total, "chunks.")
        print("    -> Indexing voi sample_size=20 la khong du.")
        print("    -> FIX: Tang sample_size len >= 500 hoac index toan bo dataset.")

    best_score = scored[0][1] if scored else 2.0
    if best_score > 1.0:
        print("[!] DENSE SCORE QUA CAO (", round(best_score, 4), ")")
        print("    -> Index khong co van ban lien quan den query nay.")
        print("    -> Dù retrieval chay dung, no chi tra ve 'gan nhat' chu khong phai 'lien quan'.")
        print("    -> Day la van de DATA, khong phai van de CODE.")

    best_bm25 = scores[top_indices[0]] if len(top_indices) > 0 else 0
    if best_bm25 == 0:
        print("[!] BM25 SCORE = 0 het")
        print("    -> Khong co tu khoa nao trong query khop voi index.")
        print("    -> Tokenizer chi split space — co the mat stopword hoac bo dau.")


if __name__ == "__main__":
    main()
