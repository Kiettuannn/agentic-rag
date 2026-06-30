
# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# from src.ingestion.loader import load_documents
# from src.ingestion.cleaner import clean_documents
# from src.ingestion.chunker import chunk_documents

# config = {
#     "dataset": {
#         "name": "th1nhng0/vietnamese-legal-documents"
#     },
#     "chunking": {
#         "chunk_size": 1200,
#         "chunk_overlap": 200
#     }
# }

# # docs = load_documents(config, sample_size=10)

# # print("=" * 80)
# # print("TOTAL DOCS:", len(docs))

# # for i, doc in enumerate(docs):
# #     print(f"\nDOCUMENT {i+1}")
# #     print("-" * 80)

# #     print("METADATA:")
# #     print(doc.metadata)

# #     print("\nCONTENT PREVIEW:")
# #     print(doc.page_content[:500])




# # docs = load_documents(config, sample_size=2)

# # print("\nBEFORE CLEANING:")
# # print(docs[0].page_content[:500])

# # cleaned_docs = clean_documents(docs)

# # print("\nAFTER CLEANING:")
# # print(cleaned_docs[0].page_content[:500])


# # Pipeline
# docs = load_documents(config, sample_size=2)
# cleaned_docs = clean_documents(docs)
# chunks = chunk_documents(cleaned_docs, config)

# # Create logs folder
# log_dir = Path("logs")
# log_dir.mkdir(exist_ok=True)

# log_file = log_dir / "chunk_output.txt"

# with open(log_file, "w", encoding="utf-8") as f:
#     f.write(f"TOTAL CHUNKS: {len(chunks)}\n")
#     f.write("=" * 100 + "\n")

#     for i, chunk in enumerate(chunks):
#         f.write(f"\nCHUNK {i+1}\n")
#         f.write("-" * 100 + "\n")

#         f.write("METADATA:\n")
#         f.write(f"{chunk.metadata}\n\n")

#         f.write("CONTENT:\n")
#         f.write(chunk.page_content)
#         f.write("\n\n")
#         f.write("=" * 100 + "\n")

# print(f"Chunk logs saved to: {log_file}")



# from src.llm import LLMClient

# def main():
#     llm = LLMClient()

#     print("Base URL:", llm.base_url)
#     print("Model:", llm.model)
#     print("API key exists:", bool(llm.api_key))

#     print(llm.invoke("hello").content)

# if __name__ == "__main__":
#     main()


from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

from configs.setting import load_config

from src.llm import LLMClient
from src.agents.orchestrator import RAGOrchestrator

from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.graph import build_graph
from src.retrieval.retriever import Retriever

from src.ingestion.loader import load_relationships


config = load_config()


def main():
    print("=" * 60)
    print("LOAD PERSISTED INDEXES")
    print("=" * 60)

    # Load Chroma từ disk
    store = ChromaStore()
    store.load()

    print("Chroma loaded")

    # Load BM25 từ disk
    bm25 = BM25Index()
    bm25.load()

    print("BM25 loaded")

    # Lấy tập doc_ids đã được index trong Chroma
    collection = store.store._collection
    all_meta = collection.get(include=["metadatas"])
    indexed_doc_ids = {m.get("doc_id") for m in all_meta["metadatas"] if m.get("doc_id")}
    print(f"Indexed doc_ids: {len(indexed_doc_ids)} unique docs trong Chroma")

    # Load toàn bộ relationships rồi filter.
    # Chỉ giữ relationships có ít nhất 1 đầu trong indexed_doc_ids.
    # Tránh build graph 153k nodes từ 897k relationships không cần thiết.
    all_relationships = load_relationships(config=config)
    filtered_relationships = [
        r for r in all_relationships
        if r.get("doc_id") in indexed_doc_ids
        or r.get("other_doc_id") in indexed_doc_ids
    ]
    print(f"Filtered: {len(filtered_relationships)}/{len(all_relationships)} relationships liên quan đến indexed docs")

    graph = build_graph(filtered_relationships)

    print(
        f"Graph nodes: {graph.number_of_nodes()}, "
        f"edges: {graph.number_of_edges()}"
    )


    retriever = Retriever(
        store=store,
        bm25_index=bm25,
        graph=graph
    )

    llm = LLMClient.from_config(config)

    orchestrator = RAGOrchestrator(
        retriever=retriever,
        llm=llm
    )

    print("=" * 60)
    print("TEST ORCHESTRATOR ONLY")
    print("=" * 60)

    # Dùng query liên quan đến data đã index để test end-to-end
    # Sau khi pipeline OK, thay bằng bất kỳ câu hỏi pháp lý nào
    # CÂU HỎI KHÓ ĐỂ ÉP RETRY
    # Cố tình dùng ngôn ngữ hỏi về hồ sơ của Quyết định 100. 
    # Nhưng nội dung QĐ 100 không hề chứa hồ sơ (mà nằm ở TT01).
    # LLM sẽ tìm Dense/Hybrid -> Ra QĐ100 -> Thấy thiếu thông tin -> Báo Insufficient -> Retry sang Graph.
    result = orchestrator.run(
        "Tôi là sinh viên nghèo. Cho tôi hỏi theo cập nhật mới nhất hiện nay thì mức hỗ trợ tiền điện sinh viên của Quyết định 100 là bao nhiêu tiền, và tôi phải nộp những loại giấy tờ gì?"
    )

    print(f"\nStrategy: {result['strategy']}")
    print(f"Reason: {result['strategy_reason']}")
    print(f"Retry: {result['retry_count']}")
    print(f"Reflection: {result['reflection']}")
    print(f"\nANSWER:\n{result['final_answer']}")


if __name__ == "__main__":
    main()


