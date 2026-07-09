from langchain_core.tools import tool
from langchain_core.documents import Document

from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.graph import graph_search

# Convert to str for reading easy (LLM)
def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No documents found."
    parts = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id", "unknown")
        title = doc.metadata.get("title", "")
        chunk_idx = doc.metadata.get("chunk_index", -1)

        parts.append(
            f"[van ban: {doc_id} | {title} | chunk {chunk_idx}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)

# Factory pattern
def create_retrieval_tools(store, bm25_index, graph):
    @tool
    def aggregate_search(question: str, k: int = 5) -> str:
        """
        Tìm kiếm tổng hợp tài liệu pháp luật (kết hợp cả từ khóa và ngữ nghĩa).
        ĐÂY LÀ CÔNG CỤ MẶC ĐỊNH. Dùng cho mọi câu hỏi về khái niệm, quy định,
        mức phạt, hoặc khi không chắc chắn nên dùng công cụ nào.
        """

        docs = hybrid_search(store=store, query=question,bm25_index=bm25_index, k=k)
        return _format_docs(docs)

    @tool
    def keyword_search(keyword: str, k: int = 5) -> str:
        """
        Tìm kiếm chính xác tài liệu pháp luật theo từ khóa / số hiệu.
        CHỈ DÙNG khi user hỏi ĐÍCH DANH số hiệu văn bản (Ví dụ: 'Nghị định 105', 'Luật Doanh nghiệp').
        """
        docs = bm25_index.search(query=keyword, k=k)
        return _format_docs(docs)
    @tool
    def related_document_search(question: str, k: int = 5) -> str:
        """
        Tìm kiếm văn bản có mối quan hệ (sửa đổi, thay thế, hướng dẫn thi hành).
        Dùng khi hỏi "Luật nào sửa đổi Nghị định X?".
        """
        docs = graph_search(store=store, query=question, k=k, bm25_index=bm25_index, neo4j_driver=graph)
        return _format_docs(docs)

    return [aggregate_search, keyword_search, related_document_search]
