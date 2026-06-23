from __future__ import annotations
import os

# Load raw legal docs
from src.ingestion.loader import load_documents, load_relationships

# Clean HTML/legal text
from src.ingestion.cleaner import clean_documents

# Chunk legal docs
from src.ingestion.chunker import chunk_documents

# Dense vector store
from src.indexing.chroma_store import ChromaStore

# BM25 lexical index
from src.indexing.bm25_index import BM25Index

# Build legal graph
from src.retrieval.graph import build_graph

# Unified retriever
from src.retrieval.retriever import Retriever

# Answer generator
from src.generation.answer import AnswerGenerator

# LLM client (ví dụ OpenAI-compatible)
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

config = {
    "dataset": {
        "name": "th1nhng0/vietnamese-legal-documents"
    },
    "chunking": {
        "chunk_size": 1200,
        "chunk_overlap": 200
    }
}


def main():
    """
    End-to-end pipeline test.
    """

    

    print("=" * 60)
    print("STEP 1: LOAD DOCUMENTS")
    print("=" * 60)

    # chỉ lấy sample nhỏ để test nhanh
    docs = load_documents(config=config, sample_size=20)

    print(f"Loaded {len(docs)} documents")

    print("=" * 60)
    print("STEP 2: CLEAN DOCUMENTS")
    print("=" * 60)

    cleaned_docs = clean_documents(docs)

    print(f"Cleaned {len(cleaned_docs)} documents")

    print("=" * 60)
    print("STEP 3: CHUNK DOCUMENTS")
    print("=" * 60)

    chunks = chunk_documents(cleaned_docs, config=config)

    print(f"Created {len(chunks)} chunks")

    print("=" * 60)
    print("STEP 4: BUILD CHROMA")
    print("=" * 60)

    store = ChromaStore()
    store.build(chunks)

    print("Chroma built successfully")

    print("=" * 60)
    print("STEP 5: BUILD BM25")
    print("=" * 60)

    bm25 = BM25Index()
    bm25.build(chunks)

    print("BM25 built successfully")

    print("=" * 60)
    print("STEP 6: BUILD GRAPH")
    print("=" * 60)

    # TODO:
    # thay bằng relationship loader thật của bạn
    relationships = load_relationships(config=config, sample_size=20)

    graph = build_graph(relationships)

    print(
        f"Graph nodes: {graph.number_of_nodes()}, "
        f"edges: {graph.number_of_edges()}"
    )

    print("=" * 60)
    print("STEP 7: CREATE RETRIEVER")
    print("=" * 60)

    retriever = Retriever(
        store=store,
        bm25_index=bm25,
        graph=graph
    )

    print("Retriever ready")

    print("=" * 60)
    print("STEP 8: CREATE LLM")
    print("=" * 60)

    # dùng provider bạn đang dùng
    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY")
    )

    class SimpleLLM:
        """
        Adapter để unify interface với .invoke()
        """

        def invoke(self, prompt: str):
            response = client.chat.completions.create(
                model="z-ai/glm-5.2-free",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            class Result:
                content = response.choices[0].message.content

            return Result()

    llm = SimpleLLM()

    print("LLM ready")

    print("=" * 60)
    print("STEP 9: CREATE ANSWER GENERATOR")
    print("=" * 60)

    generator = AnswerGenerator(
        retriever=retriever,
        llm=llm
    )

    print("Answer generator ready")

    print("=" * 60)
    print("STEP 10: QUERY")
    print("=" * 60)

    result = generator.generate(
        query="Điều kiện chuyển nhượng quyền sử dụng đất là gì?",
        strategy="hybrid",
        k=5
    )

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSOURCES:\n")

    for doc in result["documents"]:
        print(doc.metadata)


if __name__ == "__main__":
    main()