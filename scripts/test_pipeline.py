from __future__ import annotations
import os

# Load raw legal docs
# from src.ingestion.loader import load_documents, load_relationships

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

from configs.setting import load_config, get_llm_settings
from src.llm import LLMClient
from src.agents.orchestrator import RAGOrchestrator


# Load config
config = load_config()


def main():

    print("=" * 60)
    print("STEP 1: LOAD DOCUMENTS")
    print("=" * 60)

    docs = load_documents(config=config, sample_size=100)  # Load a sample of 10 documents for testing

    print(f"Loaded {len(docs)} documents")

    print("=" * 60)
    print("STEP 2: CLEAN DOCUMENTS")
    print("=" * 60)

    # num_workers = max(1, os.cpu_count() - 2)

    cleaned_docs = clean_documents(docs, workers=2)

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
    bm25.save()  # Save BM25 index to disk

    print("BM25 built successfully")

    print("=" * 60)
    print("STEP 6: BUILD GRAPH")
    print("=" * 60)

    # TODO:
    # thay bằng relationship loader thật của bạn
    relationships = load_relationships(config=config,sample_size=100)

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

    llm = LLMClient.from_config(config)
    orchestrator = RAGOrchestrator(retriever=retriever, llm=llm)

    print("=" * 60)
    print("STEP 9: TEST ORCHESTRATOR")
    print("=" * 60)

    result = orchestrator.run("Đối tượng nộp thuế?")
    print(f"\nStrategy chọn: {result['strategy']}")
    print(f"Lý do: {result['strategy_reason']}")
    print(f"Retry count: {result['retry_count']}")
    print(f"Reflection: {result['reflection']}")
    print(f"\nANSWER:\n{result['final_answer']}")

if __name__ == "__main__":
    main()