# Agentic RAG System

## Overview
This project implements an Agentic Retrieval-Augmented Generation (RAG) system designed for processing Vietnamese legal documents. It utilizes a multi-strategy retrieval approach, orchestrated by an autonomous agent workflow, to answer queries based on context extracted from a vector database, a keyword index, and a knowledge graph.

## Architecture

The system is built on a modular architecture encompassing the following components:

- **Agent Orchestrator (LangGraph):** Manages the workflow state, starting from query analysis, tool selection, retrieval execution, answer generation, and result reflection. It incorporates a retry mechanism if the initial answer is deemed insufficient based on the LLM evaluation.
- **Large Language Models (LLM):** Utilizes OpenAI-compatible APIs for query analysis, tool calling, answer generation, and reflection. The implementation supports both standard models and reasoning models.
- **Retrieval Engine:** Implements three distinct retrieval strategies:
  - **BM25 (Keyword Search):** Optimized for exact matches, such as specific document numbers or titles.
  - **ChromaDB (Vector Search):** Used for semantic search and conceptual queries based on document embeddings.
  - **Neo4j (Graph Search):** Utilized for finding relationships between documents, such as amendments and replacements.
  - **Hybrid Search:** Combines BM25 and Vector Search for comprehensive results.
- **API Server (FastAPI):** Exposes a RESTful endpoint for chat interactions and serves the static frontend files.
- **Frontend UI (HTML/CSS/JS):** A web interface for user interaction with the system.

## Project Structure

- `configs/`: Configuration files (e.g., `config.yml`, `setting.py`).
- `data/`: Local storage for ChromaDB vectors, the BM25 index, and Neo4j volume mounts.
- `scripts/`: Data ingestion scripts for various storage backends (`ingest.py`, `ingest_neo4j.py`, `run_benchmark.py`).
- `src/`: Core application source code.
  - `agents/`: LangGraph orchestrator and state definitions.
  - `api/`: FastAPI server implementation.
  - `indexing/`: Logic for creating and loading ChromaDB and BM25 indices.
  - `ingestion/`: Data loading and chunking pipelines.
  - `retrieval/`: Implementations of retrieval strategies (dense, sparse, graph, hybrid).
  - `tools/`: LangChain tools wrapping the retrieval functions for agent usage.
  - `utils/`: Helper functions and utilities.
- `ui/`: Static assets for the web frontend.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (required for Neo4j)

## Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables. Create a `.env` file in the root directory with the following variables:
   ```env
   LLM_BASE_URL=<your_llm_base_url>
   LLM_API_KEY=<your_llm_api_key>
   LLM_MODEL=<your_llm_model>
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=admin123
   ```

## Setup and Ingestion

1. Start the Neo4j database using Docker Compose:
   ```bash
   docker-compose up -d
   ```
2. Ingest document data into ChromaDB and the BM25 index:
   ```bash
   python scripts/ingest.py
   ```
3. Ingest document relationships into Neo4j:
   ```bash
   python scripts/ingest_neo4j.py
   ```

## Usage

1. Start the FastAPI server:
   ```bash
   uvicorn src.api.server:app --host 0.0.0.0 --port 8000
   ```
2. Access the web interface by navigating to `http://localhost:8000` in a web browser.
3. The API endpoint for programmatic access is available at `POST /api/chat`.
   ```json
   // Request format
   {
       "query": "Nội dung câu hỏi",
       "history": []
   }
   ```

## Configuration

System parameters can be adjusted in `configs/config.yml`. Configurable options include dataset configuration, chunk size and overlap for text splitting, persist paths for indices, retrieval parameters, and LLM generation settings.
