import os
import sys

# Tat log rac va sua loi font tieng viet tren terminal windowns
sys.stdout.reconfigure(encoding='utf-8')
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager

from configs.setting import load_config
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index
from src.ingestion.loader import load_relationships
from src.retrieval.graph import build_graph
from src.agents.orchestrator import RAGOrchestrator
from src.llm import LLMClient
from src.retrieval.retriever import Retriever
class ChatRequest(BaseModel):
  query: str


# Bien toan cuc luu Orchestrator de dung chung cho cac request
global_orchestrator = None

# Ham Lifecycle: Chay 1 lan khi khoi dong Server
@asynccontextmanager
async def lifespan(app: FastAPI):
  global global_orchestrator
  print("="*50)
  print("Dang khoi dong server va load du lieu ...")
  print("="*50)

  config = load_config()

  store = ChromaStore()
  store.load()
  print("Da tai xong Chroma Vector DB")

  bm25 = BM25Index()
  bm25.load()
  print("Da tai xong BM25 Index")

  relationships = load_relationships(config=config)
  graph = build_graph(relationships)
  print("Da tai xong Knowledgge Graph")

  # Khoi tao Retriever va LLM
  print("Dang khoi tao Retriever va LLM ...")
  retriever = Retriever(store=store, bm25_index=bm25, graph=graph)
  llm = LLMClient().from_config(config)

  # Khoi tao Orchestrator
  global_orchestrator = RAGOrchestrator(retriever=retriever, llm=llm)
  print("SERVER DA SAN SANG TAI CONG 8000")

  yield

  print("Dang tat server ...")


# Khoi tao FastAPI app
app = FastAPI(lifespan=lifespan)

# API Endpoint
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
  if not global_orchestrator:
    return {"error": "Server is not ready yet."}

  print(f"\n[USER]: {request.query}")

  # Goi ham run cua Agent
  result = global_orchestrator.run(request.query)

  return {
    "answer": result.get("final_answer", ""),
    "strategy": result.get("strategy", ""),
    "retry_count": result.get("retry_count", 0),
  }

os.makedirs("ui", exist_ok=True)
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")