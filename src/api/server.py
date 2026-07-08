import os
import sys

from src.utils.helpers import extract_sources

# Tat log rac va sua loi font tieng viet tren terminal windowns
sys.stdout.reconfigure(encoding='utf-8')
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_core.messages import AIMessage, ToolMessage

from configs.setting import load_config
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index
from src.ingestion.loader import load_relationships
from src.retrieval.graph import build_graph
from src.agents.orchestrator import RAGOrchestrator
from src.llm import LLMClient
from src.retrieval.retriever import Retriever
from src.llm import create_langchain_llm
from src.tools.retrieval_tools import create_retrieval_tools

class ChatRequest(BaseModel):
  query: str
  history: list = []


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
  llm = LLMClient.from_config(config)

  # Khởi tạo ChatOpenAI mới và Tools
  langchain_llm = create_langchain_llm(config)
  tools = create_retrieval_tools(store=store, bm25_index=bm25, graph=graph)

  # Khoi tao Orchestrator
  global_orchestrator = RAGOrchestrator(
    retriever=retriever,
    llm=llm,
    langchain_llm=langchain_llm,
    tools=tools
  )
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
  result = global_orchestrator.run(request.query, history=request.history)

  # Trích xuất Agent Logs (Quá trình gọi Tool)
  agent_logs = []
  messages = result.get("messages", [])
  for msg in messages:
      if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
          for tc in msg.tool_calls:
              # Format tham số arguments cho đẹp
              args_str = ", ".join([f"{k}='{v}'" for k, v in tc["args"].items()])
              agent_logs.append(f"> Đang gọi công cụ: {tc['name']}({args_str})")
      elif isinstance(msg, ToolMessage):
          doc_len = len(msg.content)
          agent_logs.append(f"  └─ Đã nhận kết quả ({doc_len} ký tự).")

  return {
    "answer": result.get("final_answer") or "",
    "strategy": result.get("strategy") or "",
    "strategy_reason": result.get("strategy_reason") or "",
    "reflection_reason": result.get("reflection_reason") or "",
    "thought_process": result.get("thought_process") or [],
    "retry_count": result.get("retry_count") or 0,
    "search_query": result.get("search_query") or request.query,
    "sources": extract_sources(result.get("documents") or []),
    "agent_logs": agent_logs
  }

os.makedirs("ui", exist_ok=True)
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")