from __future__ import annotations

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str = "configs/config.yml") -> dict:
  path = Path(config_path)

  if not path.exists():
    raise FileNotFoundError(f"Config file not found at {config_path}")
  with open(path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
  
  return config


def get_llm_settings() -> dict:
  return {
    "base_url": os.getenv("LLM_BASE_URL"),
    "api_key": os.getenv("LLM_API_KEY"),
    "model": os.getenv("LLM_MODEL", "z-ai/glm-5.2-free"),
  }

def get_paths(config: dict) -> dict:
  return {
    "chroma_persist__path": os.getenv(
      "CHROMA_PERSIST_PATH",
      config["indexing"]["chroma_persist_path"],
    ),
    "bm25_persist_path": os.getenv(
      "BM25_PERSIST_PATH",
      config["indexing"]["bm25_persist_path"],
    )
  }