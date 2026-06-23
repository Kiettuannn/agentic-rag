from __future__ import annotations

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
  def __init__(
    self,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
  ):
    self.base_url = base_url or os.getenv("LLM_BASE_URL")
    self.api_key = api_key or os.getenv("LLM_API_KEY")
    self.model = model or os.getenv("LLM_MODEL", "z-ai/glm-5.2-free")
    self.temperature = temperature
    self.max_tokens = max_tokens

    if not self.api_key:
      raise ValueError("LLM API key is required. Please set it in the environment variable LLM_API_KEY.")
    
    self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

  
  def invoke(self, prompt: str) -> str:
    """
    Send a prompt, require return JSON
    Use for query_analyzer node in agent (need parse JSON)
    """
    response = self.client.chat.completions.create(
      model=self.model,
      messages=[{"role": "user", "content": prompt}],
      temperature=0.0,
      max_tokens=512,
      response_format={"type": "json_object"},
    )
    return response.choices[0].message.content
  
  @classmethod
  def from_config(cls, config: dict) -> "LLMClient":
    """
    Factory method: create LLMClient from config dict
    Usage:
      config = load_config()
      llm = LLMClient.from_config(config)
    """
    llm_cfg = config.get("llm", {})
    return cls(
      temperature=llm_cfg.get("temperature", 0.1),
      max_tokens=llm_cfg.get("max_tokens", 2048),
    )
  

class _LLMResponse:
  """A simple wrapper for LLM response content"""
  def __init__(self, content: str):
    self.content = content

  def __repr__(self):
    return f"_LLMResponse(content={self.content[:100]!r}...)"