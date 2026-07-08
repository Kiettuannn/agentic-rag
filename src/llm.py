from __future__ import annotations

import os
import re
import ast
import json
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

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
    self.model = model or os.getenv("LLM_MODEL", "")
    self.temperature = temperature
    self.max_tokens = max_tokens

    if not self.api_key:
      raise ValueError("LLM API key is required. Please set it in the environment variable LLM_API_KEY.")
    
    self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)


  def invoke(self, prompt: str) -> _LLMResponse:
    response = self.client.chat.completions.create(
      model=self.model,
      messages=[{"role": "user", "content": prompt}],
      temperature=self.temperature,
      max_tokens=max(8192, self.max_tokens)  # Reasoning models need many tokens
    )
    choice = response.choices[0]
    content = choice.message.content

    # Fallback cho reasoning models: content=None khi model dùng hết token
    # cho reasoning. Lấy từ reasoning field nếu có.
    if content is None:
      reasoning = getattr(choice.message, "reasoning", None) or ""
      content = reasoning if reasoning else ""

    return _LLMResponse(content=content)

  def invoke_json(self, prompt: str) -> str:
    """
    Gửi prompt, trả về chuỗi JSON hợp lệ.

    Tương thích với cả normal models và reasoning models (chain-of-thought).
    Reasoning models (stepfun, deepseek-r1...) đặt answer trong field
    'reasoning' thay vì 'content' khi token budget bị giới hạn.
    """
    json_prompt = (
      prompt
      + '\n\nYêu cầu bắt buộc: Chỉ trả về JSON object hợp lệ, '
      'dùng double quotes ("), không có text thừa, không có markdown.'
    )

    response = self.client.chat.completions.create(
      model=self.model,
      messages=[{"role": "user", "content": json_prompt}],
      temperature=0.1,
      max_tokens=8192,   # Reasoning models cần nhiều token cho thinking
    )

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    raw = choice.message.content or ""

    # --- Fallback cho reasoning models ---
    # Khi content=None (model dùng hết token cho reasoning),
    # thử tìm JSON trong reasoning field
    if not raw.strip():
      reasoning = getattr(choice.message, "reasoning", None) or ""
      if reasoning:
        # JSON thường ở cuối reasoning: "Vậy JSON sẽ là {...}"
        extracted = _extract_json(reasoning)
        if extracted.strip().startswith("{"):
          return extracted

      # Vẫn không có gì — log để debug
      usage = getattr(response, "usage", None)
      print(
        f"[invoke_json] WARNING: Empty response!\n"
        f"  finish_reason = {finish_reason!r}\n"
        f"  usage         = {usage}\n"
        f"  model         = {self.model!r}\n"
        f"  has_reasoning = {bool(reasoning)}"
      )

    return _extract_json(raw)

  
  @classmethod
  def from_config(cls, config: dict) -> "LLMClient":
    """
    Factory method: create LLMClient from config dict.
    Usage:
      config = load_config()
      llm = LLMClient.from_config(config)
    """
    llm_cfg = config.get("llm", {})
    return cls(
      temperature=llm_cfg.get("temperature", 0.1),
      max_tokens=llm_cfg.get("max_tokens", 2048),
    )
  

def _extract_json(raw: str) -> str:
  """
  Trích xuất JSON hợp lệ từ output thô của LLM.

  Thứ tự ưu tiên:
  1. Tìm JSON trong markdown code block ```json ... ```
  2. Tìm JSON object đầu tiên bằng regex { ... }
  3. Fallback: ast.literal_eval để handle Python dict (single quotes)
  4. Trả về raw nếu tất cả đều thất bại (để caller xử lý lỗi)
  """
  # Bước 1: Lột bỏ markdown code fence nếu có
  code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
  if code_block:
    raw = code_block.group(1).strip()

  # Bước 2: Tìm JSON object { ... } đầu tiên trong text
  # Dùng balanced brace matching thay vì regex đơn giản
  json_str = _extract_first_json_object(raw)
  if json_str:
    # Thử parse trực tiếp
    try:
      json.loads(json_str)
      return json_str
    except json.JSONDecodeError:
      pass

    # Bước 3: Fallback — xử lý single quotes (Python dict style)
    try:
      py_obj = ast.literal_eval(json_str)
      if isinstance(py_obj, dict):
        return json.dumps(py_obj, ensure_ascii=False)
    except (ValueError, SyntaxError):
      pass

  # Bước 4: Trả về raw (caller sẽ catch JSONDecodeError)
  return raw.strip()


def _extract_first_json_object(text: str) -> str:
  """
  Tìm JSON object đầu tiên trong text bằng cách đếm dấu ngoặc nhọn.
  Chính xác hơn regex vì handle nested objects.
  """
  start = text.find("{")
  if start == -1:
    return ""

  depth = 0
  in_string = False
  escape_next = False

  for i, ch in enumerate(text[start:], start=start):
    if escape_next:
      escape_next = False
      continue
    if ch == "\\" and in_string:
      escape_next = True
      continue
    if ch == '"' and not escape_next:
      in_string = not in_string
      continue
    if in_string:
      continue
    if ch == "{":
      depth += 1
    elif ch == "}":
      depth -= 1
      if depth == 0:
        return text[start:i + 1]

  return ""


class _LLMResponse:
  """A simple wrapper for LLM response content"""
  def __init__(self, content: str):
    self.content = content

  def __repr__(self):
    return f"_LLMResponse(content={self.content[:100]!r}...)"


def create_langchain_llm(config: dict = None) -> ChatOpenAI:
  """Create ChatOpenAI instance for tool calling"""
  return ChatOpenAI(
    base_url = os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL", ""),
    temperature=0.1,
  )
