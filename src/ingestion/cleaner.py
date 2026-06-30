import os
import re
import unicodedata
from typing import List
from concurrent.futures import ProcessPoolExecutor

from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_core.documents import Document



def _strip_html(html: str) -> str:
  """
  Remove HTML tags while preserving readable text structure.

  Tai sao nen dung BeautifulSoup thay vi regex de strip HTML:
  - an toan cho nested tags
  - dam bao cau truc van ban duoc giu nguyen  
  """
  soup = BeautifulSoup(html, "lxml")

  # Chuyen block tags thanh newline de giu cau truc van ban
  text = soup.get_text(separator="\n")

  return text

def _normalize(text: str) -> str:
  """
  Normalize text gon gang de co the tim kiem de dang
  Steps:
  - Unicode NFC normalization (tuc la chuyen cac ky tu ve dang chuan)
  - Remove excessive whitespace (tuc la thay nhieu khoang trang lien tiep bang 1 khoang trang duy nhat)
  - Collapse multiple empty lines (tuc la thay nhieu dong trong lien tiep bang 1 dong trong duy nhat)
  """

  # Normalize unicode (quan trong cho tieng Viet)
  text = unicodedata.normalize("NFC", text)

  # Remove non-breaking spaces
  text = text.replace("\xa0", " ")

  # Remove excessive spaces/tabs
  text = re.sub(r"[ \t]+", " ", text)

  # Collapse too many newlines into max 2
  text = re.sub(r"\n{3,}", "\n\n", text)

  # Strip leading/trailing spaces
  text = text.strip()

  return text

def clean_document(doc: Document) -> Document:
  """
  Clean a single document.

  Pipeline:
  raw HTML -> strip HTML -> normalize unicode -> clean whitespace
  """
  raw = doc.page_content
  # Only parse HTML if document looks like HTML
  if "<" in raw and ">" in raw:
    raw = _strip_html(raw)

  cleaned = _normalize(raw)
  return Document(page_content=cleaned, metadata=doc.metadata)


def clean_documents(docs: List[Document], workers: int = 1) -> List[Document]:
  """
  Clean multiple documents

  workers = 1: -> sequential processing

  workers > 1: -> parallel processing using ProcessPoolExecutor

  """
  
  # Sequential mode (easy to debug)
  if workers <= 1:
    return [
      clean_document(doc) for doc in tqdm(docs, desc="Cleaning documents", unit="doc")
    ]
  
  # Parallel mode (faster for large datasets)
  with ProcessPoolExecutor(max_workers=workers) as executor:
    cleaned_docs = list(
      tqdm(
        executor.map(
          clean_document,
          docs,
          chunksize=64
        ),
        total=len(docs),
        desc="Cleaning documents",
        unit="doc"
      )
    )
  return cleaned_docs

