import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_splitter(config) -> RecursiveCharacterTextSplitter:
  """Builds a text splitter .
  Priority
  1. Dieu
  2. Khoan
  3. Diem
  4. Paragraph
  5. Line
  6. Word
  """
  return RecursiveCharacterTextSplitter(
    separators=[
      "\nĐiều ",
      "\nKhoản ",
      "\nĐiểm ",
      "\nMục ",
      "\nChương ",
      "\n\n",
      "\n",
      " ",
      "",
    ],
    chunk_size=config["chunking"]["chunk_size"],
    chunk_overlap=config["chunking"]["chunk_overlap"],
  )

def extract_article_number(text: str):
  """
  Try extracting article number from the text.
  Example: "Điều 1. Quyền và nghĩa vụ của công dân" -> "1"
  """
  match = re.search(r"Điều\s+(\d+)", text)
  if match:
      return match.group(1)
  return None

def chunk_documents(docs: List[Document], config) -> List[Document]:
  """Split cleaned documents into legal chunks."""

  splitter = build_splitter(config)

  chunks = splitter.split_documents(docs)

  for i, chunk in enumerate(chunks):
    #  Track chunk order
    chunk.metadata["chunk_index"] = i

    # Kepp chunk size for debugging
    chunk.metadata["chunk_length"] = len(chunk.page_content)

    # Save article number if found
    article_number = extract_article_number(chunk.page_content)
    if article_number:
      chunk.metadata["article_number"] = article_number
  
  return chunks

