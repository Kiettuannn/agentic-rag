from __future__ import annotations

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Model dung de convert tu text sang vector embeddings
from sentence_transformers import SentenceTransformer


class Embedder:
  """
  Embedding text

  - Load model embedding
  - Embed docs
  - Embed query

  Class nay ko luu vector, no chi tao vector
  """
  def __init__(
    self,
    model_name: str = "intfloat/multilingual-e5-base",
  ):
    self.model_name = model_name

    # Load model vam ram, nen chi load 1 lan
    self.model = SentenceTransformer(model_name)

    self.langchain_embedding = HuggingFaceEmbeddings(model_name=model_name)
  
  def embed_documents(
    self,
    documents: list[Document],
  ) -> list[list[float]]:
    """
    Embed documents

    Args:
      documents: list of Document

    Returns:
      list of vector embeddings
    """

    texts = [
      doc.page_content for doc in documents
    ]
    # normalize_embeddings=True: chuan hoa vector ve khoang -1 den 1, giup cosine similarity hieu qua hon
    embeddings = self.model.encode(
      texts,
      convert_to_numpy=True,
      normalize_embeddings=True,)
    
    # Convert numpy array to list
    # De serialize, debug, compare
    return embeddings.tolist()
  
  def embed_query(
    self,
    query: str,
  ) -> list[float]:
    """
    Embed query

    Args:
      query: string

    Returns:
      vector embedding
    """
    query_text = f"query: {query}"

    embedding = self.model.encode(
      query_text,
      convert_to_numpy=True,
      normalize_embeddings=True,
    )
    return embedding.tolist()
