from __future__ import annotations

from src.retrieval.retriever import Retriever

class AnswerGenerator:
  def __init__(
    self,
    retriever: Retriever,
    llm
  ):
    self.retriever = retriever
    self.llm = llm

  def format_context(
    self, documents
  ) -> str:
    
    context_parts = []

    for doc in documents:
      doc_id = doc.metadata.get("doc_id", "unknown")
      chunk_index = doc.metadata.get("chunk_index", -1)

      context_parts.append(
        f"[Doc: {doc_id}, Chunk: {chunk_index}]\n"
        f"{doc.page_content}\n"
      )

    return "\n\n".join(context_parts)
  
  def build_prompt(
    self,
    query: str,
    context: str
  ) -> str:
    """
    Build grounded legal QA prompt.
    """

    return f"""
      Bạn là trợ lý pháp lý chuyên về văn bản pháp luật Việt Nam.

      Chỉ trả lời dựa trên context được cung cấp.

      Nếu không đủ thông tin:
      hãy nói rõ là không tìm thấy căn cứ.

      Context:
      {context}

      Question:
      {query}

      Answer:
      """
  
  def generate(
      self,
      query: str,
      strategy: str = "hybrid",
      k: int = 5
  ):
    # Step 1: Retrieve relevant documents
    documents = self.retriever.retrieve(query=query, strategy=strategy, k=k)

    print("=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    for i, doc in enumerate(documents, start=1):
      print(f"\nChunk {i}")
      print(f"Metadata: {doc.metadata}")
      print("-" * 50)

      print(doc.page_content[:300])

      print("-" * 50)

    # Step 2: Format context
    context = self.format_context(documents)

    # Step 3: Build prompt
    prompt = self.build_prompt(query=query, context=context)

    # Step 4: Generate answer using LLM
    response = self.llm.invoke(prompt)
    
    return {
      "answer": response.content,
      "documents": documents,
    }
    
