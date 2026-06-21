
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import load_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents

config = {
    "dataset": {
        "name": "th1nhng0/vietnamese-legal-documents"
    },
    "chunking": {
        "chunk_size": 1200,
        "chunk_overlap": 200
    }
}

# docs = load_documents(config, sample_size=10)

# print("=" * 80)
# print("TOTAL DOCS:", len(docs))

# for i, doc in enumerate(docs):
#     print(f"\nDOCUMENT {i+1}")
#     print("-" * 80)

#     print("METADATA:")
#     print(doc.metadata)

#     print("\nCONTENT PREVIEW:")
#     print(doc.page_content[:500])




# docs = load_documents(config, sample_size=2)

# print("\nBEFORE CLEANING:")
# print(docs[0].page_content[:500])

# cleaned_docs = clean_documents(docs)

# print("\nAFTER CLEANING:")
# print(cleaned_docs[0].page_content[:500])


# Pipeline
docs = load_documents(config, sample_size=2)
cleaned_docs = clean_documents(docs)
chunks = chunk_documents(cleaned_docs, config)

# Create logs folder
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / "chunk_output.txt"

with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"TOTAL CHUNKS: {len(chunks)}\n")
    f.write("=" * 100 + "\n")

    for i, chunk in enumerate(chunks):
        f.write(f"\nCHUNK {i+1}\n")
        f.write("-" * 100 + "\n")

        f.write("METADATA:\n")
        f.write(f"{chunk.metadata}\n\n")

        f.write("CONTENT:\n")
        f.write(chunk.page_content)
        f.write("\n\n")
        f.write("=" * 100 + "\n")

print(f"Chunk logs saved to: {log_file}")