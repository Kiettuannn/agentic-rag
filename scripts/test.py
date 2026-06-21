
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import load_documents

config = {
    "dataset": {
        "name": "th1nhng0/vietnamese-legal-documents"
    }
}

docs = load_documents(config, sample_size=10)

print("=" * 80)
print("TOTAL DOCS:", len(docs))

for i, doc in enumerate(docs):
    print(f"\nDOCUMENT {i+1}")
    print("-" * 80)

    print("METADATA:")
    print(doc.metadata)

    print("\nCONTENT PREVIEW:")
    print(doc.page_content[:500])

