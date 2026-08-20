from pathlib import Path
from src.config import DOCUMENTS_DIR


def load_documents():
    documents = []

    documents_path = Path(DOCUMENTS_DIR)

    for file_path in documents_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "filename": file_path.name,
            "content": text
        })

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"Loaded documents: {len(docs)}")

    for doc in docs:
        print(f"\nFile: {doc['filename']}")
        print(doc["content"][:300])