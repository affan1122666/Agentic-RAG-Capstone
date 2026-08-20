import chromadb
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_documents
from src.document_loader import load_documents


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"


def create_vector_store():
    # Load documents and create chunks
    documents = load_documents()
    chunks = chunk_documents(documents)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Create or get collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Add chunks to ChromaDB
    ids = []
    texts = []
    metadatas = []

    for chunk in chunks:
        chunk_id = f"{chunk['filename']}_{chunk['chunk_id']}"

        ids.append(chunk_id)
        texts.append(chunk["content"])
        metadatas.append({
            "filename": chunk["filename"],
            "chunk_id": chunk["chunk_id"]
        })

    embeddings = model.encode(texts).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database path: {CHROMA_PATH}")


if __name__ == "__main__":
    create_vector_store()