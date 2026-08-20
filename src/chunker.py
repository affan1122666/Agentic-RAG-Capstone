from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_loader import load_documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []

    for document in documents:
        text_chunks = splitter.split_text(document["content"])

        for index, chunk in enumerate(text_chunks):
            chunks.append({
                "filename": document["filename"],
                "chunk_id": index,
                "content": chunk
            })

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks:
        print(f"\n--- Chunk {chunk['chunk_id']} ---")
        print(chunk["content"])