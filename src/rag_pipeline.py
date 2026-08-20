from src.retriever import Retriever
from src.gemini_client import generate_answer


class RAGPipeline:
    def __init__(self, top_k=2):
        self.retriever = Retriever(top_k=top_k)

    def answer(self, question):
        # Step 1: Retrieve relevant chunks
        results = self.retriever.retrieve(question)

        # Step 2: Build context
        context_parts = []

        for result in results:
            context_parts.append(
                f"Source: {result['metadata']['filename']}\n"
                f"{result['content']}"
            )

        context = "\n\n".join(context_parts)

        # Step 3: Create grounded prompt
        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.
If the answer cannot be found in the context, clearly say:
"I don't have enough information in the provided documents."

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

        # Step 4: Generate answer with Gemini
        answer = generate_answer(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                result["metadata"]["filename"]
                for result in results
            ]
        }


if __name__ == "__main__":
    rag = RAGPipeline(top_k=2)

    question = "What is supervised learning?"

    result = rag.answer(question)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")