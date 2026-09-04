import re
import time
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"

MAX_DISTANCE = 1.20


# ============================================================
# CACHED EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():

    start = time.perf_counter()

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    elapsed = time.perf_counter() - start

    print(
        f"Embedding model loaded in "
        f"{elapsed:.3f} seconds"
    )

    return model


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:

    def __init__(self, top_k=2):

        self.top_k = top_k

        # ----------------------------------------------------
        # Load cached embedding model
        # ----------------------------------------------------

        model_start = time.perf_counter()

        self.model = get_embedding_model()

        model_time = (
            time.perf_counter()
            - model_start
        )

        print(
            f"Retriever model initialization: "
            f"{model_time:.3f} seconds"
        )

        # ----------------------------------------------------
        # Connect to ChromaDB
        # ----------------------------------------------------

        chroma_start = time.perf_counter()

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        chroma_time = (
            time.perf_counter()
            - chroma_start
        )

        print(
            f"ChromaDB initialization: "
            f"{chroma_time:.3f} seconds"
        )

    # ========================================================
    # TOKENIZATION
    # ========================================================

    @staticmethod
    def _tokenize(text):

        if not text:
            return set()

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        words = text.split()

        stop_words = {
            "what",
            "is",
            "are",
            "the",
            "a",
            "an",
            "in",
            "on",
            "of",
            "to",
            "for",
            "and",
            "or",
            "does",
            "do",
            "how",
            "can",
            "this",
            "that",
            "with",
            "from",
            "by"
        }

        return {
            word
            for word in words
            if word not in stop_words
            and len(word) > 2
        }

    # ========================================================
    # KEYWORD SCORE
    # ========================================================

    def _keyword_score(
        self,
        query,
        document
    ):

        query_words = self._tokenize(
            query
        )

        document_words = self._tokenize(
            document
        )

        if not query_words:
            return 0.0

        overlap = (
            query_words
            & document_words
        )

        return (
            len(overlap)
            / len(query_words)
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(self, query):

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # Empty query
        # ----------------------------------------------------

        if not query or not query.strip():
            return []

        query = query.strip()

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        embedding_start = time.perf_counter()

        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False
        ).tolist()

        embedding_time = (
            time.perf_counter()
            - embedding_start
        )

        print(
            f"\nQuery embedding: "
            f"{embedding_time:.3f} seconds"
        )

        # ----------------------------------------------------
        # ChromaDB search
        # ----------------------------------------------------

        chroma_start = time.perf_counter()

        search_k = max(
            self.top_k,
            5
        )

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=search_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        chroma_time = (
            time.perf_counter()
            - chroma_start
        )

        print(
            f"ChromaDB query: "
            f"{chroma_time:.3f} seconds"
        )

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if not results:
            return []

        documents = results.get(
            "documents",
            [[]]
        )

        metadatas = results.get(
            "metadatas",
            [[]]
        )

        distances = results.get(
            "distances",
            [[]]
        )

        if (
            not documents
            or not documents[0]
        ):
            return []

        candidates = []

        # ====================================================
        # BUILD CANDIDATES
        # ====================================================

        scoring_start = time.perf_counter()

        for i, document in enumerate(
            documents[0]
        ):

            if not document:
                continue

            metadata = {}

            if (
                metadatas
                and metadatas[0]
                and i < len(metadatas[0])
            ):

                metadata = (
                    metadatas[0][i]
                    or {}
                )

            distance = None

            if (
                distances
                and distances[0]
                and i < len(distances[0])
            ):

                distance = distances[0][i]

            keyword_score = (
                self._keyword_score(
                    query,
                    document
                )
            )

            if distance is not None:

                semantic_score = max(
                    0.0,
                    1.0 - (
                        distance / 2.0
                    )
                )

            else:

                semantic_score = 0.0

            combined_score = (
                0.60 * semantic_score
                +
                0.40 * keyword_score
            )

            candidates.append({

                "content": document,

                "metadata": metadata,

                "distance": distance,

                "keyword_score": keyword_score,

                "semantic_score": semantic_score,

                "combined_score": combined_score

            })

        scoring_time = (
            time.perf_counter()
            - scoring_start
        )

        print(
            f"Relevance scoring: "
            f"{scoring_time:.3f} seconds"
        )

        # ====================================================
        # FILTER
        # ====================================================

        relevant = []

        for candidate in candidates:

            distance = candidate[
                "distance"
            ]

            keyword_score = candidate[
                "keyword_score"
            ]

            strong_keyword_match = (
                keyword_score >= 0.30
            )

            semantic_match = (
                distance is not None
                and distance <= MAX_DISTANCE
            )

            if (
                strong_keyword_match
                or semantic_match
            ):

                relevant.append(
                    candidate
                )

        # ====================================================
        # SORT
        # ====================================================

        relevant.sort(
            key=lambda item:
                item["combined_score"],
            reverse=True
        )

        # ====================================================
        # RETURN TOP K
        # ====================================================

        final_results = []

        for candidate in relevant[
            :self.top_k
        ]:

            final_results.append({

                "content":
                    candidate["content"],

                "metadata":
                    candidate["metadata"],

                "distance":
                    candidate["distance"]

            })

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"Total retrieve() time: "
            f"{total_time:.3f} seconds"
        )

        return final_results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVER PERFORMANCE TEST")
    print("=" * 60)

    retriever = Retriever(
        top_k=2
    )

    query = (
        "What is supervised learning?"
    )

    results = retriever.retrieve(
        query
    )

    print("\n" + "-" * 60)
    print("RESULT")
    print("-" * 60)

    print(
        f"Relevant chunks: "
        f"{len(results)}"
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result.get(
            "metadata",
            {}
        )

        print(
            f"\n--- Result {i} ---"
        )

        print(
            "File:",
            metadata.get(
                "filename",
                "Unknown"
            )
        )

        print(
            "Chunk ID:",
            metadata.get(
                "chunk_id",
                "Unknown"
            )
        )

        print(
            "Distance:",
            result.get(
                "distance"
            )
        )

        print(
            "Content:"
        )

        print(
            result.get(
                "content",
                ""
            )
        )

    print("\n" + "=" * 60)
    print("PERFORMANCE TEST COMPLETED")
    print("=" * 60)