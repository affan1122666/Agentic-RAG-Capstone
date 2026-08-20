import re

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"

# Semantic distance threshold.
# We keep this relatively relaxed because keyword matching
# will provide an additional relevance check.
MAX_DISTANCE = 1.20


class Retriever:

    def __init__(self, top_k=2):

        self.top_k = top_k

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # ----------------------------------------------------
        # Connect to existing ChromaDB
        # ----------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

    # ========================================================
    # TOKENIZATION
    # ========================================================

    @staticmethod
    def _tokenize(text):

        if not text:
            return set()

        text = text.lower()

        # Remove punctuation
        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        words = text.split()

        # Remove common stop words
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

    def retrieve(
        self,
        query
    ):

        # ----------------------------------------------------
        # Empty query
        # ----------------------------------------------------

        if not query or not query.strip():

            return []

        query = query.strip()

        # ----------------------------------------------------
        # Convert query into embedding
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            [query]
        ).tolist()

        # ----------------------------------------------------
        # Search more candidates than top_k
        #
        # This gives keyword matching more candidates to
        # evaluate.
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # Combined relevance
            # ------------------------------------------------
            #
            # Lower semantic distance is better.
            # Higher keyword score is better.
            #
            # We convert distance into a rough semantic score.
            # ------------------------------------------------

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

        # ====================================================
        # FILTER RELEVANT RESULTS
        # ====================================================

        relevant = []

        for candidate in candidates:

            distance = candidate[
                "distance"
            ]

            keyword_score = candidate[
                "keyword_score"
            ]

            # ------------------------------------------------
            # Rule 1:
            # Strong keyword match is enough.
            #
            # Example:
            # "common tasks in supervised learning"
            #
            # overlaps with:
            # supervised, learning, common, tasks
            # ------------------------------------------------

            strong_keyword_match = (
                keyword_score >= 0.30
            )

            # ------------------------------------------------
            # Rule 2:
            # Good semantic match can also qualify.
            # ------------------------------------------------

            semantic_match = (
                distance is not None
                and distance <= MAX_DISTANCE
            )

            # ------------------------------------------------
            # Accept if either retrieval signal is strong.
            # ------------------------------------------------

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

            # Keep the original expected fields.
            final_results.append({

                "content":
                    candidate["content"],

                "metadata":
                    candidate["metadata"],

                "distance":
                    candidate["distance"]

            })

        return final_results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVER TEST")
    print("=" * 60)

    retriever = Retriever(
        top_k=2
    )

    # ========================================================
    # TEST 1
    # ========================================================

    query = (
        "What is supervised learning?"
    )

    results = retriever.retrieve(
        query
    )

    print("\n" + "-" * 60)
    print("TEST 1: DIRECT QUERY")
    print("-" * 60)

    print("\nQuery:")
    print(query)

    print(
        f"\nRelevant chunks returned: "
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

    # ========================================================
    # TEST 2
    # ========================================================

    query = (
        "What are the common tasks "
        "in supervised learning?"
    )

    results = retriever.retrieve(
        query
    )

    print("\n" + "-" * 60)
    print("TEST 2: SEMANTIC + KEYWORD QUERY")
    print("-" * 60)

    print("\nQuery:")
    print(query)

    print(
        f"\nRelevant chunks returned: "
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

    # ========================================================
    # TEST 3
    # ========================================================

    query = (
        "What is the history of "
        "quantum computing in medieval Europe?"
    )

    results = retriever.retrieve(
        query
    )

    print("\n" + "-" * 60)
    print("TEST 3: UNRELATED QUERY")
    print("-" * 60)

    print("\nQuery:")
    print(query)

    print(
        f"\nRelevant chunks returned: "
        f"{len(results)}"
    )

    if not results:

        print(
            "\nNO RELEVANT LOCAL "
            "INFORMATION FOUND."
        )

    # ========================================================
    # TEST 4
    # ========================================================

    query = ""

    results = retriever.retrieve(
        query
    )

    print("\n" + "-" * 60)
    print("TEST 4: EMPTY QUERY")
    print("-" * 60)

    print(
        f"\nReturned chunks: "
        f"{len(results)}"
    )

    if not results:

        print(
            "EMPTY QUERY HANDLED CORRECTLY."
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVER TEST COMPLETED")
    print("=" * 60)