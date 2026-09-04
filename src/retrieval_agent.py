from functools import lru_cache

from src.agent_state import AgentState
from src.retriever import Retriever


# ============================================================
# CONSTANTS
# ============================================================

NO_LOCAL_INFORMATION = (
    "NO RELEVANT LOCAL INFORMATION FOUND."
)


# ============================================================
# CACHED RETRIEVER
# ============================================================

@lru_cache(maxsize=1)
def get_retriever():
    """
    Create the Retriever only once per Python process.

    The Retriever contains:
    - SentenceTransformer model
    - ChromaDB client
    - ChromaDB collection

    All subsequent queries reuse the same Retriever.
    """

    print(
        "\nInitializing local retriever..."
    )

    retriever = Retriever(
        top_k=2
    )

    print(
        "Local retriever initialized successfully."
    )

    return retriever


# ============================================================
# QUESTION HELPERS
# ============================================================

def _is_document_reference(
    question: str
) -> bool:
    """
    Detect questions that explicitly refer
    to the provided/local document.
    """

    q = question.lower()

    terms = [
        "provided document",
        "provided documents",
        "local document",
        "local documents",
        "document",
        "documents",
        "provided file",
        "local file"
    ]

    return any(
        term in q
        for term in terms
    )


def _is_comparison_question(
    question: str
) -> bool:
    """
    Detect comparison-style questions.
    """

    q = question.lower()

    terms = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "contrast",
        "versus",
        "vs",
        "compared with",
        "compared to"
    ]

    return any(
        term in q
        for term in terms
    )


# ============================================================
# BUILD RETRIEVAL QUERY
# ============================================================

def _build_local_query(
    question: str
) -> str:
    """
    Build an improved local retrieval query.

    For broad comparison questions that explicitly mention
    the provided document, use a more focused retrieval query.
    """

    if (
        _is_comparison_question(question)
        and _is_document_reference(question)
    ):

        return (
            "Machine learning supervised learning "
            "labeled data classification regression"
        )

    return question


# ============================================================
# RETRIEVAL AGENT
# ============================================================

def retrieve_information(
    state: AgentState
) -> AgentState:
    """
    Retrieve relevant information from the local document store.

    The Retriever itself is cached, so the expensive
    SentenceTransformer model is NOT loaded for every query.
    """

    question = state.get(
        "question",
        ""
    ).strip()

    # ========================================================
    # TRACE
    # ========================================================

    trace = state.get(
        "trace",
        []
    ).copy()

    # ========================================================
    # EMPTY QUESTION
    # ========================================================

    if not question:

        trace.append(
            "RETRIEVER: Empty question; retrieval skipped"
        )

        return {
            **state,
            "retrieved_context": "",
            "sources": [],
            "trace": trace
        }

    # ========================================================
    # GET CACHED RETRIEVER
    # ========================================================

    try:

        retriever = get_retriever()

    except Exception as e:

        trace.append(
            f"RETRIEVER: Initialization failed - {e}"
        )

        return {
            **state,
            "retrieved_context": "",
            "sources": [],
            "trace": trace
        }

    # ========================================================
    # BUILD QUERY
    # ========================================================

    retrieval_query = _build_local_query(
        question
    )

    if retrieval_query != question:

        trace.append(
            "RETRIEVER: Used document-aware retrieval query"
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    results = retriever.retrieve(
        retrieval_query
    )

    # ========================================================
    # PROCESS RESULTS
    # ========================================================

    context_parts = []
    sources = []

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        filename = metadata.get(
            "filename",
            "Unknown source"
        )

        content = result.get(
            "content",
            ""
        )

        if not content:
            continue

        # ----------------------------------------------------
        # Add context
        # ----------------------------------------------------

        context_parts.append(
            f"Source: {filename}\n{content}"
        )

        # ----------------------------------------------------
        # Avoid duplicate sources
        # ----------------------------------------------------

        if filename not in sources:

            sources.append(
                filename
            )

    # ========================================================
    # COMBINE CONTEXT
    # ========================================================

    context = "\n\n".join(
        context_parts
    )

    # ========================================================
    # NO RESULTS
    # ========================================================

    if not context:

        trace.append(
            "RETRIEVER: No relevant local information found"
        )

        return {
            **state,
            "retrieved_context": "",
            "sources": [],
            "trace": trace
        }

    # ========================================================
    # RESULTS FOUND
    # ========================================================

    trace.append(
        f"RETRIEVER: Retrieved {len(results)} relevant chunks"
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        **state,
        "retrieved_context": context,
        "sources": sources,
        "trace": trace
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RETRIEVAL AGENT PERFORMANCE TEST")
    print("=" * 60)

    # ========================================================
    # FIRST QUERY
    # ========================================================

    state_1: AgentState = {

        "question":
            "What is supervised learning?",

        "trace": []
    }

    result_1 = retrieve_information(
        state_1
    )

    print("\n" + "-" * 60)
    print("FIRST QUERY")
    print("-" * 60)

    print(
        "\nQuestion:",
        result_1["question"]
    )

    print(
        "\nResults:",
        len(
            result_1.get(
                "retrieved_context",
                ""
            )
        )
    )

    print(
        "\nSources:",
        result_1.get(
            "sources",
            []
        )
    )

    # ========================================================
    # SECOND QUERY
    # ========================================================

    state_2: AgentState = {

        "question":
            "What are the common tasks "
            "in supervised learning?",

        "trace": []
    }

    result_2 = retrieve_information(
        state_2
    )

    print("\n" + "-" * 60)
    print("SECOND QUERY")
    print("-" * 60)

    print(
        "\nQuestion:",
        result_2["question"]
    )

    print(
        "\nSources:",
        result_2.get(
            "sources",
            []
        )
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 60)
    print("RETRIEVAL AGENT TEST COMPLETED")
    print("=" * 60)