from src.agent_state import AgentState


def route_sources(state: AgentState) -> AgentState:
    """
    Decide whether the question should use:
    LOCAL, WEB, or BOTH sources.
    """

    question = state["question"].strip().lower()

    # ========================================================
    # WEB KEYWORDS
    # ========================================================

    web_keywords = [
        "latest",
        "current",
        "today",
        "recent",
        "news",
        "2026",
        "new developments",
        "recent developments",
        "current research",
        "latest research",
        "latest version",
        "price",
        "stock",
        "weather",
    ]

    # ========================================================
    # LOCAL KEYWORDS
    # ========================================================

    local_keywords = [
        "provided document",
        "provided documents",
        "local document",
        "local documents",
        "according to the document",
        "according to the provided document",
        "according to our document",
        "in the document",
        "from the document",
        "our documents",
        "our data",
        "uploaded document",
        "uploaded file",
    ]

    # ========================================================
    # COMPARISON / BOTH KEYWORDS
    # ========================================================

    both_keywords = [
        "compare",
        "comparison",
        "according to the document and",
        "document and current",
        "document and latest",
        "local and web",
        "both sources",
        "from the document and web",
        "provided document and current",
        "provided document and latest",
    ]

    # ========================================================
    # DETECT REQUIREMENTS
    # ========================================================

    requires_web = any(
        keyword in question
        for keyword in web_keywords
    )

    requires_local = any(
        keyword in question
        for keyword in local_keywords
    )

    requires_both = any(
        keyword in question
        for keyword in both_keywords
    )

    # ========================================================
    # SOURCE DECISION
    # ========================================================

    if requires_both:

        source = "BOTH"

    elif requires_web and requires_local:

        source = "BOTH"

    elif requires_web:

        source = "WEB"

    else:

        source = "LOCAL"

    # ========================================================
    # TRACE
    # ========================================================

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        f"SOURCE ROUTER: Selected {source}"
    )

    # ========================================================
    # RETURN STATE
    # ========================================================

    return {
        **state,
        "source_selection": source,
        "trace": trace,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SOURCE ROUTER TEST")
    print("=" * 60)

    test_questions = [

        "What is supervised learning?",

        "What are the latest AI developments in 2026?",

        "According to the provided document, what is supervised learning?",

        "Compare the information in the provided document with "
        "the latest AI developments in 2026.",

        "What does the document say about machine learning and "
        "how does it compare with current AI research?",
    ]

    for question in test_questions:

        state: AgentState = {
            "question": question,
            "trace": [],
        }

        result = route_sources(
            state
        )

        print("\n" + "-" * 60)

        print("\nQUESTION:")
        print(result["question"])

        print("\nSELECTED SOURCE:")
        print(result["source_selection"])

        print("\nTRACE:")

        for item in result["trace"]:
            print(item)

    print("\n" + "=" * 60)