from src.agent_state import AgentState
from src.web_search import web_search


def search_web_agent(state: AgentState) -> AgentState:
    """
    Search the web and prepare useful context for the
    draft-answer agent.
    """

    question = state["question"]

    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    results = web_search(
        question,
        max_results=5
    )

    web_context_parts = []
    web_sources = []

    # --------------------------------------------------------
    # PROCESS RESULTS
    # --------------------------------------------------------

    for index, result in enumerate(results, start=1):

        title = result.get(
            "title",
            ""
        )

        url = result.get(
            "url",
            ""
        )

        # Support both old and new search-result formats
        snippet = result.get(
            "snippet",
            ""
        )

        summary = result.get(
            "summary",
            ""
        )

        content = result.get(
            "content",
            ""
        )

        page_content = result.get(
            "page_content",
            ""
        )

        # ----------------------------------------------------
        # Choose the richest available content
        # ----------------------------------------------------

        useful_content = (
            content
            or page_content
            or snippet
            or summary
        )

        # ----------------------------------------------------
        # Build Web Context
        # ----------------------------------------------------

        web_context_parts.append(
            f"WEB RESULT {index}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {useful_content}"
        )

        if url:
            web_sources.append(
                url
            )

    # --------------------------------------------------------
    # Combine Context
    # --------------------------------------------------------

    web_context = "\n\n".join(
        web_context_parts
    )

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace = state.get(
        "trace",
        []
    )

    trace.append(
        f"WEB SEARCH: Retrieved {len(results)} web results"
    )

    # --------------------------------------------------------
    # Return Updated State
    # --------------------------------------------------------

    return {
        **state,
        "web_context": web_context,
        "web_sources": web_sources,
        "trace": trace
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    state: AgentState = {
        "question":
            "What is the latest development in AI?",

        "trace": []
    }

    result = search_web_agent(
        state
    )

    print("\n")
    print("=" * 60)
    print("WEB SEARCH AGENT TEST")
    print("=" * 60)

    print("\nQuestion:")
    print(result["question"])

    print("\nWeb Context:")
    print("=" * 60)

    print(
        result.get(
            "web_context",
            "No web context."
        )
    )

    print("\nWeb Sources:")
    print("=" * 60)

    for source in result.get(
        "web_sources",
        []
    ):
        print(
            f"- {source}"
        )

    print("\nTrace:")
    print("=" * 60)

    for item in result.get(
        "trace",
        []
    ):
        print(
            item
        )

    print("=" * 60)