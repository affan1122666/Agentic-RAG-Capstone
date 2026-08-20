from src.agent_state import AgentState


def create_plan(state: AgentState) -> AgentState:
    """
    Planner node.

    Creates a simple research plan and determines the type
    of information the question is likely to require.

    The planner itself does not call Gemini.
    """

    question = state.get("question", "").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    question_lower = question.lower()

    # ========================================================
    # DETECT INFORMATION TYPE
    # ========================================================

    web_keywords = [
        "latest",
        "current",
        "today",
        "recent",
        "news",
        "2026",
        "2025",
        "2024",
        "new development",
        "new developments",
        "recent development",
        "recent developments",
        "current research",
        "latest research",
        "latest version",
        "new version",
        "price",
        "stock",
        "weather",
        "market",
        "trend",
        "trends",
        "this year",
        "this month",
        "this week"
    ]

    local_keywords = [
        "according to my documents",
        "according to the document",
        "from the document",
        "from my documents",
        "in the provided document",
        "in the uploaded document",
        "according to the provided context"
    ]

    # ========================================================
    # SOURCE REQUIREMENT
    # ========================================================

    requires_web = any(
        keyword in question_lower
        for keyword in web_keywords
    )

    requires_local = any(
        keyword in question_lower
        for keyword in local_keywords
    )

    if requires_web and requires_local:

        source_requirement = "BOTH"

    elif requires_web:

        source_requirement = "WEB"

    else:

        source_requirement = "LOCAL"

    # ========================================================
    # RESEARCH PLAN
    # ========================================================

    plan = [
        "1. Understand the user's research question.",
        "2. Select the most appropriate information source.",
        "3. Retrieve relevant information from the selected source.",
        "4. Generate an answer grounded in the retrieved information.",
        "5. Verify the answer using the retrieved context.",
        "6. Revise the answer if verification fails.",
        "7. Produce the final verified answer."
    ]

    # ========================================================
    # TRACE
    # ========================================================

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "PLANNER: Created research plan"
    )

    trace.append(
        f"PLANNER: Source requirement = {source_requirement}"
    )

    # ========================================================
    # RETURN UPDATED STATE
    # ========================================================

    return {
        **state,
        "plan": plan,
        "source_requirement": source_requirement,
        "trace": trace
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PLANNER TEST")
    print("=" * 60)

    test_questions = [

        "What is supervised learning?",

        "What are the latest AI developments in 2026?",

        "What is the current state of AI research?",

        "According to the provided document, what is supervised learning?"
    ]

    for question in test_questions:

        state: AgentState = {
            "question": question,
            "trace": []
        }

        result = create_plan(
            state
        )

        print("\n" + "-" * 60)

        print("QUESTION:")
        print(
            result["question"]
        )

        print("\nSOURCE REQUIREMENT:")
        print(
            result["source_requirement"]
        )

        print("\nPLAN:")

        for step in result["plan"]:

            print(
                step
            )

        print("\nTRACE:")

        for item in result["trace"]:

            print(
                item
            )

    print("\n" + "=" * 60)