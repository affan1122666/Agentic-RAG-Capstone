from src.agent_state import AgentState


# ============================================================
# FINALIZER
# ============================================================

def finalize_answer(state: AgentState) -> AgentState:
    """
    Finalize the answer after critic verification.

    IMPORTANT:
    The finalizer does NOT call the LLM again.

    The critic has already verified the draft.
    Therefore, the safest behavior is to preserve the
    verified draft as the final answer.

    This prevents the finalizer from replacing a correct
    comparison answer with an unrelated answer.
    """

    question = state.get(
        "question",
        ""
    ).strip()

    draft = state.get(
        "draft_answer",
        ""
    ).strip()

    critique = state.get(
        "critique",
        ""
    ).strip()

    revision_count = state.get(
        "revision_count",
        0
    )

    # ========================================================
    # TRACE
    # ========================================================

    trace = state.get(
        "trace",
        []
    ).copy()

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if not draft:

        final_answer = (
            "The system could not generate a final answer "
            "from the available information."
        )

        trace.append(
            "FINALIZER: No verified draft available"
        )

        return {
            **state,
            "final_answer": final_answer,
            "trace": trace
        }

    # ========================================================
    # CHECK CRITIC VERDICT
    # ========================================================

    critique_upper = critique.upper()

    if "VERDICT: PASS" not in critique_upper:

        # The finalizer should normally only be reached
        # after PASS. This is an additional safety check.

        final_answer = draft

        trace.append(
            "FINALIZER: Draft preserved because "
            "no PASS verdict was detected"
        )

        return {
            **state,
            "final_answer": final_answer,
            "trace": trace
        }

    # ========================================================
    # VERIFIED DRAFT → FINAL ANSWER
    # ========================================================

    final_answer = draft

    # ========================================================
    # REMOVE ACCIDENTAL PREFIX
    # ========================================================

    if final_answer.lower().startswith(
        "final answer:"
    ):

        final_answer = final_answer[
            len("final answer:"):
        ].strip()

    # ========================================================
    # REMOVE ACCIDENTAL CRITIC FORMAT
    # ========================================================

    bad_markers = [
        "VERDICT:",
        "REASON:",
        "IMPROVEMENT:",
        "CRITIC:",
        "CRITIC EVALUATION:"
    ]

    if any(
        marker in final_answer.upper()
        for marker in bad_markers
    ):

        # Do not send it back to Gemini.
        # Use the original draft as the safest fallback.

        final_answer = draft

    # ========================================================
    # FINAL EMPTY CHECK
    # ========================================================

    if not final_answer.strip():

        final_answer = (
            "The available sources do not provide enough "
            "information to answer this question."
        )

    # ========================================================
    # TRACE
    # ========================================================

    if revision_count > 0:

        trace.append(
            f"FINALIZER: Preserved verified revised draft "
            f"as final answer (revision {revision_count})"
        )

    else:

        trace.append(
            "FINALIZER: Preserved verified draft as final answer"
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        **state,
        "final_answer": final_answer,
        "trace": trace
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("FINALIZER TEST")
    print("=" * 60)

    # ========================================================
    # TEST 1 — NORMAL ANSWER
    # ========================================================

    state_local: AgentState = {

        "question":
            "What is supervised learning?",

        "draft_answer":
            (
                "Supervised learning uses labeled data to "
                "train a model. Common tasks include "
                "classification and regression."
            ),

        "critique":
            (
                "VERDICT: PASS\n"
                "REASON: The draft is supported by the context.\n"
                "IMPROVEMENT: None"
            ),

        "revision_count":
            0,

        "trace":
            []
    }

    result_local = finalize_answer(
        state_local
    )

    print("\n" + "-" * 60)
    print("TEST 1: VERIFIED LOCAL ANSWER")

    print("\nQuestion:")
    print(
        result_local["question"]
    )

    print("\nFinal Answer:")
    print(
        result_local["final_answer"]
    )

    print("\nTrace:")

    for item in result_local["trace"]:
        print(item)

    # ========================================================
    # TEST 2 — BOTH SOURCE COMPARISON
    # ========================================================

    state_both: AgentState = {

        "question":
            (
                "Compare the information in the provided "
                "document with the latest AI developments "
                "in 2026."
            ),

        "draft_answer":
            (
                "The provided document focuses on foundational "
                "machine-learning concepts, particularly "
                "supervised learning and its use of labeled data. "
                "In contrast, the latest 2026 AI information "
                "focuses on newer developments such as agentic AI, "
                "multi-agent systems, reasoning models, scientific "
                "AI, and more efficient AI systems and hardware. "
                "Therefore, the document provides foundational "
                "concepts, while the current web information "
                "describes newer directions and applications of AI."
            ),

        "critique":
            (
                "VERDICT: PASS\n"
                "REASON: The draft compares the local and web "
                "information and is supported by the retrieved "
                "context.\n"
                "IMPROVEMENT: None"
            ),

        "revision_count":
            0,

        "trace":
            []
    }

    result_both = finalize_answer(
        state_both
    )

    print("\n" + "-" * 60)
    print("TEST 2: VERIFIED BOTH-SOURCE ANSWER")

    print("\nQuestion:")
    print(
        result_both["question"]
    )

    print("\nFinal Answer:")
    print(
        result_both["final_answer"]
    )

    print("\nTrace:")

    for item in result_both["trace"]:
        print(item)

    print("\n" + "=" * 60)