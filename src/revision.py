from src.agent_state import AgentState
from src.gemini_client import generate_answer


MAX_REVISIONS = 2


# ============================================================
# REVISION AGENT
# ============================================================

def revise_answer(state: AgentState) -> AgentState:
    """
    Revise the draft answer using critic feedback.

    Handles two situations:

    1. Normal FAIL:
       Draft is revised using retrieved context and critic feedback.

    2. NO RELEVANT INFORMATION:
       Do NOT call the LLM and do NOT create a fake answer.
       Return a safe no-information response instead.
    """

    # ========================================================
    # GET STATE
    # ========================================================

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

    retrieved_context = state.get(
        "retrieved_context",
        ""
    ).strip()

    web_context = state.get(
        "web_context",
        ""
    ).strip()

    revision_count = state.get(
        "revision_count",
        0
    )

    if not isinstance(
        revision_count,
        int
    ):
        revision_count = 0

    retrieval_status = state.get(
        "retrieval_status",
        ""
    )

    if not isinstance(
        retrieval_status,
        str
    ):
        retrieval_status = ""

    retrieval_status = retrieval_status.upper().strip()

    source_selection = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(
        source_selection,
        str
    ):
        source_selection = "LOCAL"

    source_selection = source_selection.upper().strip()

    # ========================================================
    # TRACE
    # ========================================================

    trace = state.get(
        "trace",
        []
    ).copy()

    # ========================================================
    # IMPORTANT: NO RELEVANT INFORMATION
    # ========================================================

    no_local_information = (
        retrieval_status == "NO_RELEVANT_INFORMATION"
        and not retrieved_context
    )

    no_context = (
        not retrieved_context
        and not web_context
    )

    # --------------------------------------------------------
    # LOCAL SOURCE WITH NO RELEVANT INFORMATION
    # --------------------------------------------------------

    if (
        no_local_information
        and source_selection == "LOCAL"
    ):

        safe_answer = (
            "The available local documents do not contain "
            "relevant information to answer this question."
        )

        trace.append(
            "REVISION: No relevant local information available; "
            "revision skipped"
        )

        return {
            **state,
            "draft_answer": safe_answer,
            "revision_count": revision_count,
            "trace": trace
        }

    # --------------------------------------------------------
    # NO CONTEXT AT ALL
    # --------------------------------------------------------

    if no_context:

        safe_answer = (
            "The available sources do not contain enough "
            "information to answer this question."
        )

        trace.append(
            "REVISION: No supporting context available; "
            "revision skipped"
        )

        return {
            **state,
            "draft_answer": safe_answer,
            "revision_count": revision_count,
            "trace": trace
        }

    # ========================================================
    # MAXIMUM REVISION CHECK
    # ========================================================

    if revision_count >= MAX_REVISIONS:

        trace.append(
            "REVISION: Maximum revision limit reached"
        )

        return {
            **state,
            "revision_count": revision_count,
            "trace": trace
        }

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []

    if retrieved_context:

        context_parts.append(
            "===== LOCAL CONTEXT =====\n"
            + retrieved_context
        )

    if web_context:

        context_parts.append(
            "===== WEB CONTEXT =====\n"
            + web_context
        )

    context = "\n\n".join(
        context_parts
    )

    # ========================================================
    # REVISION PROMPT
    # ========================================================

    prompt = f"""
You are the Revision Agent in an Agentic RAG system.

Your ONLY job is to improve the current draft using
the critic feedback and retrieved evidence.

============================================================
USER QUESTION
============================================================

{question}

============================================================
AVAILABLE RETRIEVED CONTEXT
============================================================

{context}

============================================================
CURRENT DRAFT
============================================================

{draft}

============================================================
CRITIC FEEDBACK
============================================================

{critique}

============================================================
REVISION RULES
============================================================

1. Answer the exact user question.

2. Use ONLY information supported by the retrieved context.

3. Remove unsupported claims.

4. Correct the specific problem identified by the critic.

5. Do not invent facts.

6. Do not use outside knowledge.

7. Preserve correct information from the current draft.

8. If the question is a comparison, actually compare
   the required sources.

9. If the available context is insufficient, explicitly
   say that the available sources do not provide enough
   information.

10. Return ONLY the revised answer.

11. Do not return:
    VERDICT:
    REASON:
    IMPROVEMENT:
    CRITIC:
    REVISION:
    FINAL ANSWER:

Generate the corrected answer now.
"""

    # ========================================================
    # CALL MODEL
    # ========================================================

    try:

        revised_answer = generate_answer(
            prompt
        )

    except Exception as e:

        revised_answer = ""

        trace.append(
            f"REVISION: Model generation failed - {e}"
        )

    # ========================================================
    # CLEAN RESPONSE
    # ========================================================

    if revised_answer is None:

        revised_answer = ""

    revised_answer = revised_answer.strip()

    # ========================================================
    # FALLBACK IF MODEL RETURNS NOTHING
    # ========================================================

    if not revised_answer:

        revised_answer = draft

    # ========================================================
    # REMOVE ACCIDENTAL CRITIC FORMAT
    # ========================================================

    revised_upper = revised_answer.upper()

    bad_markers = [
        "VERDICT:",
        "REASON:",
        "IMPROVEMENT:",
        "CRITIC:",
        "CRITIC EVALUATION:"
    ]

    if any(
        marker in revised_upper
        for marker in bad_markers
    ):

        revised_answer = draft

    # ========================================================
    # REMOVE "FINAL ANSWER:" PREFIX
    # ========================================================

    if revised_answer.lower().startswith(
        "final answer:"
    ):

        revised_answer = revised_answer[
            len("final answer:")
        ].strip()

    # ========================================================
    # FINAL EMPTY FALLBACK
    # ========================================================

    if not revised_answer:

        revised_answer = (
            "The available sources do not provide enough "
            "information to answer this question."
        )

    # ========================================================
    # UPDATE REVISION COUNT
    # ========================================================

    new_revision_count = (
        revision_count + 1
    )

    # ========================================================
    # TRACE
    # ========================================================

    trace.append(
        f"REVISION: Improved draft answer "
        f"(revision {new_revision_count})"
    )

    # ========================================================
    # RETURN UPDATED STATE
    # ========================================================

    return {
        **state,
        "draft_answer": revised_answer,
        "revision_count": new_revision_count,
        "trace": trace
    }


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("REVISION AGENT TEST")
    print("=" * 60)

    # ========================================================
    # TEST 1 — NORMAL FAIL → REVISION
    # ========================================================

    print("\n" + "-" * 60)
    print("TEST 1: NORMAL REVISION")
    print("-" * 60)

    state_normal: AgentState = {

        "question":
            "What is supervised learning?",

        "source_selection":
            "LOCAL",

        "retrieval_status":
            "RELEVANT",

        "draft_answer":
            (
                "Supervised learning uses labeled data "
                "and always achieves 100% accuracy."
            ),

        "retrieved_context":
            (
                "Supervised learning uses labeled data "
                "to train a model. Common tasks include "
                "classification and regression."
            ),

        "critique":
            (
                "VERDICT: FAIL\n"
                "REASON: The claim about always achieving "
                "100% accuracy is not supported by the "
                "retrieved context.\n"
                "IMPROVEMENT: Remove the unsupported "
                "accuracy claim."
            ),

        "trace": [],

        "revision_count": 0
    }

    result_normal = revise_answer(
        state_normal
    )

    print("\nQuestion:")
    print(
        result_normal["question"]
    )

    print("\nOriginal Draft:")
    print(
        state_normal["draft_answer"]
    )

    print("\nRevised Draft:")
    print(
        result_normal["draft_answer"]
    )

    print("\nRevision Count:")
    print(
        result_normal["revision_count"]
    )

    print("\nTrace:")

    for item in result_normal["trace"]:

        print(item)

    # ========================================================
    # TEST 2 — NO RELEVANT INFORMATION
    # ========================================================

    print("\n" + "-" * 60)
    print("TEST 2: NO RELEVANT INFORMATION")
    print("-" * 60)

    state_no_context: AgentState = {

        "question":
            "What is the history of quantum computing "
            "in medieval Europe?",

        "source_selection":
            "LOCAL",

        "retrieval_status":
            "NO_RELEVANT_INFORMATION",

        "retrieved_context":
            "",

        "draft_answer":
            (
                "The available local documents do not "
                "contain relevant information to answer "
                "this question."
            ),

        "critique":
            (
                "VERDICT: FAIL\n"
                "REASON: No retrieved context was available "
                "for verification.\n"
                "IMPROVEMENT: Retrieve supporting information "
                "before answering."
            ),

        "trace": [],

        "revision_count": 0
    }

    result_no_context = revise_answer(
        state_no_context
    )

    print("\nQuestion:")
    print(
        result_no_context["question"]
    )

    print("\nRevised Draft:")
    print(
        result_no_context["draft_answer"]
    )

    print("\nRevision Count:")
    print(
        result_no_context["revision_count"]
    )

    print("\nTrace:")

    for item in result_no_context["trace"]:

        print(item)

    # ========================================================
    # TEST 3 — MAXIMUM REVISION
    # ========================================================

    print("\n" + "-" * 60)
    print("TEST 3: MAX REVISION LIMIT")
    print("-" * 60)

    state_max: AgentState = {

        "question":
            "What is supervised learning?",

        "source_selection":
            "LOCAL",

        "retrieval_status":
            "RELEVANT",

        "draft_answer":
            "Supervised learning uses labeled data.",

        "retrieved_context":
            "Supervised learning uses labeled data "
            "to train a model.",

        "critique":
            "VERDICT: FAIL",

        "trace": [],

        "revision_count":
            MAX_REVISIONS
    }

    result_max = revise_answer(
        state_max
    )

    print("\nRevision Count:")
    print(
        result_max["revision_count"]
    )

    print("\nDraft:")
    print(
        result_max["draft_answer"]
    )

    print("\nTrace:")

    for item in result_max["trace"]:

        print(item)

    print("\n" + "=" * 60)
    print("REVISION AGENT TESTS COMPLETED")
    print("=" * 60)