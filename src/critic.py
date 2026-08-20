from src.agent_state import AgentState
from src.gemini_client import generate_answer


def critique_answer(state: AgentState) -> AgentState:
    """
    Verify the generated draft against the retrieved information.

    Important:
    If no relevant information was retrieved, the critic does NOT
    start a revision loop. The system should safely return a
    no-information response instead of hallucinating an answer.
    """

    question = state.get(
        "question",
        ""
    ).strip()

    draft = state.get(
        "draft_answer",
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

    retrieval_status = state.get(
        "retrieval_status",
        ""
    ).strip().upper()

    source_selection = state.get(
        "source_selection",
        "LOCAL"
    ).strip().upper()

    # ========================================================
    # TRACE
    # ========================================================

    trace = state.get(
        "trace",
        []
    ).copy()

    # ========================================================
    # NO RELEVANT INFORMATION
    # ========================================================

    no_local_information = (
        retrieval_status == "NO_RELEVANT_INFORMATION"
    )

    no_web_information = (
        source_selection in {"WEB", "BOTH"}
        and not web_context
    )

    no_any_information = (
        not retrieved_context
        and not web_context
    )

    if (
        no_local_information
        or no_any_information
        or no_web_information
    ):

        critique = (
            "VERDICT: PASS\n"
            "REASON: No relevant retrieved information was "
            "available, so the system correctly avoids making "
            "an unsupported claim.\n"
            "IMPROVEMENT: None"
        )

        trace.append(
            "CRITIC: No relevant information available; "
            "hallucination prevention check passed"
        )

        return {
            **state,
            "critique": critique,
            "trace": trace
        }

    # ========================================================
    # NO DRAFT
    # ========================================================

    if not draft:

        critique = (
            "VERDICT: FAIL\n"
            "REASON: No draft answer was generated despite "
            "relevant information being available.\n"
            "IMPROVEMENT: Generate a grounded draft from "
            "the retrieved information."
        )

        trace.append(
            "CRITIC: Draft verification failed - no draft available"
        )

        return {
            **state,
            "critique": critique,
            "trace": trace
        }

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []

    if retrieved_context:

        context_parts.append(
            "LOCAL CONTEXT:\n"
            + retrieved_context
        )

    if web_context:

        context_parts.append(
            "WEB CONTEXT:\n"
            + web_context
        )

    context = "\n\n".join(
        context_parts
    )

    # ========================================================
    # STRICT CRITIC PROMPT
    # ========================================================

    prompt = f"""
You are a strict verification and self-critique agent
inside an Agentic RAG research system.

Your job is to verify whether the draft answer correctly
answers the user's question using ONLY the retrieved
information.

============================================================
USER QUESTION
============================================================

{question}

============================================================
RETRIEVED CONTEXT
============================================================

{context}

============================================================
DRAFT ANSWER
============================================================

{draft}

============================================================
EVALUATION CRITERIA
============================================================

1. RELEVANCE

Does the answer directly answer the user's question?

2. GROUNDING

Are important claims supported by the retrieved context?

3. ACCURACY

Does the answer avoid unsupported or invented facts?

4. COMPLETENESS

Does it satisfy the user's explicit requirements?

5. COMPARISON

If the user asks for a comparison, does the answer
actually compare the requested information?

6. CURRENT INFORMATION

If the question asks about latest/current information,
does the answer use the web context when available?

============================================================
STRICT RULE
============================================================

Return PASS only when the answer is relevant, grounded,
accurate and satisfies the user's question.

Return FAIL if an important requirement is missing or
unsupported.

============================================================
OUTPUT FORMAT
============================================================

PASS:

VERDICT: PASS
REASON: <short explanation>
IMPROVEMENT: None

FAIL:

VERDICT: FAIL
REASON: <short explanation>
IMPROVEMENT: <specific correction>

Return ONLY this format.
"""

    # ========================================================
    # CALL MODEL
    # ========================================================

    critique = generate_answer(
        prompt
    ).strip()

    # ========================================================
    # EMPTY RESPONSE
    # ========================================================

    if not critique:

        critique = (
            "VERDICT: FAIL\n"
            "REASON: The critic did not return a valid result.\n"
            "IMPROVEMENT: Re-run verification."
        )

    # ========================================================
    # NORMALIZE VERDICT
    # ========================================================

    critique_upper = critique.upper()

    if "VERDICT: FAIL" in critique_upper:

        verdict = "FAIL"

    elif "VERDICT: PASS" in critique_upper:

        verdict = "PASS"

    else:

        verdict = "FAIL"

        critique = (
            "VERDICT: FAIL\n"
            "REASON: Invalid critic output format.\n"
            "IMPROVEMENT: Re-evaluate the draft using the "
            "required verification format."
        )

    # ========================================================
    # TRACE
    # ========================================================

    if verdict == "PASS":

        trace.append(
            "CRITIC: Verified draft answer - PASS"
        )

    else:

        trace.append(
            "CRITIC: Draft verification failed - FAIL"
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        **state,
        "critique": critique,
        "trace": trace
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("STRICT CRITIC TEST")
    print("=" * 60)

    # ========================================================
    # TEST 1 — GOOD ANSWER
    # ========================================================

    state_pass: AgentState = {

        "question":
            "What is supervised learning?",

        "draft_answer":
            (
                "Supervised learning uses labeled data "
                "to train a model. Common tasks include "
                "classification and regression."
            ),

        "retrieved_context":
            (
                "Supervised learning uses labeled data "
                "to train a model. Common tasks include "
                "classification and regression."
            ),

        "retrieval_status":
            "RELEVANT",

        "trace": [],

        "revision_count": 0
    }

    result_pass = critique_answer(
        state_pass
    )

    print("\nTEST 1: SUPPORTED ANSWER")
    print("\nCritique:")
    print(result_pass["critique"])

    print("\nTrace:")

    for item in result_pass["trace"]:
        print(item)

    # ========================================================
    # TEST 2 — UNSUPPORTED ANSWER
    # ========================================================

    state_fail: AgentState = {

        "question":
            "What is supervised learning?",

        "draft_answer":
            (
                "Supervised learning always achieves "
                "100% accuracy and requires no training data."
            ),

        "retrieved_context":
            (
                "Supervised learning uses labeled data "
                "to train a model."
            ),

        "retrieval_status":
            "RELEVANT",

        "trace": [],

        "revision_count": 0
    }

    result_fail = critique_answer(
        state_fail
    )

    print("\n" + "-" * 60)
    print("TEST 2: UNSUPPORTED ANSWER")

    print("\nCritique:")
    print(result_fail["critique"])

    print("\nTrace:")

    for item in result_fail["trace"]:
        print(item)

    # ========================================================
    # TEST 3 — NO INFORMATION
    # ========================================================

    state_no_info: AgentState = {

        "question":
            "What is the history of quantum computing in medieval Europe?",

        "draft_answer":
            "",

        "retrieved_context":
            "",

        "retrieval_status":
            "NO_RELEVANT_INFORMATION",

        "source_selection":
            "LOCAL",

        "trace": [],

        "revision_count": 0
    }

    result_no_info = critique_answer(
        state_no_info
    )

    print("\n" + "-" * 60)
    print("TEST 3: NO RELEVANT INFORMATION")

    print("\nCritique:")
    print(result_no_info["critique"])

    print("\nTrace:")

    for item in result_no_info["trace"]:
        print(item)

    print("\n" + "=" * 60)