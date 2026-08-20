from src.agent_state import AgentState
from src.critic import critique_answer
from src.revision import revise_answer


# ============================================================
# REVISION LOOP TEST
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("REVISION LOOP TEST")
    print("=" * 60)

    # ========================================================
    # TEST QUESTION
    # ========================================================

    question = "What is supervised learning?"

    print("\nQUESTION:")
    print(question)

    print("\nTEST PURPOSE:")
    print(
        "Verify that the system can detect an unsupported "
        "answer, revise it, and verify the revised answer."
    )

    # ========================================================
    # INITIAL STATE
    # ========================================================
    #
    # We intentionally create a WRONG answer.
    #
    # The retrieved context does NOT say that supervised
    # learning always achieves 100% accuracy.
    #
    # Therefore the critic should return FAIL.
    # ========================================================

    state: AgentState = {

        "question": question,

        "draft_answer": (
            "Supervised learning uses labeled data "
            "and always achieves 100% accuracy."
        ),

        "retrieved_context": (
            "Supervised learning uses labeled data "
            "to train a model. Common supervised "
            "learning tasks include classification "
            "and regression."
        ),

        "trace": [
            "TEST: Created intentionally unsupported draft"
        ],

        # IMPORTANT:
        # Start from ZERO.
        # revise_answer() will increment it to 1.
        "revision_count": 0
    }

    # ========================================================
    # STEP 1 — FIRST CRITIC
    # ========================================================

    print("\n" + "-" * 60)
    print("STEP 1: FIRST CRITIC")
    print("-" * 60)

    state = critique_answer(
        state
    )

    first_critique = state.get(
        "critique",
        ""
    )

    print(
        first_critique
    )

    # ========================================================
    # VERIFY FIRST CRITIC = FAIL
    # ========================================================

    if "VERDICT: FAIL" not in first_critique.upper():

        print(
            "\nERROR: The intentionally incorrect "
            "draft was expected to FAIL."
        )

        print(
            "\nREVISION LOOP TEST: FAILED"
        )

        return

    # ========================================================
    # STEP 2 — REVISION
    # ========================================================

    print("\n" + "-" * 60)
    print("STEP 2: REVISION")
    print("-" * 60)

    # IMPORTANT:
    # Do NOT manually increase revision_count here.
    #
    # revise_answer() itself handles the increment.
    # ========================================================

    state = revise_answer(
        state
    )

    revised_draft = state.get(
        "draft_answer",
        ""
    )

    print("\nREVISED DRAFT:")

    print(
        revised_draft
    )

    print("\nREVISION COUNT:")

    print(
        state.get(
            "revision_count",
            0
        )
    )

    # ========================================================
    # STEP 3 — SECOND CRITIC
    # ========================================================

    print("\n" + "-" * 60)
    print("STEP 3: SECOND CRITIC")
    print("-" * 60)

    state = critique_answer(
        state
    )

    final_critique = state.get(
        "critique",
        ""
    )

    print(
        final_critique
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("REVISION LOOP RESULT")
    print("=" * 60)

    print("\nREVISION COUNT:")

    print(
        state.get(
            "revision_count",
            0
        )
    )

    print("\nFINAL DRAFT:")

    print(
        state.get(
            "draft_answer",
            "No final draft generated."
        )
    )

    print("\nFINAL CRITIQUE:")

    print(
        final_critique
    )

    # ========================================================
    # TRACE
    # ========================================================

    print("\nTRACE:")

    trace = state.get(
        "trace",
        []
    )

    for index, item in enumerate(
        trace,
        start=1
    ):

        print(
            f"{index}. {item}"
        )

    # ========================================================
    # SUCCESS CONDITIONS
    # ========================================================

    revision_count = state.get(
        "revision_count",
        0
    )

    final_pass = (
        "VERDICT: PASS"
        in final_critique.upper()
    )

    revision_happened = (
        revision_count >= 1
    )

    # ========================================================
    # TEST PASSED
    # ========================================================

    if (
        revision_happened
        and final_pass
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            "REVISION LOOP TEST: PASSED"
        )

        print(
            "=" * 60
        )

        print(
            "\nThe system successfully:"
        )

        print(
            "1. Detected the unsupported answer."
        )

        print(
            "2. Returned CRITIC = FAIL."
        )

        print(
            "3. Revised the answer."
        )

        print(
            "4. Increased revision count to 1."
        )

        print(
            "5. Verified the revised answer."
        )

        print(
            "6. Returned CRITIC = PASS."
        )

    # ========================================================
    # TEST FAILED
    # ========================================================

    else:

        print(
            "\n" + "=" * 60
        )

        print(
            "REVISION LOOP TEST: FAILED"
        )

        print(
            "=" * 60
        )

        if not revision_happened:

            print(
                "\nReason: Revision did not occur."
            )

        if not final_pass:

            print(
                "\nReason: Revised answer did not PASS "
                "critic verification."
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    main()