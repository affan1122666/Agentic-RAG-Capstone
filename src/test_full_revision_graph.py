from src.agent_graph import build_agent_graph
from src.agent_state import AgentState


# ============================================================
# FULL LANGGRAPH REVISION LOOP TEST
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("FULL LANGGRAPH REVISION LOOP TEST")
    print("=" * 60)

    question = "What is supervised learning?"

    print("\nQUESTION:")
    print(question)

    print("\nTEST PURPOSE:")
    print(
        "Verify that the actual LangGraph can perform "
        "FAIL → REVISION → CRITIC → PASS."
    )

    # --------------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------------

    graph = build_agent_graph()

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    state: AgentState = {

        "question": question,

        # Intentionally unsupported draft
        # so that critic should detect the problem.
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

        "source_selection": "LOCAL",

        "sources": [
            "ml_basics.txt"
        ],

        "trace": [
            "TEST: Created intentionally unsupported draft"
        ],

        "revision_count": 0
    }

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # We start directly from the critic portion of the
    # workflow for this integration test.
    #
    # The complete graph itself normally starts from:
    #
    # START → PLANNER → ROUTER → RETRIEVER → DRAFT
    #
    # Here we are testing the self-correction section:
    #
    # DRAFT → CRITIC → REVISION → CRITIC → FINALIZER
    #
    # --------------------------------------------------------

    from src.critic import critique_answer
    from src.revision import revise_answer
    from src.finalizer import finalize_answer

    # ========================================================
    # STEP 1 — FIRST CRITIC
    # ========================================================

    print("\n" + "-" * 60)
    print("STEP 1: FIRST CRITIC")
    print("-" * 60)

    state = critique_answer(
        state
    )

    print(
        state.get(
            "critique",
            "No critique"
        )
    )

    # --------------------------------------------------------
    # Verify FAIL
    # --------------------------------------------------------

    critique = state.get(
        "critique",
        ""
    ).upper()

    if "VERDICT: FAIL" not in critique:

        print(
            "\nERROR: Critic did not detect the unsupported claim."
        )

        return

    # ========================================================
    # STEP 2 — REVISION
    # ========================================================

    print("\n" + "-" * 60)
    print("STEP 2: REVISION")
    print("-" * 60)

    state = revise_answer(
        state
    )

    print("\nREVISED DRAFT:")

    print(
        state.get(
            "draft_answer",
            ""
        )
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

    print(
        state.get(
            "critique",
            "No critique"
        )
    )

    # ========================================================
    # STEP 4 — FINALIZER
    # ========================================================

    critique = state.get(
        "critique",
        ""
    ).upper()

    if "VERDICT: PASS" in critique:

        print("\n" + "-" * 60)
        print("STEP 4: FINALIZER")
        print("-" * 60)

        state = finalize_answer(
            state
        )

        print(
            "\nFINAL ANSWER:"
        )

        print(
            state.get(
                "final_answer",
                state.get(
                    "draft_answer",
                    ""
                )
            )
        )

    # ========================================================
    # RESULT
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
            ""
        )
    )

    print("\nFINAL CRITIQUE:")

    print(
        state.get(
            "critique",
            ""
        )
    )

    # ========================================================
    # TRACE
    # ========================================================

    print("\nTRACE:")

    for index, item in enumerate(
        state.get(
            "trace",
            []
        ),
        start=1
    ):

        print(
            f"{index}. {item}"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    final_critique = state.get(
        "critique",
        ""
    ).upper()

    final_draft = state.get(
        "draft_answer",
        ""
    ).lower()

    revision_count = state.get(
        "revision_count",
        0
    )

    passed = (

        "VERDICT: FAIL" in
        (
            state.get(
                "trace",
                []
            )[1]
            if len(
                state.get(
                    "trace",
                    []
                )
            ) > 1
            else ""
        ).upper()

        or revision_count >= 1
    ) and (

        "VERDICT: PASS" in
        final_critique

    ) and (

        "always achieves 100% accuracy"
        not in final_draft
    )

    print("\n" + "=" * 60)

    if passed:

        print(
            "FULL REVISION LOOP TEST: PASSED"
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
            "4. Increased the revision count."
        )

        print(
            "5. Verified the revised answer."
        )

        print(
            "6. Returned CRITIC = PASS."
        )

        print(
            "7. Produced a corrected final answer."
        )

    else:

        print(
            "FULL REVISION LOOP TEST: FAILED"
        )

        print(
            "=" * 60
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    main()