from src.agent_state import AgentState
from src.critic import critique_answer
from src.draft_answer import generate_draft


def main():

    print("\n" + "=" * 60)
    print("AGENT REVISION LOOP TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Initial intentionally incorrect draft
    # --------------------------------------------------------

    state: AgentState = {

        "question":
            "What is supervised learning?",

        "draft_answer":
            (
                "Supervised learning always achieves "
                "100% accuracy."
            ),

        "retrieved_context":
            """
Source: ml_basics.txt

Supervised learning uses labeled data to train a model.
Common supervised learning tasks include classification
and regression.
""",

        "trace": [
            "TEST: Starting with intentionally unsupported answer"
        ],

        "revision_count": 1
    }

    # --------------------------------------------------------
    # First critic
    # --------------------------------------------------------

    state = critique_answer(
        state
    )

    print("\nFIRST CRITIC:")
    print(
        state["critique"]
    )

    # --------------------------------------------------------
    # Check FAIL
    # --------------------------------------------------------

    if "VERDICT: FAIL" not in state["critique"]:

        print(
            "\nERROR: Expected critic to return FAIL."
        )

        return

    print(
        "\nResult: FAIL detected successfully."
    )

    # --------------------------------------------------------
    # Revise draft
    # --------------------------------------------------------

    state = generate_draft(
        state
    )

    print("\nREVISED DRAFT:")
    print(
        state["draft_answer"]
    )

    # --------------------------------------------------------
    # Second critic
    # --------------------------------------------------------

    state = critique_answer(
        state
    )

    print("\nSECOND CRITIC:")
    print(
        state["critique"]
    )

    # --------------------------------------------------------
    # Check PASS
    # --------------------------------------------------------

    if "VERDICT: PASS" in state["critique"]:

        print(
            "\nResult: Revised answer passed verification."
        )

    else:

        print(
            "\nWARNING: Revised answer did not pass."
        )

    # --------------------------------------------------------
    # Final trace
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("REVISION TRACE")
    print("=" * 60)

    for step in state.get(
        "trace",
        []
    ):

        print(
            step
        )

    print("=" * 60)


if __name__ == "__main__":

    main()