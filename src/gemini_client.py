from google import genai
from google.genai import errors

from src.config import GEMINI_API_KEY, DEMO_MODE


# ============================================================
# GEMINI CLIENT
# ============================================================

def create_client():
    """
    Create Gemini API client.
    """

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set in .env"
        )

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# DEMO MODE RESPONSES
# ============================================================

def demo_response(prompt):
    """
    Deterministic responses used during development/testing.

    DEMO_MODE does not call Gemini and therefore does not
    consume Gemini API quota.
    """

    prompt_lower = prompt.lower()

    # ========================================================
    # CRITIC
    # ========================================================

    if (
        "verification and self-critique agent" in prompt_lower
        or "you are a critic agent" in prompt_lower
        or "evaluate the draft" in prompt_lower
    ):

        if (
            "100% accuracy" in prompt_lower
            or "always achieves" in prompt_lower
        ):

            return (
                "VERDICT: FAIL\n"
                "REASON: The claim about always achieving "
                "100% accuracy is not supported by the "
                "retrieved context.\n"
                "IMPROVEMENT: Remove the unsupported "
                "accuracy claim."
            )

        return (
            "VERDICT: PASS\n"
            "REASON: The draft answer is directly supported "
            "by the retrieved context and accurately answers "
            "the question.\n"
            "IMPROVEMENT: None"
        )


    # ========================================================
    # PLANNER
    # ========================================================

    if "research planning agent" in prompt_lower:

        return (
            "1. Retrieve relevant information from the "
            "available sources.\n"
            "2. Analyze the retrieved information and "
            "identify the key facts.\n"
            "3. Generate an answer grounded in the "
            "retrieved information.\n"
            "4. Verify the answer against the available "
            "sources."
        )


    # ========================================================
    # SOURCE ROUTER
    # ========================================================

    if "source selection agent" in prompt_lower:

        if (
            "latest" in prompt_lower
            or "recent" in prompt_lower
            or "current" in prompt_lower
            or "today" in prompt_lower
        ):

            return "WEB"

        return "LOCAL"


    # ========================================================
    # DRAFT ANSWER
    # ========================================================

    if (
        "answer generation agent" in prompt_lower
        or "generate a concise" in prompt_lower
    ):

        # ----------------------------------------------------
        # SUPERVISED LEARNING
        # ----------------------------------------------------

        if "supervised learning" in prompt_lower:

            return (
                "Supervised learning uses labeled data "
                "to train a model. Common supervised "
                "learning tasks include classification "
                "and regression."
            )


        # ----------------------------------------------------
        # MACHINE LEARNING + SUPERVISED + UNSUPERVISED
        # ----------------------------------------------------

        if (
            "machine learning" in prompt_lower
            and "unsupervised learning" in prompt_lower
        ):

            return (
                "Machine learning is a branch of artificial "
                "intelligence that allows computers to learn "
                "patterns from data and make predictions or "
                "decisions. Supervised learning uses labeled "
                "data for tasks such as classification and "
                "regression, while unsupervised learning uses "
                "unlabeled data to discover hidden patterns "
                "or structures."
            )


        # ----------------------------------------------------
        # ARTIFICIAL INTELLIGENCE
        # ----------------------------------------------------

        if (
            "what is ai" in prompt_lower
            or "what is artificial intelligence" in prompt_lower
        ):

            return (
                "Artificial intelligence (AI) is a field "
                "of computer science focused on creating "
                "systems that can perform tasks that "
                "normally require human intelligence."
            )


        # ----------------------------------------------------
        # 2026 / WEB RESEARCH
        # ----------------------------------------------------

        if (
            "2026" in prompt_lower
            or "latest developments" in prompt_lower
            or "latest" in prompt_lower
            or "recent developments" in prompt_lower
        ):

            return (
                "Based on the retrieved web information, "
                "major AI developments in 2026 include "
                "advanced agentic AI and multi-agent "
                "systems, stronger self-verification and "
                "memory capabilities, more efficient AI "
                "models and hardware, growing open-source "
                "reasoning models, world models and "
                "embodied AI, and increasing use of AI "
                "in scientific research and software "
                "development."
            )


        # ----------------------------------------------------
        # GENERIC ANSWER
        # ----------------------------------------------------

        return (
            "Based on the retrieved information, "
            "the answer can be summarized from the "
            "available evidence."
        )


    # ========================================================
    # FINALIZER
    # ========================================================

    if (
        "finalizer agent" in prompt_lower
        or "produce only the final answer" in prompt_lower
    ):

        # ----------------------------------------------------
        # SUPERVISED LEARNING
        # ----------------------------------------------------

        if "supervised learning" in prompt_lower:

            return (
                "Supervised learning uses labeled data "
                "to train a model. Common supervised "
                "learning tasks include classification "
                "and regression."
            )


        # ----------------------------------------------------
        # MACHINE LEARNING + UNSUPERVISED
        # ----------------------------------------------------

        if (
            "machine learning" in prompt_lower
            and "unsupervised learning" in prompt_lower
        ):

            return (
                "Machine learning allows computers to learn "
                "patterns from data. Supervised learning uses "
                "labeled data for tasks such as classification "
                "and regression, while unsupervised learning "
                "uses unlabeled data to discover hidden "
                "patterns or structures."
            )


        # ----------------------------------------------------
        # ARTIFICIAL INTELLIGENCE
        # ----------------------------------------------------

        if (
            "artificial intelligence" in prompt_lower
            or "what is ai" in prompt_lower
        ):

            return (
                "Artificial intelligence (AI) is a field "
                "of computer science focused on creating "
                "systems that can perform tasks that "
                "normally require human intelligence."
            )


        # ----------------------------------------------------
        # 2026 AI DEVELOPMENTS
        # ----------------------------------------------------

        if (
            "2026" in prompt_lower
            or "latest developments" in prompt_lower
            or "latest" in prompt_lower
        ):

            return (
                "Key AI developments in 2026 include "
                "agentic and multi-agent systems, improved "
                "AI memory and self-verification, efficient "
                "models and specialized hardware, growth of "
                "open-source reasoning models, world models "
                "and embodied AI, and broader use of AI in "
                "scientific research and software development."
            )


        # ----------------------------------------------------
        # GENERIC FINAL ANSWER
        # ----------------------------------------------------

        return (
            "The final answer is based on the "
            "verified retrieved information."
        )


    # ========================================================
    # GENERIC FALLBACK
    # ========================================================

    return (
        "Supervised learning uses labeled data "
        "to train a model."
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(prompt):
    """
    Generate a response.

    DEMO_MODE=True:
        Use deterministic local responses.

    DEMO_MODE=False:
        Call Gemini API.
    """

    # ========================================================
    # DEMO MODE
    # ========================================================

    if DEMO_MODE:

        return demo_response(
            prompt
        )


    # ========================================================
    # GEMINI MODE
    # ========================================================

    client = create_client()

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        if not response or not response.text:

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text.strip()


    except errors.ClientError as e:

        error_message = str(e)

        if "429" in error_message:

            return (
                "ERROR: Gemini API quota exceeded. "
                "Please wait for the quota to reset "
                "or use another API key/model."
            )

        return (
            "ERROR: Gemini API client error: "
            + error_message
        )


    except errors.ServerError:

        return (
            "ERROR: Gemini server temporarily unavailable. "
            "Please try again later."
        )


    except Exception as e:

        return (
            "ERROR: Unexpected Gemini error: "
            + str(e)
        )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("GEMINI CLIENT TEST")
    print("=" * 60)

    answer = generate_answer(
        "In one sentence, explain what machine learning is."
    )

    print("\nGemini Response:")
    print(answer)

    print("=" * 60)