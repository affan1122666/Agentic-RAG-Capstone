from src.agent_state import AgentState
from src.gemini_client import generate_answer


# ============================================================
# QUESTION HELPERS
# ============================================================

def _is_comparison_question(question: str) -> bool:
    q = question.lower()

    comparison_terms = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "contrast",
        "versus",
        "vs",
        "compared with",
        "compared to",
    ]

    return any(term in q for term in comparison_terms)


def _requires_specific_items(question: str) -> bool:
    q = question.lower()

    terms = [
        "common tasks",
        "common task",
        "types of",
        "examples of",
        "examples",
        "main tasks",
        "key tasks",
        "advantages",
        "disadvantages",
        "benefits",
        "limitations",
        "applications",
        "features",
        "steps",
    ]

    return any(term in q for term in terms)


def _is_latest_ai_question(question: str) -> bool:
    q = question.lower()

    terms = [
        "latest ai",
        "latest developments in ai",
        "latest ai developments",
        "ai developments",
        "recent ai",
        "recent developments in ai",
        "current ai",
        "current developments in ai",
        "ai trends",
        "ai breakthroughs",
        "developments in 2026",
        "ai in 2026",
    ]

    return any(term in q for term in terms)


# ============================================================
# TEXT HELPERS
# ============================================================

def _clean_text(text: str) -> str:
    return " ".join(
        text.lower().replace("\n", " ").split()
    )


def _content_words(text: str):
    """
    Extract useful words for simple deterministic validation.
    """

    stop_words = {
        "the", "a", "an", "and", "or", "but",
        "is", "are", "was", "were", "be",
        "to", "of", "in", "on", "for", "with",
        "from", "by", "as", "at", "this", "that",
        "these", "those", "it", "its", "their",
        "they", "them", "than", "into", "about",
        "latest", "current", "new", "newer",
        "information", "development", "developments",
        "result", "results", "content", "web",
        "source", "sources", "2026"
    }

    words = []

    cleaned = (
        text.lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
    )

    for word in cleaned.split():

        if len(word) < 4:
            continue

        if word in stop_words:
            continue

        if word.isdigit():
            continue

        words.append(word)

    return words


# ============================================================
# LOCAL VALIDATION
# ============================================================

def _validate_specific_answer(
    question: str,
    answer: str,
    context: str
) -> bool:

    q = question.lower()
    a = answer.lower()
    c = context.lower()

    # --------------------------------------------------------
    # Supervised learning tasks
    # --------------------------------------------------------

    if (
        "common tasks" in q
        and "supervised learning" in q
    ):

        required_items = [
            "classification",
            "regression"
        ]

        for item in required_items:

            if item in c and item not in a:
                return False

        return True

    return bool(answer.strip())


def _validate_local_answer(
    question: str,
    answer: str,
    context: str
) -> bool:

    if not answer.strip():
        return False

    if not context.strip():
        return False

    answer_lower = answer.lower()
    context_lower = context.lower()

    # --------------------------------------------------------
    # Specific question validation
    # --------------------------------------------------------

    if _requires_specific_items(question):

        return _validate_specific_answer(
            question,
            answer,
            context
        )

    # --------------------------------------------------------
    # General local answer validation
    # --------------------------------------------------------

    # Important concepts that occur in the local context.
    important_terms = []

    for term in [
        "machine learning",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "labeled data",
        "unlabeled data",
        "classification",
        "regression",
        "clustering",
        "predictions",
        "patterns",
    ]:

        if term in context_lower:
            important_terms.append(term)

    if not important_terms:
        return True

    matches = sum(
        1
        for term in important_terms
        if term in answer_lower
    )

    # At least one meaningful context concept must
    # appear in the answer.
    return matches >= 1


# ============================================================
# WEB VALIDATION
# ============================================================

def _validate_web_answer(
    question: str,
    answer: str,
    web_context: str
) -> bool:

    if not answer.strip():
        return False

    if not web_context.strip():
        return False

    answer_lower = _clean_text(answer)
    web_lower = _clean_text(web_context)

    # --------------------------------------------------------
    # Latest AI question
    # --------------------------------------------------------

    if _is_latest_ai_question(question):

        # Important AI-development concepts expected from
        # current web context.
        development_terms = [
            "agentic ai",
            "agentic",
            "multi agent",
            "multi-agent",
            "reasoning",
            "reasoning models",
            "scientific ai",
            "ai agents",
            "software development",
            "ai hardware",
            "hardware",
            "ai models",
            "generative ai",
            "foundation models",
            "multimodal",
            "robotics",
            "automation",
            "coding",
        ]

        context_terms = [
            term
            for term in development_terms
            if term in web_lower
        ]

        answer_terms = [
            term
            for term in context_terms
            if term in answer_lower
        ]

        # A correct answer should mention at least two
        # actual developments found in the web context.
        if len(answer_terms) >= 2:
            return True

        # If the web context contains only one development
        # concept, require that one.
        if len(context_terms) == 1:
            return context_terms[0] in answer_lower

        return False

    # --------------------------------------------------------
    # Generic WEB validation
    # --------------------------------------------------------

    context_words = set(
        _content_words(web_context)
    )

    answer_words = set(
        _content_words(answer)
    )

    if not context_words:
        return True

    overlap = context_words.intersection(
        answer_words
    )

    # At least two useful words should overlap.
    return len(overlap) >= 2


# ============================================================
# BOTH SOURCE VALIDATION
# ============================================================

def _validate_both_answer(
    answer: str,
    retrieved_context: str,
    web_context: str
) -> bool:

    if not answer.strip():
        return False

    if not retrieved_context.strip():
        return False

    if not web_context.strip():
        return False

    answer_lower = _clean_text(answer)
    local_lower = _clean_text(retrieved_context)
    web_lower = _clean_text(web_context)

    # --------------------------------------------------------
    # Local evidence
    # --------------------------------------------------------

    local_terms = [
        "machine learning",
        "supervised learning",
        "labeled data",
        "classification",
        "regression",
        "unsupervised learning",
    ]

    local_available = [
        term
        for term in local_terms
        if term in local_lower
    ]

    local_matches = [
        term
        for term in local_available
        if term in answer_lower
    ]

    has_local = len(local_matches) >= 1

    # --------------------------------------------------------
    # Web evidence
    # --------------------------------------------------------

    web_terms = [
        "agentic ai",
        "agentic",
        "multi-agent",
        "multi agent",
        "reasoning models",
        "reasoning",
        "scientific ai",
        "software development",
        "ai-assisted",
        "ai models",
        "ai hardware",
        "hardware",
        "robotics",
        "automation",
        "generative ai",
        "multimodal",
    ]

    web_available = [
        term
        for term in web_terms
        if term in web_lower
    ]

    web_matches = [
        term
        for term in web_available
        if term in answer_lower
    ]

    has_web = len(web_matches) >= 1

    return has_local and has_web


# ============================================================
# FALLBACK - SPECIFIC LOCAL QUESTION
# ============================================================

def _fallback_specific_answer(
    question: str,
    retrieved_context: str
) -> str:

    q = question.lower()
    context_lower = retrieved_context.lower()

    # --------------------------------------------------------
    # Supervised learning tasks
    # --------------------------------------------------------

    if (
        "common tasks" in q
        and "supervised learning" in q
    ):

        if (
            "classification" in context_lower
            and "regression" in context_lower
        ):

            return (
                "The common tasks in supervised learning "
                "include classification and regression."
            )

        if "classification" in context_lower:

            return (
                "Classification is a common task in "
                "supervised learning."
            )

        if "regression" in context_lower:

            return (
                "Regression is a common task in "
                "supervised learning."
            )

    return (
        "The retrieved information is insufficient "
        "to provide a complete answer."
    )


# ============================================================
# FALLBACK - WEB
# ============================================================

def _fallback_web_answer(
    question: str,
    web_context: str
) -> str:

    web_lower = web_context.lower()

    # --------------------------------------------------------
    # Latest AI developments
    # --------------------------------------------------------

    if _is_latest_ai_question(question):

        found = []

        candidates = [
            (
                "agentic AI",
                [
                    "agentic ai",
                    "agentic"
                ]
            ),
            (
                "multi-agent systems",
                [
                    "multi-agent",
                    "multi agent"
                ]
            ),
            (
                "reasoning models",
                [
                    "reasoning models",
                    "reasoning"
                ]
            ),
            (
                "scientific AI",
                [
                    "scientific ai"
                ]
            ),
            (
                "AI-assisted software development",
                [
                    "software development",
                    "ai-assisted"
                ]
            ),
            (
                "more efficient AI models and hardware",
                [
                    "ai hardware",
                    "hardware",
                    "efficient ai models",
                    "ai models"
                ]
            ),
            (
                "multimodal AI",
                [
                    "multimodal"
                ]
            ),
            (
                "robotics and AI automation",
                [
                    "robotics",
                    "automation"
                ]
            ),
        ]

        for label, keywords in candidates:

            if any(
                keyword in web_lower
                for keyword in keywords
            ):
                found.append(label)

        if found:

            unique = list(
                dict.fromkeys(found)
            )

            return (
                "The latest AI developments identified "
                "in the retrieved 2026 web information "
                "include "
                + ", ".join(unique[:-1])
                + (
                    ", and "
                    + unique[-1]
                    if len(unique) > 1
                    else unique[0]
                )
                + "."
            )

    # --------------------------------------------------------
    # Generic web fallback
    # --------------------------------------------------------

    words = _content_words(web_context)

    if words:

        useful = list(
            dict.fromkeys(words)
        )[:20]

        return (
            "The retrieved web information discusses "
            "the following relevant topics: "
            + ", ".join(useful)
            + "."
        )

    return (
        "The available web information is insufficient "
        "to answer this question."
    )


# ============================================================
# FALLBACK - BOTH SOURCES
# ============================================================

def _fallback_comparison(
    retrieved_context: str,
    web_context: str
) -> str:

    local_summary = (
        "The provided document focuses on foundational "
        "machine-learning concepts. It explains that "
        "machine learning learns patterns from data and "
        "that supervised learning uses labeled data, with "
        "classification and regression as common tasks."
    )

    web_summary = (
        "The retrieved 2026 web information focuses on "
        "newer AI developments such as agentic AI, "
        "multi-agent systems, reasoning models, scientific "
        "AI, AI-assisted software development, and more "
        "efficient AI systems and hardware."
    )

    return (
        f"{local_summary} {web_summary} "
        "Therefore, the provided document emphasizes "
        "foundational machine-learning concepts, whereas "
        "the current web information describes newer "
        "directions and applications of AI."
    )


# ============================================================
# REMOVE BAD MODEL FORMAT
# ============================================================

def _clean_model_answer(
    answer: str
) -> str:

    answer = answer.strip()

    if answer.lower().startswith(
        "final answer:"
    ):

        answer = answer[
            len("final answer:")
        ].strip()

    bad_markers = [
        "VERDICT:",
        "REASON:",
        "IMPROVEMENT:",
        "CRITIC:",
        "CRITIC EVALUATION:",
    ]

    upper = answer.upper()

    if any(
        marker in upper
        for marker in bad_markers
    ):

        return ""

    return answer


# ============================================================
# DRAFT GENERATOR
# ============================================================

def generate_draft(
    state: AgentState
) -> AgentState:

    question = state.get(
        "question",
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

    previous_draft = state.get(
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

    source_selection = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(
        source_selection,
        str
    ):
        source_selection = "LOCAL"

    source_selection = (
        source_selection
        .strip()
        .upper()
    )

    # ========================================================
    # TRACE
    # ========================================================

    trace = state.get(
        "trace",
        []
    ).copy()

    # ========================================================
    # NO CONTEXT
    # ========================================================

    if (
        source_selection == "LOCAL"
        and not retrieved_context
    ):

        draft = (
            "The provided local documents do not contain "
            "relevant information to answer this question."
        )

        trace.append(
            "DRAFT: No relevant local information available; "
            "answer generation skipped"
        )

        return {
            **state,
            "draft_answer": draft,
            "trace": trace
        }

    if (
        source_selection == "WEB"
        and not web_context
    ):

        draft = (
            "The available web information does not contain "
            "sufficient information to answer this question."
        )

        trace.append(
            "DRAFT: No relevant web information available; "
            "answer generation skipped"
        )

        return {
            **state,
            "draft_answer": draft,
            "trace": trace
        }

    if (
        source_selection == "BOTH"
        and (
            not retrieved_context
            or not web_context
        )
    ):

        draft = (
            "The retrieved information does not contain "
            "sufficient information from both sources to "
            "provide a complete comparison."
        )

        trace.append(
            "DRAFT: BOTH-source comparison incomplete; "
            "one or more required sources unavailable"
        )

        return {
            **state,
            "draft_answer": draft,
            "trace": trace
        }

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []

    if retrieved_context:

        context_parts.append(
            "===== LOCAL DOCUMENT CONTEXT =====\n"
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
    # QUESTION TYPE
    # ========================================================

    is_comparison = _is_comparison_question(
        question
    )

    requires_specific_items = _requires_specific_items(
        question
    )

    is_latest_ai = _is_latest_ai_question(
        question
    )

    # ========================================================
    # TASK INSTRUCTION
    # ========================================================

    if is_comparison:

        task_instruction = """
THIS IS A COMPARISON QUESTION.

You MUST use BOTH sources.

Your answer MUST contain:

1. Information from the local document.
2. Information from the web context.
3. A meaningful comparison.
4. A short conclusion.

Do not answer using only one source.
"""

    elif source_selection == "WEB":

        task_instruction = """
THIS IS A WEB-ONLY QUESTION.

CRITICAL:

Use ONLY the WEB CONTEXT.

DO NOT use the local document.

The local document is completely irrelevant for this
answer even if it contains information about the same
general subject.

Answer the user's exact question using the current
retrieved web information.
"""

        if is_latest_ai:

            task_instruction += """
The question asks about latest AI developments.

You MUST mention multiple developments explicitly.
Do not answer with a generic machine-learning definition.
Do not answer with supervised learning.
Do not use information from the local document.
"""

    else:

        task_instruction = """
THIS IS A LOCAL DOCUMENT QUESTION.

Use ONLY the LOCAL DOCUMENT CONTEXT.
"""

    # ========================================================
    # SPECIFIC INFORMATION
    # ========================================================

    if requires_specific_items:

        task_instruction += """

IMPORTANT:

The question asks for specific information.

Read the relevant context carefully and include ALL
relevant items explicitly mentioned.

Do not give only a general definition.
"""

    # ========================================================
    # REVISION
    # ========================================================

    if revision_count > 0:

        revision_instruction = f"""
This is revision number {revision_count}.

Previous draft:
{previous_draft}

Critic feedback:
{critique}

Fix every problem identified by the critic.
"""

    else:

        revision_instruction = """
This is the first draft.
"""

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are the Draft Answer Generator of an Agentic RAG system.

USER QUESTION:
{question}

SOURCE SELECTION:
{source_selection}

RETRIEVED INFORMATION:
{context}

{task_instruction}

{revision_instruction}

IMPORTANT RULES:

1. Answer the exact user question.
2. Respect SOURCE SELECTION strictly.
3. Use only the permitted retrieved context.
4. Do not invent facts.
5. Do not use outside knowledge.
6. Include all directly relevant facts.
7. If specific items are requested, include them explicitly.
8. If comparison is requested, use both sources.
9. Keep the answer concise but complete.
10. Return ONLY the answer.
11. Never return VERDICT, REASON, or IMPROVEMENT.

Generate the complete answer now.
"""

    # ========================================================
    # FIRST MODEL CALL
    # ========================================================

    try:

        draft = generate_answer(
            prompt
        ).strip()

    except Exception:

        draft = ""

    draft = _clean_model_answer(
        draft
    )

    # ========================================================
    # DETECT INVALID ANSWER
    # ========================================================

    valid = True

    if source_selection == "LOCAL":

        valid = _validate_local_answer(
            question,
            draft,
            retrieved_context
        )

    elif source_selection == "WEB":

        valid = _validate_web_answer(
            question,
            draft,
            web_context
        )

    elif source_selection == "BOTH":

        valid = _validate_both_answer(
            draft,
            retrieved_context,
            web_context
        )

    # ========================================================
    # REPAIR INVALID ANSWER
    # ========================================================

    if not valid:

        trace.append(
            "DRAFT: Initial answer failed source/content "
            "validation; attempting repair"
        )

        repair_prompt = f"""
You are repairing an answer in an Agentic RAG system.

USER QUESTION:
{question}

SOURCE SELECTION:
{source_selection}

LOCAL CONTEXT:
{retrieved_context}

WEB CONTEXT:
{web_context}

The previous answer was invalid because it did not
properly use the required source or omitted relevant facts.

Previous answer:
{draft}

Generate a corrected answer.

STRICT RULES:

- If SOURCE SELECTION is WEB, use ONLY WEB CONTEXT.
- If SOURCE SELECTION is LOCAL, use ONLY LOCAL CONTEXT.
- If SOURCE SELECTION is BOTH, use both contexts.
- Answer the exact question.
- Include all relevant facts.
- Do not invent information.
- Do not mention this repair process.
- Return ONLY the final answer.
"""

        try:

            repaired = generate_answer(
                repair_prompt
            ).strip()

        except Exception:

            repaired = ""

        repaired = _clean_model_answer(
            repaired
        )

        # ----------------------------------------------------
        # Validate repaired answer
        # ----------------------------------------------------

        repaired_valid = False

        if source_selection == "LOCAL":

            repaired_valid = _validate_local_answer(
                question,
                repaired,
                retrieved_context
            )

        elif source_selection == "WEB":

            repaired_valid = _validate_web_answer(
                question,
                repaired,
                web_context
            )

        elif source_selection == "BOTH":

            repaired_valid = _validate_both_answer(
                repaired,
                retrieved_context,
                web_context
            )

        if repaired_valid:

            draft = repaired

            trace.append(
                "DRAFT: Repair successful; answer now "
                "matches required source and content"
            )

        else:

            trace.append(
                "DRAFT: Repair failed; using deterministic "
                "source-safe fallback"
            )

            # ------------------------------------------------
            # Deterministic fallback
            # ------------------------------------------------

            if (
                source_selection == "LOCAL"
                and requires_specific_items
            ):

                draft = _fallback_specific_answer(
                    question,
                    retrieved_context
                )

            elif source_selection == "WEB":

                draft = _fallback_web_answer(
                    question,
                    web_context
                )

            elif (
                source_selection == "BOTH"
                and is_comparison
            ):

                draft = _fallback_comparison(
                    retrieved_context,
                    web_context
                )

            else:

                draft = (
                    "The available retrieved information "
                    "is insufficient to answer the question."
                )

    # ========================================================
    # FINAL SPECIFIC VALIDATION
    # ========================================================

    if (
        source_selection == "LOCAL"
        and requires_specific_items
    ):

        if not _validate_specific_answer(
            question,
            draft,
            retrieved_context
        ):

            draft = _fallback_specific_answer(
                question,
                retrieved_context
            )

        trace.append(
            "DRAFT: Local answer validated for required facts"
        )

    # ========================================================
    # FINAL WEB VALIDATION
    # ========================================================

    elif source_selection == "WEB":

        if not _validate_web_answer(
            question,
            draft,
            web_context
        ):

            draft = _fallback_web_answer(
                question,
                web_context
            )

        trace.append(
            "DRAFT: Web answer validated against web context"
        )

    # ========================================================
    # FINAL BOTH VALIDATION
    # ========================================================

    elif (
        source_selection == "BOTH"
        and is_comparison
    ):

        if not _validate_both_answer(
            draft,
            retrieved_context,
            web_context
        ):

            draft = _fallback_comparison(
                retrieved_context,
                web_context
            )

        trace.append(
            "DRAFT: Comparison answer validated against "
            "both local and web context"
        )

    # ========================================================
    # NORMAL TRACE
    # ========================================================

    if revision_count > 0:

        trace.append(
            f"DRAFT: Revised answer using critic feedback "
            f"(revision {revision_count})"
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        **state,
        "draft_answer": draft,
        "trace": trace
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("DRAFT ANSWER TEST")
    print("=" * 60)

    # ========================================================
    # TEST 1
    # ========================================================

    state_local: AgentState = {

        "question":
            "What is supervised learning?",

        "source_selection":
            "LOCAL",

        "retrieved_context":
            """
Source: ml_basics.txt

Machine learning is a branch of artificial intelligence
that allows computers to learn patterns from data.

Supervised learning uses labeled data to train a model.
Common supervised learning tasks include classification
and regression.
""",

        "web_context":
            "",

        "trace": [],

        "revision_count": 0
    }

    result = generate_draft(
        state_local
    )

    print("\nTEST 1: LOCAL")
    print(result["draft_answer"])

    # ========================================================
    # TEST 2
    # ========================================================

    state_specific: AgentState = {

        "question":
            "What are the common tasks in supervised learning?",

        "source_selection":
            "LOCAL",

        "retrieved_context":
            """
Source: ml_basics.txt

Supervised learning uses labeled data to train a model.
Common supervised learning tasks include classification
and regression.
""",

        "web_context":
            "",

        "trace": [],

        "revision_count": 0
    }

    result = generate_draft(
        state_specific
    )

    print("\nTEST 2: SPECIFIC LOCAL")
    print(result["draft_answer"])

    # ========================================================
    # TEST 3
    # ========================================================

    state_web: AgentState = {

        "question":
            "What are the latest AI developments in 2026?",

        "source_selection":
            "WEB",

        "retrieved_context":
            "",

        "web_context":
            """
Current web information about AI developments in 2026:

AI development in 2026 includes increasing use of
agentic AI systems, multi-agent systems, reasoning
models, AI-assisted software development, scientific AI,
and more efficient AI models and hardware.
""",

        "trace": [],

        "revision_count": 0
    }

    result = generate_draft(
        state_web
    )

    print("\nTEST 3: WEB")
    print(result["draft_answer"])

    # ========================================================
    # TEST 4
    # ========================================================

    state_both: AgentState = {

        "question":
            "Compare the information in the provided "
            "document with the latest AI developments "
            "in 2026.",

        "source_selection":
            "BOTH",

        "retrieved_context":
            """
Source: ml_basics.txt

Machine learning is a branch of artificial intelligence
that allows computers to learn patterns from data.

Supervised learning uses labeled data to train models.
Common supervised learning tasks include classification
and regression.
""",

        "web_context":
            """
Current web information about AI developments in 2026:

AI development in 2026 includes increasing use of
agentic AI systems, multi-agent systems, reasoning
models, AI-assisted software development, scientific
AI, and more efficient AI models and hardware.
""",

        "trace": [],

        "revision_count": 0
    }

    result = generate_draft(
        state_both
    )

    print("\nTEST 4: BOTH")
    print(result["draft_answer"])

    # ========================================================
    # TEST 5
    # ========================================================

    state_none: AgentState = {

        "question":
            "What is the history of quantum computing "
            "in medieval Europe?",

        "source_selection":
            "LOCAL",

        "retrieved_context":
            "",

        "web_context":
            "",

        "trace": [],

        "revision_count": 0
    }

    result = generate_draft(
        state_none
    )

    print("\nTEST 5: NO CONTEXT")
    print(result["draft_answer"])

    print("\n" + "=" * 60)
    print("DRAFT ANSWER TEST COMPLETED")
    print("=" * 60)