from langgraph.graph import StateGraph, START, END

from src.agent_state import AgentState

from src.planner import create_plan
from src.retrieval_agent import retrieve_information
from src.web_search_agent import search_web_agent
from src.source_router import route_sources
from src.draft_answer import generate_draft
from src.critic import critique_answer
from src.revision import revise_answer
from src.finalizer import finalize_answer
from src.logger import save_trace


# ============================================================
# CONSTANTS
# ============================================================

MAX_REVISIONS = 2

VALID_SOURCES = {
    "LOCAL",
    "WEB",
    "BOTH",
}


# ============================================================
# SOURCE ROUTER NODE
# ============================================================

def source_router_node(state: AgentState) -> AgentState:

    state = route_sources(state)

    source = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(source, str):
        source = "LOCAL"

    source = source.strip().upper()

    if source not in VALID_SOURCES:
        source = "LOCAL"

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        f"GRAPH ROUTER: Routing to {source} source"
    )

    return {
        **state,
        "source_selection": source,
        "trace": trace,
    }


# ============================================================
# BOTH SOURCES
# ============================================================

def retrieve_both_sources(
    state: AgentState
) -> AgentState:

    # --------------------------------------------------------
    # Retrieve from local documents
    # --------------------------------------------------------

    state = retrieve_information(state)

    # --------------------------------------------------------
    # Retrieve from web
    # --------------------------------------------------------

    state = search_web_agent(state)

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "BOTH SOURCE: Combined local and web information"
    )

    return {
        **state,
        "trace": trace,
    }


# ============================================================
# SOURCE DECISION
# ============================================================

def decide_source(state: AgentState):

    source = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(source, str):
        return "LOCAL"

    source = source.strip().upper()

    if source not in VALID_SOURCES:
        return "LOCAL"

    return source


# ============================================================
# INFORMATION AVAILABILITY
# ============================================================

def information_available(
    state: AgentState
) -> str:
    """
    Check whether enough relevant information exists
    for the selected source strategy.

    LOCAL:
        Local context must exist.

    WEB:
        Web context must exist.

    BOTH:
        BOTH local and web context must exist.
    """

    source = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(source, str):
        source = "LOCAL"

    source = source.strip().upper()

    retrieved_context = state.get(
        "retrieved_context",
        ""
    )

    web_context = state.get(
        "web_context",
        ""
    )

    if not isinstance(
        retrieved_context,
        str
    ):
        retrieved_context = ""

    if not isinstance(
        web_context,
        str
    ):
        web_context = ""

    # --------------------------------------------------------
    # LOCAL
    # --------------------------------------------------------

    if source == "LOCAL":

        if retrieved_context.strip():

            return "available"

        return "unavailable"

    # --------------------------------------------------------
    # WEB
    # --------------------------------------------------------

    if source == "WEB":

        if web_context.strip():

            return "available"

        return "unavailable"

    # --------------------------------------------------------
    # BOTH
    # --------------------------------------------------------

    if source == "BOTH":

        # IMPORTANT:
        # BOTH means BOTH sources are required.

        if (
            retrieved_context.strip()
            and web_context.strip()
        ):

            return "available"

        return "unavailable"

    return "unavailable"


# ============================================================
# NO INFORMATION NODE
# ============================================================

def no_information_node(
    state: AgentState
) -> AgentState:
    """
    Safely handle cases where required source information
    is unavailable.
    """

    source = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(source, str):
        source = "LOCAL"

    source = source.strip().upper()

    # --------------------------------------------------------
    # Message
    # --------------------------------------------------------

    if source == "WEB":

        message = (
            "The available web sources do not provide "
            "relevant information to answer this question."
        )

    elif source == "BOTH":

        message = (
            "The available local and web sources do not "
            "provide enough relevant information to answer "
            "this question."
        )

    else:

        message = (
            "The provided local documents do not contain "
            "relevant information to answer this question."
        )

    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "NO CONTEXT: No relevant information available; "
        "safe answer generated"
    )

    return {
        **state,

        "draft_answer": message,

        "final_answer": message,

        "critique": (
            "VERDICT: PASS\n"
            "REASON: No relevant retrieved information was "
            "available, so the system safely declined to "
            "invent an answer.\n"
            "IMPROVEMENT: None"
        ),

        "revision_count": 0,

        "trace": trace,
    }


# ============================================================
# CRITIC ROUTER
# ============================================================

def revision_decision(
    state: AgentState
):

    critique = state.get(
        "critique",
        ""
    )

    if not isinstance(critique, str):
        critique = ""

    critique_upper = critique.upper()

    revision_count = state.get(
        "revision_count",
        0
    )

    if not isinstance(
        revision_count,
        int
    ):
        revision_count = 0

    trace = state.get(
        "trace",
        []
    ).copy()

    # --------------------------------------------------------
    # FAIL + REVISION AVAILABLE
    # --------------------------------------------------------

    if (
        "VERDICT: FAIL" in critique_upper
        and revision_count < MAX_REVISIONS
    ):

        trace.append(
            "CRITIC ROUTER: Verification failed → REVISION"
        )

        return {
            "route": "revision",
            "state": {
                **state,
                "trace": trace,
            },
        }

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    if "VERDICT: PASS" in critique_upper:

        trace.append(
            "CRITIC ROUTER: Verification passed → FINALIZER"
        )

    # --------------------------------------------------------
    # MAXIMUM REVISIONS
    # --------------------------------------------------------

    else:

        trace.append(
            "CRITIC ROUTER: Maximum revisions reached → FINALIZER"
        )

    return {
        "route": "finalizer",
        "state": {
            **state,
            "trace": trace,
        },
    }


# ============================================================
# SIMPLE CRITIC ROUTER
# ============================================================

def route_after_critic(
    state: AgentState
):
    """
    LangGraph conditional-edge router.

    This version keeps the routing state simple and
    compatible with LangGraph conditional edges.
    """

    critique = state.get(
        "critique",
        ""
    )

    if not isinstance(
        critique,
        str
    ):
        critique = ""

    critique_upper = critique.upper()

    revision_count = state.get(
        "revision_count",
        0
    )

    if not isinstance(
        revision_count,
        int
    ):
        revision_count = 0

    trace = state.get(
        "trace",
        []
    ).copy()

    # --------------------------------------------------------
    # FAIL → REVISION
    # --------------------------------------------------------

    if (
        "VERDICT: FAIL" in critique_upper
        and revision_count < MAX_REVISIONS
    ):

        trace.append(
            "CRITIC ROUTER: Verification failed → REVISION"
        )

        state["trace"] = trace

        return "revision"

    # --------------------------------------------------------
    # PASS → FINALIZER
    # --------------------------------------------------------

    if "VERDICT: PASS" in critique_upper:

        trace.append(
            "CRITIC ROUTER: Verification passed → FINALIZER"
        )

    # --------------------------------------------------------
    # MAX REVISIONS → FINALIZER
    # --------------------------------------------------------

    else:

        trace.append(
            "CRITIC ROUTER: Maximum revisions reached → FINALIZER"
        )

    state["trace"] = trace

    return "finalizer"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_agent_graph():

    graph = StateGraph(
        AgentState
    )

    # ========================================================
    # NODES
    # ========================================================

    graph.add_node(
        "planner",
        create_plan
    )

    graph.add_node(
        "source_router",
        source_router_node
    )

    graph.add_node(
        "retriever",
        retrieve_information
    )

    graph.add_node(
        "web_search",
        search_web_agent
    )

    graph.add_node(
        "both_sources",
        retrieve_both_sources
    )

    graph.add_node(
        "draft",
        generate_draft
    )

    graph.add_node(
        "critic",
        critique_answer
    )

    graph.add_node(
        "revision",
        revise_answer
    )

    graph.add_node(
        "finalizer",
        finalize_answer
    )

    graph.add_node(
        "no_information",
        no_information_node
    )

    # ========================================================
    # START
    # ========================================================

    graph.add_edge(
        START,
        "planner"
    )

    # ========================================================
    # PLANNER → SOURCE ROUTER
    # ========================================================

    graph.add_edge(
        "planner",
        "source_router"
    )

    # ========================================================
    # SOURCE ROUTER → SOURCE
    # ========================================================

    graph.add_conditional_edges(

        "source_router",

        decide_source,

        {
            "LOCAL": "retriever",
            "WEB": "web_search",
            "BOTH": "both_sources",
        }
    )

    # ========================================================
    # LOCAL → INFORMATION CHECK
    # ========================================================

    graph.add_conditional_edges(

        "retriever",

        information_available,

        {
            "available": "draft",
            "unavailable": "no_information",
        }
    )

    # ========================================================
    # WEB → INFORMATION CHECK
    # ========================================================

    graph.add_conditional_edges(

        "web_search",

        information_available,

        {
            "available": "draft",
            "unavailable": "no_information",
        }
    )

    # ========================================================
    # BOTH → INFORMATION CHECK
    # ========================================================

    graph.add_conditional_edges(

        "both_sources",

        information_available,

        {
            "available": "draft",
            "unavailable": "no_information",
        }
    )

    # ========================================================
    # DRAFT → CRITIC
    # ========================================================

    graph.add_edge(
        "draft",
        "critic"
    )

    # ========================================================
    # CRITIC → REVISION / FINALIZER
    # ========================================================

    graph.add_conditional_edges(

        "critic",

        route_after_critic,

        {
            "revision": "revision",
            "finalizer": "finalizer",
        }
    )

    # ========================================================
    # REVISION → CRITIC
    # ========================================================

    graph.add_edge(
        "revision",
        "critic"
    )

    # ========================================================
    # NO INFORMATION → END
    # ========================================================

    graph.add_edge(
        "no_information",
        END
    )

    # ========================================================
    # FINALIZER → END
    # ========================================================

    graph.add_edge(
        "finalizer",
        END
    )

    # ========================================================
    # COMPILE
    # ========================================================

    return graph.compile()


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(
    question: str
):

    if not isinstance(
        question,
        str
    ):
        raise TypeError(
            "Question must be a string."
        )

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    agent = build_agent_graph()

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    initial_state: AgentState = {

        "question": question,

        "trace": [],

        "revision_count": 0,
    }

    # --------------------------------------------------------
    # Execute graph
    # --------------------------------------------------------

    result = agent.invoke(
        initial_state
    )

    # --------------------------------------------------------
    # Save complete execution trace
    # --------------------------------------------------------

    try:

        log_file = save_trace(
            question,
            result
        )

        print(
            f"\nTrace saved to: {log_file}"
        )

    except Exception as e:

        print(
            f"\nWarning: Could not save trace: {e}"
        )

    return result


# ============================================================
# PRINT SOURCES
# ============================================================

def print_sources(
    result
):

    sources = result.get(
        "sources",
        []
    )

    web_sources = result.get(
        "web_sources",
        []
    )

    if not isinstance(
        sources,
        list
    ):
        sources = []

    if not isinstance(
        web_sources,
        list
    ):
        web_sources = []

    all_sources = (
        sources +
        web_sources
    )

    clean_sources = []

    for source in all_sources:

        if not isinstance(
            source,
            str
        ):
            continue

        source = source.strip()

        if (
            source
            and source not in clean_sources
        ):

            clean_sources.append(
                source
            )

    if clean_sources:

        for source in clean_sources:

            print(
                f"- {source}"
            )

    else:

        print(
            "No sources recorded."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("AGENTIC RAG ASSISTANT")
    print("=" * 60)

    question = input(
        "\nEnter your research question: "
    ).strip()

    if not question:

        print(
            "\nError: Question cannot be empty."
        )

        raise SystemExit

    print(
        "\nProcessing your question..."
    )

    print(
        "Please wait..."
    )

    try:

        result = run_agent(
            question
        )

        # ====================================================
        # FINAL ANSWER
        # ====================================================

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(
            result.get(
                "final_answer",
                "No final answer generated."
            )
        )

        # ====================================================
        # SOURCES
        # ====================================================

        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)

        print_sources(
            result
        )

        # ====================================================
        # REVISION COUNT
        # ====================================================

        print("\n" + "=" * 60)
        print("REVISION COUNT")
        print("=" * 60)

        print(
            result.get(
                "revision_count",
                0
            )
        )

        # ====================================================
        # CRITIC
        # ====================================================

        print("\n" + "=" * 60)
        print("CRITIC VERIFICATION")
        print("=" * 60)

        print(
            result.get(
                "critique",
                "No critique available."
            )
        )

        # ====================================================
        # TRACE
        # ====================================================

        print("\n" + "=" * 60)
        print("AGENT TRACE")
        print("=" * 60)

        trace = result.get(
            "trace",
            []
        )

        if isinstance(
            trace,
            list
        ):

            for index, step in enumerate(
                trace,
                start=1
            ):

                print(
                    f"{index}. {step}"
                )

        print("=" * 60)

        print(
            "\nExecution completed successfully."
        )

    except Exception as e:

        print("\n" + "=" * 60)
        print("AGENT ERROR")
        print("=" * 60)

        print(
            f"{type(e).__name__}: {e}"
        )

        print("=" * 60)