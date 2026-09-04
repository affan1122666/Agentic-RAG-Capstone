import time

from langgraph.graph import StateGraph, START, END

from src.agent_state import AgentState

from src.planner import create_plan
from src.retrieval_agent import retrieve_information
from src.web_search_agent import search_web_agent
from src.source_router import route_sources
from src.draft_answer import generate_draft
from src.critic import critique_answer
from src.revision import revise_answer
from src.logger import save_trace


# ============================================================
# CONFIGURATION
# ============================================================

MAX_REVISIONS = 2

VALID_SOURCES = {
    "LOCAL",
    "WEB",
    "BOTH"
}


# ============================================================
# TIMING HELPER
# ============================================================

def _record_timing(
    state: AgentState,
    step_name: str,
    start_time: float
) -> AgentState:

    elapsed = time.perf_counter() - start_time

    timings = state.get(
        "timings",
        {}
    ).copy()

    timings[step_name] = elapsed

    return {
        **state,
        "timings": timings
    }


# ============================================================
# PLANNER NODE
# ============================================================

def planner_node(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    result = create_plan(state)

    return _record_timing(
        result,
        "Planner",
        start_time
    )


# ============================================================
# SOURCE ROUTER NODE
# ============================================================

def source_router_node(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    state = route_sources(state)

    source = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(
        source,
        str
    ):
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

    state = {
        **state,
        "source_selection": source,
        "trace": trace
    }

    return _record_timing(
        state,
        "Source Router",
        start_time
    )


# ============================================================
# RETRIEVER NODE
# ============================================================

def retriever_node(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    result = retrieve_information(
        state
    )

    return _record_timing(
        result,
        "Retriever",
        start_time
    )


# ============================================================
# WEB SEARCH NODE
# ============================================================

def web_search_node(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    result = search_web_agent(
        state
    )

    return _record_timing(
        result,
        "Web Search",
        start_time
    )


# ============================================================
# BOTH SOURCES NODE
# ============================================================

def retrieve_both_sources(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # LOCAL RETRIEVAL
    # --------------------------------------------------------

    local_start = time.perf_counter()

    state = retrieve_information(
        state
    )

    state = _record_timing(
        state,
        "Local Retrieval",
        local_start
    )

    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    web_start = time.perf_counter()

    state = search_web_agent(
        state
    )

    state = _record_timing(
        state,
        "Web Search",
        web_start
    )

    # --------------------------------------------------------
    # TRACE
    # --------------------------------------------------------

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "BOTH SOURCE: Combined local and web information"
    )

    state = {
        **state,
        "trace": trace
    }

    return _record_timing(
        state,
        "Both Sources",
        start_time
    )


# ============================================================
# SOURCE DECISION
# ============================================================

def decide_source(
    state: AgentState
):

    source = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(
        source,
        str
    ):
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

    source = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(
        source,
        str
    ):
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

        return (
            "available"
            if retrieved_context.strip()
            else "unavailable"
        )

    # --------------------------------------------------------
    # WEB
    # --------------------------------------------------------

    if source == "WEB":

        return (
            "available"
            if web_context.strip()
            else "unavailable"
        )

    # --------------------------------------------------------
    # BOTH
    # --------------------------------------------------------

    if source == "BOTH":

        return (
            "available"
            if (
                retrieved_context.strip()
                and web_context.strip()
            )
            else "unavailable"
        )

    return "unavailable"


# ============================================================
# DRAFT NODE
# ============================================================

def draft_node(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    result = generate_draft(
        state
    )

    return _record_timing(
        result,
        "Draft",
        start_time
    )


# ============================================================
# CRITIC NODE
# ============================================================

def critic_node(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    result = critique_answer(
        state
    )

    return _record_timing(
        result,
        "Critic",
        start_time
    )


# ============================================================
# REVISION NODE
# ============================================================

def revision_node(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    result = revise_answer(
        state
    )

    return _record_timing(
        result,
        "Revision",
        start_time
    )


# ============================================================
# NO INFORMATION NODE
# ============================================================

def no_information_node(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    source = state.get(
        "source_selection",
        "LOCAL"
    )

    if not isinstance(
        source,
        str
    ):
        source = "LOCAL"

    source = source.strip().upper()

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

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "NO CONTEXT: No relevant information available; "
        "safe answer generated"
    )

    state = {
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

        "trace": trace
    }

    return _record_timing(
        state,
        "No Information",
        start_time
    )


# ============================================================
# CRITIC ROUTER
# ============================================================

def route_after_critic(
    state: AgentState
):

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
    # REVISION REQUIRED
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
    # PASS
    # --------------------------------------------------------

    if "VERDICT: PASS" in critique_upper:

        trace.append(
            "CRITIC ROUTER: Verification passed → FINAL ANSWER"
        )

    # --------------------------------------------------------
    # MAX REVISIONS
    # --------------------------------------------------------

    else:

        trace.append(
            "CRITIC ROUTER: Maximum revisions reached "
            "→ FINAL ANSWER"
        )

    state["trace"] = trace

    return "final_answer"


# ============================================================
# FINAL ANSWER NODE
# ============================================================

def final_answer_node(
    state: AgentState
) -> AgentState:

    start_time = time.perf_counter()

    draft = state.get(
        "draft_answer",
        ""
    )

    if not isinstance(
        draft,
        str
    ):
        draft = ""

    trace = state.get(
        "trace",
        []
    ).copy()

    trace.append(
        "FINAL ANSWER: Critic-verified draft returned directly"
    )

    state = {
        **state,
        "final_answer": draft,
        "trace": trace
    }

    return _record_timing(
        state,
        "Final Answer",
        start_time
    )


# ============================================================
# BUILD AGENT GRAPH
# ============================================================

def build_agent_graph():

    graph = StateGraph(
        AgentState
    )

    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    graph.add_node(
        "planner",
        planner_node
    )

    graph.add_node(
        "source_router",
        source_router_node
    )

    graph.add_node(
        "retriever",
        retriever_node
    )

    graph.add_node(
        "web_search",
        web_search_node
    )

    graph.add_node(
        "both_sources",
        retrieve_both_sources
    )

    graph.add_node(
        "draft",
        draft_node
    )

    graph.add_node(
        "critic",
        critic_node
    )

    graph.add_node(
        "revision",
        revision_node
    )

    graph.add_node(
        "final_answer",
        final_answer_node
    )

    graph.add_node(
        "no_information",
        no_information_node
    )

    # --------------------------------------------------------
    # START → PLANNER
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "planner"
    )

    # --------------------------------------------------------
    # PLANNER → SOURCE ROUTER
    # --------------------------------------------------------

    graph.add_edge(
        "planner",
        "source_router"
    )

    # --------------------------------------------------------
    # SOURCE ROUTER → SOURCE
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "source_router",
        decide_source,
        {
            "LOCAL": "retriever",
            "WEB": "web_search",
            "BOTH": "both_sources"
        }
    )

    # --------------------------------------------------------
    # RETRIEVER → DRAFT / NO INFORMATION
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "retriever",
        information_available,
        {
            "available": "draft",
            "unavailable": "no_information"
        }
    )

    # --------------------------------------------------------
    # WEB SEARCH → DRAFT / NO INFORMATION
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "web_search",
        information_available,
        {
            "available": "draft",
            "unavailable": "no_information"
        }
    )

    # --------------------------------------------------------
    # BOTH SOURCES → DRAFT / NO INFORMATION
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "both_sources",
        information_available,
        {
            "available": "draft",
            "unavailable": "no_information"
        }
    )

    # --------------------------------------------------------
    # DRAFT → CRITIC
    # --------------------------------------------------------

    graph.add_edge(
        "draft",
        "critic"
    )

    # --------------------------------------------------------
    # CRITIC → REVISION / FINAL
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "revision": "revision",
            "final_answer": "final_answer"
        }
    )

    # --------------------------------------------------------
    # REVISION → CRITIC
    # --------------------------------------------------------

    graph.add_edge(
        "revision",
        "critic"
    )

    # --------------------------------------------------------
    # NO INFORMATION → END
    # --------------------------------------------------------

    graph.add_edge(
        "no_information",
        END
    )

    # --------------------------------------------------------
    # FINAL ANSWER → END
    # --------------------------------------------------------

    graph.add_edge(
        "final_answer",
        END
    )

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
    # TOTAL TIMER
    # --------------------------------------------------------

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # GRAPH BUILD TIMER
    # --------------------------------------------------------

    graph_start = time.perf_counter()

    agent = build_agent_graph()

    graph_build_time = (
        time.perf_counter()
        - graph_start
    )

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_state: AgentState = {

        "question": question,

        "trace": [],

        "revision_count": 0,

        "timings": {}
    }

    # --------------------------------------------------------
    # RUN GRAPH
    # --------------------------------------------------------

    result = agent.invoke(
        initial_state
    )

    # --------------------------------------------------------
    # TOTAL TIME
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    timings = result.get(
        "timings",
        {}
    ).copy()

    timings["Graph Build"] = (
        graph_build_time
    )

    timings["Total Execution"] = (
        total_time
    )

    result["timings"] = timings

    # --------------------------------------------------------
    # SAVE TRACE
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
        sources
        + web_sources
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
# PRINT TIMINGS
# ============================================================

def print_timings(
    result
):

    timings = result.get(
        "timings",
        {}
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EXECUTION TIMING"
    )

    print(
        "=" * 60
    )

    if not timings:

        print(
            "No timing information available."
        )

        return

    # --------------------------------------------------------
    # DISPLAY TIMINGS
    # --------------------------------------------------------

    for step_name, elapsed in timings.items():

        if not isinstance(
            elapsed,
            (int, float)
        ):
            continue

        print(
            f"{step_name:<25} "
            f"{elapsed:.3f} seconds"
        )

    # --------------------------------------------------------
    # FIND SLOWEST ACTUAL STAGE
    # --------------------------------------------------------

    stage_timings = {

        key: value

        for key, value in timings.items()

        if key not in {
            "Total Execution",
            "Graph Build",
            "Both Sources"
        }

        and isinstance(
            value,
            (int, float)
        )
    }

    if stage_timings:

        slowest_stage = max(
            stage_timings,
            key=stage_timings.get
        )

        slowest_time = (
            stage_timings[
                slowest_stage
            ]
        )

        print(
            "\n"
            + "-" * 60
        )

        print(
            "SLOWEST STAGE"
        )

        print(
            "-" * 60
        )

        print(
            f"{slowest_stage}: "
            f"{slowest_time:.3f} seconds"
        )

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total_time = timings.get(
        "Total Execution"
    )

    if isinstance(
        total_time,
        (int, float)
    ):

        print(
            "\n"
            + "-" * 60
        )

        print(
            "TOTAL RESPONSE TIME"
        )

        print(
            "-" * 60
        )

        print(
            f"{total_time:.3f} seconds"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 60
    )

    print(
        "AGENTIC RAG ASSISTANT"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = input(
        "\nEnter your research question: "
    ).strip()

    if not question:

        print(
            "\nError: Question cannot be empty."
        )

        raise SystemExit

    # --------------------------------------------------------
    # PROCESSING
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "FINAL ANSWER"
        )

        print(
            "=" * 60
        )

        print(
            result.get(
                "final_answer",
                "No final answer generated."
            )
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "SOURCES"
        )

        print(
            "=" * 60
        )

        print_sources(
            result
        )

        # ----------------------------------------------------
        # REVISION COUNT
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "REVISION COUNT"
        )

        print(
            "=" * 60
        )

        print(
            result.get(
                "revision_count",
                0
            )
        )

        # ----------------------------------------------------
        # CRITIC VERIFICATION
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "CRITIC VERIFICATION"
        )

        print(
            "=" * 60
        )

        print(
            result.get(
                "critique",
                "No critique available."
            )
        )

        # ----------------------------------------------------
        # TIMING
        # ----------------------------------------------------

        print_timings(
            result
        )

        # ----------------------------------------------------
        # AGENT TRACE
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "AGENT TRACE"
        )

        print(
            "=" * 60
        )

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

        print(
            "=" * 60
        )

        print(
            "\nExecution completed successfully."
        )

    except Exception as e:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "AGENT ERROR"
        )

        print(
            "=" * 60
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        print(
            "=" * 60
        )