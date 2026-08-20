import streamlit as st

from src.agent_graph import run_agent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .metric-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Agentic RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    An intelligent research assistant using Agentic AI, RAG,
    Web Search, Critic Verification, and Revision.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.success("🟢 Agentic RAG System Online")

    st.divider()

    st.subheader("🔄 Agent Pipeline")

    st.markdown(
        """
        **1. 🧠 Research Planner**

        Creates a research plan.

        **2. 🔀 Source Router**

        Selects Local RAG or Web Search.

        **3. 📚 Retrieval**

        Retrieves relevant information.

        **4. ✍️ Draft Generator**

        Creates a grounded answer.

        **5. 🔍 Critic**

        Verifies the generated answer.

        **6. 🔄 Revision Loop**

        Corrects unsupported information.

        **7. ✅ Finalizer**

        Produces the verified answer.
        """
    )

    st.divider()

    st.info(
        "The system automatically chooses between "
        "local documents and web search based on the question."
    )


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("🔎 Ask a Research Question")

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Examples:\n"
        "• What is supervised learning?\n"
        "• Explain machine learning and unsupervised learning.\n"
        "• What are the latest developments in AI in 2026?"
    ),
    height=140
)


# ============================================================
# RESEARCH BUTTON
# ============================================================

research_clicked = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN AGENT
# ============================================================

if research_clicked:

    if not question.strip():

        st.warning(
            "⚠️ Please enter a research question first."
        )

    else:

        with st.spinner(
            "🤖 Agent is researching your question..."
        ):

            try:

                result = run_agent(
                    question.strip()
                )

                st.session_state["result"] = result

            except Exception as e:

                st.error(
                    f"❌ Agent execution failed:\n\n{e}"
                )


# ============================================================
# DISPLAY RESULT
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()


    # ========================================================
    # SYSTEM METRICS
    # ========================================================

    source_selection = result.get(
        "source_selection",
        "UNKNOWN"
    )

    trace = result.get(
        "trace",
        []
    )

    revision_count = result.get(
        "revision_count",
        0
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Source",
            source_selection
        )


    with col2:

        st.metric(
            "Agent Steps",
            len(trace)
        )


    with col3:

        st.metric(
            "Revisions",
            revision_count
        )


    st.divider()


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    st.subheader("📝 Final Answer")

    final_answer = result.get(
        "final_answer",
        ""
    )

    if final_answer:

        st.markdown(final_answer)

    else:

        st.warning(
            "No final answer was generated."
        )


    # ========================================================
    # SOURCES
    # ========================================================

    st.subheader("📚 Sources")

    sources = result.get(
        "sources",
        []
    )

    web_sources = result.get(
        "web_sources",
        []
    )


    # --------------------------------------------------------
    # LOCAL SOURCES
    # --------------------------------------------------------

    if sources:

        st.markdown("### 📄 Local Sources")

        for source in sources:

            if isinstance(source, str):

                if (
                    source.startswith("http://")
                    or source.startswith("https://")
                ):

                    st.markdown(
                        f"- 🔗 [{source}]({source})"
                    )

                else:

                    st.markdown(
                        f"- 📄 `{source}`"
                    )

            elif isinstance(source, dict):

                title = source.get(
                    "title",
                    source.get(
                        "url",
                        "Local Source"
                    )
                )

                url = source.get(
                    "url",
                    ""
                )

                if url:

                    st.markdown(
                        f"- 🔗 [{title}]({url})"
                    )

                else:

                    st.markdown(
                        f"- 📄 `{title}`"
                    )


    # --------------------------------------------------------
    # WEB SOURCES
    # --------------------------------------------------------

    if web_sources:

        st.markdown("### 🌐 Web Sources")

        for source in web_sources:

            if isinstance(source, str):

                if (
                    source.startswith("http://")
                    or source.startswith("https://")
                ):

                    st.markdown(
                        f"- 🔗 [{source}]({source})"
                    )

                else:

                    st.markdown(
                        f"- 🔗 {source}"
                    )

            elif isinstance(source, dict):

                title = source.get(
                    "title",
                    "Web Source"
                )

                url = source.get(
                    "url",
                    ""
                )

                if url:

                    st.markdown(
                        f"- 🔗 [{title}]({url})"
                    )

                else:

                    st.markdown(
                        f"- 🌐 {title}"
                    )


    if not sources and not web_sources:

        st.info(
            "No sources were returned."
        )


    # ========================================================
    # AGENT TRACE
    # ========================================================

    with st.expander(
        "🔍 View Agent Trace"
    ):

        if trace:

            for index, step in enumerate(
                trace,
                start=1
            ):

                st.markdown(
                    f"**{index}.** {step}"
                )

        else:

            st.info(
                "No trace available."
            )


    # ========================================================
    # CRITIC / VERIFICATION
    # ========================================================

    critique = result.get(
        "critique",
        ""
    )

    if critique:

        with st.expander(
            "🛡️ View Critic Verification"
        ):

            st.markdown(critique)


    # ========================================================
    # FULL AGENT STATE
    # ========================================================

    with st.expander(
        "⚙️ View Full Agent State"
    ):

        st.json(result)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Agentic RAG Capstone Project • "
    "Planner → Router → Retrieval → Draft → Critic → Revision → Finalizer"
)