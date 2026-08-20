from pathlib import Path
from datetime import datetime

from src.config import LOGS_DIR


def save_trace(question, state):
    """
    Save the complete agent execution trace to a log file.
    """

    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(LOGS_DIR) / f"agent_trace_{timestamp}.txt"

    trace = state.get("trace", [])

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("AGENTIC RAG EXECUTION TRACE\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Question:\n{question}\n\n")

        f.write("Final Answer:\n")
        f.write(state.get("final_answer", "No final answer") + "\n\n")

        f.write("Sources:\n")
        for source in state.get("sources", []):
            f.write(f"- {source}\n")

        f.write("\nAgent Trace:\n")
        for step in trace:
            f.write(f"- {step}\n")

        f.write("\n" + "=" * 60 + "\n")

    return log_file