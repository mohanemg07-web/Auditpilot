"""LangGraph StateGraph wiring the five-node due-diligence pipeline.

Flow:
    START -> planner -> risk_analyst -> critic
    critic --(hallucination & iteration < MAX)--> risk_analyst   (loop)
           --(otherwise)--> memo_writer -> END

State is a plain TypedDict using LangGraph's default last-value reducers, which is
the correct semantics for this state shape (there is no chat-message channel that
would require the `add_messages` reducer). LangSmith tracing is automatic via the
`LANGCHAIN_*` env vars exported by ``backend.config``.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.agents.critic import critic_node, route_after_critic
from backend.agents.memo_writer import memo_writer_node
from backend.agents.planner import planner_node
from backend.agents.risk_analyst import risk_analyst_node


class DueDiligenceState(TypedDict, total=False):
    ticker: str
    filing_text_chunks: list[str]
    planner_output: dict[str, Any]
    risk_findings: dict[str, Any]
    critic_feedback: dict[str, Any]
    final_memo: str
    iteration: int


def build_graph():
    """Construct and compile the AuditPilot StateGraph."""
    graph = StateGraph(DueDiligenceState)

    graph.add_node("planner", planner_node)
    graph.add_node("risk_analyst", risk_analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("memo_writer", memo_writer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "risk_analyst")
    graph.add_edge("risk_analyst", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"risk_analyst": "risk_analyst", "memo_writer": "memo_writer"},
    )
    graph.add_edge("memo_writer", END)

    return graph.compile()


# Compiled graph singleton, imported by the Celery task.
compiled_graph = build_graph()

# Human-readable progress weight per node (used by the task layer).
NODE_PROGRESS = {
    "planner": 35,
    "risk_analyst": 55,
    "critic": 70,
    "memo_writer": 85,
}
