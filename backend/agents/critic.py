"""Critic node (GPT-4o-mini — chosen for cost efficiency).

Acts as a hallucination-control gate. It compares the Risk Analyst's findings
against the source chunks that were actually retrieved for each category and
flags any cited figure/percentage/claim that is not supported by the source.

The node is deliberately conservative: it is instructed to flag anything it
cannot verify. It increments the iteration counter so the graph's conditional
edge can decide whether to loop back to the Risk Analyst.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents import invoke_json, make_llm
from backend.config import settings

SYSTEM_PROMPT = """You are a conservative fact-checking critic auditing a risk \
analyst's work against the SOURCE EXCERPTS from a SEC 10-K filing.

Your job is hallucination control. For every claimed figure, percentage, dollar \
amount, or factual statement, verify it appears in (or is directly implied by) \
the provided source excerpts.

Be conservative: if a claim cannot be clearly verified against the sources, FLAG \
it. It is better to flag an uncertain item than to let a hallucination pass.

Respond ONLY with a JSON object of this exact shape:
{
  "hallucination_detected": true | false,
  "flagged_items": [
    {"category": "credit_risk", "claim": "the unsupported claim", "reason": "why"}
  ],
  "corrections": {
    "credit_risk": "concise instruction on how to fix this category's findings"
  },
  "faithfulness_score": 0.0
}
- faithfulness_score is your 0.0-1.0 self-assessment of how well the analyst's \
claims are grounded in the sources (1.0 = fully grounded).
- corrections maps a risk-category key to a short corrective instruction; only \
include categories that need fixing.
"""


def _build_review_payload(risk_findings: dict[str, Any]) -> str:
    blocks = []
    for cat_key, finding in risk_findings.items():
        sources = finding.get("_retrieved_sources", [])
        source_text = "\n\n".join(f"- {s}" for s in sources)
        claims = {
            "summary": finding.get("summary", ""),
            "findings": finding.get("findings", []),
            "evidence": finding.get("evidence", []),
        }
        blocks.append(
            f"### Category: {cat_key}\n"
            f"ANALYST CLAIMS:\n{claims}\n\n"
            f"SOURCE EXCERPTS (ground truth):\n{source_text}"
        )
    return "\n\n========\n\n".join(blocks)


def critic_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: audit findings, flag hallucinations, bump iteration."""
    risk_findings = state.get("risk_findings", {})
    payload = _build_review_payload(risk_findings)

    llm = make_llm(settings.critic_model, temperature=0.0)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Audit the following risk findings against their source excerpts. "
                "Flag any claim not supported by the sources.\n\n" + payload
            )
        ),
    ]

    feedback = invoke_json(
        llm,
        messages,
        default={
            "hallucination_detected": False,
            "flagged_items": [],
            "corrections": {},
            "faithfulness_score": 1.0,
        },
    )

    feedback.setdefault("hallucination_detected", False)
    feedback.setdefault("flagged_items", [])
    feedback.setdefault("corrections", {})
    feedback.setdefault("faithfulness_score", 1.0)

    iteration = int(state.get("iteration", 0)) + 1
    return {"critic_feedback": feedback, "iteration": iteration}


def route_after_critic(state: dict[str, Any]) -> str:
    """Conditional edge: loop back to the analyst or proceed to the memo."""
    feedback = state.get("critic_feedback", {}) or {}
    iteration = int(state.get("iteration", 0))
    if feedback.get("hallucination_detected") and iteration < settings.max_critic_iterations:
        return "risk_analyst"
    return "memo_writer"
