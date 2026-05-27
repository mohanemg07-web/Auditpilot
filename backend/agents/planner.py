"""Planner node (GPT-4o).

Reads the ticker plus the first N filing chunks and produces a structured
analysis plan: focus areas, key questions, and prioritized risks. This plan
steers the Risk Analyst's retrieval queries downstream.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents import invoke_json, make_llm
from backend.config import settings

SYSTEM_PROMPT = """You are a senior financial analyst planning a due diligence \
review of a public company based on its SEC 10-K filing.

Given excerpts from the filing, produce a focused analysis plan. Identify the \
most material focus areas, the key questions an investor must answer, and which \
risk categories deserve the most scrutiny for THIS company specifically.

The six risk categories under consideration are: Market Risk, Credit Risk, \
Liquidity Risk, Operational Risk, Regulatory/Legal Risk, and Strategic Risk.

Respond ONLY with a JSON object of the exact shape:
{
  "focus_areas": ["..."],
  "key_questions": ["..."],
  "risk_priorities": ["..."]
}
- focus_areas: 4-7 concrete themes specific to this company.
- key_questions: 5-8 pointed questions a due-diligence analyst must resolve.
- risk_priorities: an ordered subset/ranking of the six risk categories, most \
material first.
"""


def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: produce the analysis plan."""
    ticker = state["ticker"]
    chunks: list[str] = state.get("filing_text_chunks", [])
    context = "\n\n---\n\n".join(chunks[: settings.planner_context_chunks])

    llm = make_llm(settings.planner_model, temperature=0.2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Company ticker: {ticker}\n\n"
                f"Filing excerpts (first {settings.planner_context_chunks} chunks):\n\n"
                f"{context}"
            )
        ),
    ]

    plan = invoke_json(
        llm,
        messages,
        default={"focus_areas": [], "key_questions": [], "risk_priorities": []},
    )

    # Normalize shape so downstream nodes can rely on the keys existing.
    plan.setdefault("focus_areas", [])
    plan.setdefault("key_questions", [])
    plan.setdefault("risk_priorities", list(settings.risk_categories.values()))

    return {"planner_output": plan}
