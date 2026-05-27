"""Risk Analyst node (GPT-4o).

Extracts findings for exactly six risk categories. For each category it runs a
RAG retrieval against ChromaDB (top-5 chunks scoped to the ticker), then asks the
model to produce findings grounded in — and citing — that evidence (specific
figures, percentages, dollar amounts).

On Critic feedback loops (iteration > 0) the prior critic corrections are injected
so the analyst can fix flagged hallucinations.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents import invoke_json, make_llm
from backend.config import settings
from backend.tools import vector_store

# Retrieval prompts seed each category's RAG query.
_CATEGORY_QUERIES = {
    "market_risk": "market risk interest rate foreign currency commodity price equity exposure",
    "credit_risk": "credit risk counterparty default allowance for credit losses receivables",
    "liquidity_risk": "liquidity risk cash flow working capital debt maturities credit facilities",
    "operational_risk": "operational risk supply chain manufacturing cybersecurity business continuity",
    "regulatory_legal_risk": "regulatory legal risk litigation compliance government regulation investigations",
    "strategic_risk": "strategic risk competition market share product roadmap acquisitions reputation",
}

SYSTEM_PROMPT = """You are a meticulous financial risk analyst examining a SEC \
10-K filing. You are analyzing ONE risk category at a time.

Rules:
- Ground EVERY claim in the provided source excerpts. Do not invent figures.
- Quote specific numbers: dollar amounts, percentages, dates, and balances when \
the source contains them.
- If the source lacks evidence for this category, say so honestly rather than \
fabricating.

Respond ONLY with a JSON object of this exact shape:
{
  "summary": "2-4 sentence assessment of this risk category for the company",
  "findings": ["specific finding with cited figures", "..."],
  "evidence": ["short verbatim quote or figure copied from the source", "..."],
  "severity": "Low | Moderate | High",
  "confidence": "Low | Medium | High"
}
"""


def _format_chunks(chunks: list[dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[Source {i} | {c.get('section', 'General')} | "
            f"filed {c.get('filing_date', '')}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(lines)


def _analyze_category(
    ticker: str,
    cat_key: str,
    cat_label: str,
    plan: dict[str, Any],
    corrections: dict[str, Any] | None,
) -> dict[str, Any]:
    query = _CATEGORY_QUERIES[cat_key]
    chunks = vector_store.retrieve(query, ticker=ticker, n_results=settings.retrieval_top_k)
    source_block = _format_chunks(chunks)

    correction_note = ""
    if corrections and cat_key in corrections:
        correction_note = (
            "\n\nIMPORTANT — a prior review flagged issues in your earlier analysis "
            f"of this category. Apply these corrections and re-ground your claims:\n"
            f"{corrections[cat_key]}\n"
        )

    llm = make_llm(settings.risk_model, temperature=0.1)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Company: {ticker}\n"
                f"Risk category to analyze: {cat_label}\n"
                f"Analyst focus areas: {plan.get('focus_areas', [])}\n"
                f"{correction_note}\n"
                f"Source excerpts retrieved for this category:\n\n{source_block}"
            )
        ),
    ]

    findings = invoke_json(
        llm,
        messages,
        default={
            "summary": "",
            "findings": [],
            "evidence": [],
            "severity": "Moderate",
            "confidence": "Low",
        },
    )
    # Attach the retrieved sources so the Critic can verify against them.
    findings["_retrieved_sources"] = [c["text"] for c in chunks]
    return findings


def risk_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: populate findings for all six risk categories."""
    ticker = state["ticker"]
    plan = state.get("planner_output", {})
    critic_feedback = state.get("critic_feedback", {}) or {}
    corrections = critic_feedback.get("corrections") if critic_feedback else None

    risk_findings: dict[str, Any] = {}
    for cat_key, cat_label in settings.risk_categories.items():
        risk_findings[cat_key] = _analyze_category(
            ticker, cat_key, cat_label, plan, corrections
        )

    return {"risk_findings": risk_findings}
