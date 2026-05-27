"""Memo Writer node (GPT-4o, streaming).

Synthesizes the planner output, risk findings, and critic corrections into a
professional markdown due-diligence memo with the required six-section structure.

Streaming: the node streams the model's tokens. If a ``token_callback`` is found
in the LangGraph ``config["configurable"]`` mapping, each token is forwarded to it
(the Celery task wires this to a Redis pub/sub channel for SSE). The fully
assembled markdown is written to ``final_memo`` in the state.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.agents import make_llm
from backend.config import settings

SYSTEM_PROMPT = """You are a senior financial analyst writing a formal due \
diligence memo for an investment committee, based strictly on a SEC 10-K filing \
analysis. Write in clear, professional, objective prose.

Use cited figures (dollar amounts, percentages, dates) wherever the analysis \
provides them. Do NOT introduce facts that are not present in the supplied \
analysis. Where the critic flagged corrections, use the corrected values.

Produce a complete markdown memo with EXACTLY these sections and headings:

## Executive Summary
## Company Overview
## Risk Analysis
### Market Risk
### Credit Risk
### Liquidity Risk
### Operational Risk
### Regulatory/Legal Risk
### Strategic Risk
## Key Findings
## Recommendation
## Appendix: Data Sources

Aim for thoroughness and completeness (target 4.3/5.0 on a completeness rubric). \
Each risk subsection must summarize findings with cited evidence.
"""


def _build_user_prompt(state: dict[str, Any]) -> str:
    ticker = state["ticker"]
    plan = state.get("planner_output", {})
    risk_findings = state.get("risk_findings", {})
    critic_feedback = state.get("critic_feedback", {}) or {}

    # Strip the bulky retrieved-source text from the findings handed to the writer.
    trimmed: dict[str, Any] = {}
    for cat, finding in risk_findings.items():
        trimmed[cat] = {k: v for k, v in finding.items() if k != "_retrieved_sources"}

    return (
        f"Company ticker: {ticker}\n\n"
        f"## Analysis Plan\n{plan}\n\n"
        f"## Risk Findings (six categories)\n{trimmed}\n\n"
        f"## Critic Review\n"
        f"hallucination_detected: {critic_feedback.get('hallucination_detected')}\n"
        f"faithfulness_score: {critic_feedback.get('faithfulness_score')}\n"
        f"corrections: {critic_feedback.get('corrections', {})}\n\n"
        "Write the full due diligence memo now, following the required section "
        "structure exactly."
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def _stream_memo(messages: list, on_token: Callable[[str], None] | None) -> str:
    llm = make_llm(settings.memo_model, temperature=0.3, streaming=True)
    parts: list[str] = []
    for chunk in llm.stream(messages):
        token = chunk.content or ""
        if token:
            parts.append(token)
            if on_token is not None:
                try:
                    on_token(token)
                except Exception:
                    # Never let a streaming-sink failure abort memo generation.
                    pass
    return "".join(parts)


def memo_writer_node(
    state: dict[str, Any], config: RunnableConfig = None
) -> dict[str, Any]:
    # NOTE: annotated as bare `RunnableConfig` (not `RunnableConfig | None`) on
    # purpose: with `from __future__ import annotations` active, LangGraph's node
    # signature check only injects the runtime config when the stringified
    # annotation is exactly "RunnableConfig" or "Optional[RunnableConfig]". That
    # injected config carries our live token-streaming callback.
    """LangGraph node: stream and assemble the final markdown memo."""
    on_token: Callable[[str], None] | None = None
    if config:
        on_token = (config.get("configurable") or {}).get("token_callback")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(state)),
    ]
    memo = _stream_memo(messages, on_token)
    return {"final_memo": memo}
