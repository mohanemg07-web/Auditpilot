"""LangGraph agent nodes and shared LLM helpers for AuditPilot."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings


def make_llm(model: str, *, temperature: float = 0.1, streaming: bool = False) -> ChatOpenAI:
    """Build a ChatOpenAI client with our standard settings."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
        api_key=settings.openai_api_key or None,
        timeout=90,
        max_retries=0,  # we manage retries via tenacity at the call site
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def invoke_json(llm: ChatOpenAI, messages: list, *, default: Any = None) -> Any:
    """Invoke an LLM expected to return JSON, with backoff + tolerant parsing.

    Requests JSON object mode where supported, then robustly extracts the first
    JSON object from the response.
    """
    bound = llm.bind(response_format={"type": "json_object"})
    resp = bound.invoke(messages)
    content = resp.content if hasattr(resp, "content") else str(resp)
    return parse_json(content, default=default)


def parse_json(text: str, *, default: Any = None) -> Any:
    """Best-effort extraction of a JSON object from a model response."""
    if not text:
        return default if default is not None else {}
    text = text.strip()
    # Strip markdown code fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return default if default is not None else {}
