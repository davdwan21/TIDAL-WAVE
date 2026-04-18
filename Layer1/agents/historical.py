"""Historical precedent agent grounded by Google Search via Gemini."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.genai import errors as genai_errors

from agents.parser import ParsedIntent
from llm import research_with_search
from schema import Source

logger = logging.getLogger(__name__)


async def historical_agent(parsed: ParsedIntent) -> dict[str, Any]:
    """
    Retrieve historical analogs for similar policies and summarize observed outcomes.

    Input:
        parsed: Structured policy intent from parser agent.

    Output:
        Dictionary with findings, sources, suggested_parameters, and reasoning_trace.

    Failure modes:
        On Gemini API errors, returns fallback findings and empty sources without raising.
    """
    query = (
        "Find historical precedents for marine/coastal policies similar to "
        f"{parsed.action_type} of {parsed.target_activity} in {parsed.scope_geographic}. "
        "Prioritize California Current, US West Coast, and comparable fisheries/pollution interventions. "
        "Report implementation outcomes and realistic timescales."
    )

    try:
        result = await asyncio.to_thread(research_with_search, query)
    except genai_errors.APIError as err:
        logger.warning("historical_agent: Gemini API error: %s", err)
        return {
            "findings": [
                "Live historical grounding unavailable due to API quota/error; unable to retrieve precedents."
            ],
            "sources": [],
            "suggested_parameters": [],
            "reasoning_trace": ["🏛️ Historical agent unavailable (API quota/error); no precedent links returned."],
        }

    text = (result.get("text") or "").strip()
    raw_sources = result.get("sources") or []
    findings = [
        segment.strip()
        for segment in text.replace("\n", " ").split(". ")
        if segment.strip()
    ][:3]
    if not findings:
        findings = ["No concrete precedent findings were parsed from grounded text output."]

    sources: list[Source] = [
        Source(
            title=(src.get("title") or "Untitled source"),
            url=src.get("url"),
            excerpt=(text[:220] + "...") if len(text) > 220 else text or "Grounded response without excerpt.",
        )
        for src in raw_sources
    ]

    suggestions: list[dict[str, Any]] = []
    if parsed.action_type in {"increase", "deregulate", "expand"}:
        suggestions.append(
            {
                "target": "anchovy.catch_rate",
                "operation": "add",
                "value": 0.1,
                "reason": "Historical deregulation analogs often increase near-term extraction rates.",
            }
        )

    return {
        "findings": findings,
        "sources": sources,
        "suggested_parameters": suggestions,
        "reasoning_trace": [
            f"🏛️ Historical agent found {len(sources)} precedent sources and {len(findings)} outcome notes."
        ],
    }
