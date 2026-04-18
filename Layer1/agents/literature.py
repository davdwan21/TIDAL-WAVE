"""Literature research agent grounded by Google Search via Gemini."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.genai import errors as genai_errors

from agents.parser import ParsedIntent
from llm import research_with_search
from schema import Source

logger = logging.getLogger(__name__)


async def literature_agent(parsed: ParsedIntent) -> dict[str, Any]:
    """
    Research ecological literature relevant to the parsed policy intent.

    Input:
        parsed: Structured policy intent from parser agent.

    Output:
        Dictionary with findings, sources, suggested_parameters, and reasoning_trace.

    Failure modes:
        On Gemini API errors, returns fallback findings and empty sources without raising.
    """
    species_clause = ", ".join(parsed.affected_species[:5]) or "coastal food web species"
    query = (
        "Find authoritative and peer-reviewed evidence on ecological impacts of "
        f"{parsed.action_type} policies targeting {parsed.target_activity} in "
        f"{parsed.scope_geographic}. Focus on {species_clause}. "
        "Summarize observed outcomes and effect direction with concise evidence-oriented language."
    )

    try:
        result = await asyncio.to_thread(research_with_search, query)
    except genai_errors.APIError as err:
        logger.warning("literature_agent: Gemini API error: %s", err)
        return {
            "findings": [
                "Live literature grounding unavailable due to API quota/error; using placeholder evidence note."
            ],
            "sources": [],
            "suggested_parameters": [],
            "reasoning_trace": ["📚 Literature agent unavailable (API quota/error); no external sources returned."],
        }

    text = (result.get("text") or "").strip()
    raw_sources = result.get("sources") or []

    findings = [
        segment.strip()
        for segment in text.replace("\n", " ").split(". ")
        if segment.strip()
    ][:3]
    if not findings:
        findings = ["No concrete literature findings were parsed from grounded text output."]

    sources: list[Source] = [
        Source(
            title=(src.get("title") or "Untitled source"),
            url=src.get("url"),
            excerpt=(text[:220] + "...") if len(text) > 220 else text or "Grounded response without excerpt.",
        )
        for src in raw_sources
    ]

    suggestions: list[dict[str, Any]] = []
    if parsed.action_type in {"ban", "restrict", "regulate"}:
        suggestions.append(
            {
                "target": "fishing_fleet.effort_level",
                "operation": "multiply",
                "value": 0.8,
                "reason": "Literature typically reports reduced extraction pressure under restrictive policy regimes.",
            }
        )

    return {
        "findings": findings,
        "sources": sources,
        "suggested_parameters": suggestions,
        "reasoning_trace": [
            f"📚 Literature agent found {len(sources)} grounded sources and {len(findings)} key findings."
        ],
    }
