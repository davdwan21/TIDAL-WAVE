"""Skeptic agent: challenges research findings and proposed parameter shifts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from google.genai import errors as genai_errors
from pydantic import BaseModel, Field, ValidationError

from agents.parser import ParsedIntent
from llm import chat_json
from valid_parameters import VALID_TARGETS, valid_targets_prompt_block

logger = logging.getLogger(__name__)


class _Adjustment(BaseModel):
    target: str
    suggested_value: float
    reason: str


class _SkepticLLMOut(BaseModel):
    concerns: list[str] = Field(default_factory=list)
    adjustments: list[_Adjustment] = Field(default_factory=list)


def _filter_adjustments(raw: list[_Adjustment]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for adj in raw:
        if adj.target not in VALID_TARGETS:
            logger.warning("skeptic_agent: dropping adjustment with invalid target %r", adj.target)
            continue
        out.append(
            {
                "target": adj.target,
                "suggested_value": adj.suggested_value,
                "reason": adj.reason,
            }
        )
    return out


async def skeptic_agent(
    parsed: ParsedIntent,
    research_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Review parallel research outputs for overstatements and weak evidence.

    Input:
        parsed — structured policy intent.
        research_outputs — list of dicts from literature, historical, and dataset agents.

    Output:
        Dict with ``concerns``, ``adjustments`` (valid targets only), and ``reasoning_trace``.

    Failure modes:
        On ``ENABLE_SKEPTIC=false`` returns a skip payload without calling Gemini.
        On ``APIError`` or malformed JSON, returns conservative concerns and empty adjustments.
    """
    if os.getenv("ENABLE_SKEPTIC", "true").strip().lower() in {"0", "false", "no"}:
        return {
            "concerns": ["Skeptic review skipped (quota mode)"],
            "adjustments": [],
            "reasoning_trace": [
                "🤔 Skeptic review was skipped because ENABLE_SKEPTIC is false, so 0 concerns and 0 adjustments were produced."
            ],
        }

    targets_block = valid_targets_prompt_block()
    payload = json.dumps(
        {
            "parsed": parsed.model_dump(),
            "research": [
                {
                    "findings": r.get("findings", [])[:8],
                    "suggested_parameters": r.get("suggested_parameters", [])[:8],
                }
                for r in research_outputs
            ],
        },
        ensure_ascii=False,
    )[:12000]

    prompt = (
        "You are a scientific skeptic. Review the research findings and suggested parameters. "
        "Flag overstatements, weak evidence, or magnitudes inconsistent with similar historical cases. "
        "Only flag concerns backed by the research text below; do not invent citations.\n\n"
        "For any adjustment you propose, the ``target`` MUST be exactly one of these valid simulation keys "
        "(verbatim, dot-notation):\n"
        f"{targets_block}\n\n"
        "Return ONLY valid JSON with this exact shape:\n"
        '{"concerns": ["string", ...], "adjustments": [{"target": "<one of the keys above>", '
        '"suggested_value": <float>, "reason": "string"}, ...]}\n\n'
        f"Context JSON:\n{payload}"
    )

    try:
        raw_dict = await asyncio.to_thread(chat_json, prompt, "gemini-2.5-flash", 0.7)
    except genai_errors.APIError as err:
        logger.warning("skeptic_agent: Gemini API error: %s", err)
        return {
            "concerns": ["Skeptic review unavailable due to Gemini API quota or transport error."],
            "adjustments": [],
            "reasoning_trace": [
                "🤔 Skeptic agent could not reach Gemini; no automated critique was produced this run."
            ],
        }
    except json.JSONDecodeError as err:
        logger.warning("skeptic_agent: JSON decode error: %s", err)
        return {
            "concerns": ["Skeptic output failed validation; treating adjustments as empty for safety."],
            "adjustments": [],
            "reasoning_trace": ["🤔 Skeptic agent returned unusable JSON after retries; skipped adjustments."],
        }

    try:
        model = _SkepticLLMOut.model_validate(raw_dict)
    except ValidationError as err:
        logger.warning("skeptic_agent: schema validation failed: %s", err)
        return {
            "concerns": ["Skeptic JSON did not match the expected schema."],
            "adjustments": [],
            "reasoning_trace": ["🤔 Skeptic agent produced malformed structured output; skipped adjustments."],
        }

    adjustments = _filter_adjustments(list(model.adjustments))
    concerns = list(model.concerns)
    n_c, n_a = len(concerns), len(adjustments)
    return {
        "concerns": concerns,
        "adjustments": adjustments,
        "reasoning_trace": [
            f"🤔 Skeptic agent recorded {n_c} concern(s) and {n_a} evidence-backed adjustment hint(s) for synthesis."
        ],
    }
