"""Synthesizer agent: merges research + skeptic into a validated PolicyInterpretation."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google.genai import errors as genai_errors
from pydantic import BaseModel, Field, ValidationError

from agents.parser import ParsedIntent
from llm import chat_json
from schema import ParameterDelta, PolicyInterpretation, Source
from valid_parameters import valid_targets_prompt_block

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    """Recursively convert Pydantic models (e.g. ``Source``) into JSON-serializable structures."""
    if hasattr(obj, "model_dump"):
        return _json_safe(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    return obj


class _SynthLLMOut(BaseModel):
    """LLM payload (reasoning_trace is assembled server-side)."""

    plain_english_summary: str
    parameter_deltas: list[ParameterDelta]
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[Source]
    warnings: list[str] = Field(default_factory=list)


def _confidence_from_skeptic(base: float, skeptic_output: dict[str, Any]) -> float:
    n = len(skeptic_output.get("concerns") or [])
    adjusted = float(base) - 0.07 * min(n, 8)
    return max(0.05, min(0.95, adjusted))


def _merge_sources_dedupe_url(*source_lists: list[Source]) -> list[Source]:
    seen: set[str] = set()
    merged: list[Source] = []
    for lst in source_lists:
        for src in lst:
            key = (src.url or "").strip() or f"title:{src.title}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(src)
    return merged


async def synthesize(
    parsed: ParsedIntent,
    research_outputs: list[dict[str, Any]],
    skeptic_output: dict[str, Any],
    prior_reasoning_trace: list[str],
) -> PolicyInterpretation:
    """
    Produce the final PolicyInterpretation using Gemini JSON mode.

    Input:
        parsed — parser output.
        research_outputs — literature, historical, dataset dicts.
        skeptic_output — concerns and adjustments from skeptic_agent.
        prior_reasoning_trace — emoji-prefixed trace from earlier pipeline stages.

    Output:
        Validated PolicyInterpretation with merged sources and server-side reasoning_trace tail.

    Failure modes:
        On API or validation errors after one retry, raises ``RuntimeError`` for the pipeline wrapper.
    """
    targets_block = valid_targets_prompt_block()
    schema_hint = (
        '{"plain_english_summary": "string", '
        '"parameter_deltas": [{"target": "<one line from VALID list>", '
        '"operation": "multiply|add|set", "value": 0.0, "rationale": "string"}], '
        '"confidence": 0.0, '
        '"sources": [{"title": "string", "url": "string or null", "excerpt": "string"}], '
        '"warnings": ["string"]}'
    )

    research_blob = json.dumps(
        _json_safe(
            {
                "parsed": parsed.model_dump(),
                "research": research_outputs,
                "skeptic": skeptic_output,
            }
        ),
        ensure_ascii=False,
    )[:14000]

    prompt = (
        "You are the lead integrator for a marine policy → simulation-parameter pipeline.\n"
        "USE ONLY these parameter target names for every ParameterDelta.target. Do not invent new ones.\n"
        f"{targets_block}\n\n"
        "Allowed operations per delta: multiply, add, set.\n\n"
        "Return ONLY valid JSON matching this shape (omit reasoning_trace; the server appends it):\n"
        f"{schema_hint}\n\n"
        "The JSON object MUST include these keys exactly: "
        "plain_english_summary, parameter_deltas, confidence (0..1), sources, warnings (array, may be empty).\n"
        "plain_english_summary must be 2–4 sentences, judge-ready, citing the strongest research themes.\n"
        "Include 3–8 parameter_deltas when justified; each rationale must reference research or skeptic notes.\n"
        "Incorporate skeptic concerns by lowering confidence and tightening magnitudes when concerns are strong.\n\n"
        f"Context JSON:\n{research_blob}"
    )

    def _call() -> dict[str, Any]:
        return chat_json(prompt, "gemini-2.5-flash", 0.3)

    try:
        raw = await asyncio.to_thread(_call)
        model = _SynthLLMOut.model_validate(raw)
    except genai_errors.APIError as err:
        logger.exception("synthesize: Gemini API error on first attempt: %s", err)
        raise RuntimeError(f"Gemini API error during synthesis: {err}") from err
    except (json.JSONDecodeError, ValidationError) as err:
        logger.warning("synthesize: first validation failed: %s", err)
        retry_prompt = (
            f"{prompt}\n\nPrevious attempt failed validation with:\n{err!s}\n"
            "Return ONLY JSON matching the schema; use only targets from the VALID list."
        )

        def _retry() -> dict[str, Any]:
            return chat_json(retry_prompt, "gemini-2.5-flash", 0.3)

        try:
            raw2 = await asyncio.to_thread(_retry)
            model = _SynthLLMOut.model_validate(raw2)
        except (genai_errors.APIError, json.JSONDecodeError, ValidationError) as err2:
            logger.exception("synthesize: failed after retry: %s", err2)
            raise RuntimeError(f"Synthesis failed validation: {err2}") from err2

    research_sources: list[Source] = []
    for block in research_outputs:
        for src in block.get("sources") or []:
            if isinstance(src, Source):
                research_sources.append(src)

    merged_sources = _merge_sources_dedupe_url(research_sources, list(model.sources))
    conf = _confidence_from_skeptic(model.confidence, skeptic_output)

    synth_line = (
        f"✅ Synthesizer merged {len(merged_sources)} unique sources into "
        f"{len(model.parameter_deltas)} parameter delta(s) at confidence {conf:.2f}."
    )
    full_trace = list(prior_reasoning_trace) + [synth_line]

    return PolicyInterpretation(
        plain_english_summary=model.plain_english_summary,
        parameter_deltas=list(model.parameter_deltas),
        confidence=conf,
        sources=merged_sources,
        reasoning_trace=full_trace,
        warnings=list(model.warnings),
    )
