"""Policy interpretation orchestration: parse → research → skeptic → synthesize."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from functools import lru_cache
from time import perf_counter
from typing import Any, Awaitable, Callable

from agents.dataset import dataset_agent
from agents.historical import historical_agent
from agents.literature import literature_agent
from agents.parser import ParsedIntent, parse_policy
from agents.skeptic import skeptic_agent
from agents.synthesizer import synthesize
from canned_policies import match_canned_policy
from schema import ParameterDelta, PolicyInterpretation, PolicyRequest, Source

logger = logging.getLogger(__name__)

# functools.lru_cache cannot wrap async ``interpret_policy`` without blocking the event loop; we keep an
# equivalent in-process LRU (max 50) keyed by policy text, region, and relevant env flags, and use
# ``lru_cache`` here to normalize cache key tuples per the project convention.
_INTERPRET_CACHE: OrderedDict[tuple[str, str, str, str], PolicyInterpretation] = OrderedDict()
_CACHE_MAX: int = 50


@lru_cache(maxsize=50)
def _interpret_cache_key(
    policy_text: str,
    region: str,
    demo_mode: str,
    enable_skeptic: str,
) -> tuple[str, str, str, str]:
    """Normalize cache key components (lru_cache dedupes identical signatures across calls)."""
    return (policy_text.strip(), region.strip(), demo_mode, enable_skeptic)


def clear_interpret_cache() -> None:
    """Clear the interpretation LRU (for tests)."""
    _INTERPRET_CACHE.clear()


def _cache_get(key: tuple[str, str, str, str]) -> PolicyInterpretation | None:
    if key not in _INTERPRET_CACHE:
        return None
    _INTERPRET_CACHE.move_to_end(key)
    return _INTERPRET_CACHE[key].model_copy(deep=True)


def _cache_put(key: tuple[str, str, str, str], value: PolicyInterpretation) -> None:
    _INTERPRET_CACHE[key] = value.model_copy(deep=True)
    _INTERPRET_CACHE.move_to_end(key)
    while len(_INTERPRET_CACHE) > _CACHE_MAX:
        _INTERPRET_CACHE.popitem(last=False)


async def _run_research_agent(
    fn: Callable[[ParsedIntent], Awaitable[dict[str, Any]]],
    parsed: ParsedIntent,
    timeout_seconds: float = 12.0,
) -> dict[str, Any]:
    """Run one research agent with timeout and safe fallback payload."""
    name = getattr(fn, "__name__", "research_agent")
    try:
        return await asyncio.wait_for(fn(parsed), timeout=timeout_seconds)
    except TimeoutError:
        logger.warning("%s timed out after %.1fs", name, timeout_seconds)
        return {
            "findings": [f"{name} timed out before returning findings."],
            "sources": [],
            "suggested_parameters": [],
            "reasoning_trace": [
                f"📊 {name} timed out at {timeout_seconds:.1f}s and returned 0 sources, 0 findings, and 0 parameter hints."
            ],
        }


def _failure_interpretation(
    request: PolicyRequest,
    err: BaseException,
    partial_trace: list[str],
) -> PolicyInterpretation:
    """Return a valid PolicyInterpretation after a hard pipeline failure (≥5 trace lines, all emoji roles)."""
    prior_n = len(partial_trace)
    err_name = type(err).__name__
    trace = list(partial_trace)
    trace.append(
        f"📚 Literature merge did not finish after {err_name}, so 0 new literature URLs were added in recovery."
    )
    trace.append(
        f"🏛️ Historical merge did not finish after {err_name}, so 0 new precedent URLs were added in recovery."
    )
    trace.append(
        f"📊 Dataset merge did not finish after {err_name}, so 0 new CalCOFI/iNaturalist merges ran in recovery."
    )
    trace.append(
        f"🤔 Skeptic integration did not finish after {err_name}, so 0 skeptic-backed adjustments were applied."
    )
    trace.append(
        f"✅ Failure handler returned confidence 0.0 with 1 placeholder delta after {prior_n} prior trace line(s): "
        f"{str(err)[:160]!r}."
    )
    return PolicyInterpretation(
        plain_english_summary=(
            "The policy interpreter encountered an internal error while calling Gemini or validating output; "
            "review warnings and retry with a shorter policy text, or set DEMO_MODE=true for canned demos."
        ),
        parameter_deltas=[
            ParameterDelta(
                target="ocean.pollution_index",
                operation="set",
                value=1.0,
                rationale="Neutral placeholder delta so Layer 2 receives a schema-valid payload after failure.",
            )
        ],
        confidence=0.0,
        sources=[
            Source(
                title="Layer 1 failure handler",
                url=None,
                excerpt=str(err)[:400],
            )
        ],
        reasoning_trace=trace[:25],
        warnings=[f"Pipeline failure: {err_name}: {err}"],
    )


def _ensure_min_trace_entries(trace: list[str], min_len: int = 5) -> list[str]:
    """Append concise, numeric pipeline summaries until the trace meets UI minimums."""
    out = list(trace)
    if len(out) >= min_len:
        return out
    out.append(
        f"📋 Trace padding: pipeline held {len(out)} line(s) before adding neutral summaries to reach {min_len} entries."
    )
    if len(out) >= min_len:
        return out
    out.append(
        f"📚 Trace padding: literature block contributed {sum(1 for x in trace if x.startswith('📚'))} explicit line(s)."
    )
    if len(out) >= min_len:
        return out
    out.append(
        f"🏛️ Trace padding: historical block contributed {sum(1 for x in trace if x.startswith('🏛️'))} explicit line(s)."
    )
    if len(out) >= min_len:
        return out
    out.append(
        f"📊 Trace padding: dataset block contributed {sum(1 for x in trace if x.startswith('📊'))} explicit line(s)."
    )
    return out


async def interpret_policy(request: PolicyRequest) -> PolicyInterpretation:
    """
    Full async pipeline: parse → parallel research → skeptic → synthesize.

    ``DEMO_MODE=true`` uses ``canned_policies`` first. Otherwise an in-process LRU (50) keyed by
    policy text, region, and env flags caches successful live results.
    """
    t0 = perf_counter()
    demo_on = os.getenv("DEMO_MODE", "").strip().lower() == "true"
    skeptic_on = os.getenv("ENABLE_SKEPTIC", "true").strip().lower() not in {"0", "false", "no"}
    demo_key = "true" if demo_on else "false"
    skeptic_key = "true" if skeptic_on else "false"
    cache_key = _interpret_cache_key(request.policy_text, request.region, demo_key, skeptic_key)

    if demo_on:
        canned = match_canned_policy(request.policy_text)
        if canned is not None:
            logger.info("interpret_policy: DEMO_MODE canned hit (policy_len=%s)", len(request.policy_text))
            return canned

    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("interpret_policy: LRU cache hit for region=%r policy_len=%s", request.region, len(request.policy_text))
        return cached

    partial_trace: list[str] = [
        f"📋 Received policy for region {request.region!r} with {len(request.policy_text)} characters of text.",
    ]

    try:
        parsed: ParsedIntent = await asyncio.to_thread(parse_policy, request.policy_text)
        partial_trace.extend(
            [
                (
                    f"📋 Parser extracted action_type={parsed.action_type!r}, target_activity={parsed.target_activity!r}, "
                    f"and scope_geographic={parsed.scope_geographic!r}."
                ),
                (
                    f"📋 Parser mapped_species lists {len(parsed.mapped_species)} canonical token(s) and "
                    f"dataset_match_species lists {len(parsed.dataset_match_species)} CSV join key(s)."
                ),
            ]
        )

        research_start = perf_counter()
        literature_output, historical_output, dataset_output = await asyncio.gather(
            _run_research_agent(literature_agent, parsed),
            _run_research_agent(historical_agent, parsed),
            _run_research_agent(dataset_agent, parsed),
        )
        research_elapsed = perf_counter() - research_start
        partial_trace.extend(
            literature_output.get("reasoning_trace", [])
            + historical_output.get("reasoning_trace", [])
            + dataset_output.get("reasoning_trace", [])
        )
        partial_trace.append(
            f"📊 Parallel research phase finished in {research_elapsed:.2f}s across 3 agents (literature, historical, dataset)."
        )

        research_outputs = [literature_output, historical_output, dataset_output]
        skeptic_output = await skeptic_agent(parsed, research_outputs)
        partial_trace.extend(skeptic_output.get("reasoning_trace", []))

        result = await synthesize(parsed, research_outputs, skeptic_output, partial_trace)

        warn = list(result.warnings)
        if research_elapsed > 15.0:
            warn.append(f"Research phase soft budget 15s was exceeded at {research_elapsed:.2f}s.")

        result = result.model_copy(update={"warnings": warn})

        elapsed = perf_counter() - t0
        if elapsed > 30.0:
            logger.warning("interpret_policy: total latency %.2fs exceeded 30s target", elapsed)
            warn2 = list(result.warnings) + [f"Total interpret latency {elapsed:.2f}s exceeded 30s."]
            result = result.model_copy(update={"warnings": warn2})

        trace = _ensure_min_trace_entries(list(result.reasoning_trace), min_len=5)
        result = result.model_copy(update={"reasoning_trace": trace})

        _cache_put(cache_key, result)
        logger.info("interpret_policy: completed in %.2fs (cache_miss)", elapsed)
        return result

    except Exception as err:
        logger.exception("interpret_policy: pipeline failure")
        return _failure_interpretation(request, err, partial_trace)
    finally:
        elapsed = perf_counter() - t0
        if elapsed > 30.0:
            logger.warning("interpret_policy: wall clock %.2fs (post-run)", elapsed)
