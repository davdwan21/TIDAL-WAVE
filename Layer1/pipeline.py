"""Policy interpretation orchestration (Step 4: parser + parallel research agents)."""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, Awaitable, Callable

from agents.dataset import dataset_agent
from agents.historical import historical_agent
from agents.literature import literature_agent
from agents.parser import ParsedIntent, parse_policy
from schema import ParameterDelta, PolicyInterpretation, PolicyRequest, Source


async def _run_research_agent(
    fn: Callable[[ParsedIntent], Awaitable[dict[str, Any]]],
    parsed: ParsedIntent,
    timeout_seconds: float = 12.0,
) -> dict[str, Any]:
    """Run one research agent with timeout and safe fallback payload."""
    try:
        return await asyncio.wait_for(fn(parsed), timeout=timeout_seconds)
    except TimeoutError:
        return {
            "findings": ["Agent timed out before returning findings."],
            "sources": [],
            "suggested_parameters": [],
            "reasoning_trace": [f"{getattr(fn, '__name__', 'agent')} timed out at {timeout_seconds:.1f}s."],
        }


async def interpret_policy(request: PolicyRequest) -> PolicyInterpretation:
    """
    Interpret a policy: parse with Gemini, then return a stub interpretation (Steps 4–6 pending).

    Input:
        request — user policy text and optional region.

    Output:
        PolicyInterpretation — schema-valid; parameter deltas still illustrative.

    Failure modes:
        Parser falls back to an ``unclear`` intent on empty input or repeated validation failure;
        missing ``GEMINI_API_KEY`` raises when parsing is attempted.
    """
    parsed: ParsedIntent = await asyncio.to_thread(parse_policy, request.policy_text)
    research_start = perf_counter()
    literature_output, historical_output, dataset_output = await asyncio.gather(
        _run_research_agent(literature_agent, parsed),
        _run_research_agent(historical_agent, parsed),
        _run_research_agent(dataset_agent, parsed),
    )
    research_elapsed = perf_counter() - research_start

    species_preview = (
        ", ".join(parsed.affected_species[:5])
        if parsed.affected_species
        else "(none named)"
    )
    research_sources: list[Source] = []
    for output in (literature_output, historical_output, dataset_output):
        for src in output.get("sources", []):
            if isinstance(src, Source):
                research_sources.append(src)

    reasoning_trace = [
        f"Received policy text for region {request.region!r}.",
        (
            f"Parser: action_type={parsed.action_type!r}, "
            f"target_activity={parsed.target_activity!r}, "
            f"scope_geographic={parsed.scope_geographic!r}."
        ),
        (
            f"Parser: scope_temporal={parsed.scope_temporal!r}, magnitude={parsed.magnitude!r}, "
            f"affected_species={species_preview}."
        ),
        *literature_output.get("reasoning_trace", []),
        *historical_output.get("reasoning_trace", []),
        *dataset_output.get("reasoning_trace", []),
        f"Parallel research completed in {research_elapsed:.2f}s.",
        "Synthesizer not wired — emitting a single illustrative parameter delta.",
    ]
    return PolicyInterpretation(
        plain_english_summary=(
            f"Parsed intent: {parsed.action_type} affecting {parsed.target_activity} "
            f"({parsed.scope_geographic}). Parallel literature/historical/dataset signals are now included; "
            "anchovy mortality multiplier shown as a placeholder ecological lever."
        ),
        parameter_deltas=[
            ParameterDelta(
                target="anchovy.mortality_rate",
                operation="multiply",
                value=0.6,
                rationale=(
                    "Illustrative delta pending full pipeline: reduced fishing pressure "
                    f"often lowers forage-fish mortality under policies like {parsed.action_type!r}."
                ),
            )
        ],
        confidence=0.75,
        sources=research_sources
        or [
            Source(
                title="CalCOFI and fisheries reference (mock)",
                url="https://example.org/calcofi-mock",
                excerpt="Fallback source used when research agents return no source objects.",
            )
        ],
        reasoning_trace=reasoning_trace,
        warnings=[
            "Skeptic and synthesizer agents are not yet wired — parameter deltas remain placeholders.",
            *(
                [f"Research phase exceeded latency target: {research_elapsed:.2f}s (> 15s)."]
                if research_elapsed > 15.0
                else []
            ),
        ],
    )
