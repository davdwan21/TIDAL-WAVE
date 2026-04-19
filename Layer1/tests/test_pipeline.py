"""Pipeline orchestration tests (async, mocked LLM, no skips)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import mock

import pytest

from agents.parser import ParsedIntent, TRAWLING_DATASET_MATCH
from schema import PolicyRequest


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("ENABLE_SKEPTIC", raising=False)
    from pipeline import clear_interpret_cache

    clear_interpret_cache()
    yield
    clear_interpret_cache()


def test_demo_mode_returns_canned_trawling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    from pipeline import interpret_policy

    async def run() -> Any:
        return await interpret_policy(
            PolicyRequest(
                policy_text="Ban commercial trawling within 50 miles of California coast",
                region="socal",
            )
        )

    r = asyncio.run(run())
    assert any("demo" in w.lower() for w in r.warnings)
    assert 0.0 <= r.confidence <= 1.0
    assert len(r.sources) >= 3
    assert len(r.parameter_deltas) >= 3
    joined = "\n".join(r.reasoning_trace)
    for emoji in ("📋", "📚", "🏛️", "📊", "🤔", "✅"):
        assert emoji in joined


def test_enable_skeptic_false_skips_skeptic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_SKEPTIC", "false")

    parsed = ParsedIntent(
        action_type="ban",
        target_activity="commercial_trawling",
        scope_geographic="california",
        scope_temporal="5 years",
        magnitude="50 miles",
        affected_species=["anchovy"],
        mapped_species=["anchovy", "sardine", "fishing_fleet"],
        dataset_match_species=list(TRAWLING_DATASET_MATCH),
    )

    async def fake_lit(p: ParsedIntent) -> dict[str, Any]:
        return {
            "findings": ["f1"],
            "sources": [],
            "suggested_parameters": [],
            "reasoning_trace": ["📚 Literature stub returned 0 sources and 1 finding."],
        }

    async def fake_hist(p: ParsedIntent) -> dict[str, Any]:
        return {
            "findings": ["h1"],
            "sources": [],
            "suggested_parameters": [],
            "reasoning_trace": ["🏛️ Historical stub returned 0 sources and 1 finding."],
        }

    async def fake_data(p: ParsedIntent) -> dict[str, Any]:
        return {
            "findings": ["d1"],
            "sources": [],
            "suggested_parameters": [],
            "reasoning_trace": ["📊 Dataset stub returned 0 sources and 1 finding."],
        }

    def fake_parse(text: str) -> ParsedIntent:
        assert "trawl" in text.lower()
        return parsed

    synth_payload = {
        "plain_english_summary": "Stub synthesis merges mocked research without live Gemini.",
        "parameter_deltas": [
            {
                "target": "anchovy.mortality_rate",
                "operation": "multiply",
                "value": 0.9,
                "rationale": "Lower mortality under trawl ban (mocked).",
            }
        ],
        "confidence": 0.82,
        "sources": [{"title": "Mock", "url": "https://example.org/mock", "excerpt": "ex"}],
        "warnings": [],
    }

    chat = mock.MagicMock(return_value=synth_payload)
    monkeypatch.setattr("pipeline.parse_policy", fake_parse)
    monkeypatch.setattr("pipeline.literature_agent", fake_lit)
    monkeypatch.setattr("pipeline.historical_agent", fake_hist)
    monkeypatch.setattr("pipeline.dataset_agent", fake_data)
    monkeypatch.setattr("agents.synthesizer.chat_json", chat)

    from pipeline import interpret_policy

    r = asyncio.run(
        interpret_policy(
            PolicyRequest(policy_text="Ban commercial trawling near California", region="socal"),
        )
    )
    assert chat.call_count == 1
    joined = "\n".join(r.reasoning_trace)
    assert "🤔" in joined
    assert "skipped" in joined.lower()


def test_lru_cache_second_call_skips_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("ENABLE_SKEPTIC", "false")

    parsed = ParsedIntent(
        action_type="ban",
        target_activity="commercial_trawling",
        scope_geographic="california",
        scope_temporal="5 years",
        magnitude="50 miles",
        affected_species=["anchovy"],
        mapped_species=["anchovy", "sardine", "fishing_fleet"],
        dataset_match_species=list(TRAWLING_DATASET_MATCH),
    )

    async def stub_agent(_: ParsedIntent) -> dict[str, Any]:
        return {
            "findings": ["x"],
            "sources": [],
            "suggested_parameters": [],
            "reasoning_trace": ["📚 stub"],
        }

    fake_parse = mock.MagicMock(return_value=parsed)

    synth_payload = {
        "plain_english_summary": "Cached run stub.",
        "parameter_deltas": [
            {
                "target": "sardine.mortality_rate",
                "operation": "multiply",
                "value": 0.88,
                "rationale": "r",
            }
        ],
        "confidence": 0.77,
        "sources": [{"title": "t", "url": "https://example.org/a", "excerpt": "e"}],
        "warnings": [],
    }

    chat = mock.MagicMock(return_value=synth_payload)
    monkeypatch.setattr("pipeline.parse_policy", fake_parse)
    monkeypatch.setattr("pipeline.literature_agent", stub_agent)
    monkeypatch.setattr("pipeline.historical_agent", stub_agent)
    monkeypatch.setattr("pipeline.dataset_agent", stub_agent)
    monkeypatch.setattr("agents.synthesizer.chat_json", chat)

    from pipeline import interpret_policy

    req = PolicyRequest(policy_text="Identical cache policy text", region="socal")
    r1 = asyncio.run(interpret_policy(req))
    r2 = asyncio.run(interpret_policy(req))
    assert chat.call_count == 1
    assert fake_parse.call_count == 1
    assert r1.model_dump() == r2.model_dump()
