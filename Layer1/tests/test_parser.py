"""Parser agent tests (mocked Gemini JSON; no network, no skips)."""

from __future__ import annotations

from typing import Any

import pytest

from agents.parser import ParsedIntent, is_api_fallback_intent, parse_policy


def _fish_json() -> dict[str, Any]:
    return {
        "action_type": "ban",
        "target_activity": "commercial_fishing",
        "scope_geographic": "within 12 nautical miles of the california coast",
        "scope_temporal": "next decade",
        "magnitude": "100 percent prohibition inside zone",
        "affected_species": ["nearshore fish stocks"],
        "mapped_species": ["anchovy", "sardine", "fishing_fleet", "pelican"],
    }


def _mpa_json() -> dict[str, Any]:
    return {
        "action_type": "establish",
        "target_activity": "marine_protected_area",
        "scope_geographic": "off san diego california coastal zone",
        "scope_temporal": "immediate",
        "magnitude": "20 percent coastal zone no-take",
        "affected_species": ["rockfish", "kelp forest species"],
        "mapped_species": ["anchovy", "sardine", "pelican", "sea_lion", "zooplankton", "phytoplankton"],
    }


def _runoff_json() -> dict[str, Any]:
    return {
        "action_type": "regulate",
        "target_activity": "agricultural_runoff",
        "scope_geographic": "coastal watersheds",
        "scope_temporal": "winter storms",
        "magnitude": "cap nitrogen application rates",
        "affected_species": ["phytoplankton", "zooplankton"],
        "mapped_species": ["phytoplankton", "zooplankton", "coastal_community"],
    }


def _quota_json() -> dict[str, Any]:
    return {
        "action_type": "increase",
        "target_activity": "sardine_catch_quota",
        "scope_geographic": "california current",
        "scope_temporal": "next season",
        "magnitude": "25 percent quota increase",
        "affected_species": ["sardine"],
        "mapped_species": ["sardine", "fishing_fleet"],
    }


def _vague_json() -> dict[str, Any]:
    return {
        "action_type": "unclear",
        "target_activity": "unspecified",
        "scope_geographic": "unspecified",
        "scope_temporal": "unspecified",
        "magnitude": "vague intent",
        "affected_species": [],
        "mapped_species": [],
    }


def test_parser_fishing_ban(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agents.parser.chat_json", lambda *a, **k: _fish_json())
    text = (
        "Ban all commercial fishing within 12 nautical miles of the California coast "
        "for the next decade to protect nearshore fish stocks."
    )
    p: ParsedIntent = parse_policy(text)
    assert not is_api_fallback_intent(p)
    assert p.action_type.lower() in {"ban", "restrict", "prohibit", "phase_out", "moratorium"}
    assert "fish" in p.target_activity.lower() or "commercial" in p.target_activity.lower()
    assert "california" in p.scope_geographic.lower() or "coast" in p.scope_geographic.lower()


def test_parser_marine_protected_area(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agents.parser.chat_json", lambda *a, **k: _mpa_json())
    text = (
        "Establish a large no-take marine protected area off San Diego covering "
        "roughly 20% of the coastal zone, effective immediately."
    )
    p = parse_policy(text)
    assert not is_api_fallback_intent(p)
    assert p.action_type.lower() in {"establish", "create", "designate", "expand", "implement"}
    assert "mpa" in p.target_activity.lower() or "protected" in p.target_activity.lower() or "marine" in p.target_activity.lower()
    assert "san diego" in p.scope_geographic.lower() or "california" in p.scope_geographic.lower() or "coastal" in p.scope_geographic.lower()


def test_parser_pollution_regulation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agents.parser.chat_json", lambda *a, **k: _runoff_json())
    text = (
        "Regulate agricultural runoff from coastal watersheds: cap nitrogen application "
        "rates during winter storms to reduce eutrophication and harmful algal blooms."
    )
    p = parse_policy(text)
    assert not is_api_fallback_intent(p)
    assert "runoff" in p.target_activity.lower() or "agricultur" in p.target_activity.lower() or "nutrient" in p.target_activity.lower() or "pollution" in p.target_activity.lower()
    assert p.action_type.lower() in {"regulate", "restrict", "require", "limit", "control", "reduce"}


def test_parser_fishing_quota_increase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agents.parser.chat_json", lambda *a, **k: _quota_json())
    text = (
        "Increase the commercial sardine catch quota by 25% next season to support "
        "the fishing fleet while monitoring stock biomass."
    )
    p = parse_policy(text)
    assert not is_api_fallback_intent(p)
    assert "increase" in p.action_type.lower() or "raise" in p.action_type.lower() or "expand" in p.action_type.lower()
    assert "quota" in p.magnitude.lower() or "25" in p.magnitude or "catch" in p.target_activity.lower() or "sardine" in p.target_activity.lower()


def test_parser_vague_ambiguous_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agents.parser.chat_json", lambda *a, **k: _vague_json())
    text = "We should probably do something better for the ocean soon."
    p = parse_policy(text)
    assert not is_api_fallback_intent(p)
    assert (
        p.action_type.lower() == "unclear"
        or "unspecified" in p.target_activity.lower()
        or "unspecified" in p.magnitude.lower()
        or "unspecified" in p.scope_geographic.lower()
        or "vague" in p.magnitude.lower()
    )
