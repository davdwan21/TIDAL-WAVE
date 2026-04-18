"""Parser agent tests — integration cases call live Gemini (``GEMINI_API_KEY``)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agents.parser import ParsedIntent, is_api_fallback_intent, parse_policy


pytestmark = pytest.mark.integration


def _load_dotenv_for_tests() -> None:
    """Ensure Layer1/.env is loaded when pytest is run without shell-exported key."""
    try:
        from dotenv import load_dotenv

        layer1_root = Path(__file__).resolve().parents[1]
        load_dotenv(layer1_root / ".env")
    except ImportError:
        pass


@pytest.fixture(scope="module")
def gemini_configured() -> None:
    _load_dotenv_for_tests()
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        pytest.skip(
            "GEMINI_API_KEY not set — load Layer1/.env or export the key for integration tests."
        )


def _assert_not_fallback(p: ParsedIntent) -> None:
    assert not is_api_fallback_intent(p), (
        "Parser returned API fallback intent for a structured policy input. "
        "Gemini likely failed (quota/outage/transport) and this test should fail loudly."
    )


def test_parser_fishing_ban(gemini_configured: None) -> None:
    text = (
        "Ban all commercial fishing within 12 nautical miles of the California coast "
        "for the next decade to protect nearshore fish stocks."
    )
    p: ParsedIntent = parse_policy(text)
    _assert_not_fallback(p)
    assert p.action_type.lower() in {"ban", "restrict", "prohibit", "phase_out", "moratorium"}
    assert "fish" in p.target_activity.lower() or "commercial" in p.target_activity.lower()
    assert "california" in p.scope_geographic.lower() or "coast" in p.scope_geographic.lower()
    assert len(p.affected_species) >= 0


def test_parser_marine_protected_area(gemini_configured: None) -> None:
    text = (
        "Establish a large no-take marine protected area off San Diego covering "
        "roughly 20% of the coastal zone, effective immediately."
    )
    p = parse_policy(text)
    _assert_not_fallback(p)
    assert p.action_type.lower() in {"establish", "create", "designate", "expand", "implement"}
    assert "mpa" in p.target_activity.lower() or "protected" in p.target_activity.lower() or "marine" in p.target_activity.lower()
    assert "san diego" in p.scope_geographic.lower() or "california" in p.scope_geographic.lower() or "coastal" in p.scope_geographic.lower()


def test_parser_pollution_regulation(gemini_configured: None) -> None:
    text = (
        "Regulate agricultural runoff from coastal watersheds: cap nitrogen application "
        "rates during winter storms to reduce eutrophication and harmful algal blooms."
    )
    p = parse_policy(text)
    _assert_not_fallback(p)
    assert "runoff" in p.target_activity.lower() or "agricultur" in p.target_activity.lower() or "nutrient" in p.target_activity.lower() or "pollution" in p.target_activity.lower()
    assert p.action_type.lower() in {"regulate", "restrict", "require", "limit", "control", "reduce"}


def test_parser_fishing_quota_increase(gemini_configured: None) -> None:
    text = (
        "Increase the commercial sardine catch quota by 25% next season to support "
        "the fishing fleet while monitoring stock biomass."
    )
    p = parse_policy(text)
    _assert_not_fallback(p)
    assert "increase" in p.action_type.lower() or "raise" in p.action_type.lower() or "expand" in p.action_type.lower()
    assert "quota" in p.magnitude.lower() or "25" in p.magnitude or "catch" in p.target_activity.lower() or "sardine" in p.target_activity.lower()


def test_parser_vague_ambiguous_policy(gemini_configured: None) -> None:
    text = "We should probably do something better for the ocean soon."
    p = parse_policy(text)
    _assert_not_fallback(p)
    assert (
        p.action_type.lower() == "unclear"
        or "unspecified" in p.target_activity.lower()
        or "unspecified" in p.magnitude.lower()
        or "unspecified" in p.scope_geographic.lower()
        or "vague" in p.magnitude.lower()
    )
