"""Parser agent: natural-language policy → structured ParsedIntent (Gemini JSON)."""

from __future__ import annotations

import json
import logging
from typing import Any

from google.genai import errors as genai_errors
from pydantic import BaseModel, Field, ValidationError, field_validator

from llm import chat_json

logger = logging.getLogger(__name__)

# Canonical names for dataset matching and sim-facing hints (subset of ecosystem cast + pressures).
CANONICAL_SIMULATION_SPECIES: frozenset[str] = frozenset(
    {
        "anchovy",
        "sardine",
        "zooplankton",
        "phytoplankton",
        "pelican",
        "sea_lion",
        "leopard_shark",
        "market_squid",
        "coastal_community",
        "fishing_fleet",
    }
)

# Species keys that exist in stub CalCOFI / iNaturalist CSVs (not LLM output); excludes fishing_fleet.
_DATASET_JOIN_KEYS: frozenset[str] = frozenset(
    {
        "anchovy",
        "sardine",
        "zooplankton",
        "phytoplankton",
        "pelican",
        "sea_lion",
        "leopard_shark",
        "market_squid",
        "coastal_community",
    }
)

TRAWLING_DATASET_MATCH: tuple[str, ...] = ("anchovy", "sardine", "zooplankton", "phytoplankton")


class ParsedIntent(BaseModel):
    """Structured intent extracted from free-text policy language."""

    action_type: str = Field(
        ...,
        description="High-level policy verb, e.g. ban, restrict, establish, increase, regulate, unclear.",
    )
    target_activity: str = Field(
        ...,
        description="Primary activity or object being regulated (e.g. commercial_trawling, mpa, runoff).",
    )
    scope_geographic: str = Field(
        ...,
        description="Where the policy applies (region, distance from shore, jurisdiction).",
    )
    scope_temporal: str = Field(
        ...,
        description="Duration or phase: permanent, seasonal, multi-year, immediate, unspecified.",
    )
    magnitude: str = Field(
        ...,
        description="Strength, caps, or extent (percent, distance, tonnage) or 'unspecified'.",
    )
    affected_species: list[str] = Field(
        default_factory=list,
        description="Species or groups explicitly or reasonably implied; empty if none.",
    )
    mapped_species: list[str] = Field(
        default_factory=list,
        description=(
            "Full canonical list from the policy (Gemini), used for reasoning and downstream agents; "
            f"tokens must be from: {', '.join(sorted(CANONICAL_SIMULATION_SPECIES))}."
        ),
    )
    dataset_match_species: list[str] = Field(
        default_factory=list,
        description=(
            "Subset of canonical names used only for CalCOFI/iNaturalist CSV row joins; "
            "computed in parse_policy (not returned by Gemini). For trawling policies this is fixed to "
            f"{list(TRAWLING_DATASET_MATCH)}."
        ),
    )

    @field_validator("mapped_species", mode="before")
    @classmethod
    def normalize_mapped_species(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for item in v:
            if not isinstance(item, str):
                continue
            key = item.strip().lower().replace(" ", "_")
            if key in CANONICAL_SIMULATION_SPECIES:
                out.append(key)
        seen: set[str] = set()
        ordered: list[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                ordered.append(x)
        return ordered


def _supplement_mapped_for_fisheries(parsed: ParsedIntent) -> ParsedIntent:
    """Merge common canonical tokens for fisheries policies without replacing Gemini's mapped_species."""
    ta = parsed.target_activity.lower()
    if not any(k in ta for k in ("trawl", "trawling", "fishing", "commercial")):
        return parsed
    extras = ["anchovy", "sardine", "zooplankton", "phytoplankton", "fishing_fleet"]
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(parsed.mapped_species) + extras:
        if item in CANONICAL_SIMULATION_SPECIES and item not in seen:
            seen.add(item)
            merged.append(item)
    return parsed.model_copy(update={"mapped_species": merged})


def _compute_dataset_match_species(parsed: ParsedIntent) -> list[str]:
    ta = parsed.target_activity.lower()
    if any(k in ta for k in ("trawl", "trawling")):
        return list(TRAWLING_DATASET_MATCH)
    out: list[str] = []
    seen: set[str] = set()
    for m in parsed.mapped_species:
        if m in _DATASET_JOIN_KEYS and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _finalize_parsed_intent(parsed: ParsedIntent) -> ParsedIntent:
    """Apply fisheries merge hints, then attach CSV join keys."""
    merged = _supplement_mapped_for_fisheries(parsed)
    return merged.model_copy(update={"dataset_match_species": _compute_dataset_match_species(merged)})


def _parser_prompt(policy_text: str) -> str:
    """Build the full user prompt with instructions and few-shot JSON examples."""
    canonical_list = ", ".join(sorted(CANONICAL_SIMULATION_SPECIES))
    schema_hint = (
        "Return a single JSON object with exactly these keys: "
        "action_type, target_activity, scope_geographic, scope_temporal, magnitude, "
        "affected_species (array of free-form strings), mapped_species (array of canonical strings). "
        "Strings elsewhere are plain text. Use lowercase snake_case for action_type and target_activity where helpful. "
        f"mapped_species MUST contain only these canonical tokens (no other strings): {canonical_list}. "
        "Keep affected_species as human-readable names from the policy (e.g. groundfish, rockfish). "
        "For commercial fishing or trawling affecting the food web, mapped_species should usually include "
        "anchovy, sardine, zooplankton, phytoplankton, fishing_fleet, and often pelican and sea_lion for coastal California. "
        "For vague policies, mapped_species may be empty. "
        "If the policy is vague, set action_type to 'unclear' and explain uncertainty in magnitude "
        "or scope fields with plain language."
    )
    few_shots = """
Few-shot example 1 (input → output JSON):
INPUT: Ban commercial bottom trawling within 50 miles of the California coast for five years.
OUTPUT:
{"action_type":"ban","target_activity":"commercial_bottom_trawling","scope_geographic":"within 50 miles of California coast","scope_temporal":"five years","magnitude":"100 percent prohibition inside zone","affected_species":["groundfish","benthic invertebrates","commercial target fish"],"mapped_species":["anchovy","sardine","zooplankton","phytoplankton","fishing_fleet","pelican","sea_lion"]}

Few-shot example 2 (input → output JSON):
INPUT: Establish a no-take marine protected area covering 12% of Southern California state waters.
OUTPUT:
{"action_type":"establish","target_activity":"marine_protected_area","scope_geographic":"Southern California state waters","scope_temporal":"unspecified duration in text","magnitude":"12 percent area no-take","affected_species":["rockfish","kelp forest species","sea lions"],"mapped_species":["anchovy","sardine","pelican","sea_lion","zooplankton","phytoplankton"]}

Few-shot example 3 (input → output JSON):
INPUT: Require farms in coastal watersheds to cut fertilizer use by 30% to reduce algal blooms.
OUTPUT:
{"action_type":"regulate","target_activity":"agricultural_runoff","scope_geographic":"coastal watersheds","scope_temporal":"ongoing compliance","magnitude":"30 percent reduction in fertilizer use","affected_species":["phytoplankton","zooplankton","fish larvae"],"mapped_species":["phytoplankton","zooplankton","coastal_community"]}
"""
    return (
        "You are a marine and coastal sustainability policy parser for the US West Coast context.\n"
        f"{schema_hint}\n"
        f"{few_shots}\n"
        "Now parse the following policy. Respond with ONLY the JSON object, no markdown fences.\n\n"
        f"INPUT: {policy_text.strip()}\n"
        "OUTPUT:"
    )


_FALLBACK_INTENT_SHAPE = ParsedIntent(
    action_type="unclear",
    target_activity="unspecified",
    scope_geographic="unspecified",
    scope_temporal="unspecified",
    magnitude="unspecified",
    affected_species=[],
    mapped_species=[],
    dataset_match_species=[],
)


def _fallback_intent(policy_text: str, reason: str) -> ParsedIntent:
    logger.warning("parse_policy: using fallback intent (%s)", reason)
    stripped = policy_text.strip()
    if any(k in stripped.lower() for k in ("trawl", "trawling")):
        return _FALLBACK_INTENT_SHAPE.model_copy(
            update={"dataset_match_species": list(TRAWLING_DATASET_MATCH)}
        )
    return _FALLBACK_INTENT_SHAPE.model_copy(deep=True)


def is_api_fallback_intent(parsed: ParsedIntent) -> bool:
    """True when ``parse_policy`` substituted the conservative unclear default (e.g. API/quota failure)."""
    return (
        parsed.action_type == "unclear"
        and parsed.target_activity == "unspecified"
        and parsed.scope_geographic == "unspecified"
        and parsed.scope_temporal == "unspecified"
        and parsed.magnitude == "unspecified"
        and not parsed.affected_species
        and not parsed.mapped_species
    )


def parse_policy(policy_text: str) -> ParsedIntent:
    """
    Parse free-text policy into structured fields using Gemini JSON mode.

    Input:
        policy_text — raw user policy string (any length).

    Output:
        ParsedIntent — validated structured intent, including ``dataset_match_species`` filled for CSV joins.

    Failure modes:
        JSON decode errors are retried once inside ``chat_json`` with a stricter tail prompt.
        Gemini ``APIError`` (rate limits, outages) returns the same conservative fallback intent.
        If Pydantic validation fails after one additional structured retry, returns that fallback.
        ``RuntimeError`` from a missing API key is not caught and propagates.
    """
    stripped = policy_text.strip()
    if not stripped:
        return _fallback_intent(policy_text, "empty policy text")

    base_prompt = _parser_prompt(stripped)

    def _validate(data: dict[str, Any]) -> ParsedIntent:
        return ParsedIntent.model_validate(data)

    try:
        data = chat_json(base_prompt)
        return _finalize_parsed_intent(_validate(data))
    except genai_errors.APIError as err:
        logger.warning("parse_policy: Gemini API error, using fallback: %s", err)
        return _fallback_intent(stripped, "gemini api")
    except json.JSONDecodeError as err:
        logger.warning("parse_policy: JSON failure after llm retries: %s", err)
        return _fallback_intent(stripped, "json decode")
    except ValidationError as err:
        logger.warning("parse_policy: validation failed, retrying with errors: %s", err)
        retry_prompt = (
            f"{base_prompt}\n\n"
            "Your previous JSON failed schema validation. Fix it. Required shape: "
            '{"action_type": string, "target_activity": string, "scope_geographic": string, '
            '"scope_temporal": string, "magnitude": string, "affected_species": [strings...], '
            '"mapped_species": [canonical strings only from the allowed list]}. '
            "Return ONLY valid JSON, no markdown."
        )
        try:
            data2 = chat_json(retry_prompt)
            return _finalize_parsed_intent(_validate(data2))
        except genai_errors.APIError as err_api:
            logger.warning("parse_policy: API error on validation retry: %s", err_api)
            return _fallback_intent(stripped, "gemini api on retry")
        except (json.JSONDecodeError, ValidationError) as err2:
            logger.warning("parse_policy: retry failed: %s", err2)
            return _fallback_intent(stripped, "validation after retry")
