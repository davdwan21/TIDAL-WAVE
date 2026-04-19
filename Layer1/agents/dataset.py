"""Dataset agent reading local CalCOFI/iNaturalist CSV summaries (no LLM calls)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from agents.parser import TRAWLING_DATASET_MATCH, ParsedIntent
from schema import Source

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_CALCOFI_FILE = _DATA_DIR / "calcofi_summary.csv"
_INAT_FILE = _DATA_DIR / "inaturalist_summary.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_species_key(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def _row_matches_mapped(row_species: str, mapped_species: list[str]) -> bool:
    if not mapped_species:
        return False
    key = _normalize_species_key(row_species)
    mapped_set = {_normalize_species_key(m) for m in mapped_species}
    return key in mapped_set


async def dataset_agent(parsed: ParsedIntent) -> dict[str, Any]:
    """
    Gather baseline signals from local CalCOFI and iNaturalist summary CSV files.

    Input:
        parsed: Structured policy intent from parser agent.

    Output:
        Dictionary with findings, sources, baseline_values, trend_indicators, and reasoning_trace.

    Failure modes:
        Missing CSV files return empty lists with an explanatory finding; no exception raised.
    """
    calcofi_rows = _read_csv(_CALCOFI_FILE)
    inat_rows = _read_csv(_INAT_FILE)

    join_keys = list(parsed.dataset_match_species)
    calcofi_species_keys = {_normalize_species_key(row.get("species", "")) for row in calcofi_rows}
    inat_species_keys = {_normalize_species_key(row.get("species", "")) for row in inat_rows}
    mapped_for_calcofi = [m for m in join_keys if _normalize_species_key(m) in calcofi_species_keys]
    mapped_for_inat = [m for m in join_keys if _normalize_species_key(m) in inat_species_keys]

    relevant_calcofi = [
        row for row in calcofi_rows if _row_matches_mapped(row.get("species", ""), mapped_for_calcofi)
    ]
    relevant_inat = [
        row for row in inat_rows if _row_matches_mapped(row.get("species", ""), mapped_for_inat)
    ]

    baseline_values = list(relevant_calcofi)
    trend_indicators = list(relevant_inat)

    findings: list[str] = []
    for row in baseline_values:
        findings.append(
            f"CalCOFI baseline: {row.get('species', 'unknown')} {row.get('metric', 'metric')}={row.get('value', 'n/a')} ({row.get('trend', 'trend n/a')})."
        )
    for row in trend_indicators:
        findings.append(
            f"iNaturalist trend: {row.get('species', 'unknown')} sightings={row.get('recent_observations', 'n/a')} ({row.get('trend', 'trend n/a')})."
        )

    if not findings:
        findings = [
            "No local dataset rows matched dataset_match_species; check parser join keys and CSV species column."
        ]

    sources = [
        Source(
            title="CalCOFI local summary CSV",
            url=None,
            excerpt=f"Rows considered: {len(relevant_calcofi)} from {len(calcofi_rows)} total.",
        ),
        Source(
            title="iNaturalist local summary CSV",
            url=None,
            excerpt=f"Rows considered: {len(relevant_inat)} from {len(inat_rows)} total.",
        ),
    ]

    suggested_parameters: list[dict[str, Any]] = []
    if "runoff" in parsed.target_activity:
        suggested_parameters.append(
            {
                "target": "ocean.nutrient_level",
                "operation": "add",
                "value": -0.15,
                "reason": "Runoff-focused policies are usually linked to lower nutrient loading trends.",
            }
        )
    mapped_full = list(parsed.mapped_species)
    trawling_dataset = tuple(parsed.dataset_match_species) == TRAWLING_DATASET_MATCH
    if (
        "fishing_fleet" in mapped_full
        or "trawl" in parsed.target_activity
        or "fish" in parsed.target_activity
        or trawling_dataset
    ):
        suggested_parameters.append(
            {
                "target": "fishing_fleet.effort_level",
                "operation": "multiply",
                "value": 0.85,
                "reason": "Trawling or fleet-focused policies typically imply lower realized fishing effort in the near term.",
            }
        )

    return {
        "findings": findings,
        "sources": sources,
        "baseline_values": baseline_values,
        "trend_indicators": trend_indicators,
        "suggested_parameters": suggested_parameters,
        "reasoning_trace": [
            f"📊 Dataset agent matched {len(relevant_calcofi)} CalCOFI rows and {len(relevant_inat)} iNaturalist rows "
            f"(dataset_match_species={join_keys}; mapped_species={mapped_full}; "
            f"CalCOFI keys={mapped_for_calcofi}; iNat keys={mapped_for_inat})."
        ],
    }
