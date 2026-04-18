"""Dataset agent reading local CalCOFI/iNaturalist CSV summaries (no LLM calls)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from agents.parser import ParsedIntent
from schema import Source

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_CALCOFI_FILE = _DATA_DIR / "calcofi_summary.csv"
_INAT_FILE = _DATA_DIR / "inaturalist_summary.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def _is_relevant(species_name: str, affected_species: list[str]) -> bool:
    if not affected_species:
        return True
    candidate = _normalize(species_name)
    return any(token in candidate or candidate in token for token in (_normalize(v) for v in affected_species))


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

    relevant_calcofi = [row for row in calcofi_rows if _is_relevant(row.get("species", ""), parsed.affected_species)]
    relevant_inat = [row for row in inat_rows if _is_relevant(row.get("species", ""), parsed.affected_species)]

    baseline_values = relevant_calcofi[:3]
    trend_indicators = relevant_inat[:3]

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
            "No local dataset rows matched parsed species; using empty baseline/trend placeholders."
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

    return {
        "findings": findings,
        "sources": sources,
        "baseline_values": baseline_values,
        "trend_indicators": trend_indicators,
        "suggested_parameters": suggested_parameters,
        "reasoning_trace": [
            f"📊 Dataset agent matched {len(baseline_values)} CalCOFI rows and {len(trend_indicators)} iNaturalist rows."
        ],
    }
