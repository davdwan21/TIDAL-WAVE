"""Simulation parameter contract: approved targets and operations (Layer 1 ↔ Layer 2)."""

from __future__ import annotations

from functools import lru_cache

# Dot-notation keys the ecosystem sim accepts. Do not invent names elsewhere.
VALID_TARGETS: set[str] = {
    # Species population dynamics — rates are typically 0–2+ as multipliers vs baseline;
    # mortality/reproduction often interpreted as relative scale factors unless sim docs say otherwise.
    "phytoplankton.growth_rate",
    "zooplankton.growth_rate",
    "zooplankton.mortality_rate",
    "anchovy.mortality_rate",
    "anchovy.reproduction_rate",
    "anchovy.catch_rate",
    "sardine.mortality_rate",
    "sardine.reproduction_rate",
    "pelican.mortality_rate",
    "pelican.reproduction_rate",
    "sea_lion.mortality_rate",
    "sea_lion.reproduction_rate",
    "leopard_shark.mortality_rate",
    # Ocean state — temperature often °C delta or scale; pH small deltas (~0.01–0.1);
    # dissolved_oxygen, nutrient_level, pollution_index as relative or index scales per sim.
    "ocean.temperature",
    "ocean.ph",
    "ocean.dissolved_oxygen",
    "ocean.nutrient_level",
    "ocean.pollution_index",
    # Human pressure — catch_rate / effort_level / runoff / consumption as multipliers or index.
    "fishing_fleet.catch_rate",
    "fishing_fleet.effort_level",
    "coastal_community.runoff_rate",
    "coastal_community.consumption_rate",
    # Policy zones — coverage as 0–100 percent of domain or comparable unit per sim.
    "protected_area.coverage_percent",
}

VALID_OPERATIONS: set[str] = {"multiply", "add", "set"}


def validate_delta(target: str, operation: str) -> bool:
    """Return True if *target* and *operation* are allowed for a ParameterDelta."""
    return target in VALID_TARGETS and operation in VALID_OPERATIONS


@lru_cache(maxsize=1)
def valid_targets_prompt_block() -> str:
    """Stable newline-separated VALID_TARGETS for LLM prompts (lru_cache per project convention)."""
    return "\n".join(sorted(VALID_TARGETS))
