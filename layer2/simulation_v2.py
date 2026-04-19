#!/usr/bin/env python3
"""
FATHOM v2 — California Current Ecosystem Simulation
Ecologically accurate model with proper predator-prey dynamics
"""

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import json

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeciesParams:
    """Ecological parameters for a species"""
    name: str
    intrinsic_growth: float      # r: growth rate when resources unlimited
    carrying_capacity: float     # K: max population given resources
    consumption_rate: float      # α: how much prey consumed per predator
    conversion_efficiency: float # e: prey biomass → predator biomass conversion
    mortality_rate: float        # m: natural death rate
    temp_optimum: float         # Optimal temperature
    temp_tolerance: float       # Temperature tolerance range
    
    # Environmental sensitivity
    nutrient_dependence: float  # How much growth depends on nutrients (0-1)
    pollution_sensitivity: float # Mortality increase per pollution unit


# ══════════════════════════════════════════════════════════════════════════════
# SPECIES DEFINITIONS (based on California Current ecology)
# ══════════════════════════════════════════════════════════════════════════════

SPECIES_PARAMS = {
    "phytoplankton": SpeciesParams(
        name="Phytoplankton",
        intrinsic_growth=0.8,        # Fast growth in good conditions
        carrying_capacity=100.0,
        consumption_rate=0.0,        # Primary producer
        conversion_efficiency=0.0,
        mortality_rate=0.15,         # Natural die-off
        temp_optimum=14.0,           # Optimal for cold-water diatoms
        temp_tolerance=4.0,
        nutrient_dependence=0.9,     # Highly nutrient-dependent
        pollution_sensitivity=0.3
    ),
    
    "zooplankton": SpeciesParams(
        name="Zooplankton",
        intrinsic_growth=0.5,
        carrying_capacity=80.0,      # Limited by phytoplankton
        consumption_rate=0.012,      # Grazing rate on phytoplankton
        conversion_efficiency=0.3,   # ~30% conversion efficiency
        mortality_rate=0.2,
        temp_optimum=15.0,
        temp_tolerance=5.0,
        nutrient_dependence=0.0,     # Gets nutrients from prey
        pollution_sensitivity=0.25
    ),
    
    "anchovy": SpeciesParams(
        name="Anchovy",
        intrinsic_growth=0.4,
        carrying_capacity=70.0,
        consumption_rate=0.015,      # Filter-feeding rate
        conversion_efficiency=0.25,
        mortality_rate=0.15,
        temp_optimum=14.0,           # Prefers cold water
        temp_tolerance=3.0,          # Narrow tolerance
        nutrient_dependence=0.0,
        pollution_sensitivity=0.2
    ),
    
    "sardine": SpeciesParams(
        name="Sardine",
        intrinsic_growth=0.35,
        carrying_capacity=70.0,
        consumption_rate=0.013,
        conversion_efficiency=0.25,
        mortality_rate=0.15,
        temp_optimum=17.0,           # Prefers warmer water than anchovy
        temp_tolerance=4.0,          # Broader tolerance
        nutrient_dependence=0.0,
        pollution_sensitivity=0.2
    ),
    
    "sea_lion": SpeciesParams(
        name="Sea Lion",
        intrinsic_growth=0.15,       # Slow reproduction (K-selected)
        carrying_capacity=60.0,
        consumption_rate=0.008,      # Lower consumption rate (larger predator)
        conversion_efficiency=0.15,  # Lower efficiency (endothermic)
        mortality_rate=0.08,
        temp_optimum=16.0,
        temp_tolerance=6.0,
        nutrient_dependence=0.0,
        pollution_sensitivity=0.35   # Bioaccumulation
    ),
    
    "kelp": SpeciesParams(
        name="Kelp",
        intrinsic_growth=0.6,
        carrying_capacity=90.0,
        consumption_rate=0.0,        # Primary producer
        conversion_efficiency=0.0,
        mortality_rate=0.1,
        temp_optimum=13.0,           # Cold-water species
        temp_tolerance=4.0,
        nutrient_dependence=0.7,
        pollution_sensitivity=0.25
    ),
    
    "urchin": SpeciesParams(
        name="Urchin",
        intrinsic_growth=0.3,
        carrying_capacity=70.0,
        consumption_rate=0.018,      # High grazing rate
        conversion_efficiency=0.2,
        mortality_rate=0.12,
        temp_optimum=15.0,
        temp_tolerance=5.0,
        nutrient_dependence=0.0,
        pollution_sensitivity=0.15
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# ECOSYSTEM STATE
# ══════════════════════════════════════════════════════════════════════════════

class EcosystemState:
    """Manages all species populations and environmental conditions"""
    
    def __init__(self):
        # Population levels (0-100 scale)
        self.populations = {
            "phytoplankton": 65.0,
            "zooplankton": 55.0,
            "anchovy": 50.0,
            "sardine": 45.0,
            "sea_lion": 50.0,
            "kelp": 60.0,
            "urchin": 40.0,
        }
        
        # Environmental state (baseline California Current)
        self.environment = {
            "temperature": 16.2,      # °C
            "nutrients": 0.6,         # 0-1 scale
            "pH": 8.05,              # pH units
            "salinity": 33.4,        # PSU
            "fishing_pressure": 0.2, # 0-1 scale
            "pollution_index": 0.3,  # 0-1 scale
        }
        
        # Track consumption for feedback
        self.consumption_this_tick = {species: 0.0 for species in self.populations}
        
    def get_population(self, species: str) -> float:
        """Get current population for a species"""
        return self.populations.get(species, 0.0)
    
    def set_population(self, species: str, value: float):
        """Set population (clamped 0-100)"""
        self.populations[species] = np.clip(value, 0.0, 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# ECOLOGICAL EQUATIONS
# ══════════════════════════════════════════════════════════════════════════════

def temperature_stress(current_temp: float, optimum: float, tolerance: float) -> float:
    """
    Calculate temperature stress multiplier (0-1)
    Returns 1.0 at optimum, decreases with distance from optimum
    Uses Gaussian curve for realistic gradual stress
    """
    deviation = abs(current_temp - optimum)
    stress = np.exp(-0.5 * (deviation / tolerance) ** 2)
    return stress


def nutrient_limitation(nutrients: float, dependence: float) -> float:
    """
    Michaelis-Menten style nutrient limitation
    Returns growth multiplier based on nutrient availability
    """
    if dependence == 0:
        return 1.0
    
    # Half-saturation constant
    k_half = 0.3
    nutrient_effect = nutrients / (nutrients + k_half)
    
    # Blend between nutrient-limited and nutrient-independent
    return (1 - dependence) + dependence * nutrient_effect


def logistic_growth(population: float, carrying_capacity: float, growth_rate: float) -> float:
    """
    Logistic growth model: dN/dt = r*N*(1 - N/K)
    Returns the growth term (before other factors)
    """
    if population <= 0:
        return 0.0
    
    crowding_factor = 1.0 - (population / carrying_capacity)
    crowding_factor = max(0.0, crowding_factor)  # Can't be negative
    
    return growth_rate * population * crowding_factor


def predation_rate(predator_pop: float, prey_pop: float, attack_rate: float) -> float:
    """
    Type II functional response (Holling)
    Predation saturates as prey becomes very abundant
    Returns amount of prey consumed
    """
    if prey_pop <= 0 or predator_pop <= 0:
        return 0.0
    
    # Handling time creates saturation
    handling_time = 0.5
    consumption = (attack_rate * predator_pop * prey_pop) / (1 + attack_rate * handling_time * prey_pop)
    
    # Can't consume more than available
    return min(consumption, prey_pop)


def fishing_mortality(population: float, fishing_pressure: float, base_rate: float = 0.3) -> float:
    """
    Fishing mortality increases with population (more fish = more catch)
    """
    if population <= 0:
        return 0.0
    
    # Fishing is more effective on larger populations
    catchability = 0.01 * population
    mortality = base_rate * fishing_pressure * catchability * population
    
    return mortality


# ══════════════════════════════════════════════════════════════════════════════
# SPECIES UPDATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def update_primary_producer(species_name: str, state: EcosystemState, dt: float = 1.0) -> Dict:
    """
    Update phytoplankton or kelp (primary producers)
    Returns dict with population change breakdown
    """
    params = SPECIES_PARAMS[species_name]
    pop = state.get_population(species_name)
    env = state.environment
    
    # Environmental effects
    temp_mult = temperature_stress(env["temperature"], params.temp_optimum, params.temp_tolerance)
    nutrient_mult = nutrient_limitation(env["nutrients"], params.nutrient_dependence)
    
    # Growth
    growth = logistic_growth(pop, params.carrying_capacity, params.intrinsic_growth)
    growth *= temp_mult * nutrient_mult
    
    # Natural mortality
    natural_death = params.mortality_rate * pop
    
    # Pollution mortality
    pollution_death = params.pollution_sensitivity * env["pollution_index"] * pop
    
    # Consumption by predators (will be set by predator updates)
    consumed = state.consumption_this_tick[species_name]
    
    # Net change
    delta = (growth - natural_death - pollution_death - consumed) * dt
    new_pop = pop + delta
    
    state.set_population(species_name, new_pop)
    
    return {
        "growth": growth * dt,
        "natural_death": natural_death * dt,
        "pollution_death": pollution_death * dt,
        "consumed": consumed * dt,
        "net_change": delta,
        "temp_stress": temp_mult,
        "nutrient_limit": nutrient_mult
    }


def update_consumer(species_name: str, prey_name: str, state: EcosystemState, dt: float = 1.0) -> Dict:
    """
    Update a consumer species (herbivore or carnivore)
    Returns dict with population change breakdown
    """
    params = SPECIES_PARAMS[species_name]
    pop = state.get_population(species_name)
    prey_pop = state.get_population(prey_name)
    env = state.environment
    
    # Environmental effects
    temp_mult = temperature_stress(env["temperature"], params.temp_optimum, params.temp_tolerance)
    
    # Predation (food intake)
    consumed = predation_rate(pop, prey_pop, params.consumption_rate)
    
    # Record consumption for prey update
    state.consumption_this_tick[prey_name] += consumed
    
    # Growth from food (conversion of prey biomass)
    food_growth = consumed * params.conversion_efficiency
    
    # Natural mortality
    natural_death = params.mortality_rate * pop
    
    # Starvation (additional mortality when food is scarce)
    starvation_threshold = 0.2  # Below 20% of prey K, starvation kicks in
    if prey_pop < (SPECIES_PARAMS[prey_name].carrying_capacity * starvation_threshold):
        starvation = 0.15 * pop * (1 - prey_pop / (SPECIES_PARAMS[prey_name].carrying_capacity * starvation_threshold))
    else:
        starvation = 0.0
    
    # Pollution mortality
    pollution_death = params.pollution_sensitivity * env["pollution_index"] * pop
    
    # Temperature stress mortality (beyond just reduced growth)
    temp_death = (1 - temp_mult) * 0.1 * pop
    
    # Net change
    delta = (food_growth - natural_death - starvation - pollution_death - temp_death) * dt
    new_pop = pop + delta
    
    state.set_population(species_name, new_pop)
    
    return {
        "food_growth": food_growth * dt,
        "consumed_prey": consumed * dt,
        "natural_death": natural_death * dt,
        "starvation": starvation * dt,
        "pollution_death": pollution_death * dt,
        "temp_death": temp_death * dt,
        "net_change": delta,
        "temp_stress": temp_mult
    }


def update_top_predator(state: EcosystemState, dt: float = 1.0) -> Dict:
    """
    Sea lions eat both anchovy and sardine
    """
    params = SPECIES_PARAMS["sea_lion"]
    pop = state.get_population("sea_lion")
    anchovy_pop = state.get_population("anchovy")
    sardine_pop = state.get_population("sardine")
    env = state.environment
    
    # Environmental effects
    temp_mult = temperature_stress(env["temperature"], params.temp_optimum, params.temp_tolerance)
    
    # Predation on both prey types (sea lions are generalists)
    consumed_anchovy = predation_rate(pop * 0.5, anchovy_pop, params.consumption_rate)
    consumed_sardine = predation_rate(pop * 0.5, sardine_pop, params.consumption_rate)
    
    total_consumed = consumed_anchovy + consumed_sardine
    
    # Record consumption
    state.consumption_this_tick["anchovy"] += consumed_anchovy
    state.consumption_this_tick["sardine"] += consumed_sardine
    
    # Growth from food
    food_growth = total_consumed * params.conversion_efficiency
    
    # Natural mortality
    natural_death = params.mortality_rate * pop
    
    # Starvation when BOTH prey are scarce
    total_prey = anchovy_pop + sardine_pop
    if total_prey < 30:
        starvation = 0.2 * pop * (1 - total_prey / 30)
    else:
        starvation = 0.0
    
    # Pollution (bioaccumulation)
    pollution_death = params.pollution_sensitivity * env["pollution_index"] * pop
    
    # Fishing mortality (sea lions as bycatch)
    fishing_death = fishing_mortality(pop, env["fishing_pressure"], base_rate=0.05)
    
    # Net change
    delta = (food_growth - natural_death - starvation - pollution_death - fishing_death) * dt
    new_pop = pop + delta
    
    state.set_population("sea_lion", new_pop)
    
    return {
        "food_growth": food_growth * dt,
        "consumed_anchovy": consumed_anchovy * dt,
        "consumed_sardine": consumed_sardine * dt,
        "natural_death": natural_death * dt,
        "starvation": starvation * dt,
        "pollution_death": pollution_death * dt,
        "fishing_death": fishing_death * dt,
        "net_change": delta,
        "temp_stress": temp_mult
    }


def update_fish_with_competition(fish_name: str, competitor_name: str, state: EcosystemState, dt: float = 1.0) -> Dict:
    """
    Update anchovy or sardine with interspecific competition
    """
    params = SPECIES_PARAMS[fish_name]
    pop = state.get_population(fish_name)
    competitor_pop = state.get_population(competitor_name)
    prey_pop = state.get_population("zooplankton")
    env = state.environment
    
    # Environmental effects
    temp_mult = temperature_stress(env["temperature"], params.temp_optimum, params.temp_tolerance)
    
    # Predation (food intake)
    # Competition reduces effective prey availability
    competition_factor = 1.0 / (1.0 + 0.3 * competitor_pop / params.carrying_capacity)
    effective_consumption_rate = params.consumption_rate * competition_factor
    
    consumed = predation_rate(pop, prey_pop, effective_consumption_rate)
    state.consumption_this_tick["zooplankton"] += consumed
    
    # Growth from food
    food_growth = consumed * params.conversion_efficiency
    
    # Natural mortality
    natural_death = params.mortality_rate * pop
    
    # Starvation
    if prey_pop < 15:
        starvation = 0.2 * pop * (1 - prey_pop / 15)
    else:
        starvation = 0.0
    
    # Pollution mortality
    pollution_death = params.pollution_sensitivity * env["pollution_index"] * pop
    
    # Temperature stress mortality
    temp_death = (1 - temp_mult) * 0.12 * pop
    
    # Fishing mortality (major factor for commercial species)
    fishing_death = fishing_mortality(pop, env["fishing_pressure"], base_rate=0.4)
    
    # Net change
    delta = (food_growth - natural_death - starvation - pollution_death - temp_death - fishing_death) * dt
    new_pop = pop + delta
    
    state.set_population(fish_name, new_pop)
    
    return {
        "food_growth": food_growth * dt,
        "consumed_prey": consumed * dt,
        "natural_death": natural_death * dt,
        "starvation": starvation * dt,
        "pollution_death": pollution_death * dt,
        "temp_death": temp_death * dt,
        "fishing_death": fishing_death * dt,
        "competition_factor": competition_factor,
        "net_change": delta,
        "temp_stress": temp_mult
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_tick(state: EcosystemState, verbose: bool = True) -> Dict:
    """
    Run one time step of the simulation
    Returns breakdown of all population changes
    """
    # Reset consumption tracking
    state.consumption_this_tick = {species: 0.0 for species in state.populations}
    
    # Update in trophic order (bottom-up)
    results = {}
    
    # Primary producers
    results["phytoplankton"] = update_primary_producer("phytoplankton", state)
    results["kelp"] = update_primary_producer("kelp", state)
    
    # Primary consumers
    results["zooplankton"] = update_consumer("zooplankton", "phytoplankton", state)
    results["urchin"] = update_consumer("urchin", "kelp", state)
    
    # Secondary consumers (with competition)
    results["anchovy"] = update_fish_with_competition("anchovy", "sardine", state)
    results["sardine"] = update_fish_with_competition("sardine", "anchovy", state)
    
    # Top predator
    results["sea_lion"] = update_top_predator(state)
    
    return results


def apply_policy(state: EcosystemState, policy_text: str):
    """
    Parse and apply a policy to environmental parameters
    Simplified version - expand as needed
    """
    policy_lower = policy_text.lower()
    
    # Temperature policies
    if "warming" in policy_lower or "temperature" in policy_lower:
        if "reduce" in policy_lower or "cool" in policy_lower:
            state.environment["temperature"] -= 1.0
            print(f"  → Temperature reduced to {state.environment['temperature']:.1f}°C")
        else:
            # Extract degree change if specified
            import re
            match = re.search(r'(\d+\.?\d*)\s*(?:degree|°c)', policy_lower)
            if match:
                change = float(match.group(1))
                state.environment["temperature"] += change
                print(f"  → Temperature increased to {state.environment['temperature']:.1f}°C")
    
    # Fishing policies
    if "fishing" in policy_lower or "harvest" in policy_lower:
        if "ban" in policy_lower or "stop" in policy_lower or "zero" in policy_lower:
            state.environment["fishing_pressure"] = 0.0
            print(f"  → Fishing banned (pressure = 0.0)")
        elif "reduce" in policy_lower or "limit" in policy_lower:
            state.environment["fishing_pressure"] *= 0.5
            print(f"  → Fishing reduced to {state.environment['fishing_pressure']:.2f}")
        elif "increase" in policy_lower or "expand" in policy_lower:
            state.environment["fishing_pressure"] = min(1.0, state.environment["fishing_pressure"] * 1.5)
            print(f"  → Fishing increased to {state.environment['fishing_pressure']:.2f}")
    
    # Pollution policies
    if "pollution" in policy_lower or "clean" in policy_lower:
        if "reduce" in policy_lower or "clean" in policy_lower:
            state.environment["pollution_index"] *= 0.5
            print(f"  → Pollution reduced to {state.environment['pollution_index']:.2f}")
        elif "increase" in policy_lower:
            state.environment["pollution_index"] = min(1.0, state.environment["pollution_index"] * 1.5)
            print(f"  → Pollution increased to {state.environment['pollution_index']:.2f}")
    
    # Nutrient policies
    if "nutrient" in policy_lower or "fertilizer" in policy_lower or "upwelling" in policy_lower:
        if "reduce" in policy_lower:
            state.environment["nutrients"] *= 0.7
            print(f"  → Nutrients reduced to {state.environment['nutrients']:.2f}")
        elif "increase" in policy_lower or "enhance" in policy_lower:
            state.environment["nutrients"] = min(1.0, state.environment["nutrients"] * 1.3)
            print(f"  → Nutrients increased to {state.environment['nutrients']:.2f}")


def get_trend(change: float) -> str:
    """Convert population change to trend descriptor"""
    if change > 2:
        return "↑↑ surging"
    elif change > 0.5:
        return "↑ growing"
    elif change > -0.5:
        return "→ stable"
    elif change > -2:
        return "↓ declining"
    else:
        return "↓↓ crashing"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FATHOM v2 - California Current Ecosystem Simulation"
    )
    parser.add_argument(
        "--policy", 
        type=str, 
        default="",
        help="Natural language policy to apply (e.g., 'ban fishing', 'increase temperature by 2 degrees')"
    )
    parser.add_argument(
        "--years", 
        type=int, 
        default=10,
        help="Number of years to simulate"
    )
    parser.add_argument(
        "--json", 
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Initialize ecosystem
    state = EcosystemState()
    
    # Apply policy if provided
    if args.policy:
        print("=" * 70)
        print("  POLICY APPLICATION")
        print("=" * 70)
        print(f"Policy: {args.policy}\n")
        apply_policy(state, args.policy)
        print()
    
    # Print header
    if not args.json:
        print("=" * 70)
        print("  FATHOM v2 — California Current Ecosystem Simulation")
        print("  Ecologically accurate predator-prey dynamics")
        print("=" * 70)
        print(f"\nStarting conditions:")
        print(f"  Temperature: {state.environment['temperature']:.1f}°C")
        print(f"  Nutrients: {state.environment['nutrients']:.2f}")
        print(f"  Fishing pressure: {state.environment['fishing_pressure']:.2f}")
        print(f"  Pollution: {state.environment['pollution_index']:.2f}\n")
    
    # Track history for JSON output
    history = []
    
    # Run simulation
    for year in range(1, args.years + 1):
        results = run_tick(state)
        
        if args.json:
            # Store for JSON output
            year_data = {
                "year": year,
                "populations": dict(state.populations),
                "environment": dict(state.environment),
                "changes": {k: v["net_change"] for k, v in results.items()}
            }
            history.append(year_data)
        else:
            # Human-readable output
            print(f"{'─' * 70}")
            print(f"YEAR {year}")
            print(f"{'─' * 70}")
            
            # Print each species with key metrics
            for species in ["phytoplankton", "zooplankton", "anchovy", "sardine", "sea_lion", "kelp", "urchin"]:
                pop = state.populations[species]
                change = results[species]["net_change"]
                trend = get_trend(change)
                
                # Icon
                icons = {
                    "phytoplankton": "🌿",
                    "zooplankton": "🦐",
                    "anchovy": "🐟",
                    "sardine": "🐠",
                    "sea_lion": "🦭",
                    "kelp": "🌱",
                    "urchin": "🦔"
                }
                
                print(f"{icons[species]} {SPECIES_PARAMS[species].name:15s} │ "
                      f"pop: {pop:5.1f}/100 │ {trend:12s} │ Δ{change:+5.1f}")
                
                # Show key limiting factors
                details = []
                if "temp_stress" in results[species]:
                    stress = results[species]["temp_stress"]
                    if stress < 0.7:
                        details.append(f"temp stress: {stress:.2f}")
                
                if "consumed_prey" in results[species]:
                    consumed = results[species]["consumed_prey"]
                    if consumed > 0:
                        details.append(f"ate {consumed:.1f} prey")
                
                if species in ["anchovy", "sardine"] and "fishing_death" in results[species]:
                    fishing = results[species]["fishing_death"]
                    if fishing > 1:
                        details.append(f"fished: -{fishing:.1f}")
                
                if "starvation" in results[species] and results[species]["starvation"] > 0.5:
                    details.append(f"starving: -{results[species]['starvation']:.1f}")
                
                if details:
                    print(f"   └─ {' | '.join(details)}")
            
            print()
    
    # Final summary
    if args.json:
        print(json.dumps(history, indent=2))
    else:
        print("=" * 70)
        print("FINAL STATE")
        print("=" * 70)
        for species, pop in state.populations.items():
            print(f"{SPECIES_PARAMS[species].name:15s}: {pop:5.1f}/100")
        print("=" * 70)


if __name__ == "__main__":
    main()