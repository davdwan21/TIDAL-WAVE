#!/usr/bin/env python3
"""
FATHOM — California Current Ecosystem Simulation
Integrated version: Proper ecological math + original visualization pipeline
"""

import argparse
import numpy as np
import re
import time
import webbrowser
import os
import json
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import database_fetch
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False
    print("  [Note: database_fetch not available, using simplified policy parsing]")

# ══════════════════════════════════════════════════════════════════════════════
# SPECIES PARAMETERS (Ecological model)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeciesParams:
    """Ecological parameters for a species"""
    name: str
    intrinsic_growth: float
    carrying_capacity: float
    consumption_rate: float
    conversion_efficiency: float
    mortality_rate: float
    temp_optimum: float
    temp_tolerance: float
    nutrient_dependence: float
    pollution_sensitivity: float


SPECIES_PARAMS = {
    "phytoplankton": SpeciesParams(
        name="Phytoplankton",
        intrinsic_growth=0.8,
        carrying_capacity=100.0,
        consumption_rate=0.0,
        conversion_efficiency=0.0,
        mortality_rate=0.15,
        temp_optimum=14.0,
        temp_tolerance=4.0,
        nutrient_dependence=0.9,
        pollution_sensitivity=0.3
    ),
    "zooplankton": SpeciesParams(
        name="Zooplankton",
        intrinsic_growth=0.5,
        carrying_capacity=80.0,
        consumption_rate=0.012,
        conversion_efficiency=0.3,
        mortality_rate=0.2,
        temp_optimum=15.0,
        temp_tolerance=5.0,
        nutrient_dependence=0.0,
        pollution_sensitivity=0.25
    ),
    "anchovy": SpeciesParams(
        name="Anchovy",
        intrinsic_growth=0.4,
        carrying_capacity=70.0,
        consumption_rate=0.015,
        conversion_efficiency=0.25,
        mortality_rate=0.15,
        temp_optimum=14.0,
        temp_tolerance=3.0,
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
        temp_optimum=17.0,
        temp_tolerance=4.0,
        nutrient_dependence=0.0,
        pollution_sensitivity=0.2
    ),
    "sea_lion": SpeciesParams(
        name="Sea Lion",
        intrinsic_growth=0.15,
        carrying_capacity=60.0,
        consumption_rate=0.008,
        conversion_efficiency=0.15,
        mortality_rate=0.08,
        temp_optimum=16.0,
        temp_tolerance=6.0,
        nutrient_dependence=0.0,
        pollution_sensitivity=0.35
    ),
    "kelp": SpeciesParams(
        name="Kelp",
        intrinsic_growth=0.6,
        carrying_capacity=90.0,
        consumption_rate=0.0,
        conversion_efficiency=0.0,
        mortality_rate=0.1,
        temp_optimum=13.0,
        temp_tolerance=4.0,
        nutrient_dependence=0.7,
        pollution_sensitivity=0.25
    ),
    "urchin": SpeciesParams(
        name="Urchin",
        intrinsic_growth=0.3,
        carrying_capacity=70.0,
        consumption_rate=0.018,
        conversion_efficiency=0.2,
        mortality_rate=0.12,
        temp_optimum=15.0,
        temp_tolerance=5.0,
        nutrient_dependence=0.0,
        pollution_sensitivity=0.15
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# ECOLOGICAL EQUATIONS
# ══════════════════════════════════════════════════════════════════════════════

def temperature_stress(current_temp: float, optimum: float, tolerance: float) -> float:
    """Calculate temperature stress multiplier (0-1)"""
    deviation = abs(current_temp - optimum)
    stress = np.exp(-0.5 * (deviation / tolerance) ** 2)
    return stress


def nutrient_limitation(nutrients: float, dependence: float) -> float:
    """Michaelis-Menten style nutrient limitation"""
    if dependence == 0:
        return 1.0
    k_half = 0.3
    nutrient_effect = nutrients / (nutrients + k_half)
    return (1 - dependence) + dependence * nutrient_effect


def logistic_growth(population: float, carrying_capacity: float, growth_rate: float) -> float:
    """Logistic growth model: dN/dt = r*N*(1 - N/K)"""
    if population <= 0:
        return 0.0
    crowding_factor = 1.0 - (population / carrying_capacity)
    crowding_factor = max(0.0, crowding_factor)
    return growth_rate * population * crowding_factor


def predation_rate(predator_pop: float, prey_pop: float, attack_rate: float) -> float:
    """Type II functional response (Holling)"""
    if prey_pop <= 0 or predator_pop <= 0:
        return 0.0
    handling_time = 0.5
    consumption = (attack_rate * predator_pop * prey_pop) / (1 + attack_rate * handling_time * prey_pop)
    return min(consumption, prey_pop)


def fishing_mortality(population: float, fishing_pressure: float, base_rate: float = 0.3) -> float:
    """Fishing mortality increases with population"""
    if population <= 0:
        return 0.0
    catchability = 0.01 * population
    mortality = base_rate * fishing_pressure * catchability * population
    return mortality


# ══════════════════════════════════════════════════════════════════════════════
# BEHAVIOR NARRATION (for UI compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def generate_behavior_narrative(species_name: str, change: float, temp_stress: float, 
                                prey_pop: float = None, consumed: float = None) -> tuple:
    """
    Generate behavior names and reasons based on ecological changes
    This maintains compatibility with your existing UI while using real math
    """
    params = SPECIES_PARAMS[species_name]
    
    # Primary producers (phytoplankton, kelp)
    if species_name in ["phytoplankton", "kelp"]:
        if change > 8:
            action = "bloom"
            reason = "Optimal conditions for rapid growth"
        elif change > 2:
            action = "grow"
            reason = "Favorable environmental conditions"
        elif change > -2:
            action = "persist"
            reason = "Maintaining steady state"
        elif change > -10:
            action = "recede"
            if temp_stress < 0.6:
                reason = "Temperature stress limiting growth"
            else:
                reason = "Resource constraints slowing growth"
        else:
            action = "collapse" if species_name == "kelp" else "die_off"
            reason = "Severe environmental stress and high mortality"
    
    # Herbivores/primary consumers (zooplankton, urchin)
    elif species_name in ["zooplankton", "urchin"]:
        if change > 8:
            action = "reproduce" if species_name == "zooplankton" else "barren_expand"
            reason = "Abundant food enabling rapid population growth"
        elif change > 2:
            action = "graze" if species_name == "zooplankton" else "graze_kelp"
            reason = "Feeding successfully on available prey"
        elif change > -2:
            action = "swarm" if species_name == "zooplankton" else "reproduce"
            reason = "Maintaining population under current conditions"
        elif change > -10:
            action = "disperse" if species_name == "zooplankton" else "retreat"
            if prey_pop and prey_pop < 20:
                reason = "Food scarcity forcing dispersal"
            else:
                reason = "Environmental pressure reducing population"
        else:
            action = "starve"
            reason = "Severe food shortage causing population crash"
    
    # Fish (anchovy, sardine)
    elif species_name in ["anchovy", "sardine"]:
        if change > 8:
            action = "spawn"
            reason = "Excellent conditions for reproduction"
        elif change > 2:
            action = "feed_aggressively"
            reason = "Taking advantage of abundant zooplankton"
        elif change > -2:
            action = "school"
            reason = "Maintaining cohesion under moderate conditions"
        elif change > -10:
            if temp_stress < 0.5:
                action = "migrate_north" if species_name == "anchovy" else "migrate_south"
                reason = "Seeking optimal temperature zones"
            else:
                action = "scatter"
                reason = "Dispersing due to resource scarcity"
        else:
            action = "decline"
            if prey_pop and prey_pop < 15:
                reason = "Starvation from zooplankton collapse"
            else:
                reason = "Multiple stressors causing rapid decline"
    
    # Top predator (sea lion)
    elif species_name == "sea_lion":
        if change > 8:
            action = "thrive"
            reason = "Both prey species abundant, optimal hunting"
        elif change > 2:
            action = "hunt"
            reason = "Successfully feeding on fish populations"
        elif change > -2:
            action = "compete"
            reason = "Moderate prey availability, competing for food"
        elif change > -5:
            action = "haul_out"
            reason = "Conserving energy due to limited prey"
        elif change > -10:
            action = "migrate"
            reason = "Following declining fish populations"
        else:
            action = "starve"
            reason = "Catastrophic prey collapse, severe starvation"
    
    else:
        action = "persist"
        reason = "Continuing under current conditions"
    
    return action, reason


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE MANAGEMENT
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
        
        # Environment (baseline California Current)
        if HAS_DATABASE:
            self.environment = database_fetch.BASELINE_ENVIRONMENT.copy()
        else:
            self.environment = {
                "temperature": 16.2,
                "nutrients": 0.6,
                "pH": 8.05,
                "salinity": 33.4,
                "fishing_pressure": 0.2,
                "pollution_index": 0.3
            }
        
        # Track consumption for feedback
        self.consumption_this_tick = {species: 0.0 for species in self.populations}
        
        # For UI compatibility - track behaviors
        self.behaviors = {species: {"action": None, "reason": None} for species in self.populations}
    
    def get_population(self, species: str) -> float:
        return self.populations.get(species, 0.0)
    
    def set_population(self, species: str, value: float):
        self.populations[species] = np.clip(value, 0.0, 100.0)
    
    def get_agent_dict(self, species: str) -> dict:
        """Return agent dict compatible with existing UI"""
        return {
            "population": int(round(self.populations[species])),
            "last_action": self.behaviors[species]["action"],
            "health_trend": self._get_health_trend(species)
        }
    
    def _get_health_trend(self, species: str) -> str:
        """Determine health trend from behavior"""
        action = self.behaviors[species]["action"]
        if not action:
            return "stable"
        
        # Map actions to trends
        positive_actions = ["bloom", "grow", "reproduce", "spawn", "feed_aggressively", 
                          "thrive", "hunt", "graze", "barren_expand"]
        negative_actions = ["die_off", "collapse", "starve", "decline", "migrate_north", 
                          "migrate_south", "haul_out"]
        
        if action in positive_actions:
            return "improving"
        elif action in negative_actions:
            return "declining"
        else:
            return "stable"


# ══════════════════════════════════════════════════════════════════════════════
# SPECIES UPDATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def update_primary_producer(species_name: str, state: EcosystemState, dt: float = 1.0) -> Dict:
    """Update phytoplankton or kelp"""
    params = SPECIES_PARAMS[species_name]
    pop = state.get_population(species_name)
    env = state.environment
    
    # Environmental effects
    temp_mult = temperature_stress(env["temperature"], params.temp_optimum, params.temp_tolerance)
    nutrient_mult = nutrient_limitation(env["nutrients"], params.nutrient_dependence)
    
    # Growth
    growth = logistic_growth(pop, params.carrying_capacity, params.intrinsic_growth)
    growth *= temp_mult * nutrient_mult
    
    # Mortality
    natural_death = params.mortality_rate * pop
    pollution_death = params.pollution_sensitivity * env["pollution_index"] * pop
    consumed = state.consumption_this_tick[species_name]
    
    # Net change
    delta = (growth - natural_death - pollution_death - consumed) * dt
    new_pop = pop + delta
    state.set_population(species_name, new_pop)
    
    # Generate narrative
    action, reason = generate_behavior_narrative(species_name, delta, temp_mult)
    state.behaviors[species_name] = {"action": action, "reason": reason}
    
    return {"net_change": delta, "temp_stress": temp_mult}


def update_consumer(species_name: str, prey_name: str, state: EcosystemState, dt: float = 1.0) -> Dict:
    """Update a consumer species"""
    params = SPECIES_PARAMS[species_name]
    pop = state.get_population(species_name)
    prey_pop = state.get_population(prey_name)
    env = state.environment
    
    # Environmental effects
    temp_mult = temperature_stress(env["temperature"], params.temp_optimum, params.temp_tolerance)
    
    # Predation
    consumed = predation_rate(pop, prey_pop, params.consumption_rate)
    state.consumption_this_tick[prey_name] += consumed
    
    # Growth from food
    food_growth = consumed * params.conversion_efficiency
    
    # Mortality
    natural_death = params.mortality_rate * pop
    
    # Starvation
    starvation_threshold = 0.2
    if prey_pop < (SPECIES_PARAMS[prey_name].carrying_capacity * starvation_threshold):
        starvation = 0.15 * pop * (1 - prey_pop / (SPECIES_PARAMS[prey_name].carrying_capacity * starvation_threshold))
    else:
        starvation = 0.0
    
    pollution_death = params.pollution_sensitivity * env["pollution_index"] * pop
    temp_death = (1 - temp_mult) * 0.1 * pop
    
    # Net change
    delta = (food_growth - natural_death - starvation - pollution_death - temp_death) * dt
    new_pop = pop + delta
    state.set_population(species_name, new_pop)
    
    # Generate narrative
    action, reason = generate_behavior_narrative(species_name, delta, temp_mult, prey_pop, consumed)
    state.behaviors[species_name] = {"action": action, "reason": reason}
    
    return {"net_change": delta, "temp_stress": temp_mult}


def update_fish_with_competition(fish_name: str, competitor_name: str, state: EcosystemState, dt: float = 1.0) -> Dict:
    """Update anchovy or sardine with competition"""
    params = SPECIES_PARAMS[fish_name]
    pop = state.get_population(fish_name)
    competitor_pop = state.get_population(competitor_name)
    prey_pop = state.get_population("zooplankton")
    env = state.environment
    
    # Environmental effects
    temp_mult = temperature_stress(env["temperature"], params.temp_optimum, params.temp_tolerance)
    
    # Competition
    competition_factor = 1.0 / (1.0 + 0.3 * competitor_pop / params.carrying_capacity)
    effective_consumption_rate = params.consumption_rate * competition_factor
    
    consumed = predation_rate(pop, prey_pop, effective_consumption_rate)
    state.consumption_this_tick["zooplankton"] += consumed
    
    # Growth from food
    food_growth = consumed * params.conversion_efficiency
    
    # Mortality
    natural_death = params.mortality_rate * pop
    
    # Starvation
    if prey_pop < 15:
        starvation = 0.2 * pop * (1 - prey_pop / 15)
    else:
        starvation = 0.0
    
    pollution_death = params.pollution_sensitivity * env["pollution_index"] * pop
    temp_death = (1 - temp_mult) * 0.12 * pop
    fishing_death = fishing_mortality(pop, env["fishing_pressure"], base_rate=0.4)
    
    # Net change
    delta = (food_growth - natural_death - starvation - pollution_death - temp_death - fishing_death) * dt
    new_pop = pop + delta
    state.set_population(fish_name, new_pop)
    
    # Generate narrative
    action, reason = generate_behavior_narrative(fish_name, delta, temp_mult, prey_pop, consumed)
    state.behaviors[fish_name] = {"action": action, "reason": reason}
    
    return {"net_change": delta, "temp_stress": temp_mult}


def update_top_predator(state: EcosystemState, dt: float = 1.0) -> Dict:
    """Sea lions eat both anchovy and sardine"""
    params = SPECIES_PARAMS["sea_lion"]
    pop = state.get_population("sea_lion")
    anchovy_pop = state.get_population("anchovy")
    sardine_pop = state.get_population("sardine")
    env = state.environment
    
    # Environmental effects
    temp_mult = temperature_stress(env["temperature"], params.temp_optimum, params.temp_tolerance)
    
    # Predation on both prey types
    consumed_anchovy = predation_rate(pop * 0.5, anchovy_pop, params.consumption_rate)
    consumed_sardine = predation_rate(pop * 0.5, sardine_pop, params.consumption_rate)
    total_consumed = consumed_anchovy + consumed_sardine
    
    state.consumption_this_tick["anchovy"] += consumed_anchovy
    state.consumption_this_tick["sardine"] += consumed_sardine
    
    # Growth from food
    food_growth = total_consumed * params.conversion_efficiency
    
    # Mortality
    natural_death = params.mortality_rate * pop
    
    # Starvation
    total_prey = anchovy_pop + sardine_pop
    if total_prey < 30:
        starvation = 0.2 * pop * (1 - total_prey / 30)
    else:
        starvation = 0.0
    
    pollution_death = params.pollution_sensitivity * env["pollution_index"] * pop
    fishing_death = fishing_mortality(pop, env["fishing_pressure"], base_rate=0.05)
    
    # Net change
    delta = (food_growth - natural_death - starvation - pollution_death - fishing_death) * dt
    new_pop = pop + delta
    state.set_population("sea_lion", new_pop)
    
    # Generate narrative
    action, reason = generate_behavior_narrative("sea_lion", delta, temp_mult, total_prey, total_consumed)
    state.behaviors["sea_lion"] = {"action": action, "reason": reason}
    
    return {"net_change": delta, "temp_stress": temp_mult}


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_tick(state: EcosystemState) -> Dict:
    """Run one time step of the simulation"""
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


def apply_policy_to_environment(policy_text: str, state: EcosystemState):
    """Apply a natural language policy to environmental parameters"""
    if not policy_text:
        return
    
    print(f"Applying policy: {policy_text}")
    
    # Try database_fetch if available
    if HAS_DATABASE:
        try:
            result = database_fetch.parse_policy(policy_text, baseline=state.environment)
            state.environment.update(result["environment"])
            print(f"Policy parse confidence: {result['confidence']:.2%}")
            print(f"Policy summary: {result['summary']}")
            return
        except Exception as exc:
            print(f"  [policy parse error] {exc}")
            print("  Falling back to manual policy application.")
    
    # Manual policy parsing
    policy_lower = policy_text.lower()
    
    # Temperature
    if "warming" in policy_lower or "temperature" in policy_lower:
        if "reduce" in policy_lower or "cool" in policy_lower:
            state.environment["temperature"] -= 1.0
            print(f"  → Temperature reduced to {state.environment['temperature']:.1f}°C")
        else:
            import re
            match = re.search(r'(\d+\.?\d*)\s*(?:degree|°c)', policy_lower)
            if match:
                change = float(match.group(1))
                state.environment["temperature"] += change
                print(f"  → Temperature increased to {state.environment['temperature']:.1f}°C")
    
    # Fishing
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
    
    # Pollution
    if "pollution" in policy_lower or "clean" in policy_lower:
        if "reduce" in policy_lower or "clean" in policy_lower:
            state.environment["pollution_index"] *= 0.5
            print(f"  → Pollution reduced to {state.environment['pollution_index']:.2f}")
        elif "increase" in policy_lower:
            state.environment["pollution_index"] = min(1.0, state.environment["pollution_index"] * 1.5)
            print(f"  → Pollution increased to {state.environment['pollution_index']:.2f}")
    
    # Nutrients
    if "nutrient" in policy_lower or "fertilizer" in policy_lower or "upwelling" in policy_lower:
        if "reduce" in policy_lower:
            state.environment["nutrients"] *= 0.7
            print(f"  → Nutrients reduced to {state.environment['nutrients']:.2f}")
        elif "increase" in policy_lower or "enhance" in policy_lower:
            state.environment["nutrients"] = min(1.0, state.environment["nutrients"] * 1.3)
            print(f"  → Nutrients increased to {state.environment['nutrients']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION (Your original beautiful UI)
# ══════════════════════════════════════════════════════════════════════════════

simulation_history = []

def record_year(year, state: EcosystemState):
    """Record the state of a simulation year for later visualization"""
    agents_snapshot = {}
    for species in state.populations:
        agents_snapshot[species] = state.get_agent_dict(species)
    
    behaviors_snapshot = {
        species: state.behaviors[species].copy() 
        for species in state.populations
    }
    
    simulation_history.append({
        'year': year,
        'agents': agents_snapshot,
        'environment': state.environment.copy(),
        'behaviors': behaviors_snapshot
    })


def generate_final_map():
    """Generate the interactive HTML visualization"""
    years_data = json.dumps(simulation_history, indent=2)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TIDAL WAVE - California Current Ecosystem Simulation</title>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Crimson Pro', serif;
            background: linear-gradient(135deg, #0a1828 0%, #1a3a52 50%, #2a5a7a 100%);
            color: #e8f4f8;
            min-height: 100vh;
            padding: 2rem;
            overflow-x: hidden;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 2rem;
            position: relative;
        }}
        
        h1 {{
            font-size: 3.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #4dd0e1 0%, #80deea 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 0 40px rgba(77, 208, 225, 0.3);
        }}
        
        .subtitle {{
            font-family: 'Space Mono', monospace;
            font-size: 0.9rem;
            color: #80deea;
            opacity: 0.8;
            letter-spacing: 0.1em;
        }}
        
        .map-container {{
            background: rgba(10, 24, 40, 0.6);
            border-radius: 1.5rem;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(77, 208, 225, 0.2);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        }}
        
        .map-title {{
            font-family: 'Space Mono', monospace;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: #80deea;
            margin-bottom: 1rem;
            text-align: center;
        }}
        
        #regionMap {{
            height: 400px;
            border-radius: 1rem;
            overflow: hidden;
            border: 2px solid rgba(77, 208, 225, 0.3);
        }}
        
        .controls {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }}
        
        .year-nav {{
            display: flex;
            align-items: center;
            gap: 1rem;
            background: rgba(10, 24, 40, 0.6);
            padding: 1rem 2rem;
            border-radius: 2rem;
            border: 1px solid rgba(77, 208, 225, 0.3);
        }}
        
        .nav-button {{
            background: rgba(77, 208, 225, 0.2);
            border: 1px solid rgba(77, 208, 225, 0.4);
            color: #4dd0e1;
            font-family: 'Space Mono', monospace;
            font-size: 1rem;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 700;
        }}
        
        .nav-button:hover:not(:disabled) {{
            background: rgba(77, 208, 225, 0.3);
            border-color: rgba(77, 208, 225, 0.6);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(77, 208, 225, 0.3);
        }}
        
        .nav-button:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}
        
        .year-display {{
            font-family: 'Space Mono', monospace;
            font-size: 1.8rem;
            color: #80deea;
            font-weight: 700;
            min-width: 120px;
            text-align: center;
        }}
        
        .year-dots {{
            display: flex;
            gap: 0.5rem;
        }}
        
        .year-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: rgba(77, 208, 225, 0.3);
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }}
        
        .year-dot.active {{
            background: #4dd0e1;
            box-shadow: 0 0 12px rgba(77, 208, 225, 0.6);
            transform: scale(1.3);
        }}
        
        .year-dot:hover {{
            background: #80deea;
            transform: scale(1.2);
        }}
        
        .ocean-map {{
            background: linear-gradient(180deg, rgba(26, 58, 82, 0.4) 0%, rgba(10, 24, 40, 0.6) 100%);
            border-radius: 1.5rem;
            padding: 3rem;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(77, 208, 225, 0.2);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        }}
        
        .ocean-map::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 30%, rgba(77, 208, 225, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(128, 222, 234, 0.03) 0%, transparent 50%);
            pointer-events: none;
        }}
        
        .depth-layers {{
            display: grid;
            gap: 2rem;
            position: relative;
            z-index: 1;
        }}
        
        .layer {{
            background: rgba(255, 255, 255, 0.03);
            border-left: 4px solid;
            padding: 2rem;
            border-radius: 0.75rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }}
        
        .layer:hover {{
            background: rgba(255, 255, 255, 0.06);
            transform: translateX(10px);
        }}
        
        .layer.surface {{ border-color: #4dd0e1; }}
        .layer.mid-water {{ border-color: #26c6da; }}
        .layer.benthic {{ border-color: #00acc1; }}
        
        .layer-title {{
            font-family: 'Space Mono', monospace;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: #80deea;
            margin-bottom: 1.5rem;
            opacity: 0.7;
        }}
        
        .species-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }}
        
        .species-card {{
            background: rgba(10, 24, 40, 0.4);
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid rgba(77, 208, 225, 0.15);
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .species-card:hover {{
            border-color: rgba(77, 208, 225, 0.4);
            box-shadow: 0 10px 30px rgba(77, 208, 225, 0.1);
        }}
        
        .species-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        
        .species-icon {{
            font-size: 2.5rem;
            line-height: 1;
        }}
        
        .species-name {{
            font-size: 1.4rem;
            font-weight: 600;
            color: #e8f4f8;
        }}
        
        .population-bar {{
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
            overflow: hidden;
            margin: 1rem 0;
            position: relative;
        }}
        
        .population-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4dd0e1, #80deea);
            border-radius: 1rem;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            box-shadow: 0 0 10px rgba(77, 208, 225, 0.5);
        }}
        
        .population-fill::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer 2s infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        
        .population-value {{
            font-family: 'Space Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            color: #4dd0e1;
            margin: 0.5rem 0;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            font-family: 'Space Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.5rem;
        }}
        
        .status-improving {{
            background: rgba(102, 187, 106, 0.2);
            color: #66bb6a;
            border: 1px solid rgba(102, 187, 106, 0.3);
        }}
        
        .status-stable {{
            background: rgba(77, 208, 225, 0.2);
            color: #4dd0e1;
            border: 1px solid rgba(77, 208, 225, 0.3);
        }}
        
        .status-declining {{
            background: rgba(239, 83, 80, 0.2);
            color: #ef5350;
            border: 1px solid rgba(239, 83, 80, 0.3);
        }}
        
        .behavior {{
            font-size: 0.9rem;
            color: #b0bec5;
            margin-top: 0.75rem;
            font-style: italic;
            line-height: 1.4;
        }}
        
        .behavior strong {{
            color: #80deea;
            font-style: normal;
        }}
        
        .environment-panel {{
            background: rgba(10, 24, 40, 0.6);
            border-radius: 1rem;
            padding: 2rem;
            margin-top: 3rem;
            border: 1px solid rgba(77, 208, 225, 0.2);
        }}
        
        .env-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}
        
        .env-stat {{
            text-align: center;
        }}
        
        .env-label {{
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #80deea;
            opacity: 0.7;
            margin-bottom: 0.5rem;
        }}
        
        .env-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #4dd0e1;
            font-family: 'Space Mono', monospace;
        }}
        
        .env-unit {{
            font-size: 1rem;
            color: #b0bec5;
            margin-left: 0.25rem;
        }}
        
        .leaflet-container {{
            background: #1a3a52;
        }}
        
        .leaflet-popup-content-wrapper {{
            background: rgba(10, 24, 40, 0.95);
            color: #e8f4f8;
            border: 1px solid rgba(77, 208, 225, 0.3);
            border-radius: 0.5rem;
        }}
        
        .leaflet-popup-tip {{
            background: rgba(10, 24, 40, 0.95);
        }}
        
        .leaflet-popup-content {{
            font-family: 'Crimson Pro', serif;
            margin: 1rem;
        }}
        
        .leaflet-popup-content h3 {{
            color: #4dd0e1;
            margin-bottom: 0.5rem;
        }}
        
        @media (max-width: 768px) {{
            h1 {{
                font-size: 2.5rem;
            }}
            
            .ocean-map {{
                padding: 1.5rem;
            }}
            
            .species-grid {{
                grid-template-columns: 1fr;
            }}
            
            .controls {{
                flex-direction: column;
            }}
            
            #regionMap {{
                height: 300px;
            }}
        }}
        
        .wave {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 200%;
            height: 100px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 100'%3E%3Cpath fill='rgba(77, 208, 225, 0.05)' d='M0,50 Q300,0 600,50 T1200,50 L1200,100 L0,100 Z'/%3E%3C/svg%3E");
            background-size: 1200px 100px;
            animation: wave 20s linear infinite;
            pointer-events: none;
            z-index: 0;
        }}
        
        @keyframes wave {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-1200px); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.4s ease-in;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <div class="wave"></div>
    <div class="container">
        <header>
            <h1>TIDAL WAVE</h1>
            <div class="subtitle">Ecologically Accurate California Current Simulation</div>
        </header>
        
        <div class="controls">
            <div class="year-nav">
                <button class="nav-button" id="prevBtn" onclick="changeYear(-1)">← PREV</button>
                <div class="year-display" id="yearDisplay">YEAR 1</div>
                <button class="nav-button" id="nextBtn" onclick="changeYear(1)">NEXT →</button>
            </div>
            <div class="year-dots" id="yearDots"></div>
        </div>
        
        <div class="map-container">
            <div class="map-title">California Current Ecosystem Region</div>
            <div id="regionMap"></div>
        </div>
        
        <div id="mapContainer"></div>
    </div>
    
    <script>
        const simulationData = {years_data};
        let currentYearIndex = 0;
        
        function initRegionMap() {{
            const map = L.map('regionMap').setView([35.5, -121.5], 6);
            
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                attribution: 'Tiles &copy; Esri',
                maxZoom: 13,
                minZoom: 5
            }}).addTo(map);
            
            const californiaCurrentZone = [
                [41.0, -125.0],
                [41.0, -121.0],
                [32.5, -117.0],
                [32.5, -122.0],
                [41.0, -125.0]
            ];
            
            L.polygon(californiaCurrentZone, {{
                color: '#4dd0e1',
                fillColor: '#4dd0e1',
                fillOpacity: 0.15,
                weight: 2,
                dashArray: '5, 10'
            }}).addTo(map).bindPopup('<h3>California Current System</h3><p>Cold-water upwelling zone supporting rich marine biodiversity</p>');
            
            const locations = [
                {{
                    name: 'Monterey Bay',
                    coords: [36.8, -121.9],
                    description: 'Major upwelling center and kelp forest habitat'
                }},
                {{
                    name: 'Channel Islands',
                    coords: [34.0, -119.8],
                    description: 'Critical breeding grounds for sea lions and seabirds'
                }},
                {{
                    name: 'Point Conception',
                    coords: [34.45, -120.47],
                    description: 'Biogeographic boundary - warm and cold currents meet'
                }},
                {{
                    name: 'San Diego',
                    coords: [32.7, -117.2],
                    description: 'Southern extent of simulation region'
                }}
            ];
            
            locations.forEach(loc => {{
                L.circleMarker(loc.coords, {{
                    radius: 6,
                    fillColor: '#80deea',
                    color: '#4dd0e1',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.8
                }}).addTo(map).bindPopup(`<h3>${{loc.name}}</h3><p>${{loc.description}}</p>`);
            }});
        }}
        
        function renderYear(index) {{
            const data = simulationData[index];
            const agents = data.agents;
            const env = data.environment;
            const behaviors = data.behaviors;
            
            const html = `
                <div class="ocean-map fade-in">
                    <div class="depth-layers">
                        <div class="layer surface">
                            <div class="layer-title">Surface Waters (0-50m)</div>
                            <div class="species-grid">
                                <div class="species-card">
                                    <div class="species-header">
                                        <span class="species-icon">🌿</span>
                                        <span class="species-name">Phytoplankton</span>
                                    </div>
                                    <div class="population-value">${{agents.phytoplankton.population}}<span style="font-size: 1rem; color: #b0bec5;">/100</span></div>
                                    <div class="population-bar">
                                        <div class="population-fill" style="width: ${{agents.phytoplankton.population}}%"></div>
                                    </div>
                                    <span class="status-badge status-${{agents.phytoplankton.health_trend}}">${{agents.phytoplankton.health_trend}}</span>
                                    <div class="behavior"><strong>${{behaviors.phytoplankton.action}}</strong> — ${{behaviors.phytoplankton.reason}}</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="layer mid-water">
                            <div class="layer-title">Mid-Water Zone (50-200m)</div>
                            <div class="species-grid">
                                <div class="species-card">
                                    <div class="species-header">
                                        <span class="species-icon">🦐</span>
                                        <span class="species-name">Zooplankton</span>
                                    </div>
                                    <div class="population-value">${{agents.zooplankton.population}}<span style="font-size: 1rem; color: #b0bec5;">/100</span></div>
                                    <div class="population-bar">
                                        <div class="population-fill" style="width: ${{agents.zooplankton.population}}%"></div>
                                    </div>
                                    <span class="status-badge status-${{agents.zooplankton.health_trend}}">${{agents.zooplankton.health_trend}}</span>
                                    <div class="behavior"><strong>${{behaviors.zooplankton.action}}</strong> — ${{behaviors.zooplankton.reason}}</div>
                                </div>
                                
                                <div class="species-card">
                                    <div class="species-header">
                                        <span class="species-icon">🐟</span>
                                        <span class="species-name">Anchovy</span>
                                    </div>
                                    <div class="population-value">${{agents.anchovy.population}}<span style="font-size: 1rem; color: #b0bec5;">/100</span></div>
                                    <div class="population-bar">
                                        <div class="population-fill" style="width: ${{agents.anchovy.population}}%"></div>
                                    </div>
                                    <span class="status-badge status-${{agents.anchovy.health_trend}}">${{agents.anchovy.health_trend}}</span>
                                    <div class="behavior"><strong>${{behaviors.anchovy.action}}</strong> — ${{behaviors.anchovy.reason}}</div>
                                </div>
                                
                                <div class="species-card">
                                    <div class="species-header">
                                        <span class="species-icon">🐟</span>
                                        <span class="species-name">Sardine</span>
                                    </div>
                                    <div class="population-value">${{agents.sardine.population}}<span style="font-size: 1rem; color: #b0bec5;">/100</span></div>
                                    <div class="population-bar">
                                        <div class="population-fill" style="width: ${{agents.sardine.population}}%"></div>
                                    </div>
                                    <span class="status-badge status-${{agents.sardine.health_trend}}">${{agents.sardine.health_trend}}</span>
                                    <div class="behavior"><strong>${{behaviors.sardine.action}}</strong> — ${{behaviors.sardine.reason}}</div>
                                </div>
                                
                                <div class="species-card">
                                    <div class="species-header">
                                        <span class="species-icon">🦭</span>
                                        <span class="species-name">Sea Lion</span>
                                    </div>
                                    <div class="population-value">${{agents.sea_lion.population}}<span style="font-size: 1rem; color: #b0bec5;">/100</span></div>
                                    <div class="population-bar">
                                        <div class="population-fill" style="width: ${{agents.sea_lion.population}}%"></div>
                                    </div>
                                    <span class="status-badge status-${{agents.sea_lion.health_trend}}">${{agents.sea_lion.health_trend}}</span>
                                    <div class="behavior"><strong>${{behaviors.sea_lion.action}}</strong> — ${{behaviors.sea_lion.reason}}</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="layer benthic">
                            <div class="layer-title">Benthic Zone (Seafloor)</div>
                            <div class="species-grid">
                                <div class="species-card">
                                    <div class="species-header">
                                        <span class="species-icon">🌱</span>
                                        <span class="species-name">Kelp</span>
                                    </div>
                                    <div class="population-value">${{agents.kelp.population}}<span style="font-size: 1rem; color: #b0bec5;">/100</span></div>
                                    <div class="population-bar">
                                        <div class="population-fill" style="width: ${{agents.kelp.population}}%"></div>
                                    </div>
                                    <span class="status-badge status-${{agents.kelp.health_trend}}">${{agents.kelp.health_trend}}</span>
                                    <div class="behavior"><strong>${{behaviors.kelp.action}}</strong> — ${{behaviors.kelp.reason}}</div>
                                </div>
                                
                                <div class="species-card">
                                    <div class="species-header">
                                        <span class="species-icon">🦔</span>
                                        <span class="species-name">Urchin</span>
                                    </div>
                                    <div class="population-value">${{agents.urchin.population}}<span style="font-size: 1rem; color: #b0bec5;">/100</span></div>
                                    <div class="population-bar">
                                        <div class="population-fill" style="width: ${{agents.urchin.population}}%"></div>
                                    </div>
                                    <span class="status-badge status-${{agents.urchin.health_trend}}">${{agents.urchin.health_trend}}</span>
                                    <div class="behavior"><strong>${{behaviors.urchin.action}}</strong> — ${{behaviors.urchin.reason}}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="environment-panel fade-in">
                    <h2 style="font-size: 1.5rem; margin-bottom: 1rem; color: #80deea;">Environmental Conditions</h2>
                    <div class="env-grid">
                        <div class="env-stat">
                            <div class="env-label">Temperature</div>
                            <div class="env-value">${{env.temperature}}<span class="env-unit">°C</span></div>
                        </div>
                        <div class="env-stat">
                            <div class="env-label">Nutrients</div>
                            <div class="env-value">${{env.nutrients.toFixed(2)}}<span class="env-unit">/1.0</span></div>
                        </div>
                        <div class="env-stat">
                            <div class="env-label">pH Level</div>
                            <div class="env-value">${{env.pH.toFixed(2)}}</div>
                        </div>
                        <div class="env-stat">
                            <div class="env-label">Pollution</div>
                            <div class="env-value">${{env.pollution_index.toFixed(2)}}<span class="env-unit">/1.0</span></div>
                        </div>
                        <div class="env-stat">
                            <div class="env-label">Fishing Pressure</div>
                            <div class="env-value">${{env.fishing_pressure.toFixed(2)}}<span class="env-unit">/1.0</span></div>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('mapContainer').innerHTML = html;
            document.getElementById('yearDisplay').textContent = `YEAR ${{data.year}}`;
            
            document.getElementById('prevBtn').disabled = index === 0;
            document.getElementById('nextBtn').disabled = index === simulationData.length - 1;
            
            updateDots();
        }}
        
        function changeYear(delta) {{
            const newIndex = currentYearIndex + delta;
            if (newIndex >= 0 && newIndex < simulationData.length) {{
                currentYearIndex = newIndex;
                renderYear(currentYearIndex);
            }}
        }}
        
        function jumpToYear(index) {{
            currentYearIndex = index;
            renderYear(currentYearIndex);
        }}
        
        function updateDots() {{
            const dotsContainer = document.getElementById('yearDots');
            dotsContainer.innerHTML = '';
            
            simulationData.forEach((_, index) => {{
                const dot = document.createElement('div');
                dot.className = 'year-dot' + (index === currentYearIndex ? ' active' : '');
                dot.onclick = () => jumpToYear(index);
                dot.title = `Year ${{index + 1}}`;
                dotsContainer.appendChild(dot);
            }});
        }}
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') changeYear(-1);
            if (e.key === 'ArrowRight') changeYear(1);
        }});
        
        initRegionMap();
        renderYear(0);
    </script>
</body>
</html>"""
    
    return html


def display_final_map():
    """Generate and open the final interactive map"""
    html_content = generate_final_map()
    
    output_path = os.path.join(os.getcwd(), 'fathom_simulation.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    webbrowser.open('file://' + os.path.abspath(output_path))
    print(f"\n✓ Interactive map generated: {output_path}")
    print(f"✓ Opened in browser (use ← → arrows or dots to navigate years)")
    
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FATHOM - California Current Ecosystem Simulation (Integrated)"
    )
    parser.add_argument(
        "--policy", 
        type=str, 
        default="",
        help="Natural language policy to apply"
    )
    parser.add_argument(
        "--years", 
        type=int, 
        default=10,
        help="Number of years to simulate"
    )
    parser.add_argument(
        "--no-browser", 
        action="store_true",
        help="Don't open browser visualization"
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
        apply_policy_to_environment(args.policy, state)
        print()
    
    # Print header
    print("=" * 70)
    print("  TIDAL WAVE — California Current Ecosystem Simulation")
    print("  Ecologically Accurate Predator-Prey Dynamics")
    print("=" * 70)
    print(f"\nStarting conditions:")
    print(f"  Temperature: {state.environment['temperature']:.1f}°C")
    print(f"  Nutrients: {state.environment['nutrients']:.2f}")
    print(f"  Fishing pressure: {state.environment['fishing_pressure']:.2f}")
    print(f"  Pollution: {state.environment['pollution_index']:.2f}\n")
    
    # Run simulation
    for year in range(1, args.years + 1):
        print(f"{'─' * 70}")
        print(f"YEAR {year}")
        print(f"{'─' * 70}")
        
        # Run the tick
        results = run_tick(state)
        
        # Print results
        icons = {
            "phytoplankton": "🌿",
            "zooplankton": "🦐",
            "anchovy": "🐟",
            "sardine": "🐠",
            "sea_lion": "🦭",
            "kelp": "🌱",
            "urchin": "🦔"
        }
        
        for species in ["phytoplankton", "zooplankton", "anchovy", "sardine", "sea_lion", "kelp", "urchin"]:
            agent_dict = state.get_agent_dict(species)
            behavior = state.behaviors[species]
            
            print(f"{icons[species]} {SPECIES_PARAMS[species].name:15s} | "
                  f"{behavior['action']:<20} | pop: {agent_dict['population']:>3}/100 | "
                  f"{agent_dict['health_trend']}")
            print(f"   → {behavior['reason']}")
        
        print()
        
        # Record for visualization
        record_year(year, state)
        
        time.sleep(0.3)
    
    # Final summary
    print("=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    for species in state.populations:
        agent_dict = state.get_agent_dict(species)
        print(f"{SPECIES_PARAMS[species].name:15s}: {agent_dict['population']:>3}/100 ({agent_dict['health_trend']})")
    print("=" * 70)
    
    # Generate visualization
    if not args.no_browser:
        display_final_map()


if __name__ == "__main__":
    main()