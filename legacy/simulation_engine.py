"""
Task 2: Simulation Engine
Rule-based ecosystem simulation for Southern California coastal waters.

Simulates 6 functional groups over a 10-year time horizon based on 5 core variables.
"""

from typing import Dict, List, TypedDict
import copy


class SimulationState(TypedDict):
    """State of all variables and agents at a given timestep"""
    year: int
    # Environmental variables (0.0 to 1.0 scale, 0.5 = baseline)
    nutrient_load: float
    dissolved_oxygen: float
    water_temperature: float
    habitat_quality: float
    fishing_pressure: float
    # Agent populations (0.0 to 1.0 scale, 0.5 = baseline)
    phytoplankton: float
    zooplankton: float
    forage_fish: float
    predator_fish: float
    seabirds: float
    marine_mammals: float


class SimulationResult(TypedDict):
    """Complete simulation output"""
    timeline: List[SimulationState]
    impacts: Dict[str, str]  # agent -> "increase" | "decrease" | "stable"
    summary: str


# Baseline state (pre-policy)
BASELINE_STATE = {
    "year": 0,
    # Environmental variables at baseline (0.5 = neutral)
    "nutrient_load": 0.5,
    "dissolved_oxygen": 0.5,
    "water_temperature": 0.5,
    "habitat_quality": 0.5,
    "fishing_pressure": 0.5,
    # Agent populations at baseline (0.5 = healthy equilibrium)
    "phytoplankton": 0.5,
    "zooplankton": 0.5,
    "forage_fish": 0.5,
    "predator_fish": 0.5,
    "seabirds": 0.5,
    "marine_mammals": 0.5,
}


def apply_policy_levers(baseline: Dict, levers: Dict[str, float]) -> Dict:
    """
    Apply policy changes to baseline state.
    
    Args:
        baseline: Initial state
        levers: Policy changes (e.g. {"nutrient_load": -0.3, "habitat_quality": 0.1})
    
    Returns:
        New state with policy applied
    """
    state = copy.deepcopy(baseline)
    
    for lever, change in levers.items():
        if lever in state:
            # Apply change and clamp to [0, 1]
            state[lever] = max(0.0, min(1.0, state[lever] + change))
    
    return state


def simulate_step(state: Dict) -> Dict:
    """
    Simulate one year of ecosystem dynamics.
    
    Rules (simplified food web):
    1. Phytoplankton: ↑ nutrients (short-term), ↓ if too much oxygen depletion
    2. Zooplankton: follows phytoplankton (with lag), needs oxygen
    3. Forage fish: eats zooplankton, needs oxygen + habitat, ↓ by fishing
    4. Predator fish: eats forage fish, needs oxygen + habitat, ↓ by fishing
    5. Seabirds: eats forage fish, needs habitat
    6. Marine mammals: eats forage + predator fish, needs oxygen + habitat
    """
    new_state = copy.deepcopy(state)
    new_state["year"] += 1
    
    # Environmental constraints
    oxygen = state["dissolved_oxygen"]
    nutrients = state["nutrient_load"]
    temp = state["water_temperature"]
    habitat = state["habitat_quality"]
    fishing = state["fishing_pressure"]
    
    # RULE 1: Phytoplankton
    # ↑ with nutrients (bloom), but ↓ if oxygen too low (die-off from hypoxia)
    phyto_change = 0.0
    phyto_change += (nutrients - 0.5) * 0.3  # Nutrients drive growth
    phyto_change -= (0.5 - oxygen) * 0.2  # Low oxygen hurts them too
    new_state["phytoplankton"] = clamp(state["phytoplankton"] + phyto_change)
    
    # RULE 2: Zooplankton
    # Follows phytoplankton (food source), needs oxygen
    zoo_change = 0.0
    zoo_change += (state["phytoplankton"] - state["zooplankton"]) * 0.2  # Track food
    zoo_change -= (0.5 - oxygen) * 0.25  # Oxygen-sensitive
    new_state["zooplankton"] = clamp(state["zooplankton"] + zoo_change)
    
    # RULE 3: Forage fish (anchovy, sardine)
    # Eats zooplankton, very sensitive to oxygen, impacted by fishing
    forage_change = 0.0
    forage_change += (state["zooplankton"] - 0.5) * 0.25  # Food availability
    forage_change -= (0.5 - oxygen) * 0.4  # VERY oxygen sensitive
    forage_change += (habitat - 0.5) * 0.15  # Habitat quality helps
    forage_change -= (fishing - 0.5) * 0.3  # Fishing pressure
    forage_change -= (temp - 0.5) * 0.2  # Temperature stress
    new_state["forage_fish"] = clamp(state["forage_fish"] + forage_change)
    
    # RULE 4: Predator fish (tuna, larger fish)
    # Eats forage fish, needs oxygen + habitat, impacted by fishing
    predator_change = 0.0
    predator_change += (state["forage_fish"] - 0.5) * 0.3  # Food source
    predator_change -= (0.5 - oxygen) * 0.3  # Oxygen need
    predator_change += (habitat - 0.5) * 0.2  # Habitat quality
    predator_change -= (fishing - 0.5) * 0.35  # Fishing pressure (often targeted)
    predator_change -= (temp - 0.5) * 0.15  # Temperature stress
    new_state["predator_fish"] = clamp(state["predator_fish"] + predator_change)
    
    # RULE 5: Seabirds
    # Eats forage fish primarily, needs coastal habitat
    seabird_change = 0.0
    seabird_change += (state["forage_fish"] - 0.5) * 0.35  # Heavily dependent on forage fish
    seabird_change += (habitat - 0.5) * 0.25  # Nesting/roosting habitat
    new_state["seabirds"] = clamp(state["seabirds"] + seabird_change)
    
    # RULE 6: Marine mammals
    # Eats both forage and predator fish, needs oxygen + habitat
    mammal_change = 0.0
    mammal_change += (state["forage_fish"] - 0.5) * 0.2  # Diet component
    mammal_change += (state["predator_fish"] - 0.5) * 0.2  # Diet component
    mammal_change -= (0.5 - oxygen) * 0.2  # Oxygen need (for prey too)
    mammal_change += (habitat - 0.5) * 0.2  # Coastal habitat quality
    new_state["marine_mammals"] = clamp(state["marine_mammals"] + mammal_change)
    
    # ENVIRONMENTAL FEEDBACK: High nutrients + low oxygen creates hypoxia
    # (Nutrient bloom → phytoplankton die-off → oxygen depletion)
    if nutrients > 0.6 and new_state["phytoplankton"] > 0.6:
        oxygen_depletion = -0.05
        new_state["dissolved_oxygen"] = clamp(new_state["dissolved_oxygen"] + oxygen_depletion)
    
    return new_state


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def run_simulation(levers: Dict[str, float], years: int = 10) -> SimulationResult:
    """
    Run the full ecosystem simulation.
    
    Args:
        levers: Policy changes from Task 1 parser
        years: Number of years to simulate (default 10)
    
    Returns:
        SimulationResult with timeline and impact summary
    """
    # Apply policy to baseline
    initial_state = apply_policy_levers(BASELINE_STATE, levers)
    
    # Run simulation
    timeline = [initial_state]
    current_state = initial_state
    
    for _ in range(years):
        current_state = simulate_step(current_state)
        timeline.append(current_state)
    
    # Calculate impacts (compare final state to baseline)
    final_state = timeline[-1]
    impacts = {}
    agents = ["phytoplankton", "zooplankton", "forage_fish", "predator_fish", "seabirds", "marine_mammals"]
    
    for agent in agents:
        change = final_state[agent] - BASELINE_STATE[agent]
        if change > 0.05:
            impacts[agent] = "increase"
        elif change < -0.05:
            impacts[agent] = "decrease"
        else:
            impacts[agent] = "stable"
    
    # Generate summary
    winners = [a for a, impact in impacts.items() if impact == "increase"]
    losers = [a for a, impact in impacts.items() if impact == "decrease"]
    
    if winners and not losers:
        summary = f"Ecosystem improved: {', '.join(winners)} populations increased."
    elif losers and not winners:
        summary = f"Ecosystem declined: {', '.join(losers)} populations decreased."
    elif winners and losers:
        summary = f"Mixed impacts: {', '.join(winners)} increased but {', '.join(losers)} decreased."
    else:
        summary = "Ecosystem remained stable with minimal change."
    
    return {
        "timeline": timeline,
        "impacts": impacts,
        "summary": summary
    }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("TASK 2: SIMULATION ENGINE TEST")
    print("=" * 60)
    
    # Test case: Reduce runoff by 30%
    test_levers = {
        "nutrient_load": -0.3,  # 30% reduction
        "habitat_quality": 0.1,  # 10% improvement
    }
    
    print(f"\nPolicy levers: {test_levers}")
    print("-" * 60)
    
    result = run_simulation(test_levers, years=10)
    
    print("\nYear-by-year results:")
    print(f"{'Year':<6} {'Nutrients':<10} {'Oxygen':<10} {'Forage':<10} {'Predator':<10} {'Seabirds':<10}")
    print("-" * 60)
    
    for state in result["timeline"][::2]:  # Every 2 years for brevity
        print(
            f"{state['year']:<6} "
            f"{state['nutrient_load']:<10.2f} "
            f"{state['dissolved_oxygen']:<10.2f} "
            f"{state['forage_fish']:<10.2f} "
            f"{state['predator_fish']:<10.2f} "
            f"{state['seabirds']:<10.2f}"
        )
    
    print("\n" + "=" * 60)
    print("ECOSYSTEM IMPACTS")
    print("=" * 60)
    
    for agent, impact in result["impacts"].items():
        symbol = "↑" if impact == "increase" else "↓" if impact == "decrease" else "→"
        print(f"{symbol} {agent.replace('_', ' ').title():<20} {impact}")
    
    print("\n" + "-" * 60)
    print(f"Summary: {result['summary']}")
    print("=" * 60)