import argparse
from groq import Groq
import copy
import os
import re
import sys
import time
import anchovy as anchovy_module
import sardine as sardine_module
import kelp as kelp_module
import urchin as urchin_module
import sealion as sealion_module
import database_fetch

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Environment state (CalCOFI California Current baseline) ──────────────────
environment = database_fetch.BASELINE_ENVIRONMENT.copy()

# ── Agent states ─────────────────────────────────────────────────────────────
phytoplankton = {
    "population": 65,
    "last_action": None,
    "health_trend": "stable"
}

zooplankton = {
    "population": 55,
    "last_action": None,
    "health_trend": "stable"
}

anchovy = {
    "population": 50,
    "last_action": None,
    "health_trend": "stable"
}

sardine = {
    "population": 45,
    "last_action": None,
    "health_trend": "stable"
}

sea_lion = {
    "population": 50,
    "last_action": None,
    "health_trend": "stable"
}

kelp = {
    "population": 60,
    "last_action": None,
    "health_trend": "stable"
}

urchin = {
    "population": 40,
    "last_action": None,
    "health_trend": "stable"
}


def get_initial_agents():
    return {
        "phytoplankton": {
            "population": 65,
            "last_action": None,
            "health_trend": "stable"
        },
        "zooplankton": {
            "population": 55,
            "last_action": None,
            "health_trend": "stable"
        },
        "anchovy": {
            "population": 50,
            "last_action": None,
            "health_trend": "stable"
        },
        "sardine": {
            "population": 45,
            "last_action": None,
            "health_trend": "stable"
        },
        "sea_lion": {
            "population": 50,
            "last_action": None,
            "health_trend": "stable"
        },
        "kelp": {
            "population": 60,
            "last_action": None,
            "health_trend": "stable"
        },
        "urchin": {
            "population": 40,
            "last_action": None,
            "health_trend": "stable"
        }
    }

# ── Behavior deltas ──────────────────────────────────────────────────────────
PHYTOPLANKTON_BEHAVIORS = {
    "bloom":          +15,
    "die_off":        -20,
    "persist":          0,
    "migrate_depth":   -5,
}

ZOOPLANKTON_BEHAVIORS = {
    "graze":      +12,
    "swarm":       +3,
    "disperse":    -3,
    "starve":     -18,
    "reproduce":  +15,
}

# ── Prompt builders ──────────────────────────────────────────────────────────
def build_phytoplankton_prompt(agent, env):
    temp = env['temperature']
    nutrient = env['nutrients']

    if temp > 18:
        temp_status = "TOO HIGH - critically stressful, die_off likely"
    elif temp > 16:
        temp_status = "ELEVATED - mild stress, avoid bloom"
    else:
        temp_status = "OPTIMAL - good for growth"

    if nutrient > 0.7:
        nutrient_status = "ABUNDANT - bloom or reproduce conditions"
    elif nutrient > 0.4:
        nutrient_status = "MODERATE - persist or graze conditions"
    else:
        nutrient_status = "SCARCE - risk of die_off"

    return f"""You are a phytoplankton colony in the California Current. Pick a survival behavior RIGHT NOW.

TEMPERATURE ({temp}°C): {temp_status}
NUTRIENTS ({nutrient}/1.0): {nutrient_status}
pH: {env['pH']} (optimal 8.1-8.3)
POLLUTION: {env['pollution_index']}/1.0
YOUR POPULATION: {agent['population']}/100
YOUR LAST ACTION: {agent['last_action'] or 'none'}

DECISION RULES — follow strictly:
- If temp TOO HIGH → die_off
- If nutrients SCARCE and temp ELEVATED → die_off or migrate_depth
- If nutrients ABUNDANT and temp OPTIMAL → bloom
- If nutrients MODERATE → persist or migrate_depth
- If population already high (>80) → persist instead of bloom

Pick exactly one from: bloom, die_off, persist, migrate_depth

BEHAVIOR: [one word]
REASON: [one sentence first person]"""


def build_zooplankton_prompt(agent, env, phyto):
    food = phyto['population']
    if food < 20:
        food_status = "CRITICALLY LOW - you must pick starve"
    elif food < 40:
        food_status = "LOW - food is scarce, avoid graze or reproduce"
    elif food < 70:
        food_status = "MODERATE - some food available"
    else:
        food_status = "ABUNDANT - graze or reproduce are good choices"

    temp = env['temperature']
    if temp > 16:
        temp_status = "TOO HIGH - stressful even with food"
    else:
        temp_status = "OPTIMAL"

    return f"""You are a zooplankton colony in the California Current. Pick a survival behavior RIGHT NOW.

FOOD (phytoplankton {food}/100): {food_status}
PHYTOPLANKTON TREND: {phyto['health_trend']} (last action: {phyto['last_action']})
TEMPERATURE ({temp}°C): {temp_status}
YOUR POPULATION: {agent['population']}/100
YOUR LAST ACTION: {agent['last_action'] or 'none'}

DECISION RULES — follow strictly:
- If food CRITICALLY LOW → starve
- If food LOW → disperse or swarm
- If food MODERATE and temp OPTIMAL → graze
- If food ABUNDANT and temp OPTIMAL → reproduce or graze
- If temp TOO HIGH regardless of food → swarm or disperse

Pick exactly one from: graze, swarm, disperse, starve, reproduce

BEHAVIOR: [one word]
REASON: [one sentence first person]"""


# ── Tick functions ────────────────────────────────────────────────────────────
def parse_response(raw, behaviors, default):
    if not isinstance(raw, str):
        print(f"  [validation warning] non-string response, using default '{default}'")
        return default, "No reason provided."

    behavior = default
    reason = "No reason provided."
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    # Parse explicit behavior label first
    for line in lines:
        if line.upper().startswith("BEHAVIOR:"):
            candidate = line.split(":", 1)[1].strip().lower().replace(" ", "_")
            if candidate in behaviors:
                behavior = candidate
                break

    # Backward-compatible fallback by substring search
    if behavior == default:
        for b in behaviors:
            if re.search(rf"\b{re.escape(b)}\b", raw.lower()):
                behavior = b
                break

    # Extract reason safely
    for line in lines:
        if line.upper().startswith("REASON:"):
            candidate_reason = line.split(":", 1)[1].strip()
            if len(candidate_reason) >= 10:
                reason = candidate_reason
            break

    if behavior not in behaviors:
        print(f"  [validation warning] invalid behavior response, using default '{default}'. Raw: {raw!r}")
        behavior = default

    if raw.lower().count("behavior:") != 1:
        print(f"  [validation warning] unexpected behavior format; using '{behavior}'. Raw: {raw!r}")

    return behavior, reason


def update_agent(agent, behavior, deltas):
    delta = deltas[behavior]
    agent["population"] = max(0, min(100, agent["population"] + delta))
    agent["last_action"] = behavior
    agent["health_trend"] = "improving" if delta > 0 else "declining" if delta < 0 else "stable"
    return agent


def clamp_environment(env: dict):
    env["temperature"] = max(10.0, min(22.0, float(env.get("temperature", 16.2))))
    env["nutrients"] = max(0.0, min(1.0, float(env.get("nutrients", 0.6))))
    env["pH"] = max(7.8, min(8.4, float(env.get("pH", 8.05))))
    env["salinity"] = max(30.0, min(36.0, float(env.get("salinity", 33.4))))
    env["fishing_pressure"] = max(0.0, min(1.0, float(env.get("fishing_pressure", 0.2))))
    env["pollution_index"] = max(0.0, min(1.0, float(env.get("pollution_index", 0.3))))
    return env


def apply_cross_species_feedback(env: dict, agents: dict):
    if agents["urchin"]["last_action"] == "barren_expand":
        env["pollution_index"] = min(1.0, env["pollution_index"] + 0.06)
        env["nutrients"] = min(1.0, env["nutrients"] + 0.04)
    elif agents["urchin"]["last_action"] == "starve":
        env["pollution_index"] = max(0.0, env["pollution_index"] - 0.05)

    if agents["kelp"]["population"] < 30:
        env["temperature"] = min(22.0, env["temperature"] + 0.15)
        env["pollution_index"] = min(1.0, env["pollution_index"] + 0.04)
        env["nutrients"] = max(0.0, env["nutrients"] - 0.03)
    elif agents["kelp"]["population"] > 70:
        env["pollution_index"] = max(0.0, env["pollution_index"] - 0.04)

    if agents["anchovy"]["population"] > 70 and agents["sardine"]["population"] > 70:
        env["fishing_pressure"] = min(1.0, env["fishing_pressure"] + 0.05)

    if agents["phytoplankton"]["population"] > 80 and agents["zooplankton"]["population"] < 30:
        env["nutrients"] = max(0.0, env["nutrients"] - 0.05)

    if agents["phytoplankton"]["population"] < 20 and agents["zooplankton"]["population"] > 60:
        env["pollution_index"] = min(1.0, env["pollution_index"] + 0.03)

    clamp_environment(env)
    return env

def apply_policy_to_environment(policy_text: str, env: dict):
    """Apply a natural language policy to the environment state."""
    if not policy_text:
        return env, None

    print(f"Applying policy: {policy_text}")
    try:
        result = database_fetch.parse_policy(policy_text, baseline=env)
        env.update(result["environment"])
        print(f"Policy parse confidence: {result['confidence']:.2%}")
        print(f"Policy summary: {result['summary']}")
        return env, result
    except Exception as exc:
        print(f"  [policy parse error] {exc}")
        print("  Falling back to manual policy application.")
        fallback_env = database_fetch.apply_policy_manually(policy_text, baseline=env)
        env.update(fallback_env)
        return env, None

def run_simulation(env: dict, agents: dict, ticks: int = 5, verbose: bool = False):
    env = clamp_environment(env.copy())
    agents = copy.deepcopy(agents)

    if verbose:
        print("\n=== Running simulation ===")
        print(f"Starting environment: temp={env['temperature']}°C | nutrients={env['nutrients']} | fishing={env['fishing_pressure']}")

    for tick_num in range(1, ticks + 1):
        if verbose:
            print(f"\n── YEAR {tick_num} {'─' * 50}")

        agents["phytoplankton"], p_behavior, p_reason = tick_phytoplankton(agents["phytoplankton"], env)
        agents["zooplankton"], z_behavior, z_reason = tick_zooplankton(agents["zooplankton"], env, agents["phytoplankton"])
        agents["anchovy"], a_behavior, a_reason = anchovy_module.tick(agents["anchovy"], env, agents["zooplankton"])
        agents["sardine"], s_behavior, s_reason = sardine_module.tick(agents["sardine"], env, agents["zooplankton"], agents["anchovy"])
        agents["sea_lion"], sl_behavior, sl_reason = sealion_module.tick(agents["sea_lion"], env, agents["anchovy"], agents["sardine"])
        agents["urchin"], u_behavior, u_reason = urchin_module.tick(agents["urchin"], env, agents["kelp"])
        agents["kelp"], k_behavior, k_reason = kelp_module.tick(agents["kelp"], env, agents["urchin"])

        env = apply_cross_species_feedback(env, agents)

        if verbose:
            print(f"🌿 Phytoplankton | {p_behavior:<20} | pop: {agents['phytoplankton']['population']:>3}/100 | {agents['phytoplankton']['health_trend']}")
            print(f"   → {p_reason}")
            print(f"🦐 Zooplankton   | {z_behavior:<20} | pop: {agents['zooplankton']['population']:>3}/100 | {agents['zooplankton']['health_trend']}")
            print(f"   → {z_reason}")
            print(f"🐟 Anchovy       | {a_behavior:<20} | pop: {agents['anchovy']['population']:>3}/100 | {agents['anchovy']['health_trend']}")
            print(f"   → {a_reason}")
            print(f"🐟 Sardine       | {s_behavior:<20} | pop: {agents['sardine']['population']:>3}/100 | {agents['sardine']['health_trend']}")
            print(f"   → {s_reason}")
            print(f"🦁 Sea Lion      | {sl_behavior:<20} | pop: {agents['sea_lion']['population']:>3}/100 | {agents['sea_lion']['health_trend']}")
            print(f"   → {sl_reason}")
            print(f"🌱 Kelp          | {k_behavior:<20} | pop: {agents['kelp']['population']:>3}/100 | {agents['kelp']['health_trend']}")
            print(f"   → {k_reason}")
            print(f"🦀 Urchin        | {u_behavior:<20} | pop: {agents['urchin']['population']:>3}/100 | {agents['urchin']['health_trend']}")
            print(f"   → {u_reason}")

        if verbose:
            print(f"  post-feedback: temp={env['temperature']:.2f}°C | nutrients={env['nutrients']:.3f} | pollution={env['pollution_index']:.3f}")
            time.sleep(0.3)

    return env, agents


def print_summary(env: dict, agents: dict):
    clamp_environment(env)
    print("\nFINAL STATE")
    print(f"Environment: temp={env['temperature']:.2f}°C | nutrients={env['nutrients']:.3f} | fishing={env['fishing_pressure']:.3f} | pollution={env['pollution_index']:.3f}")
    for name in ["phytoplankton", "zooplankton", "anchovy", "sardine", "sea_lion", "kelp", "urchin"]:
        agent = agents[name]
        print(f"{name.replace('_', ' ').title()}: {agent['population']}/100 ({agent['health_trend']})")


def print_comparison(base_env: dict, base_agents: dict, policy_env: dict, policy_agents: dict):
    print("\nENVIRONMENT COMPARISON")
    metrics = ["temperature", "nutrients", "pH", "salinity", "fishing_pressure", "pollution_index"]
    for key in metrics:
        before = base_env[key]
        after = policy_env[key]
        delta = after - before
        print(f"  {key:15} {before:>6.3f} → {after:>6.3f}  ({delta:+.3f})")

    print("\nSPECIES POPULATION COMPARISON")
    for name in ["phytoplankton", "zooplankton", "anchovy", "sardine", "sea_lion", "kelp", "urchin"]:
        before = base_agents[name]["population"]
        after = policy_agents[name]["population"]
        delta = after - before
        print(f"  {name.replace('_', ' ').title():15} {before:>3} → {after:>3}  ({delta:+})")


def tick_phytoplankton(agent, env):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": build_phytoplankton_prompt(agent, env)}],
            max_tokens=100
        )
        raw = response.choices[0].message.content
    except Exception as e:
        print(f"  [Groq error - phytoplankton]: {e}")
        raw = "BEHAVIOR: persist\nREASON: Defaulting due to error."

    behavior, reason = parse_response(raw, PHYTOPLANKTON_BEHAVIORS, "persist")
    agent = update_agent(agent, behavior, PHYTOPLANKTON_BEHAVIORS)
    return agent, behavior, reason


def tick_zooplankton(agent, env, phyto):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": build_zooplankton_prompt(agent, env, phyto)}],
            max_tokens=100
        )
        raw = response.choices[0].message.content
    except Exception as e:
        print(f"  [Groq error - zooplankton]: {e}")
        raw = "BEHAVIOR: disperse\nREASON: Defaulting due to error."

    behavior, reason = parse_response(raw, ZOOPLANKTON_BEHAVIORS, "disperse")
    agent = update_agent(agent, behavior, ZOOPLANKTON_BEHAVIORS)
    return agent, behavior, reason


# ── Main simulation loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the California Current ecosystem simulation.")
    parser.add_argument("--policy", type=str, default="",
                        help="Natural language policy to apply before running the simulation.")
    parser.add_argument("--contrast", action="store_true",
                        help="Run baseline and policy simulations side-by-side for comparison.")
    parser.add_argument("--ticks", type=int, default=5,
                        help="Number of simulation ticks to run.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-tick results for the simulation.")
    args = parser.parse_args()

    if args.contrast and not args.policy:
        print("Contrast mode requires --policy. Running baseline only.")

    baseline_env = database_fetch.BASELINE_ENVIRONMENT.copy()
    baseline_agents = get_initial_agents()

    if args.contrast:
        print("Running baseline simulation...")
        base_env_final, base_agents_final = run_simulation(baseline_env, baseline_agents, ticks=args.ticks, verbose=args.verbose)

        if args.policy:
            print("Applying policy and running policy simulation...")
            policy_env = database_fetch.BASELINE_ENVIRONMENT.copy()
            policy_agents = get_initial_agents()
            apply_policy_to_environment(args.policy, policy_env)
            policy_env_final, policy_agents_final = run_simulation(policy_env, policy_agents, ticks=args.ticks, verbose=args.verbose)

            print("=== Policy Contrast Summary ===")
            print_comparison(base_env_final, base_agents_final, policy_env_final, policy_agents_final)
        else:
            print("=== Baseline Summary ===")
            print_summary(base_env_final, base_agents_final)

        sys.exit(0)

    env = database_fetch.BASELINE_ENVIRONMENT.copy()
    agents = get_initial_agents()

    if args.policy:
        apply_policy_to_environment(args.policy, env)

    run_simulation(env, agents, ticks=args.ticks, verbose=True)
