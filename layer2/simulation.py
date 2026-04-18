import ollama
import re
import time
import anchovy as anchovy_module
import sardine as sardine_module
import kelp as kelp_module
import urchin as urchin_module
import sealion as sealion_module

# ── Environment state (CalCOFI California Current baseline) ──────────────────
environment = {
    "temperature": 16.2,      # celsius, optimal 12-16
    "nutrients": 0.6,         # 0-1 normalized
    "pH": 8.05,               # optimal 8.1-8.3
    "salinity": 33.4,         # PSU, optimal 32-34
    "fishing_pressure": 0.2,  # 0-1 normalized
    "pollution_index": 0.3    # 0-1 normalized
}

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
    behavior = default
    for b in behaviors:
        if b in raw.lower():
            behavior = b
            break
    reason_match = re.search(r'REASON:\s*(.+)', raw)
    reason = reason_match.group(1).strip() if reason_match else "No reason provided."
    return behavior, reason


def update_agent(agent, behavior, deltas):
    delta = deltas[behavior]
    agent["population"] = max(0, min(100, agent["population"] + delta))
    agent["last_action"] = behavior
    agent["health_trend"] = "improving" if delta > 0 else "declining" if delta < 0 else "stable"
    return agent


def tick_phytoplankton(agent, env):
    try:
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": build_phytoplankton_prompt(agent, env)}]
        )
        raw = response["message"]["content"]
    except Exception as e:
        print(f"  [Ollama error - phytoplankton]: {e}")
        raw = "BEHAVIOR: persist\nREASON: Defaulting due to error."

    behavior, reason = parse_response(raw, PHYTOPLANKTON_BEHAVIORS, "persist")
    agent = update_agent(agent, behavior, PHYTOPLANKTON_BEHAVIORS)
    return agent, behavior, reason


def tick_zooplankton(agent, env, phyto):
    try:
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": build_zooplankton_prompt(agent, env, phyto)}]
        )
        raw = response["message"]["content"]
    except Exception as e:
        print(f"  [Ollama error - zooplankton]: {e}")
        raw = "BEHAVIOR: disperse\nREASON: Defaulting due to error."

    behavior, reason = parse_response(raw, ZOOPLANKTON_BEHAVIORS, "disperse")
    agent = update_agent(agent, behavior, ZOOPLANKTON_BEHAVIORS)
    return agent, behavior, reason


# ── Main simulation loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  FATHOM — California Current Ecosystem Simulation")
    print("=" * 60)
    print(f"Starting conditions: temp={environment['temperature']}°C | "
          f"nutrients={environment['nutrients']} | "
          f"fishing={environment['fishing_pressure']}\n")

    for tick_num in range(1, 6):
        print(f"── YEAR {tick_num} {'─' * 50}")

        # Each species runs in order — bottom of food chain first
        # State from lower species feeds into the next one
        phytoplankton, p_behavior, p_reason = tick_phytoplankton(phytoplankton, environment)
        zooplankton, z_behavior, z_reason   = tick_zooplankton(zooplankton, environment, phytoplankton)
        anchovy, a_behavior, a_reason       = anchovy_module.tick(anchovy, environment, zooplankton)
        sardine, s_behavior, s_reason       = sardine_module.tick(sardine, environment, zooplankton, anchovy)
        sea_lion, sl_behavior, sl_reason    = sealion_module.tick(sea_lion, environment, anchovy, sardine)
        urchin, u_behavior, u_reason       = urchin_module.tick(urchin, environment, kelp)
        kelp, k_behavior, k_reason         = kelp_module.tick(kelp, environment, urchin)

        # Print results
        print(f"🌿 Phytoplankton | {p_behavior:<20} | pop: {phytoplankton['population']:>3}/100 | {phytoplankton['health_trend']}")
        print(f"   → {p_reason}")
        print(f"🦐 Zooplankton   | {z_behavior:<20} | pop: {zooplankton['population']:>3}/100 | {zooplankton['health_trend']}")
        print(f"   → {z_reason}")
        print(f"🐟 Anchovy       | {a_behavior:<20} | pop: {anchovy['population']:>3}/100 | {anchovy['health_trend']}")
        print(f"   → {a_reason}")
        print(f"🐟 Sardine       | {s_behavior:<20} | pop: {sardine['population']:>3}/100 | {sardine['health_trend']}")
        print(f"   → {s_reason}")
        print(f"🦁 Sea Lion      | {sl_behavior:<20} | pop: {sea_lion['population']:>3}/100 | {sea_lion['health_trend']}")
        print(f"   → {sl_reason}")
        print(f"🌱 Kelp          | {k_behavior:<20} | pop: {kelp['population']:>3}/100 | {kelp['health_trend']}")
        print(f"   → {k_reason}")
        print(f"🦀 Urchin        | {u_behavior:<20} | pop: {urchin['population']:>3}/100 | {urchin['health_trend']}")
        print(f"   → {u_reason}")
        print()

        time.sleep(0.3)

    print("=" * 60)
    print("FINAL STATE")
    print(f"🌿 Phytoplankton: {phytoplankton['population']}/100 ({phytoplankton['health_trend']})")
    print(f"🦐 Zooplankton:   {zooplankton['population']}/100 ({zooplankton['health_trend']})")
    print(f"🐟 Anchovy:       {anchovy['population']}/100 ({anchovy['health_trend']})")
    print(f"🐟 Sardine:       {sardine['population']}/100 ({sardine['health_trend']})")
    print(f"🦁 Sea Lion:      {sea_lion['population']}/100 ({sea_lion['health_trend']})")
    print(f"🌱 Kelp:          {kelp['population']}/100 ({kelp['health_trend']})")
    print(f"🦀 Urchin:        {urchin['population']}/100 ({urchin['health_trend']})")
    print("=" * 60)