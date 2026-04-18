import ollama
import re
import time

# ── Anchovy agent state ──────────────────────────────────────────────────────
anchovy = {
    "population": 50,
    "last_action": None,
    "health_trend": "stable"
}

# ── Environment state (CalCOFI California Current baseline) ──────────────────
environment = {
    "temperature": 16.2,
    "nutrients": 0.6,
    "pH": 8.05,
    "salinity": 33.4,
    "fishing_pressure": 0.2,
    "pollution_index": 0.3
}

# ── Zooplankton state (read-only input from layer below) ─────────────────────
# In full simulation this comes live from zooplankton tick each round
# Manually adjust here to test cascade behavior
zooplankton_state = {
    "population": 55,
    "last_action": "graze",
    "health_trend": "improving"
}

# ── Behavior deltas ──────────────────────────────────────────────────────────
BEHAVIORS = {
    "feed_aggressively": +15,
    "school":             +5,
    "scatter":            -3,
    "spawn":             +12,
    "migrate_north":      -8,
    "decline":           -15,
}


def build_prompt(agent, env, zoo):
    food = zoo['population']
    if food < 15:
        food_status = "CRITICALLY LOW - you must decline or migrate_north"
    elif food < 35:
        food_status = "LOW - scarce food, school for safety"
    elif food < 65:
        food_status = "MODERATE - feed cautiously"
    else:
        food_status = "ABUNDANT - feed aggressively or spawn"

    temp = env['temperature']
    if temp > 18:
        temp_status = "TOO HIGH - migrate north immediately"
    elif temp > 16:
        temp_status = "ELEVATED - mild stress, avoid spawning"
    else:
        temp_status = "OPTIMAL"

    fishing = env['fishing_pressure']
    if fishing > 0.7:
        fishing_status = "HEAVY - school tightly or scatter to survive"
    elif fishing > 0.4:
        fishing_status = "MODERATE - be cautious"
    else:
        fishing_status = "LOW - relatively safe"

    return f"""You are an anchovy school in the California Current. Pick a survival behavior RIGHT NOW.

FOOD (zooplankton {food}/100): {food_status}
ZOOPLANKTON TREND: {zoo['health_trend']} (last action: {zoo['last_action']})
TEMPERATURE ({temp}°C): {temp_status}
FISHING PRESSURE: {fishing_status}
YOUR POPULATION: {agent['population']}/100
YOUR LAST ACTION: {agent['last_action'] or 'none'}

DECISION RULES — follow strictly:
- If food CRITICALLY LOW → decline or migrate_north
- If food LOW and fishing HEAVY → scatter or decline
- If food LOW and fishing LOW → school
- If food MODERATE and temp OPTIMAL → feed_aggressively or school
- If food ABUNDANT and temp OPTIMAL and fishing LOW → spawn
- If temp TOO HIGH → migrate_north regardless of food

Pick exactly one from: feed_aggressively, school, scatter, spawn, migrate_north, decline

BEHAVIOR: [one word]
REASON: [one sentence first person]"""


def tick(agent, env, zoo):
    prompt = build_prompt(agent, env, zoo)

    try:
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"]
    except Exception as e:
        print(f"Ollama error: {e}")
        raw = "BEHAVIOR: school\nREASON: Defaulting due to error."

    # Parse behavior
    behavior = "school"
    for b in BEHAVIORS:
        if b in raw.lower():
            behavior = b
            break

    # Parse reason
    reason_match = re.search(r'REASON:\s*(.+)', raw)
    reason = reason_match.group(1).strip() if reason_match else "No reason provided."

    # Apply delta and clamp
    delta = BEHAVIORS[behavior]
    agent["population"] = max(0, min(100, agent["population"] + delta))
    agent["last_action"] = behavior
    agent["health_trend"] = "improving" if delta > 0 else "declining" if delta < 0 else "stable"

    return agent, behavior, reason


# ── Main loop ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Anchovy Simulation ===")
    print(f"Zooplankton context: pop={zooplankton_state['population']}, action={zooplankton_state['last_action']}\n")

    for i in range(1, 6):
        anchovy, behavior, reason = tick(anchovy, environment, zooplankton_state)
        print(f"Tick {i} | Action: {behavior} | Population: {anchovy['population']}/100 | Trend: {anchovy['health_trend']}")
        print(f"Reason: {reason}")
        print("---")
        time.sleep(0.5)