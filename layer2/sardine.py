import ollama
import re
import time

# ── Sardine agent state ──────────────────────────────────────────────────────
sardine = {
    "population": 45,
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
zooplankton_state = {
    "population": 55,
    "last_action": "graze",
    "health_trend": "improving"
}

# ── Anchovy state (sardines compete directly with anchovy) ───────────────────
# Key difference from anchovy: sardines THRIVE when anchovies are weak
# When anchovy migrates north or declines, sardines dominate the food supply
anchovy_state = {
    "population": 50,
    "last_action": "school",
    "health_trend": "stable"
}

# ── Behavior deltas ──────────────────────────────────────────────────────────
# Sardines tolerate warmer water than anchovy (optimal 14-20°C vs anchovy 12-16°C)
# This is what drives the natural anchovy/sardine alternation in California Current
BEHAVIORS = {
    "feed_aggressively": +15,
    "school":             +5,
    "scatter":            -3,
    "spawn":             +12,
    "dominate":          +18,   # unique to sardine — surge when anchovy is weak
    "migrate_south":      -8,   # sardines go south unlike anchovy which goes north
    "decline":           -15,
}


def build_prompt(agent, env, zoo, anchovy):
    food = zoo['population']
    if food < 15:
        food_status = "CRITICALLY LOW - you must decline or migrate_south"
    elif food < 35:
        food_status = "LOW - scarce food, school for safety"
    elif food < 65:
        food_status = "MODERATE - feed cautiously"
    else:
        food_status = "ABUNDANT - feed aggressively or spawn"

    temp = env['temperature']
    # Sardines tolerate warmer water — key biological difference from anchovy
    if temp > 22:
        temp_status = "TOO HIGH - even sardines struggle, migrate_south"
    elif temp > 20:
        temp_status = "WARM - sardines can handle this, anchovy cannot"
    elif temp > 14:
        temp_status = "OPTIMAL for sardines - good conditions"
    else:
        temp_status = "COLD - sardines prefer warmer water, mild stress"

    fishing = env['fishing_pressure']
    if fishing > 0.7:
        fishing_status = "HEAVY - school tightly or scatter to survive"
    elif fishing > 0.4:
        fishing_status = "MODERATE - be cautious"
    else:
        fishing_status = "LOW - relatively safe"

    # Competition signal — this drives the alternation dynamic
    anchovy_pop = anchovy['population']
    if anchovy_pop < 25:
        competition_status = "VERY WEAK - anchovy is collapsing, this is your chance to dominate"
    elif anchovy_pop < 45:
        competition_status = "WEAK - anchovy struggling, you can expand aggressively"
    elif anchovy_pop < 65:
        competition_status = "MODERATE - sharing food supply with anchovy"
    else:
        competition_status = "STRONG - anchovy dominant, compete carefully or school"

    return f"""You are a sardine school in the California Current. Pick a survival behavior RIGHT NOW.

FOOD (zooplankton {food}/100): {food_status}
ZOOPLANKTON TREND: {zoo['health_trend']} (last action: {zoo['last_action']})
TEMPERATURE ({temp}°C): {temp_status}
FISHING PRESSURE: {fishing_status}
ANCHOVY COMPETITION ({anchovy_pop}/100): {competition_status}
YOUR POPULATION: {agent['population']}/100
YOUR LAST ACTION: {agent['last_action'] or 'none'}

KEY BIOLOGY: Unlike anchovy, you THRIVE in warmer water. When anchovy declines or migrates 
north due to heat, YOU fill that gap. This natural alternation has been documented in the 
California Current for decades.

DECISION RULES — follow strictly:
- If food CRITICALLY LOW → decline or migrate_south
- If food LOW and fishing HEAVY → scatter or decline
- If anchovy VERY WEAK and food MODERATE+ → dominate (this is your opportunity)
- If anchovy WEAK and temp OPTIMAL/WARM → feed_aggressively or spawn
- If food ABUNDANT and temp OPTIMAL and fishing LOW → spawn
- If temp TOO HIGH → migrate_south
- If anchovy STRONG and food LOW → school conservatively

Pick exactly one from: feed_aggressively, school, scatter, spawn, dominate, migrate_south, decline

BEHAVIOR: [one word]
REASON: [one sentence first person]"""


def tick(agent, env, zoo, anchovy):
    prompt = build_prompt(agent, env, zoo, anchovy)

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
    print("=== Sardine Simulation ===")
    print(f"Zooplankton context: pop={zooplankton_state['population']}, action={zooplankton_state['last_action']}")
    print(f"Anchovy competition: pop={anchovy_state['population']}, trend={anchovy_state['health_trend']}\n")

    for i in range(1, 6):
        sardine, behavior, reason = tick(sardine, environment, zooplankton_state, anchovy_state)
        print(f"Tick {i} | Action: {behavior} | Population: {sardine['population']}/100 | Trend: {sardine['health_trend']}")
        print(f"Reason: {reason}")
        print("---")
        time.sleep(0.5)