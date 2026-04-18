import ollama
import os
import re
import time

# ── Zooplankton agent state ──────────────────────────────────────────────────
zooplankton = {
    "population": 10,
    "last_action": None,
    "health_trend": "stable"
}

# ── Environment state (CalCOFI California Current baseline) ──────────────────
environment = {
    "temperature": 16.2,      # celsius, optimal 12-16
    "nutrients": 0.6,         # 0-1 normalized
    "pH": 8.05,               # optimal 8.1-8.3
    "salinity": 33.4,         # PSU, optimal 32-34
    "fishing_pressure": 0.2,
    "pollution_index": 0.3
}

# ── Phytoplankton state (read-only input from Layer below) ───────────────────
# In the full simulation this comes live from phytoplankton.py each tick
# For now we simulate it as a static snapshot you can manually adjust
phytoplankton_state = {
    "population": 65,
    "last_action": "bloom",
    "health_trend": "improving"
}

# ── Behavior deltas ──────────────────────────────────────────────────────────
BEHAVIORS = {
    "graze":      +12,   # food abundant, eat and grow
    "swarm":      +3,    # cluster for safety, slight growth
    "disperse":   -3,    # spread out, costs energy
    "starve":     -18,   # phytoplankton too low
    "reproduce":  +15,   # optimal conditions, rapid growth
}


def build_prompt(zooplankton, environment, phytoplankton_state):
    # Derive explicit food status so the LLM doesn't have to infer it
    food = phytoplankton_state['population']
    if food < 20:
        food_status = "CRITICALLY LOW - you are starving, you must pick starve"
    elif food < 40:
        food_status = "LOW - food is scarce, avoid graze or reproduce"
    elif food < 70:
        food_status = "MODERATE - some food available"
    else:
        food_status = "ABUNDANT - plenty of food, graze or reproduce are good choices"

    temp = environment['temperature']
    if temp > 16:
        temp_status = "TOO HIGH - stressful conditions even if food is available"
    elif temp > 14:
        temp_status = "SLIGHTLY ELEVATED - mild stress"
    else:
        temp_status = "OPTIMAL"

    return f"""You are a zooplankton colony in the California Current. You must pick a survival behavior RIGHT NOW.

FOOD AVAILABILITY (phytoplankton population: {food}/100): {food_status}
TEMPERATURE ({temp}°C): {temp_status}
YOUR POPULATION: {zooplankton['population']}/100
YOUR LAST ACTION: {zooplankton['last_action'] or 'none'}

DECISION RULES — follow these strictly:
- If food is CRITICALLY LOW → you MUST pick starve
- If food is LOW → pick disperse or swarm
- If food is MODERATE and temp is OPTIMAL → pick graze
- If food is ABUNDANT and temp is OPTIMAL → pick reproduce or graze
- If temp is TOO HIGH regardless of food → pick disperse or swarm

Pick exactly one behavior from: graze, swarm, disperse, starve, reproduce

Respond in this exact format only:
BEHAVIOR: [one word from the list above]
REASON: [one sentence in first person]"""


def tick(zooplankton, environment, phytoplankton_state):
    prompt = build_prompt(zooplankton, environment, phytoplankton_state)

    try:
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"]
    except Exception as e:
        print(f"Ollama error: {e}")
        raw = "BEHAVIOR: disperse\nREASON: Defaulting due to error."

    # Parse behavior
    behavior = "disperse"  # safe default
    for b in BEHAVIORS:
        if b in raw.lower():
            behavior = b
            break

    # Parse reason
    reason_match = re.search(r'REASON:\s*(.+)', raw)
    reason = reason_match.group(1).strip() if reason_match else "No reason provided."

    # Apply delta and clamp
    delta = BEHAVIORS[behavior]
    zooplankton["population"] = max(0, min(100, zooplankton["population"] + delta))
    zooplankton["last_action"] = behavior

    # Update health trend
    if delta > 0:
        zooplankton["health_trend"] = "improving"
    elif delta < 0:
        zooplankton["health_trend"] = "declining"
    else:
        zooplankton["health_trend"] = "stable"

    return zooplankton, behavior, reason


# ── Main loop ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Zooplankton Simulation ===")
    print(f"Phytoplankton context: pop={phytoplankton_state['population']}, action={phytoplankton_state['last_action']}\n")

    for i in range(1, 6):
        zooplankton, behavior, reason = tick(zooplankton, environment, phytoplankton_state)
        print(f"Tick {i} | Action: {behavior} | Population: {zooplankton['population']}/100 | Trend: {zooplankton['health_trend']}")
        print(f"Reason: {reason}")
        print("---")
        time.sleep(0.5)