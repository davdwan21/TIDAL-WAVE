import requests
import os
import re
import time

# Ollama setup - assuming llama3.1 is running locally

# Agent state
agent = {
    "population": 65,
    "last_action": None,
    "health_trend": "stable"
}

# Environment state (CalCOFI California Current baseline)
environment = {
    "temperature": 16.2,      # celsius, optimal 12-16
    "nutrients": 0.6,         # 0-1 normalized
    "pH": 8.05,               # optimal 8.1-8.3
    "salinity": 33.4,         # PSU, optimal 32-34
    "fishing_pressure": 0.2,  # 0-1 normalized
    "pollution_index": 0.3    # 0-1 normalized
}

# Behavior deltas — LLM picks behavior, we apply delta
BEHAVIORS = {
    "bloom": 15,
    "die_off": -20,
    "persist": 0,
    "migrate_depth": -5
}


def build_prompt(agent, environment):
    return f"""You are a phytoplankton colony in the California Current ecosystem.

Current environment:
- Temperature: {environment['temperature']}°C (optimal range: 12-16°C)
- Nutrients: {environment['nutrients']}/1.0
- pH: {environment['pH']} (optimal: 8.1-8.3)
- Salinity: {environment['salinity']} PSU (optimal: 32-34 PSU)
- Fishing pressure: {environment['fishing_pressure']}/1.0
- Pollution index: {environment['pollution_index']}/1.0

Your current state:
- Population index: {agent['population']}/100
- Last action: {agent['last_action'] or 'none'}
- Health trend: {agent['health_trend']}

Choose ONE behavior from this exact list:
- bloom
- die_off
- persist
- migrate_depth

Respond in this exact format:
BEHAVIOR: [behavior name]
REASON: [one sentence first person explanation]"""


def tick(agent, environment):
    prompt = build_prompt(agent, environment)

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.1",
            "prompt": prompt,
            "options": {"num_predict": 100},
            "stream": False
        })
        response.raise_for_status()
        raw = response.json()["response"]
    except Exception as e:
        print(f"API error: {e}")
        raw = "BEHAVIOR: persist\nREASON: Defaulting due to API error."

    # Parse behavior
    behavior = "persist"  # safe default
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

    # Update health trend
    if delta > 0:
        agent["health_trend"] = "improving"
    elif delta < 0:
        agent["health_trend"] = "declining"
    else:
        agent["health_trend"] = "stable"

    return agent, behavior, reason


# Main loop — 5 ticks
if __name__ == "__main__":
    print("=== Phytoplankton Simulation ===\n")
    for i in range(1, 6):
        agent, behavior, reason = tick(agent, environment)
        print(f"Year {i} | Action: {behavior} | Population: {agent['population']}/100 | Trend: {agent['health_trend']}")
        print(f"Reason: {reason}")
        print("---")
        time.sleep(0.5)