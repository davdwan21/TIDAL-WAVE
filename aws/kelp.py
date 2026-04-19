from groq import Groq
import os
import re
import time

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Kelp agent state ─────────────────────────────────────────────────────────
kelp = {
    "population": 60,
    "last_action": None,
    "health_trend": "stable"
}

# ── Environment state ────────────────────────────────────────────────────────
environment = {
    "temperature": 16.2,
    "nutrients": 0.6,
    "pH": 8.05,
    "salinity": 33.4,
    "fishing_pressure": 0.2,
    "pollution_index": 0.3
}

# ── Urchin state (kelp's primary threat) ─────────────────────────────────────
urchin_state = {
    "population": 40,
    "last_action": "graze_kelp",
    "health_trend": "stable"
}

# ── Behavior deltas ──────────────────────────────────────────────────────────
BEHAVIORS = {
    "grow":      +15,   # urchin low + nutrients good = kelp forest thrives
    "hold":        0,   # stable conditions, maintain coverage
    "recede":    -12,   # urchin pressure increasing
    "collapse":  -25,   # urchin barrens forming, catastrophic loss
    "recover":   +20,   # urchin pressure removed, rapid regrowth
}


def build_prompt(agent, env, urchin):
    threat = urchin['population']
    if threat > 75:
        threat_status = "CRITICAL - urchin barrens forming, you must collapse"
    elif threat > 55:
        threat_status = "HIGH - heavy grazing pressure, recede"
    elif threat > 35:
        threat_status = "MODERATE - manageable grazing pressure"
    else:
        threat_status = "LOW - urchins controlled, you can grow or recover"

    temp = env['temperature']
    if temp > 20:
        temp_status = "TOO HIGH - heat stress on kelp, weakens growth"
    elif temp > 17:
        temp_status = "ELEVATED - mild stress"
    else:
        temp_status = "OPTIMAL for kelp growth"

    nutrients = env['nutrients']
    if nutrients > 0.6:
        nutrient_status = "ABUNDANT - great growing conditions"
    elif nutrients > 0.3:
        nutrient_status = "MODERATE - adequate for maintenance"
    else:
        nutrient_status = "SCARCE - stunted growth"

    last_urchin_action = urchin['last_action']
    if last_urchin_action == "barren_expand":
        urchin_action_status = "EXPANDING AGGRESSIVELY - immediate threat to your forest"
    elif last_urchin_action == "retreat":
        urchin_action_status = "RETREATING - pressure is lifting, opportunity to recover"
    elif last_urchin_action == "starve":
        urchin_action_status = "STARVING - urchin threat collapsing, recover now"
    else:
        urchin_action_status = "GRAZING STEADILY - normal pressure"

    return f"""You are a kelp forest in the California Current. Pick a survival behavior RIGHT NOW.

URCHIN THREAT ({threat}/100): {threat_status}
URCHIN LAST ACTION: {urchin_action_status}
TEMPERATURE ({temp}°C): {temp_status}
NUTRIENTS ({nutrients}/1.0): {nutrient_status}
pH: {env['pH']} (optimal 8.1-8.3, lower = acidification stress)
YOUR POPULATION: {agent['population']}/100
YOUR LAST ACTION: {agent['last_action'] or 'none'}

KEY BIOLOGY: You are the ecosystem indicator. Your health determines habitat for 
hundreds of species. Urchins are your primary threat — when they barren_expand 
you collapse. But when urchin pressure lifts you recover rapidly.

DECISION RULES — follow strictly:
- If urchin CRITICAL or barren_expanding → collapse
- If urchin HIGH → recede
- If urchin LOW and nutrients ABUNDANT and temp OPTIMAL → grow
- If urchin RETREATING or STARVING → recover (take the opportunity)
- If urchin MODERATE and temp OPTIMAL → hold or grow
- If temp TOO HIGH → recede regardless of urchin pressure

Pick exactly one from: grow, hold, recede, collapse, recover

BEHAVIOR: [one word]
REASON: [one sentence first person]"""


def validate_behavior(raw, allowed, default):
    if not isinstance(raw, str):
        print(f"[validation warning] non-string response, using default '{default}'")
        return default

    behavior = default
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in lines:
        if line.upper().startswith("BEHAVIOR:"):
            candidate = line.split(":", 1)[1].strip().lower().replace(" ", "_")
            if candidate in allowed:
                return candidate

    for b in allowed:
        if re.search(rf"\b{re.escape(b)}\b", raw.lower()):
            behavior = b
            break

    if raw.lower().count("behavior:") != 1:
        print(f"[validation warning] unexpected behavior format, using '{behavior}'. Raw: {raw!r}")

    if behavior not in allowed:
        print(f"[validation warning] invalid behavior response, using default '{default}'. Raw: {raw!r}")
        return default

    return behavior


def extract_reason(raw):
    if not isinstance(raw, str):
        return "No reason provided."

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in lines:
        if line.upper().startswith("REASON:"):
            candidate = line.split(":", 1)[1].strip()
            if len(candidate) >= 10:
                return candidate
            break
    return "No reason provided."


def tick(agent, env, urchin):
    prompt = build_prompt(agent, env, urchin)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        raw = response.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")
        raw = "BEHAVIOR: hold\nREASON: Defaulting due to error."

    behavior = validate_behavior(raw, BEHAVIORS, "hold")
    reason = extract_reason(raw)

    delta = BEHAVIORS[behavior]
    agent["population"] = max(0, min(100, agent["population"] + delta))
    agent["last_action"] = behavior
    agent["health_trend"] = "improving" if delta > 0 else "declining" if delta < 0 else "stable"

    return agent, behavior, reason


if __name__ == "__main__":
    print("=== Kelp Simulation ===")
    print(f"Urchin threat: pop={urchin_state['population']}, action={urchin_state['last_action']}\n")

    for i in range(1, 6):
        kelp, behavior, reason = tick(kelp, environment, urchin_state)
        print(f"Tick {i} | Action: {behavior} | Population: {kelp['population']}/100 | Trend: {kelp['health_trend']}")
        print(f"Reason: {reason}")
        print("---")
        time.sleep(0.5)