from groq import Groq
import os
import re
import time

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Urchin agent state ───────────────────────────────────────────────────────
urchin = {
    "population": 40,
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

# ── Kelp state (urchin's food source) ────────────────────────────────────────
kelp_state = {
    "population": 60,
    "last_action": "grow",
    "health_trend": "stable"
}

# ── Behavior deltas ──────────────────────────────────────────────────────────
BEHAVIORS = {
    "graze_kelp":      +12,   # eat kelp, population grows
    "barren_expand":   +20,   # kelp abundant + no predators = urchin boom
    "retreat":         -15,   # being harvested or predated
    "reproduce":       +10,   # stable conditions, slow growth
    "starve":          -18,   # kelp collapsed, nothing to eat
}


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

    return behavior


def extract_reason(raw):
    match = re.search(r'REASON:\s*(.+)', raw, flags=re.IGNORECASE)
    if match:
        reason = match.group(1).strip()
        if len(reason) >= 10:
            return reason
    return "No reason provided."


def build_prompt(agent, env, kelp):
    food = kelp['population']
    if food < 15:
        food_status = "CRITICALLY LOW - kelp forest collapsing, you must starve"
    elif food < 35:
        food_status = "LOW - kelp scarce, graze carefully"
    elif food < 65:
        food_status = "MODERATE - enough kelp to graze"
    else:
        food_status = "ABUNDANT - kelp forest healthy, expand aggressively"

    fishing = env['fishing_pressure']
    if fishing > 0.7:
        fishing_status = "HEAVY - urchin harvesting active, retreat"
    elif fishing > 0.4:
        fishing_status = "MODERATE - some harvesting pressure"
    else:
        fishing_status = "LOW - minimal predation, you can expand"

    return f"""You are a sea urchin population in the California Current. Pick a survival behavior RIGHT NOW.

FOOD (kelp {food}/100): {food_status}
KELP TREND: {kelp['health_trend']} (last action: {kelp['last_action']})
FISHING/HARVEST PRESSURE: {fishing_status}
YOUR POPULATION: {agent['population']}/100
YOUR LAST ACTION: {agent['last_action'] or 'none'}

KEY BIOLOGY: You are the primary threat to kelp forests. When unchecked you create 
"urchin barrens" — dead zones where kelp forests used to be. You are kept in check 
by fishing/harvesting pressure and natural predators like sea otters.

DECISION RULES — follow strictly:
- If kelp CRITICALLY LOW → starve (you ate everything)
- If kelp ABUNDANT and fishing LOW → barren_expand (boom conditions)
- If kelp ABUNDANT and fishing HEAVY → retreat (being harvested)
- If kelp MODERATE and fishing LOW → graze_kelp or reproduce
- If kelp MODERATE and fishing HEAVY → retreat
- If kelp LOW → graze_kelp cautiously

Pick exactly one from: graze_kelp, barren_expand, retreat, reproduce, starve

BEHAVIOR: [one word]
REASON: [one sentence first person]"""


def tick(agent, env, kelp):
    prompt = build_prompt(agent, env, kelp)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        raw = response.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")
        raw = "BEHAVIOR: graze_kelp\nREASON: Defaulting due to error."

    behavior = validate_behavior(raw, BEHAVIORS, "graze_kelp")
    reason = extract_reason(raw)

    delta = BEHAVIORS[behavior]
    agent["population"] = max(0, min(100, agent["population"] + delta))
    agent["last_action"] = behavior
    agent["health_trend"] = "improving" if delta > 0 else "declining" if delta < 0 else "stable"

    return agent, behavior, reason


if __name__ == "__main__":
    print("=== Urchin Simulation ===")
    print(f"Kelp context: pop={kelp_state['population']}, action={kelp_state['last_action']}\n")

    for i in range(1, 6):
        urchin, behavior, reason = tick(urchin, environment, kelp_state)
        print(f"Tick {i} | Action: {behavior} | Population: {urchin['population']}/100 | Trend: {urchin['health_trend']}")
        print(f"Reason: {reason}")
        print("---")
        time.sleep(0.5)