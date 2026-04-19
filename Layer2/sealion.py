from groq import Groq
import os
import re
import time

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Sea Lion agent state ─────────────────────────────────────────────────────
sea_lion = {
    "population": 50,
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

# ── Prey states (sea lion reads both anchovy and sardine) ────────────────────
anchovy_state = {
    "population": 50,
    "last_action": "school",
    "health_trend": "stable"
}

sardine_state = {
    "population": 45,
    "last_action": "feed_aggressively",
    "health_trend": "improving"
}

# ── Behavior deltas ──────────────────────────────────────────────────────────
BEHAVIORS = {
    "hunt":       +12,   # fish abundant, active hunting, population grows
    "thrive":     +18,   # both anchovy and sardine abundant, boom conditions
    "compete":     +3,   # fighting other predators for limited fish
    "haul_out":   -8,    # retreat to land, ocean conditions poor
    "starve":     -20,   # fish populations crashed, severe decline
    "migrate":    -10,   # follow fish populations elsewhere
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


def build_prompt(agent, env, anchovy, sardine):
    # Combined prey availability — sea lions eat both
    total_prey = (anchovy['population'] + sardine['population']) / 2

    if total_prey < 15:
        prey_status = "CRITICALLY LOW - both anchovy and sardine collapsed, you must starve or migrate"
    elif total_prey < 35:
        prey_status = "LOW - fish scarce, haul_out or compete for what remains"
    elif total_prey < 60:
        prey_status = "MODERATE - enough fish to hunt"
    else:
        prey_status = "ABUNDANT - both species available, thrive conditions"

    anchovy_action = anchovy['last_action']
    sardine_action = sardine['last_action']

    # Schooling fish are easier to hunt — important behavioral signal
    if anchovy_action == "school" or sardine_action == "school":
        hunt_ease = "EASY - fish are schooling, concentrated and easy to catch"
    elif anchovy_action == "scatter" or sardine_action == "scatter":
        hunt_ease = "HARD - fish scattered, expensive to hunt"
    elif anchovy_action == "migrate_north" and sardine_action == "migrate_south":
        hunt_ease = "VERY HARD - fish migrating in opposite directions, prey dispersed"
    else:
        hunt_ease = "MODERATE - normal hunting conditions"

    temp = env['temperature']
    if temp > 20:
        temp_status = "TOO HIGH - warm water reduces fish density, harder to hunt"
    elif temp > 17:
        temp_status = "ELEVATED - mild impact on hunting"
    else:
        temp_status = "OPTIMAL"

    pollution = env['pollution_index']
    if pollution > 0.7:
        pollution_status = "HIGH - affecting your health and prey quality"
    elif pollution > 0.4:
        pollution_status = "MODERATE - some impact"
    else:
        pollution_status = "LOW - clean conditions"

    return f"""You are a California sea lion colony. Pick a survival behavior RIGHT NOW.

PREY AVAILABILITY (avg {total_prey:.0f}/100): {prey_status}
- Anchovy: {anchovy['population']}/100 (last action: {anchovy_action}, trend: {anchovy['health_trend']})
- Sardine: {sardine['population']}/100 (last action: {sardine_action}, trend: {sardine['health_trend']})
HUNTING CONDITIONS: {hunt_ease}
TEMPERATURE ({temp}°C): {temp_status}
POLLUTION: {pollution_status}
YOUR POPULATION: {agent['population']}/100
YOUR LAST ACTION: {agent['last_action'] or 'none'}

KEY BIOLOGY: You are the top predator and ecosystem health indicator. You eat 
both anchovy and sardine — if both collapse simultaneously you face a crisis. 
When fish school you can hunt efficiently. When fish scatter or migrate you 
burn more energy than you gain.

DECISION RULES — follow strictly:
- If prey CRITICALLY LOW → starve or migrate
- If prey LOW and hunting HARD → haul_out (conserve energy on land)
- If prey LOW and hunting EASY → compete (fight for what's left)
- If prey MODERATE and temp OPTIMAL → hunt
- If prey ABUNDANT and hunting EASY → thrive
- If prey ABUNDANT but hunting HARD → hunt (still worth it)
- If temp TOO HIGH → haul_out or migrate

Pick exactly one from: hunt, thrive, compete, haul_out, starve, migrate

BEHAVIOR: [one word]
REASON: [one sentence first person]"""


def tick(agent, env, anchovy, sardine):
    prompt = build_prompt(agent, env, anchovy, sardine)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        raw = response.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")
        raw = "BEHAVIOR: haul_out\nREASON: Defaulting due to error."

    behavior = validate_behavior(raw, BEHAVIORS, "haul_out")
    reason = extract_reason(raw)

    delta = BEHAVIORS[behavior]
    agent["population"] = max(0, min(100, agent["population"] + delta))
    agent["last_action"] = behavior
    agent["health_trend"] = "improving" if delta > 0 else "declining" if delta < 0 else "stable"

    return agent, behavior, reason


if __name__ == "__main__":
    print("=== Sea Lion Simulation ===")
    print(f"Anchovy: pop={anchovy_state['population']}, action={anchovy_state['last_action']}")
    print(f"Sardine: pop={sardine_state['population']}, action={sardine_state['last_action']}\n")

    for i in range(1, 6):
        sea_lion, behavior, reason = tick(sea_lion, environment, anchovy_state, sardine_state)
        print(f"Tick {i} | Action: {behavior} | Population: {sea_lion['population']}/100 | Trend: {sea_lion['health_trend']}")
        print(f"Reason: {reason}")
        print("---")
        time.sleep(0.5)