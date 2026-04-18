# Layer 1: Policy Interpreter — Cursor Build Context

> **Read this entire file before writing any code.** It defines the project, architecture, conventions, and the exact build order. Every step below maps to a concrete deliverable. Do not skip ahead. Do not invent features outside this spec.

---

## 1. Project Context

This is **Layer 1** of a hackathon project called **"See the Future"** at DataHacks 2026 (UCSD, April 18–19, 2026). The theme is **sustainability**.

### What the product does
The user is a policy maker. They type a natural-language policy (e.g., *"Ban commercial trawling within 50 miles of the California coast"*). Layer 1 interprets that policy, grounds it in real data and literature, and outputs concrete simulation parameter changes. Layer 2 (built by another teammate, not in this repo) runs an agent-based ecosystem simulation using those parameters.

### Datasets in use
- **CalCOFI** (California Cooperative Oceanic Fisheries Investigations) — quantitative ocean chemistry, plankton, larvae baselines
- **iNaturalist** — geotagged species observations for charismatic coastal species

### Agent cast for the simulation (informs valid parameter targets)
- Phytoplankton
- Zooplankton / krill
- Anchovy / sardine schools
- Brown pelican
- California sea lion
- Leopard shark or market squid
- Commercial fishing fleet
- Coastal communities

### What Layer 1 is NOT
- NOT the simulation itself (that's Layer 2, rule-based / ODE-driven, built separately)
- NOT the frontend visualization
- NOT responsible for running the sim — only for producing parameter deltas

---

## 2. Tech Stack (Locked — do not change)

- **Language:** Python 3.10+
- **LLM provider:** Google Gemini 2.5 Flash (via `google-genai` SDK) — **only provider**. No OpenAI, no Anthropic, no Orthogonal.
- **Web grounding:** Gemini's built-in Google Search tool (`types.Tool(google_search=types.GoogleSearch())`)
- **Web framework:** FastAPI with Uvicorn
- **Data validation:** Pydantic v2
- **Env management:** python-dotenv
- **Manual HTTP (if ever needed):** requests + beautifulsoup4

### Installed packages (already in requirements.txt)
```
google-genai
pydantic
fastapi
uvicorn
requests
beautifulsoup4
python-dotenv
```

### Environment variables (loaded from `.env`)
```
GEMINI_API_KEY=<present>
```

---

## 3. Architecture Overview

```
         USER POLICY TEXT
               │
               ▼
       ┌───────────────┐
       │ Parser Agent  │  Gemini, JSON mode
       └───────┬───────┘
               │ parsed intent (structured)
               ▼
     ┌─────────┼─────────┐
     ▼         ▼         ▼
  ┌──────┐ ┌──────┐ ┌────────┐
  │ Lit  │ │ Hist │ │Dataset │  3 research agents
  │Agent │ │Agent │ │ Agent  │  run in parallel
  └───┬──┘ └───┬──┘ └────┬───┘
      │       │         │
      └───────┼─────────┘
              ▼
      ┌───────────────┐
      │ Skeptic Agent │  Challenges findings
      └───────┬───────┘
              ▼
      ┌───────────────┐
      │  Synthesizer  │  Gemini, JSON mode
      └───────┬───────┘
              ▼
      PolicyInterpretation  ← returned via FastAPI endpoint
```

---

## 4. File Layout (Target)

```
Layer1/
├── .env                      # loaded, not tracked
├── .gitignore                # at repo root, not here
├── requirements.txt
├── main.py                   # FastAPI app, POST /interpret
├── schema.py                 # Pydantic models — SINGLE SOURCE OF TRUTH
├── llm.py                    # Gemini wrapper (chat_json, chat_text, research_with_search)
├── pipeline.py               # Orchestrates: parse → research → skeptic → synthesize
├── canned_policies.py        # Demo-safe pre-computed outputs
├── valid_parameters.py       # List of valid ParameterDelta.target names
├── agents/
│   ├── __init__.py
│   ├── parser.py
│   ├── literature.py
│   ├── historical.py
│   ├── dataset.py
│   ├── skeptic.py
│   └── synthesizer.py
├── data/                     # Local CalCOFI / iNaturalist CSV stubs for dataset agent
│   └── README.md             # explains what goes here
└── tests/
    ├── test_schema.py
    ├── test_parser.py
    └── test_pipeline.py
```

---

## 5. Conventions & Quality Bar

### Code conventions
- **Type hints everywhere.** Every function signature types all params and return.
- **Pydantic v2 syntax** (use `model_dump()`, not `.dict()`; use `BaseModel` with type-annotated fields).
- **Async for I/O** — research agents must run concurrently via `asyncio.gather`.
- **No bare `except:`** — always catch specific exceptions and log.
- **Docstrings** on every agent function explaining: input, output, failure modes.
- **No magic strings** — parameter target names come from `valid_parameters.py`.

### LLM call conventions
- **Always use JSON mode** (`response_mime_type="application/json"`) when expecting structured output.
- **Always retry once** on `json.JSONDecodeError` with an explicit "return ONLY valid JSON" instruction.
- **Always include the valid parameter list in prompts** that produce `ParameterDelta` objects — the LLM must not invent parameter names.
- **Temperature:** 0.3 for parsing/synthesis (deterministic), 0.7 for skeptic (more creative challenges).
- **Timeout:** 30 seconds per LLM call.

### Demo-safety conventions
- Every LLM call must have a fallback path. If Gemini fails, return a graceful error in the response, never crash.
- Canned policies take precedence when `DEMO_MODE=true` is set in env.
- Reasoning trace must always populate, even on partial failures — judges see this live.

---

## 6. Build Order — Execute Steps 1 through 9 in Order

### ⚠️ Rules for Cursor
1. Complete each step fully before starting the next.
2. After each step, stop and print a summary: **what was built, what files changed, what to verify manually.**
3. Do not combine steps. Do not scope-creep.
4. If a step depends on a human decision (e.g., Step 1 requires naming conventions), **ask the human for input before coding**.

---

### STEP 1 — Lock the I/O Contract ⚠️ HUMAN INPUT REQUIRED FIRST

**Deliverable:** `schema.py` and `valid_parameters.py`

**Before writing code**, print the following to the user and wait for their response:

> "Step 1 requires agreeing on valid parameter names with the simulation teammate. I'm proposing the list below. Please:
> 1. Review it.
> 2. Walk to your simulation teammate and confirm these exact names will be accepted.
> 3. Paste back any additions/renames, or say 'approved' to proceed.
>
> **Proposed valid parameter targets** (dot-notation: `entity.attribute`):
>
> **Species population dynamics:**
> - `phytoplankton.growth_rate`
> - `zooplankton.growth_rate`, `zooplankton.mortality_rate`
> - `anchovy.mortality_rate`, `anchovy.reproduction_rate`, `anchovy.catch_rate`
> - `sardine.mortality_rate`, `sardine.reproduction_rate`
> - `pelican.mortality_rate`, `pelican.reproduction_rate`
> - `sea_lion.mortality_rate`, `sea_lion.reproduction_rate`
> - `leopard_shark.mortality_rate`
>
> **Ocean state variables:**
> - `ocean.temperature`, `ocean.ph`, `ocean.dissolved_oxygen`, `ocean.nutrient_level`, `ocean.pollution_index`
>
> **Human pressure:**
> - `fishing_fleet.catch_rate`, `fishing_fleet.effort_level`
> - `coastal_community.runoff_rate`, `coastal_community.consumption_rate`
>
> **Policy zones:**
> - `protected_area.coverage_percent`
>
> Valid operations: `multiply`, `add`, `set`.
> Valid value range: any float (but document expected magnitudes per parameter in comments).
>
> Awaiting your approval or edits."

**Once approved, build:**

1. `valid_parameters.py` containing:
   - `VALID_TARGETS: set[str]` — the approved list
   - `VALID_OPERATIONS: set[str] = {"multiply", "add", "set"}`
   - A helper `validate_delta(target, operation) -> bool`

2. `schema.py` with Pydantic v2 models:
   - `PolicyRequest` — input from frontend: `policy_text: str`, `region: str = "socal"`
   - `ParameterDelta` — `target: str`, `operation: str`, `value: float`, `rationale: str` — validate `target` against `VALID_TARGETS` and `operation` against `VALID_OPERATIONS` with Pydantic validators
   - `Source` — `title: str`, `url: Optional[str]`, `excerpt: str`
   - `PolicyInterpretation` — output schema: `plain_english_summary: str`, `parameter_deltas: List[ParameterDelta]`, `confidence: float` (0–1, validated), `sources: List[Source]`, `reasoning_trace: List[str]`, `warnings: List[str]`

3. `tests/test_schema.py` — pytest tests covering: valid case, invalid target rejected, invalid operation rejected, confidence out of range rejected.

**Stop condition:** User runs `pytest tests/test_schema.py` and all tests pass. Summary printed.

---

### STEP 2 — Stub the Pipeline End-to-End with Fake Data

**Deliverable:** `main.py`, `pipeline.py`, initial `llm.py` shell

**Goal:** Return a *valid* `PolicyInterpretation` with hardcoded mock data. No real LLM calls yet. This unblocks any frontend/integration work immediately.

1. `llm.py` — create the Gemini client and expose:
   - `chat_json(prompt: str, model: str = "gemini-2.5-flash", temperature: float = 0.3) -> dict`
   - `chat_text(prompt: str, model: str = "gemini-2.5-flash", temperature: float = 0.3) -> str`
   - `research_with_search(query: str, model: str = "gemini-2.5-flash") -> dict` — returns `{"text": str, "sources": List[dict]}` using Google Search grounding

2. `pipeline.py` — sync function `interpret_policy(request: PolicyRequest) -> PolicyInterpretation` that returns a realistic mock:
   - One `ParameterDelta` (use a real valid target like `anchovy.mortality_rate`, operation `multiply`, value `0.6`)
   - One mock `Source`
   - A reasoning_trace with 3–4 strings
   - confidence 0.75

3. `main.py` — FastAPI app with:
   - `POST /interpret` — accepts `PolicyRequest`, returns `PolicyInterpretation`
   - `GET /health` — returns `{"status": "ok"}`
   - CORS enabled for all origins (hackathon, not prod)

**Stop condition:** User runs `uvicorn main:app --reload`, opens `/docs`, and can POST a test request that returns a valid response. Summary printed.

---

### STEP 3 — Build the Parser Agent

**Deliverable:** `agents/parser.py`

1. Define `ParsedIntent` as a TypedDict or Pydantic model with fields: `action_type`, `target_activity`, `scope_geographic`, `scope_temporal`, `magnitude`, `affected_species: list[str]`.

2. `parse_policy(policy_text: str) -> ParsedIntent` — uses `chat_json` with a detailed system prompt instructing Gemini to return exactly this schema as JSON. Prompt must include 2–3 few-shot examples.

3. Handle JSON decode errors with one retry using a stricter prompt.

4. Wire into `pipeline.py` — replace the mock's "parsing" step with a real call.

5. `tests/test_parser.py` — 5 test cases covering: fishing ban, MPA establishment, pollution regulation, fishing quota increase, vague/ambiguous policy. Use real Gemini calls but mark as integration tests.

**Stop condition:** Parser correctly extracts structured intent for the 5 test cases. Summary printed.

---

### STEP 4 — Build the Research Agents (Parallel)

**Deliverable:** `agents/literature.py`, `agents/historical.py`, `agents/dataset.py`

1. **`literature_agent(parsed: ParsedIntent) -> dict`** — uses `research_with_search` to find peer-reviewed or authoritative sources on the ecological impact of the proposed action. Returns `{"findings": List[str], "sources": List[Source], "suggested_parameters": List[dict]}`.

2. **`historical_agent(parsed: ParsedIntent) -> dict`** — uses `research_with_search` to find historical precedents for similar policies and their observed outcomes. Same return shape.

3. **`dataset_agent(parsed: ParsedIntent) -> dict`** — does NOT call the LLM directly. Reads a stub CSV from `data/calcofi_summary.csv` and `data/inaturalist_summary.csv`, extracts numbers relevant to `parsed.affected_species`, and returns baseline values + trend indicators. For MVP, hardcode 2–3 rows in each CSV with plausible values.

4. In `pipeline.py`, convert `interpret_policy` to async and run the three research agents via `asyncio.gather`.

5. Each agent appends to the `reasoning_trace`.

**Stop condition:** All three agents run in parallel, complete under 15s total, and contribute to the final output. Summary printed.

---

### STEP 5 — Build the Skeptic Agent

**Deliverable:** `agents/skeptic.py`

1. `skeptic_agent(parsed: ParsedIntent, research_outputs: list[dict]) -> dict` — uses `chat_json` with temperature 0.7. System prompt: "You are a scientific skeptic. Review these findings and proposed parameter changes. Flag overstatements, missing evidence, or magnitudes inconsistent with similar historical cases. Be rigorous but fair."

2. Returns `{"concerns": List[str], "adjustments": List[{"target": str, "suggested_value": float, "reason": str}]}`.

3. In `pipeline.py`, wire after research agents, before synthesizer.

4. Skeptic output appends prominently to `reasoning_trace` with a 🤔 emoji prefix (judges see this).

**Stop condition:** Skeptic produces meaningful critiques on test cases and doesn't rubber-stamp findings. Summary printed.

---

### STEP 6 — Build the Synthesizer

**Deliverable:** `agents/synthesizer.py`

1. `synthesize(parsed, research_outputs, skeptic_output) -> PolicyInterpretation` — uses `chat_json` with temperature 0.3.

2. System prompt MUST:
   - Include the full `VALID_TARGETS` list from `valid_parameters.py`
   - Instruct the LLM to produce ONLY valid parameter targets
   - Include the target JSON schema inline
   - Instruct to incorporate skeptic adjustments

3. Validate the response against the `PolicyInterpretation` Pydantic model. If validation fails, retry once with the validation error in the retry prompt.

4. Wire as the final step in `pipeline.py`.

**Stop condition:** End-to-end pipeline produces validated `PolicyInterpretation` objects. No invalid parameter targets slip through. Summary printed.

---

### STEP 7 — Wire It All Together + Polish Error Handling

**Deliverable:** Polished `pipeline.py`, updated `main.py`

1. Full async orchestration in `pipeline.py`:
```python
async def interpret_policy(request: PolicyRequest) -> PolicyInterpretation:
    parsed = await parse_policy(request.policy_text)
    lit, hist, data = await asyncio.gather(
        literature_agent(parsed),
        historical_agent(parsed),
        dataset_agent(parsed),
    )
    skeptic = await skeptic_agent(parsed, [lit, hist, data])
    return await synthesize(parsed, [lit, hist, data], skeptic)
```

2. Wrap the whole pipeline in a try/except that produces a valid (but flagged) `PolicyInterpretation` even on failure, with `confidence=0.0` and warnings explaining what broke.

3. Add request-level logging (use Python's `logging`, not `print`).

4. Add a `/interpret` endpoint latency target: log if it exceeds 30s.

**Stop condition:** User can POST real policies and consistently get valid responses under 30s. Summary printed.

---

### STEP 8 — Canned Fallbacks (CRITICAL FOR DEMO)

**Deliverable:** `canned_policies.py`

1. Hand-craft 5 `PolicyInterpretation` objects for these exact prompt templates:
   - "ban commercial fishing" / "ban commercial trawling"
   - "establish marine protected area"
   - "deregulate fishing" / "remove fishing restrictions"
   - "reduce agricultural runoff"
   - "carbon tax on ocean shipping"

2. Each canned response must be *excellent* — judge-ready: 3–5 well-reasoned parameter deltas, 3+ real-looking sources with plausible titles, a rich reasoning trace showing "multi-agent" thinking.

3. Add `fuzzy_match(text: str, keys: list[str]) -> Optional[str]` — simple keyword-based matching.

4. In `pipeline.py`, check the env flag `DEMO_MODE`. If set to `"true"`, try canned match first; fall back to live pipeline if no match.

5. Document in a comment block at the top of `canned_policies.py`: **"These are the demo-safe outputs. If live demo fails, set `DEMO_MODE=true` and type one of the trigger phrases."**

**Stop condition:** With `DEMO_MODE=true`, canned policies return instantly. With it off, live pipeline runs. Summary printed.

---

### STEP 9 — Polish the Reasoning Trace for UI

**Deliverable:** Final trace polish across all agents

1. Standardize reasoning_trace entries with emoji prefixes for visual distinction in the UI:
   - `📋` Parser
   - `📚` Literature agent
   - `🏛️` Historical agent
   - `📊` Dataset agent
   - `🤔` Skeptic
   - `✅` Synthesizer

2. Each entry is a single human-readable sentence (not raw JSON).

3. Include specific numbers where possible ("Literature agent found 4 sources, average recovery time 3.2 years").

4. Trace must always have at least 5 entries even on failure paths.

**Stop condition:** POST a test policy, read the trace aloud — it should feel like watching a research team think in real time. Summary printed.

---

## 7. Final Deliverable Checklist

When all 9 steps are complete, confirm:

- [ ] `pytest tests/` all pass
- [ ] `uvicorn main:app --reload` starts cleanly
- [ ] POST `/interpret` with "Ban commercial trawling within 50 miles of the California coast" returns valid, rich output in under 30s
- [ ] `DEMO_MODE=true` returns canned output instantly
- [ ] No hardcoded API keys anywhere in code
- [ ] `.env` still not tracked by git
- [ ] No invalid parameter targets can reach the output (Pydantic enforces this)
- [ ] Reasoning trace is demo-ready

---

## 8. How to Start

Begin with **Step 1**. Print the approval request for valid parameter names and **wait** for the human's response before writing any code. Do not proceed until they say "approved" or provide edits.
