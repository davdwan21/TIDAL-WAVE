# Tidal Wave — California Current Ecosystem Simulator

A Python-based ecosystem simulation project for the Southern California coastal environment.
The simulator combines a policy parser, a rule-based ecosystem engine, and agent-style species behavior to model how environmental actions affect the California Current system.

## What this project does

- Simulates a coastal ecosystem using core variables such as nutrient load, dissolved oxygen, temperature, habitat quality, and fishing pressure.
- Models populations for key functional groups: phytoplankton, zooplankton, forage fish, predator fish, seabirds, and marine mammals.
- Converts natural-language policy descriptions into absolute environmental parameter changes via `policy_parser.py`.
- Produces visual summaries and charts using `visualization.py`.

## Key components

- `policy_parser.py`
  - Parses policy text into environmental levers and parameter values.
  - Uses local Ollama model endpoints to translate natural language into actionable simulation input.

- `simulation_engine.py`
  - Implements the core rule-based simulation engine.
  - Simulates year-over-year ecosystem dynamics across 10 years.
  - Applies policy levers and models food web interactions.

- `visualization.py`
  - Generates visual outputs for simulation results.
  - Creates charts and summary images to help interpret model outcomes.

- `run.py`
  - Example driver script for running a policy scenario and generating visuals.

- `layer2/`
  - Contains additional ecosystem simulation components and agent behavior modules.
  - Includes species-specific behavior definitions for `anchovy`, `sardine`, `kelp`, `urchin`, `sealion`, `phytoplankton`, and `zooplankton`.
  - Includes multiple simulation versions and visualization helpers.

## Requirements

This project requires Python 3.10+ and the packages listed in `requirements.txt`.

### Install dependencies

```bash
cd /Users/shreyaanchhabra/Documents/GitHub/datahacks-26
pip install -r requirements.txt
```

### Additional dependencies

If you want to run the `layer2/simulation.py` agent-based simulator, you also need:
- `groq` Python client
- `GROQ_API_KEY` environment variable
- Local Ollama service if using `policy_parser.py`

## Running the project

### Run the sample pipeline

```bash
cd /Users/shreyaanchhabra/Documents/GitHub/datahacks-26
python run.py
```

This will execute a sample policy simulation and generate an output visualization file.

### Run the core simulation engine directly

```bash
cd /Users/shreyaanchhabra/Documents/GitHub/datahacks-26
python simulation_engine.py
```

### Run the agent-based California Current simulation

```bash
cd /Users/shreyaanchhabra/Documents/GitHub/datahacks-26/layer2
export GROQ_API_KEY=your_groq_api_key_here
python simulation.py --policy "Ban commercial fishing" --ticks 5
```

### Run the policy parser alone

```bash
cd /Users/shreyaanchhabra/Documents/GitHub/datahacks-26
python -c "from policy_parser import parse_policy; print(parse_policy('Reduce agricultural runoff by 30%'))"
```

## Project structure

```
/                    Root project folder
├── README.md
├── requirements.txt        # Python dependencies
├── run.py                  # Example simulation driver
├── policy_parser.py        # Natural language policy interpreter
├── simulation_engine.py    # Core rule-based ecosystem simulator
├── visualization.py        # Plot and image generation
├── layer2/                 # Agent-based species simulation modules
│   ├── simulation.py
│   ├── anchovy.py
│   ├── sardine.py
│   ├── kelp.py
│   ├── urchin.py
│   ├── sealion.py
│   ├── phytoplankton.py
│   ├── zooplankton.py
│   ├── database_fetch.py
│   ├── sim_v3.py
│   ├── simulation_v2.py
│   ├── simulation_vis.py
│   ├── sim_vis_2.py
│   └── ...
└── fathom_simulation.html   # Example interactive simulation page
```

## Notes

- The simulation is designed to illustrate environmental policy impact in a coastal ecosystem context.
- The project is intended for hackathon or prototyping use and is not a production scientific model.
- The policy parser uses a local language model endpoint, so the environment must include a running Ollama service if used.

## Recommended next steps

- Add a frontend interface for live policy input and visualization.
- Add preset policy scenarios such as marine protected areas, pollution reduction, or fishing limits.
- Improve agent models with stronger coupling between species and environmental variables.
- Add logging, testing, and configuration to support repeated experiments.

---

### Contact

If you want to expand the project, start by editing `run.py`, `policy_parser.py`, and `simulation_engine.py`.
