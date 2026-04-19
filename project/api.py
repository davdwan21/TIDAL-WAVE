import copy
import os
import sys
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

import database_fetch
import simulation

app = FastAPI(title="Fathom Ocean Simulator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimRequest(BaseModel):
    policy: str
    ticks: int = 10


@app.post("/simulate")
def simulate(req: SimRequest):
    baseline_env = database_fetch.BASELINE_ENVIRONMENT.copy()
    baseline_agents = simulation.get_initial_agents()

    # Year 0: pre-policy baseline (what the user starts from)
    baseline_tick = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ticks": 0,
        "policy": "",
        "environment": copy.deepcopy(baseline_env),
        "agents": copy.deepcopy(baseline_agents),
    }

    # Apply policy to derive the starting env for the simulation
    sim_env = copy.deepcopy(baseline_env)
    if req.policy.strip():
        simulation.apply_policy_to_environment(req.policy, sim_env)

    # Run simulation — returns year 0 (post-policy) through year N
    sim_timeline = simulation.run_simulation_timeline(
        sim_env,
        copy.deepcopy(baseline_agents),
        ticks=req.ticks,
        policy=req.policy,
    )

    # Return: [baseline year 0] + [simulation years 1..N]
    timeline = [baseline_tick] + sim_timeline[1:]

    return {
        "timeline": timeline,
        "policy_environment": sim_env,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
