"""FastAPI entrypoint for Layer 1 policy interpretation."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline import interpret_policy
from schema import PolicyInterpretation, PolicyRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Layer 1 Policy Interpreter", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/interpret", response_model=PolicyInterpretation)
async def interpret(request: PolicyRequest) -> PolicyInterpretation:
    return await interpret_policy(request)
