"""
app.py — FastAPI server for the Traffic Surveillance OpenEnv environment.
=========================================================================
Exposes the environment as a REST API on port 7860 for HuggingFace Spaces.
Tries the official openenv-core factory pattern; falls back to hand-rolled
FastAPI endpoints so the server works with or without the published package.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import (
    TrafficSurveillanceEnv,
    TrafficAction,
    TrafficObservation,
    TrafficState,
)

# ---------------------------------------------------------------------------
# Global env instance (single-session for HF Space; production would use
# per-session state stored in Redis / DB)
# ---------------------------------------------------------------------------
_env: Optional[TrafficSurveillanceEnv] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = TrafficSurveillanceEnv()
    yield
    _env = None


# ---------------------------------------------------------------------------
# Try openenv-core factory; fall back to manual app
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import create_fastapi_app  # type: ignore

    _env_instance = TrafficSurveillanceEnv()
    app = create_fastapi_app(
        env=_env_instance,
        action_cls=TrafficAction,
        observation_cls=TrafficObservation,
    )
    _USING_OPENENV_FACTORY = True
except ImportError:
    _USING_OPENENV_FACTORY = False
    app = FastAPI(
        title="Smart City Traffic Surveillance Analyst",
        description="OpenEnv-compliant traffic incident detection and response environment.",
        version="1.0.0",
        lifespan=lifespan,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Request / Response schemas for hand-rolled endpoints
# ===========================================================================

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: Optional[str] = "collision_tagging"


class StepRequest(BaseModel):
    action_type: str
    payload: Dict[str, Any] = {}


# ===========================================================================
# Endpoints (only registered when openenv-core factory is NOT available)
# ===========================================================================

if not _USING_OPENENV_FACTORY:

    @app.get("/", tags=["health"])
    async def root():
        """Health check — HuggingFace Space ping endpoint (must return 200)."""
        return {
            "status": "ok",
            "environment": "Smart City Traffic Surveillance Analyst",
            "version": "1.0.0",
            "timestamp": time.time(),
        }

    @app.get("/health", tags=["health"])
    async def health():
        """Secondary health check endpoint."""
        return {"status": "healthy"}

    @app.post("/reset", response_model=Dict[str, Any], tags=["environment"])
    async def reset(req: ResetRequest):
        """Reset the environment and return the initial observation."""
        if _env is None:
            raise HTTPException(status_code=503, detail="Environment not initialized")
        obs = _env.reset(
            seed=req.seed,
            episode_id=req.episode_id,
            task_name=req.task_name,
        )
        return obs.model_dump()

    @app.post("/step", response_model=Dict[str, Any], tags=["environment"])
    async def step(req: StepRequest):
        """Submit an action and advance the environment by one step."""
        if _env is None:
            raise HTTPException(status_code=503, detail="Environment not initialized")
        try:
            action = TrafficAction(action_type=req.action_type, payload=req.payload)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))
        obs = _env.step(action)
        return obs.model_dump()

    @app.get("/state", response_model=Dict[str, Any], tags=["environment"])
    async def state():
        """Return the current internal environment state."""
        if _env is None:
            raise HTTPException(status_code=503, detail="Environment not initialized")
        return _env.state.model_dump()

    @app.get("/tasks", tags=["metadata"])
    async def list_tasks():
        """List all available tasks with their metadata."""
        from tasks import TASKS
        return {
            name: {
                "task_id": t.task_id,
                "difficulty": t.difficulty,
                "description": t.description,
                "max_steps": t.max_steps,
            }
            for name, t in TASKS.items()
        }

    @app.post("/grade", tags=["grading"])
    async def grade(task_name: str):
        """Grade the current episode state for the given task."""
        if _env is None:
            raise HTTPException(status_code=503, detail="Environment not initialized")
        from tasks import run_grader
        state_dict = _env.state.model_dump()
        try:
            score = run_grader(task_name, state_dict, _env)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"task": task_name, "score": score}

else:
    # When using openenv-core factory, add the mandatory health endpoint
    @app.get("/", tags=["health"])
    async def root_factory():
        return {"status": "ok", "environment": "Smart City Traffic Surveillance Analyst"}

    @app.get("/health", tags=["health"])
    async def health_factory():
        return {"status": "healthy"}


# ===========================================================================
# Entrypoint for `python app.py` and `uv run server`
# ===========================================================================

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
