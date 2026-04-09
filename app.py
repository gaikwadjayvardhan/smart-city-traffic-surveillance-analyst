"""
app.py — FastAPI server for the Traffic Surveillance OpenEnv environment.
=========================================================================
Exposes the environment as a REST API on port 7860 for HuggingFace Spaces.

Critical fixes vs v1:
  - _env is initialized at MODULE LOAD TIME (not inside lifespan) so it is
    never None when the first request arrives.  lifespan was too late.
  - /reset accepts an empty body {} (all fields optional with defaults).
  - Keep-alive background thread prevents HF Space from going to sleep.
  - Full startup stdout logging so Docker build/run failures are visible.
"""

from __future__ import annotations

import os
import sys
import time
import threading
import traceback
from typing import Any, Dict, List, Optional

print("[BOOT] app.py starting import phase...", flush=True)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    print("[BOOT] FastAPI + Pydantic imported", flush=True)
except Exception as e:
    print(f"[FATAL] FastAPI import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    from environment import (
        TrafficSurveillanceEnv,
        TrafficAction,
        TrafficObservation,
        TrafficState,
    )
    print("[BOOT] environment.py imported", flush=True)
except Exception as e:
    print(f"[FATAL] environment.py import failed: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    from tasks import TASKS, run_grader
    print("[BOOT] tasks.py imported", flush=True)
except Exception as e:
    print(f"[FATAL] tasks.py import failed: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

# ===========================================================================
# Global env — initialized IMMEDIATELY at module load, never None
# ===========================================================================
print("[BOOT] Initializing environment instance...", flush=True)
_env: TrafficSurveillanceEnv = TrafficSurveillanceEnv()
# Pre-warm with a reset so the env is ready to serve instantly
_env.reset(seed=42, task_name="collision_tagging")
print("[BOOT] Environment initialized and pre-warmed", flush=True)

# ===========================================================================
# FastAPI app
# ===========================================================================
app = FastAPI(
    title="Smart City Traffic Surveillance Analyst",
    description=(
        "OpenEnv-compliant environment for traffic incident detection, "
        "emergency dispatch, and traffic management."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# Request schemas — all fields Optional so {} body always works
# ===========================================================================

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: Optional[str] = "collision_tagging"
    # Accept extra fields silently (some validators send additional metadata)
    model_config = {"extra": "ignore"}


class StepRequest(BaseModel):
    action_type: str = "noop"
    payload: Dict[str, Any] = {}
    model_config = {"extra": "ignore"}


# ===========================================================================
# Exception handler — always return JSON, never HTML
# ===========================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"[ERROR] Unhandled exception on {request.url}: {exc}\n{tb}", flush=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


# ===========================================================================
# Health endpoints (HF Space pings these to mark container as healthy)
# ===========================================================================

@app.get("/", tags=["health"])
async def root():
    """Primary health check — HuggingFace Space ping endpoint."""
    return {
        "status": "ok",
        "environment": "Smart City Traffic Surveillance Analyst",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@app.get("/health", tags=["health"])
async def health():
    """Secondary health check."""
    return {"status": "healthy", "timestamp": time.time()}


# ===========================================================================
# Core OpenEnv endpoints
# ===========================================================================

@app.post("/reset", tags=["environment"])
async def reset(req: ResetRequest = ResetRequest()):
    """
    Reset the environment and return the initial observation.
    Accepts an empty body {} — all fields are optional.
    """
    global _env
    try:
        obs = _env.reset(
            seed=req.seed,
            episode_id=req.episode_id,
            task_name=req.task_name or "collision_tagging",
        )
        result = obs.model_dump()
        print(f"[RESET] task={req.task_name} seed={req.seed} frame={obs.current_frame}", flush=True)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", tags=["environment"])
async def step(req: StepRequest):
    """Submit an action and advance the environment by one step."""
    global _env
    try:
        action = TrafficAction(action_type=req.action_type, payload=req.payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    try:
        obs = _env.step(action)
        result = obs.model_dump()
        print(f"[STEP] action={req.action_type} reward={obs.reward:.3f} done={obs.done}", flush=True)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", tags=["environment"])
async def state():
    """Return the current internal environment state."""
    global _env
    try:
        return _env.state.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks", tags=["metadata"])
async def list_tasks():
    """List all available tasks with their metadata."""
    return {
        name: {
            "task_id": t.task_id,
            "difficulty": t.difficulty,
            "description": t.description,
            "max_steps": t.max_steps,
            "seed": t.seed,
        }
        for name, t in TASKS.items()
    }


@app.post("/grade", tags=["grading"])
async def grade(task_name: str = "collision_tagging"):
    """Grade the current episode state for the given task."""
    global _env
    state_dict = _env.state.model_dump()
    try:
        score = run_grader(task_name, state_dict, _env)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    print(f"[GRADE] task={task_name} score={score:.4f}", flush=True)
    return {"task": task_name, "score": score}


# ===========================================================================
# Keep-alive background thread
# Pings the app every 4 minutes to prevent HF Space from sleeping.
# ===========================================================================

def _keep_alive():
    """Background thread: self-ping every 4 minutes."""
    port = int(os.environ.get("PORT", 7860))
    url = f"http://127.0.0.1:{port}/health"
    time.sleep(30)  # Wait for server to fully start before first ping
    while True:
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=5)
            print("[KEEPALIVE] ping OK", flush=True)
        except Exception as e:
            print(f"[KEEPALIVE] ping failed: {e}", flush=True)
        time.sleep(240)  # 4 minutes


# ===========================================================================
# Entrypoint
# ===========================================================================

def main():
    import uvicorn

    # Start keep-alive thread
    t = threading.Thread(target=_keep_alive, daemon=True)
    t.start()

    port = int(os.environ.get("PORT", 7860))
    print(f"[BOOT] Starting uvicorn on 0.0.0.0:{port}", flush=True)
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
