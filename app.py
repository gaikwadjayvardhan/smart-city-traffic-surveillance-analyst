"""
app.py — FastAPI server (restructured to match passing repo pattern)
=====================================================================
Key fixes vs previous versions:
  - /healthz endpoint (matches openenv.yaml endpoints spec)
  - /grader endpoint (matches openenv.yaml endpoints spec — was /grade)
  - Rewards clamped to (0.01, 0.989) — never 0.0 or 1.0 (Phase 2 compliance)
  - _env initialized at module load (never None on first request)
  - All endpoints return JSON — no HTML errors
"""

from __future__ import annotations

import os
import sys
import time
import threading
import traceback
from typing import Any, Dict, Optional

print("[BOOT] Starting app.py imports...", flush=True)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    print("[BOOT] FastAPI imported OK", flush=True)
except Exception as e:
    print(f"[FATAL] FastAPI import failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

try:
    from environment import TrafficSurveillanceEnv, TrafficAction, TrafficObservation
    print("[BOOT] environment.py imported OK", flush=True)
except Exception as e:
    print(f"[FATAL] environment.py import failed: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    from tasks import TASKS, run_grader
    print("[BOOT] tasks.py imported OK", flush=True)
except Exception as e:
    print(f"[FATAL] tasks.py import failed: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

# ===========================================================================
# Reward safety — clamp to open interval (0.01, 0.989)
# Phase 2 validation rejects rewards of exactly 0.0 or 1.0
# ===========================================================================
_SCORE_MIN = 0.01
_SCORE_MAX = 0.989


def _safe_score(raw: float) -> float:
    try:
        val = float(raw)
        return max(_SCORE_MIN, min(_SCORE_MAX, val))
    except (ValueError, TypeError):
        return _SCORE_MIN


# ===========================================================================
# Global env instance — initialized at module load (never None)
# ===========================================================================
print("[BOOT] Initializing TrafficSurveillanceEnv...", flush=True)
_env: TrafficSurveillanceEnv = TrafficSurveillanceEnv()
_env.reset(seed=42, task_name="collision_tagging")
print("[BOOT] Environment ready", flush=True)


# ===========================================================================
# FastAPI app
# ===========================================================================
app = FastAPI(
    title="Smart City Traffic Surveillance Analyst",
    description="OpenEnv-compliant traffic incident detection and response environment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Request schemas — all fields optional so {} body always works
# ===========================================================================

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: Optional[str] = "collision_tagging"
    task_id: Optional[str] = None  # alias accepted from validators
    model_config = {"extra": "ignore"}


class StepRequest(BaseModel):
    action_type: str = "noop"
    payload: Dict[str, Any] = {}
    model_config = {"extra": "ignore"}


# ===========================================================================
# Global exception handler — always JSON, never HTML
# ===========================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[ERROR] {request.url}: {exc}", flush=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


# ===========================================================================
# Health endpoints
# /healthz  — per openenv.yaml spec (passing repo uses this path)
# /health   — secondary alias
# /         — HF Space root ping
# ===========================================================================

@app.get("/healthz", tags=["health"])
async def healthz():
    """Primary health check — required by openenv.yaml endpoints spec."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/", tags=["health"])
async def root():
    return {
        "status": "ok",
        "environment": "Smart City Traffic Surveillance Analyst",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


# ===========================================================================
# Core OpenEnv endpoints
# ===========================================================================

@app.post("/reset", tags=["environment"])
async def reset(req: ResetRequest = ResetRequest()):
    """Reset environment. Accepts empty body {}."""
    global _env
    task = req.task_id or req.task_name or "collision_tagging"
    try:
        obs = _env.reset(seed=req.seed, episode_id=req.episode_id, task_name=task)
        result = obs.model_dump()
        # Clamp reward to safe range
        result["reward"] = _safe_score(result.get("reward", _SCORE_MIN))
        print(f"[RESET] task={task} frame={obs.current_frame}", flush=True)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", tags=["environment"])
async def step(req: StepRequest):
    """Submit one action."""
    global _env
    try:
        action = TrafficAction(action_type=req.action_type, payload=req.payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
    try:
        obs = _env.step(action)
        result = obs.model_dump()
        result["reward"] = _safe_score(result.get("reward", _SCORE_MIN))
        print(f"[STEP] action={req.action_type} reward={result['reward']:.3f} done={obs.done}", flush=True)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", tags=["environment"])
async def state():
    """Return current internal state."""
    global _env
    try:
        return _env.state.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/grader", tags=["grading"])
@app.post("/grader", tags=["grading"])
async def grader(task_name: str = "collision_tagging", task_id: Optional[str] = None):
    """Grade the current episode (GET or POST). Matches openenv.yaml endpoints spec."""
    global _env
    task = task_id or task_name
    state_dict = _env.state.model_dump()
    try:
        raw_score = run_grader(task, state_dict, _env)
        score = _safe_score(raw_score)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    print(f"[GRADE] task={task} raw={raw_score:.4f} clamped={score:.4f}", flush=True)
    return {"task": task, "score": score, "reward": score}


@app.get("/tasks", tags=["metadata"])
async def list_tasks():
    """List all registered tasks."""
    return {
        "tasks": [
            {
                "task_id": name,
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "max_steps": t.max_steps,
            }
            for name, t in TASKS.items()
        ]
    }


# ===========================================================================
# Keep-alive — ping self every 4 min to prevent HF Space sleeping
# ===========================================================================

def _keep_alive():
    port = int(os.environ.get("PORT", 7860))
    url = f"http://127.0.0.1:{port}/healthz"
    time.sleep(40)
    while True:
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=5)
            print("[KEEPALIVE] ping OK", flush=True)
        except Exception as e:
            print(f"[KEEPALIVE] ping failed: {e}", flush=True)
        time.sleep(240)


# ===========================================================================
# Entrypoint
# ===========================================================================

def make_app():
    """Factory function for openenv-core compatibility."""
    return app


def main():
    import uvicorn
    t = threading.Thread(target=_keep_alive, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 7860))
    print(f"[BOOT] Serving on 0.0.0.0:{port}", flush=True)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
