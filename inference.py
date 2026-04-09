"""
Inference Script — Smart City Traffic Surveillance Analyst
==========================================================
MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your HuggingFace / API key.
    IMAGE_NAME          Local Docker image name (if using from_docker_image).
    SPACE_URL           HF Space base URL (e.g. https://user-space.hf.space)
                        Falls back to http://localhost:7860 for local runs.

STDOUT FORMAT (exact — do not alter):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import time
import textwrap
import subprocess
import threading
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Credentials and config — read from env vars, never hard-coded
# ---------------------------------------------------------------------------
IMAGE_NAME    = os.getenv("IMAGE_NAME")                            # local docker image
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "sk-placeholder")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL     = os.getenv("SPACE_URL", "http://localhost:7860").rstrip("/")

BENCHMARK     = "smart-city-traffic-surveillance"
MAX_STEPS     = 20
TEMPERATURE   = 0.0
MAX_TOKENS    = 512
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    "collision_tagging",
    "near_miss_detection",
    "active_incident_management",
]

# ===========================================================================
# Exact stdout log helpers — format is enforced by the validator
# ===========================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()   # must be lowercase: true / false
    # action must be a single-line string (no embedded newlines)
    action_oneline = action.replace("\n", " ")
    print(
        f"[STEP] step={step} action={action_oneline} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ===========================================================================
# Environment HTTP client — talks to the FastAPI server
# ===========================================================================

class TrafficEnvClient:
    """
    Thin HTTP wrapper around the FastAPI environment server.

    Supports two deployment modes:
      1. LOCAL DOCKER  — starts IMAGE_NAME, waits for health, connects to :7860
      2. HF SPACE / local server — connects directly to SPACE_URL
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._container_id: Optional[str] = None
        # Lazy-import requests to avoid top-level import failures
        try:
            import requests as _req
            self._requests = _req
        except ImportError:
            print("[ERROR] 'requests' package is required. pip install requests", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Docker lifecycle (used when IMAGE_NAME is set)
    # ------------------------------------------------------------------

    @classmethod
    def from_docker_image(cls, image_name: str, port: int = 7860) -> "TrafficEnvClient":
        """Start a local Docker container and return a connected client."""
        print(f"[INFO] Starting Docker container from image: {image_name}", flush=True)
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", f"{port}:{port}", image_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] docker run failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        container_id = result.stdout.strip()
        client = cls(f"http://localhost:{port}")
        client._container_id = container_id
        client._wait_for_health(timeout=60)
        return client

    def _wait_for_health(self, timeout: int = 60) -> None:
        """Poll /health until the container is ready."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = self._requests.get(f"{self.base_url}/health", timeout=3)
                if r.status_code == 200:
                    print("[INFO] Environment server is healthy", flush=True)
                    return
            except Exception:
                pass
            time.sleep(2)
        print("[WARN] Health check timed out — proceeding anyway", flush=True)

    def close(self) -> None:
        """Stop the Docker container if we started one."""
        if self._container_id:
            subprocess.run(["docker", "stop", self._container_id], capture_output=True)
            print(f"[INFO] Container {self._container_id[:12]} stopped", flush=True)

    # ------------------------------------------------------------------
    # OpenEnv API methods
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "collision_tagging", seed: Optional[int] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"task_name": task_name}
        if seed is not None:
            payload["seed"] = seed
        r = self._requests.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._requests.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "payload": payload},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = self._requests.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()

    def grade(self, task_name: str) -> float:
        r = self._requests.post(
            f"{self.base_url}/grade",
            params={"task_name": task_name},
            timeout=10,
        )
        r.raise_for_status()
        return float(r.json().get("score", 0.0))


# ===========================================================================
# LLM integration
# ===========================================================================

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI traffic surveillance analyst connected to city intersection cameras.

AVAILABLE ACTIONS (respond with ONE valid JSON object per turn):
  {"action_type": "advance_frame", "payload": {}}
  {"action_type": "tag_collision",  "payload": {"frame_id": <int>, "vehicle_ids": ["V001","V002"]}}
  {"action_type": "tag_near_miss",  "payload": {"vehicle_ids": ["V003"], "minimum_distance": <float_metres>}}
  {"action_type": "dispatch_ems",   "payload": {"coordinates": [<lat>, <lon>], "service_type": "ambulance"}}
  {"action_type": "update_signs",   "payload": {"sign_ids": ["SIGN_N1"], "message": "ACCIDENT AHEAD", "upstream": true}}

RULES:
- Output ONLY raw JSON — no markdown, no extra text.
- Scan frames with advance_frame until you detect an event.
- Collision = bounding boxes overlap AND both vehicles speed drops to 0.
- Near-miss = vehicle enters <10 m radius of another vehicle without colliding.
- For active_incident_management: tag collision, then dispatch_ems, then update_signs.
""").strip()


def _build_prompt(obs: Dict[str, Any], task_name: str, step: int) -> str:
    frame    = obs.get("current_frame", 0)
    total    = obs.get("total_frames", 30)
    tele     = obs.get("telemetry_data", [])
    alerts   = obs.get("active_alerts", [])

    lines = [
        f"TASK: {task_name}  STEP: {step}  FRAME: {frame}/{total - 1}",
        "",
        "=== TELEMETRY ===",
    ]
    for v in tele:
        vid   = v.get("vehicle_id", "?")
        spd   = v.get("speed_kmh", 0)
        coord = v.get("coordinates", [0, 0])
        bbox  = v.get("bbox", [0, 0, 0, 0])
        lines.append(
            f"  {vid}: speed={spd}km/h  gps=({coord[0]:.5f},{coord[1]:.5f})"
            f"  bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
        )

    if alerts:
        lines.append("\n=== ACTIVE ALERTS ===")
        for a in alerts:
            lines.append(f"  [{a.get('alert_type')}] vids={a.get('vehicle_ids')} frame={a.get('frame_id','?')}")

    lines.append("\nOutput your next action JSON:")
    return "\n".join(lines)


def _call_llm(client: Any, messages: List[Dict], model: str) -> str:
    """Call the LLM with exponential backoff. Returns raw string."""
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            wait = 2 ** attempt
            print(f"[DEBUG] LLM call failed (attempt {attempt+1}): {exc}. Retrying in {wait}s", flush=True)
            time.sleep(wait)
    return ""


def _parse_action(raw: str) -> tuple[str, Dict[str, Any]]:
    """Parse LLM output → (action_type, payload). Falls back to advance_frame."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rstrip("`").strip()
    try:
        parsed = json.loads(raw)
        return parsed.get("action_type", "noop"), parsed.get("payload", {})
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                return parsed.get("action_type", "noop"), parsed.get("payload", {})
            except Exception:
                pass
    return "advance_frame", {}


# ===========================================================================
# Single-task episode runner
# ===========================================================================

def run_episode(task_name: str, env_client: TrafficEnvClient, llm_client: Any) -> Dict[str, Any]:
    """Run one full episode. Returns summary dict."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # RESET
        obs = env_client.reset(task_name=task_name)
        done = obs.get("done", False)

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Build prompt and call LLM
            user_msg = _build_prompt(obs, task_name, step)
            conversation.append({"role": "user", "content": user_msg})
            raw_response = _call_llm(llm_client, conversation, MODEL_NAME)

            if not raw_response:
                raw_response = '{"action_type":"advance_frame","payload":{}}'
                last_error = "LLM returned empty response"

            conversation.append({"role": "assistant", "content": raw_response})

            action_type, payload = _parse_action(raw_response)

            # Compact single-line action string for [STEP] log
            action_str = json.dumps({"action_type": action_type, "payload": payload},
                                    separators=(",", ":"))

            # STEP
            try:
                obs = env_client.step(action_type, payload)
                reward = float(obs.get("reward", 0.0))
                done   = bool(obs.get("done", False))
                last_error = None
            except Exception as e:
                reward     = 0.0
                done       = False
                last_error = str(e)
                print(f"[DEBUG] step() error: {e}", flush=True)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

        # Grade via API
        try:
            score = env_client.grade(task_name)
        except Exception as e:
            print(f"[DEBUG] grade() failed: {e}", flush=True)
            score = 0.0

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        last_error = str(e)
        print(f"[DEBUG] Episode exception: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "steps": steps_taken, "success": success}


# ===========================================================================
# Main — runs all tasks sequentially
# ===========================================================================

def main() -> None:
    print("=" * 60, flush=True)
    print("Smart City Traffic Surveillance Analyst — Inference", flush=True)
    print(f"Model: {MODEL_NAME}   Space: {SPACE_URL}", flush=True)
    print("=" * 60, flush=True)

    # Build LLM client (lazy import)
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai package not installed: pip install openai", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Build environment client
    if IMAGE_NAME:
        env_client = TrafficEnvClient.from_docker_image(IMAGE_NAME)
    else:
        # Connect to running HF Space or local server
        env_client = TrafficEnvClient(SPACE_URL)
        env_client._wait_for_health(timeout=30)

    results = []
    try:
        for task_name in TASKS:
            print(f"\n{'─'*60}", flush=True)
            result = run_episode(task_name, env_client, llm_client)
            results.append(result)
            print(f"[INFO] Finished {task_name}: score={result['score']:.2f}", flush=True)

    finally:
        env_client.close()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    total = 0.0
    for r in results:
        total += r["score"]
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['task']:<35} score={r['score']:.2f}", flush=True)
    avg = total / len(results) if results else 0.0
    print(f"\n  Average score: {avg:.2f}", flush=True)

    if avg == 0.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
