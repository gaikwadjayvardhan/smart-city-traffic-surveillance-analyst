"""
inference.py — Baseline Agent Evaluation Script
================================================
Runs a GPT-compatible LLM agent against all 3 tasks using the official
OpenAI Python client.  Emits structured stdout logs in the exact
[START] / [STEP] / [END] format required by the OpenEnv automated validator.

Runtime budget: <20 minutes on 2vCPU / 8 GB RAM.
Credentials:    Read from environment variables (never hard-coded).

Log format contract (do NOT alter):
  [START] task=<name> episode=<id>
  [STEP] step=<n> action=<json> reward=<f> done=<bool>
  [END] task=<name> episode=<id> score=<f> steps=<n>
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Lazy imports — only fail at call site, not at module import
# ---------------------------------------------------------------------------
def _get_openai_client():
    """Return an openai.OpenAI client configured from environment variables."""
    try:
        import openai  # type: ignore
    except ImportError as e:
        print(f"[ERROR] openai package not installed: {e}", file=sys.stderr)
        sys.exit(1)

    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "sk-placeholder")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    client = openai.OpenAI(base_url=api_base, api_key=api_key)
    return client, model_name


# ---------------------------------------------------------------------------
# Environment & task imports (lazy to avoid import-time failures)
# ---------------------------------------------------------------------------
def _import_env():
    from environment import TrafficSurveillanceEnv, TrafficAction
    from tasks import TASKS, run_grader
    return TrafficSurveillanceEnv, TrafficAction, TASKS, run_grader


# ===========================================================================
# Structured Logger — validator-compatible format
# ===========================================================================

class OpenEnvLogger:
    """
    Emits structured stdout logs in the exact format expected by the
    OpenEnv automated validator:

        [START] task=<name> episode=<id>
        [STEP]  step=<n> action=<json_str> reward=<float> done=<bool>
        [END]   task=<name> episode=<id> score=<float> steps=<int>
    """

    @staticmethod
    def start(task: str, episode: str) -> None:
        print(f"[START] task={task} episode={episode}", flush=True)

    @staticmethod
    def step(step: int, action: Dict, reward: float, done: bool) -> None:
        action_str = json.dumps(action, separators=(",", ":"))
        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward:.4f} done={done}",
            flush=True,
        )

    @staticmethod
    def end(task: str, episode: str, score: float, steps: int) -> None:
        print(
            f"[END] task={task} episode={episode} "
            f"score={score:.4f} steps={steps}",
            flush=True,
        )


log = OpenEnvLogger()


# ===========================================================================
# Observation → Prompt Builder
# ===========================================================================

SYSTEM_PROMPT = """You are an AI traffic surveillance analyst. You receive real-time telemetry data from city traffic cameras.

Your available actions are:
1. tag_collision    — {"action_type": "tag_collision", "payload": {"frame_id": <int>, "vehicle_ids": ["Vxxx", "Vyyy"]}}
2. tag_near_miss    — {"action_type": "tag_near_miss", "payload": {"vehicle_ids": ["Vxxx"], "minimum_distance": <float_metres>}}
3. dispatch_ems     — {"action_type": "dispatch_ems", "payload": {"coordinates": [lat, lon], "service_type": "ambulance"}}
4. update_signs     — {"action_type": "update_signs", "payload": {"sign_ids": ["SIGN_N1"], "message": "ACCIDENT AHEAD - SLOW DOWN", "upstream": true}}
5. advance_frame    — {"action_type": "advance_frame", "payload": {}}
6. noop             — {"action_type": "noop", "payload": {}}

Rules:
- Always respond with a valid JSON object matching EXACTLY one of the above schemas.
- Do NOT add markdown fences or extra text — output raw JSON only.
- Advance frames to scan for collisions. When you detect an event, submit the appropriate action immediately.
- For dispatch_ems, use the GPS coordinates of the collision/incident.
"""

def _build_user_prompt(obs_dict: Dict, task_description: str, step: int) -> str:
    """Convert current observation to a compact, token-efficient prompt string."""
    frame = obs_dict.get("current_frame", 0)
    total = obs_dict.get("total_frames", 30)
    telemetry = obs_dict.get("telemetry_data", [])
    alerts = obs_dict.get("active_alerts", [])
    task = obs_dict.get("task_name", "unknown")

    lines = [
        f"TASK: {task}",
        f"DESCRIPTION: {task_description}",
        f"STEP: {step}  FRAME: {frame}/{total - 1}",
        "",
        "=== TELEMETRY (current frame) ===",
    ]
    for v in telemetry:
        vid = v.get("vehicle_id", "?")
        spd = v.get("speed_kmh", 0)
        coords = v.get("coordinates", [0, 0])
        bbox = v.get("bbox", [0, 0, 0, 0])
        lines.append(
            f"  {vid}: speed={spd} km/h  gps=({coords[0]:.5f},{coords[1]:.5f})"
            f"  bbox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}]"
        )

    if alerts:
        lines.append("")
        lines.append("=== ACTIVE ALERTS ===")
        for a in alerts:
            lines.append(f"  [{a.get('alert_type','')}] vehicles={a.get('vehicle_ids',[])} frame={a.get('frame_id','N/A')}")

    lines.append("")
    lines.append("Respond with the next action JSON:")
    return "\n".join(lines)


# ===========================================================================
# LLM → Action converter
# ===========================================================================

def _parse_llm_response(raw: str) -> Dict[str, Any]:
    """
    Parse the LLM's raw text output into an action dict.
    Falls back to noop if parsing fails.
    """
    raw = raw.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        parsed = json.loads(raw)
        if "action_type" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Heuristic extraction for common formats
    import re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    return {"action_type": "noop", "payload": {}}


# ===========================================================================
# Single-episode agent loop
# ===========================================================================

def run_episode(
    task_name: str,
    max_steps: int,
    seed: int,
    client: Any,
    model_name: str,
) -> Dict[str, Any]:
    """
    Run one full episode of the agent loop.

    Returns dict with: task, episode_id, score, steps, conversation_history
    """
    TrafficSurveillanceEnv, TrafficAction, TASKS, run_grader = _import_env()

    task_spec = TASKS[task_name]
    env = TrafficSurveillanceEnv()

    # Reset with deterministic seed for reproducibility
    obs = env.reset(seed=seed, task_name=task_name)
    episode_id = obs.episode_id or f"ep_{seed}"

    log.start(task=task_name, episode=episode_id)

    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    step = 0
    done = False
    last_reward = 0.0

    while step < max_steps and not done:
        step += 1

        # Build prompt from current observation
        obs_dict = obs.model_dump()
        user_msg = _build_user_prompt(obs_dict, task_spec.description, step)
        conversation.append({"role": "user", "content": user_msg})

        # Query LLM (with exponential backoff on rate limits)
        raw_response = ""
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=conversation,
                    max_tokens=256,
                    temperature=0.0,
                    timeout=30,
                )
                raw_response = completion.choices[0].message.content or ""
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"[WARN] LLM call failed (attempt {attempt+1}): {e}. Retrying in {wait}s…", file=sys.stderr)
                time.sleep(wait)
                raw_response = ""

        if not raw_response:
            raw_response = '{"action_type": "noop", "payload": {}}'

        # Update conversation with assistant response
        conversation.append({"role": "assistant", "content": raw_response})

        # Parse and validate action
        action_dict = _parse_llm_response(raw_response)
        try:
            action = TrafficAction(**action_dict)
        except Exception:
            action = TrafficAction(action_type="noop", payload={})

        # Step the environment
        obs = env.step(action)
        last_reward = obs.reward
        done = obs.done

        log.step(step=step, action=action_dict, reward=last_reward, done=done)

    # Grade the completed episode
    state_dict = env.state.model_dump()
    score = run_grader(task_name=task_name, submission=state_dict, env=env)

    log.end(task=task_name, episode=episode_id, score=score, steps=step)

    return {
        "task": task_name,
        "episode_id": episode_id,
        "score": score,
        "steps": step,
        "cumulative_reward": state_dict.get("cumulative_reward", 0.0),
    }


# ===========================================================================
# Main entrypoint
# ===========================================================================

def main():
    """
    Run inference across all tasks and print a summary.
    Exits with code 1 if any task scores 0.0 (for CI pipelines).
    """
    print("=" * 60, flush=True)
    print("Smart City Traffic Surveillance Analyst — Inference", flush=True)
    print("=" * 60, flush=True)

    client, model_name = _get_openai_client()
    print(f"[INFO] Using model: {model_name}", flush=True)

    _, _, TASKS, _ = _import_env()

    results = []
    for task_name, task_spec in TASKS.items():
        print(f"\n[INFO] Starting task: {task_name} (seed={task_spec.seed})", flush=True)
        try:
            result = run_episode(
                task_name=task_name,
                max_steps=task_spec.max_steps,
                seed=task_spec.seed,
                client=client,
                model_name=model_name,
            )
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Task {task_name} failed: {e}", file=sys.stderr)
            results.append({"task": task_name, "score": 0.0, "steps": 0})

    # Summary table
    print("\n" + "=" * 60, flush=True)
    print("INFERENCE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    total_score = 0.0
    for r in results:
        score = r.get("score", 0.0)
        total_score += score
        print(f"  {r['task']:35s}  score={score:.4f}  steps={r.get('steps', 0)}", flush=True)

    avg_score = total_score / len(results) if results else 0.0
    print(f"\n  Average score: {avg_score:.4f}", flush=True)
    print("=" * 60, flush=True)

    # Non-zero exit if all tasks failed (useful for CI)
    if avg_score == 0.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
