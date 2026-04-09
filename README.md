---
title: Smart City Traffic Surveillance Analyst
emoji: 🚦
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
license: mit
short_description: OpenEnv RL environment for traffic incident detection & response
tags:
  - openenv
  - reinforcement-learning
  - traffic
  - fastapi
  - docker
pinned: false
---

# Smart City Traffic Surveillance Analyst

> An **OpenEnv**-compliant reinforcement-learning environment for automated traffic incident detection, emergency dispatch, and traffic management.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-0.2.3-blue)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.11-brightgreen)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)

---

## Overview

The **Smart City Traffic Surveillance Analyst** environment simulates a real-world AI-assisted traffic control centre. An agent receives structured JSON telemetry from simulated intersection cameras — vehicle bounding boxes, GPS coordinates, and speeds — and must issue API-style actions to:

- 🚗 **Detect collisions** by parsing frame-by-frame telemetry
- ⚠️ **Identify near-miss events** by computing vehicle proximity trajectories
- 🚑 **Dispatch emergency services** to crash coordinates
- 🪧 **Update upstream warning signs** to manage traffic flow

---

## File Structure

```
traffic-accident-openenv/
├── environment.py       # Core OpenEnv environment (reset, step, state)
├── tasks.py             # Task definitions + deterministic graders (Easy/Medium/Hard)
├── inference.py         # Baseline LLM agent evaluation script
├── app.py               # FastAPI server (port 7860, HF Spaces compatible)
├── openenv.yaml         # OpenEnv metadata configuration
├── Dockerfile           # Container build for HuggingFace Spaces deployment
├── requirements.txt     # Pinned Python dependencies
└── README.md            # This file
```

---

## Environment Interface

### `reset(seed, task_name, episode_id)` → `TrafficObservation`
Resets all state and generates a fresh deterministic scenario.

### `step(action)` → `TrafficObservation`
Applies an action and returns an observation with `.reward` (dense) and `.done`.

### `env.state` (property) → `TrafficState`
Returns a snapshot of internal state — no parentheses needed.

---

## Tasks

| # | Name | Difficulty | Description |
|---|------|-----------|-------------|
| 1 | `collision_tagging` | 🟢 Easy | Find the exact frame where two vehicles collide |
| 2 | `near_miss_detection` | 🟡 Medium | Identify a vehicle that breached a 10m safety radius |
| 3 | `active_incident_management` | 🔴 Hard | Sequence: tag collision → dispatch EMS → update signs |

---

## Action Space

| `action_type` | Required `payload` keys |
|---|---|
| `tag_collision` | `frame_id: int`, `vehicle_ids: List[str]` |
| `tag_near_miss` | `vehicle_ids: List[str]`, `minimum_distance: float` |
| `dispatch_ems` | `coordinates: [lat, lon]`, `service_type: str` |
| `update_signs` | `sign_ids: List[str]`, `message: str`, `upstream: bool` |
| `advance_frame` | _(empty payload)_ |
| `noop` | _(empty payload — mildly penalised)_ |

---

## Reward Signal

Dense and continuous — never purely sparse:

| Outcome | Reward |
|---------|--------|
| Correct action + correct payload | `+1.0` |
| Correct type, partial payload | `+0.2 … +0.5` |
| Wrong frame / wrong IDs | `-0.1` |
| Hallucinated vehicle IDs | `-0.3` |
| Duplicate correct action | `-0.5` |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server locally
```bash
python app.py
# Server starts at http://localhost:7860
```

### 3. Run the baseline inference agent
```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
# Optionally override API base for local models:
# export API_BASE_URL="http://localhost:11434/v1"

python inference.py
```

### 4. Build and run via Docker
```bash
docker build -t traffic-env .
docker run -p 7860:7860 \
    -e OPENAI_API_KEY=sk-... \
    -e MODEL_NAME=gpt-4o-mini \
    traffic-env
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check (returns 200 OK) |
| `GET` | `/health` | Secondary health check |
| `POST` | `/reset` | Reset environment |
| `POST` | `/step` | Submit action |
| `GET` | `/state` | Get current state |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/grade?task_name=...` | Grade current episode |

### Example: Reset
```bash
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task_name": "collision_tagging", "seed": 42}'
```

### Example: Step
```bash
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action_type": "advance_frame", "payload": {}}'
```

---

## OpenEnv Compliance Checklist

| Requirement | Status |
|-------------|--------|
| `reset()` returns `Observation` with `.done` and `.reward` | ✅ |
| `step()` returns `Observation` with `.done` and `.reward` | ✅ |
| `state` is a `@property` (no parentheses) | ✅ |
| Pydantic v2 models with strict validation | ✅ |
| Dense (non-sparse) reward signal | ✅ |
| Graders return `float` in `[0.0, 1.0]` | ✅ |
| `[START]` / `[STEP]` / `[END]` log format in inference.py | ✅ |
| FastAPI server on port 7860 | ✅ |
| `openenv.yaml` with `api_version` and `transport` | ✅ |
| Docker health check + EXPOSE 7860 | ✅ |
| Credentials from env vars (never hard-coded) | ✅ |
| Inference runs in <20 min on 2vCPU / 8GB | ✅ |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (inference) | OpenAI API key |
| `API_BASE_URL` | No | Custom API base (e.g. local Ollama) |
| `MODEL_NAME` | No | Model name (default: `gpt-4o-mini`) |
| `HF_TOKEN` | No | HuggingFace token (alternative to OPENAI_API_KEY) |
| `PORT` | No | Server port (default: `7860`) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
