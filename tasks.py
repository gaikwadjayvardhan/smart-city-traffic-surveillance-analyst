"""
tasks.py — Task Definitions & Graders
======================================
Defines 3 tasks (Easy / Medium / Hard) with deterministic graders that return
a float in [0.0, 1.0].  All graders are side-effect free and reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from environment import TrafficSurveillanceEnv, _haversine_m


# ===========================================================================
# Task Registry
# ===========================================================================

@dataclass
class TaskSpec:
    """Lightweight task descriptor used by inference.py and the server."""
    name: str
    difficulty: str  # 'easy' | 'medium' | 'hard'
    description: str
    task_id: str
    max_steps: int
    seed: int  # deterministic seed so graders are reproducible


TASKS: Dict[str, TaskSpec] = {
    "collision_tagging": TaskSpec(
        task_id="task_1_easy",
        name="collision_tagging",
        difficulty="easy",
        description=(
            "Analyse a 30-frame video sequence from city intersection cameras. "
            "Identify the exact frame where two vehicles collide (bounding boxes overlap "
            "and both speeds drop to zero). Submit action tag_collision with the correct "
            "frame_id and vehicle_ids."
        ),
        max_steps=40,
        seed=42,
    ),
    "near_miss_detection": TaskSpec(
        task_id="task_2_medium",
        name="near_miss_detection",
        difficulty="medium",
        description=(
            "Analyse vehicle trajectory coordinates across all frames. Identify the "
            "vehicle that breached a 10-metre safety radius around another vehicle "
            "without making physical contact. Submit action tag_near_miss with the "
            "correct vehicle_id and the calculated minimum_distance (in metres). "
            "Tolerance: ±1.5 m."
        ),
        max_steps=50,
        seed=99,
    ),
    "active_incident_management": TaskSpec(
        task_id="task_3_hard",
        name="active_incident_management",
        difficulty="hard",
        description=(
            "A multi-vehicle highway crash has occurred. You must complete three "
            "sequential sub-tasks in any order: "
            "(1) tag_collision with correct frame_id and vehicle_ids, "
            "(2) dispatch_ems to the crash GPS coordinates (within 200 m), "
            "(3) update_signs for at least one upstream sign with a valid warning message. "
            "All three must be correct to achieve full score."
        ),
        max_steps=60,
        seed=777,
    ),
}


# ===========================================================================
# Grader helpers
# ===========================================================================

def _set_match_score(submitted: List[str], ground_truth: List[str]) -> float:
    """Jaccard similarity between two lists of IDs."""
    s, g = set(submitted), set(ground_truth)
    if not g:
        return 0.0
    return len(s & g) / len(s | g)


def _coord_score(submitted: Tuple[float, float], gt: Tuple[float, float], max_dist_m: float = 200.0) -> float:
    """Linear falloff score: 1.0 at 0 m, 0.0 at max_dist_m."""
    dist = _haversine_m(submitted[0], submitted[1], gt[0], gt[1])
    return max(0.0, 1.0 - dist / max_dist_m)


# ===========================================================================
# Task 1 Grader — Collision Tagging (Easy)
# ===========================================================================

def grade_task1(submission: Dict[str, Any], env: TrafficSurveillanceEnv) -> float:
    """
    Grade the collision_tagging task.

    Submission schema (extracted from env.state.tagged_collisions):
        [{
            "frame_id": int,
            "vehicle_ids": ["V001", "V002"],
        }, ...]

    Scoring:
        frame_id correct  → 0.5
        vehicle_ids exact → 0.5
        Total max         → 1.0

    Returns float in [0.0, 1.0].
    """
    gt = env.get_ground_truth()
    gt_frame = gt["collision_frame"]
    gt_vids = sorted(gt["collision_vehicle_ids"])

    tagged = submission.get("tagged_collisions", [])
    if not tagged:
        return 0.0

    best_score = 0.0
    for tag in tagged:
        frame_score = 1.0 if tag.get("frame_id") == gt_frame else 0.0
        submitted_vids = sorted(tag.get("vehicle_ids", []))
        vid_score = _set_match_score(submitted_vids, gt_vids)
        score = 0.5 * frame_score + 0.5 * vid_score
        best_score = max(best_score, score)

    return round(min(1.0, max(0.0, best_score)), 4)


# ===========================================================================
# Task 2 Grader — Near-Miss Detection (Medium)
# ===========================================================================

def grade_task2(submission: Dict[str, Any], env: TrafficSurveillanceEnv) -> float:
    """
    Grade the near_miss_detection task.

    Submission schema (from env.state.tagged_near_misses):
        [{
            "vehicle_ids": ["V003"],
            "details": {"minimum_distance": 5.2},
        }, ...]

    Scoring:
        correct near-miss vehicle_id       → 0.6
        minimum_distance within ±1.5 m     → 0.4
        Total max                          → 1.0

    Returns float in [0.0, 1.0].
    """
    gt = env.get_ground_truth()
    gt_vid = gt["near_miss_vehicle_id"]
    gt_dist = gt["near_miss_min_distance_m"]

    tagged = submission.get("tagged_near_misses", [])
    if not tagged:
        return 0.0

    best_score = 0.0
    for tag in tagged:
        vids = tag.get("vehicle_ids", [])
        vid_score = 0.6 if gt_vid in vids else 0.0

        details = tag.get("details", {})
        submitted_dist = details.get("minimum_distance")
        if submitted_dist is not None:
            dist_err = abs(submitted_dist - gt_dist)
            dist_score = 0.4 if dist_err <= 1.5 else max(0.0, 0.4 * (1.0 - dist_err / 10.0))
        else:
            dist_score = 0.0

        score = vid_score + dist_score
        best_score = max(best_score, score)

    return round(min(1.0, max(0.0, best_score)), 4)


# ===========================================================================
# Task 3 Grader — Active Incident Management (Hard)
# ===========================================================================

def grade_task3(submission: Dict[str, Any], env: TrafficSurveillanceEnv) -> float:
    """
    Grade the active_incident_management task.

    All three sub-tasks are evaluated independently and averaged:

    Sub-task A (tag_collision):
        Same rubric as Task 1.

    Sub-task B (dispatch_ems):
        Coordinates within 200 m of collision point → 1.0
        Linear falloff up to 500 m                  → partial
        service_type in {ambulance, fire, ems}       → +0.2 bonus (capped at 1.0)

    Sub-task C (update_signs):
        At least 1 correct upstream sign_id          → 0.5
        Non-empty warning message                    → 0.3
        upstream=True flag set                       → 0.2

    Final score = (score_A + score_B + score_C) / 3.0

    Returns float in [0.0, 1.0].
    """
    gt = env.get_ground_truth()

    # --- Sub-task A: collision tag ---
    score_a = grade_task1(submission, env)

    # --- Sub-task B: EMS dispatch ---
    dispatched = submission.get("dispatched_ems", [])
    score_b = 0.0
    if dispatched:
        gt_coords = gt["collision_coordinates"]
        best_b = 0.0
        for dispatch in dispatched:
            details = dispatch.get("details", {})
            raw_coords = dispatch.get("coordinates") or details.get("coordinates")
            if raw_coords and len(raw_coords) == 2:
                c_score = _coord_score(tuple(raw_coords), gt_coords, max_dist_m=500.0)
                service = details.get("service_type", "")
                s_bonus = 0.2 if service in ("ambulance", "fire", "ems", "police") else 0.0
                best_b = max(best_b, min(1.0, c_score + s_bonus))
        score_b = best_b

    # --- Sub-task C: sign update ---
    updated_signs = submission.get("updated_signs", [])
    score_c = 0.0
    if updated_signs:
        gt_signs = set(gt.get("sign_ids", []))
        best_c = 0.0
        for sign_entry in updated_signs:
            details = sign_entry.get("details", {})
            submitted_sign_ids = set(details.get("sign_ids", []))
            message = details.get("message", "")
            upstream = details.get("upstream", False)

            sign_ok = 0.5 if (submitted_sign_ids & gt_signs) else 0.0
            msg_ok = 0.3 if len(message) > 3 else 0.0
            up_ok = 0.2 if upstream is True else 0.0
            best_c = max(best_c, sign_ok + msg_ok + up_ok)
        score_c = best_c

    final = (score_a + score_b + score_c) / 3.0
    return round(min(1.0, max(0.0, final)), 4)


# ===========================================================================
# Public dispatch table
# ===========================================================================

GRADERS = {
    "collision_tagging": grade_task1,
    "near_miss_detection": grade_task2,
    "active_incident_management": grade_task3,
}


def run_grader(task_name: str, submission: Dict[str, Any], env: TrafficSurveillanceEnv) -> float:
    """
    Dispatch to the correct grader.

    Args:
        task_name:  Must be a key in GRADERS.
        submission: env.state model_dump() or equivalent dict.
        env:        The environment instance (used to fetch ground truth).

    Returns:
        float in [0.0, 1.0]
    """
    if task_name not in GRADERS:
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {list(GRADERS.keys())}")
    return GRADERS[task_name](submission, env)
