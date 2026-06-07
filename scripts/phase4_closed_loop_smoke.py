#!/usr/bin/env python3
"""Bounded closed-loop selector smoke on fallback BEV map."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

RECTS = [
    ("wall_north", -6.0, 6.0, 3.92, 4.08), ("wall_south", -6.0, 6.0, -4.08, -3.92),
    ("wall_west", -6.08, -5.92, -4.0, 4.0), ("wall_east", 5.92, 6.08, -4.0, 4.0),
    ("divider_a", -2.06, -1.94, -2.6, 2.6), ("divider_b", 1.94, 2.06, -2.6, 2.6),
    ("table", -0.7, 0.7, 0.95, 1.85), ("box", 2.95, 3.85, -1.65, -0.75),
]


def occupied_at(x, y):
    return any(x0 <= x <= x1 and y0 <= y <= y1 for _, x0, x1, y0, y1 in RECTS)


def line_blocked(x0, y0, x1, y1, resolution):
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist / (resolution * 0.5)))
    for i in range(1, steps):
        t = i / steps
        if occupied_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
            return True
    return False


def make_grid(args):
    xs = [round(args.x_min + i * args.resolution, 6) for i in range(int(round((args.x_max - args.x_min) / args.resolution)) + 1)]
    ys = [round(args.y_min + i * args.resolution, 6) for i in range(int(round((args.y_max - args.y_min) / args.resolution)) + 1)]
    return {(ix, iy): {"x": x, "y": y, "state": "unknown", "observed_count": 0} for iy, y in enumerate(ys) for ix, x in enumerate(xs)}


def observe(cells, pose, args):
    px, py = pose
    newly_known = 0
    for cell in cells.values():
        x, y = cell["x"], cell["y"]
        if math.hypot(x - px, y - py) <= args.sensor_radius and not line_blocked(px, py, x, y, args.resolution):
            old = cell["state"]
            cell["observed_count"] += 1
            cell["state"] = "occupied" if occupied_at(x, y) else "known_free"
            if old == "unknown":
                newly_known += 1
    return newly_known


def visible_unknown_count(cells, pose, args):
    px, py = pose
    return sum(
        1 for cell in cells.values()
        if cell["state"] == "unknown"
        and math.hypot(cell["x"] - px, cell["y"] - py) <= args.sensor_radius
        and not line_blocked(px, py, cell["x"], cell["y"], args.resolution)
    )


def counts(cells):
    out = {"known_free": 0, "occupied": 0, "unknown": 0}
    for cell in cells.values():
        out[cell["state"]] += 1
    out["known_ratio"] = (out["known_free"] + out["occupied"]) / len(cells)
    return out


def sample_candidates(pose, cells, args):
    sx, sy = pose
    out = []
    for i in range(args.candidate_count):
        theta = 2 * math.pi * i / args.candidate_count
        cx = sx + args.candidate_radius * math.cos(theta)
        cy = sy + args.candidate_radius * math.sin(theta)
        in_bounds = args.x_min <= cx <= args.x_max and args.y_min <= cy <= args.y_max
        collision = occupied_at(cx, cy)
        path_blocked = line_blocked(sx, sy, cx, cy, args.resolution)
        valid = in_bounds and not collision and not path_blocked
        gain = visible_unknown_count(cells, (cx, cy), args) if valid else 0
        cost = math.hypot(cx - sx, cy - sy)
        score = gain - args.alpha * cost if valid else -9999.0
        out.append({"candidate_id": i, "x": round(cx, 4), "y": round(cy, 4), "yaw": round(theta, 4), "valid": valid, "information_gain": gain, "path_cost": round(cost, 4), "score": round(score, 4)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/phase4_closed_loop_smoke")
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--x-min", type=float, default=-6.0)
    parser.add_argument("--x-max", type=float, default=6.0)
    parser.add_argument("--y-min", type=float, default=-4.0)
    parser.add_argument("--y-max", type=float, default=4.0)
    parser.add_argument("--sensor-radius", type=float, default=2.8)
    parser.add_argument("--candidate-radius", type=float, default=1.5)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max-actions", type=int, default=8)
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cells = make_grid(args)
    pose = (-4.5, -2.8)
    steps = []
    observe(cells, pose, args)
    for step_id in range(args.max_actions):
        candidates = sample_candidates(pose, cells, args)
        ranked = sorted([c for c in candidates if c["valid"]], key=lambda c: c["score"], reverse=True)
        if not ranked or ranked[0]["information_gain"] <= 0:
            steps.append({"step_id": step_id, "pose_x": pose[0], "pose_y": pose[1], "failure": "no_positive_gain_candidate", **counts(cells)})
            break
        chosen = ranked[0]
        next_pose = (chosen["x"], chosen["y"])
        newly_known = observe(cells, next_pose, args)
        c = counts(cells)
        steps.append({"step_id": step_id, "pose_x": pose[0], "pose_y": pose[1], "selected_x": next_pose[0], "selected_y": next_pose[1], "selected_candidate_id": chosen["candidate_id"], "information_gain": chosen["information_gain"], "newly_known_cells": newly_known, "score": chosen["score"], "failure": "", **c})
        pose = next_pose
    summary = {"action_count": len([s for s in steps if not s.get("failure")]), "step_count": len(steps), "final_counts": counts(cells), "final_pose": {"x": pose[0], "y": pose[1]}, "coverage_increased": steps[-1]["known_ratio"] > steps[0]["known_ratio"] if len(steps) > 1 else False, "negative_scope": {"training": False, "rl": False, "checkpoint": False, "long_rollout": False}}
    with (out_dir / "closed_loop_steps.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for s in steps for k in s.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(steps)
    (out_dir / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["action_count"] >= 3 and summary["coverage_increased"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
