#!/usr/bin/env python3
"""Candidate viewpoint and information-gain smoke for fallback BEV map."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


RECTS = [
    ("wall_north", -6.0, 6.0, 3.92, 4.08),
    ("wall_south", -6.0, 6.0, -4.08, -3.92),
    ("wall_west", -6.08, -5.92, -4.0, 4.0),
    ("wall_east", 5.92, 6.08, -4.0, 4.0),
    ("divider_a", -2.06, -1.94, -2.6, 2.6),
    ("divider_b", 1.94, 2.06, -2.6, 2.6),
    ("table", -0.7, 0.7, 0.95, 1.85),
    ("box", 2.95, 3.85, -1.65, -0.75),
]
PATH = [(-4.5, -2.8), (-3.2, -2.2), (-1.0, -1.5), (1.0, -0.8), (3.0, -0.2)]


def occupied_at(x: float, y: float) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for _, x0, x1, y0, y1 in RECTS)


def line_blocked(x0: float, y0: float, x1: float, y1: float, resolution: float) -> bool:
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist / (resolution * 0.5)))
    for i in range(1, steps):
        t = i / steps
        if occupied_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t):
            return True
    return False


def build_partial_map(args):
    xs = [round(args.x_min + i * args.resolution, 6) for i in range(int(round((args.x_max - args.x_min) / args.resolution)) + 1)]
    ys = [round(args.y_min + i * args.resolution, 6) for i in range(int(round((args.y_max - args.y_min) / args.resolution)) + 1)]
    cells = {}
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            cells[(ix, iy)] = {"x": x, "y": y, "state": "unknown", "observed_count": 0}
    for px, py in PATH:
        for cell in cells.values():
            x, y = cell["x"], cell["y"]
            if math.hypot(x - px, y - py) <= args.sensor_radius and not line_blocked(px, py, x, y, args.resolution):
                cell["observed_count"] += 1
                cell["state"] = "occupied" if occupied_at(x, y) else "known_free"
    return cells


def visible_unknown_count(cells, cx, cy, args):
    count = 0
    for cell in cells.values():
        if cell["state"] != "unknown":
            continue
        x, y = cell["x"], cell["y"]
        if math.hypot(x - cx, y - cy) <= args.sensor_radius and not line_blocked(cx, cy, x, y, args.resolution):
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/phase3_candidate_smoke")
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--x-min", type=float, default=-6.0)
    parser.add_argument("--x-max", type=float, default=6.0)
    parser.add_argument("--y-min", type=float, default=-4.0)
    parser.add_argument("--y-max", type=float, default=4.0)
    parser.add_argument("--sensor-radius", type=float, default=2.8)
    parser.add_argument("--candidate-radius", type=float, default=2.0)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = build_partial_map(args)
    sx, sy = PATH[-1]
    candidates = []
    for i in range(args.candidate_count):
        theta = 2.0 * math.pi * i / args.candidate_count
        cx = sx + args.candidate_radius * math.cos(theta)
        cy = sy + args.candidate_radius * math.sin(theta)
        in_bounds = args.x_min <= cx <= args.x_max and args.y_min <= cy <= args.y_max
        collision = occupied_at(cx, cy)
        path_blocked = line_blocked(sx, sy, cx, cy, args.resolution)
        valid = in_bounds and not collision and not path_blocked
        gain = visible_unknown_count(cells, cx, cy, args) if valid else 0
        cost = math.hypot(cx - sx, cy - sy)
        score = gain - args.alpha * cost if valid else -9999.0
        yaw = math.atan2(cy - sy, cx - sx)
        candidates.append(
            {
                "candidate_id": i,
                "x": round(cx, 4),
                "y": round(cy, 4),
                "yaw": round(yaw, 4),
                "valid": valid,
                "in_bounds": in_bounds,
                "collision": collision,
                "path_blocked": path_blocked,
                "information_gain": gain,
                "path_cost": round(cost, 4),
                "score": round(score, 4),
            }
        )
    valid_candidates = [c for c in candidates if c["valid"]]
    ranked = sorted(valid_candidates, key=lambda c: c["score"], reverse=True)
    summary = {
        "start_pose": {"x": sx, "y": sy},
        "candidate_count": len(candidates),
        "valid_count": len(valid_candidates),
        "invalid_count": len(candidates) - len(valid_candidates),
        "best_candidate": ranked[0] if ranked else None,
        "top5_candidates": ranked[:5],
        "alpha": args.alpha,
        "sensor_radius": args.sensor_radius,
        "candidate_radius": args.candidate_radius,
        "negative_scope": {"training": False, "rl": False, "checkpoint": False, "rollout": False},
    }
    with (out_dir / "candidate_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)
    (out_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ranked and ranked[0]["information_gain"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
