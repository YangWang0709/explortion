#!/usr/bin/env python3
"""Lightweight BEV partial-map smoke for the fallback indoor scene.

This deterministic smoke establishes the data contract for known_free,
occupied, unknown, observed_count, and frontier cells. It does not train,
rollout, checkpoint, or run RL.
"""

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

PATH = [
    (-4.5, -2.8),
    (-3.2, -2.2),
    (-1.0, -1.5),
    (1.0, -0.8),
    (3.0, -0.2),
]


def occupied_at(x: float, y: float) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for _, x0, x1, y0, y1 in RECTS)


def line_blocked(x0: float, y0: float, x1: float, y1: float, resolution: float) -> bool:
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist / (resolution * 0.5)))
    for i in range(1, steps):
        t = i / steps
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        if occupied_at(x, y):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/phase2_mapping_smoke")
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--x-min", type=float, default=-6.0)
    parser.add_argument("--x-max", type=float, default=6.0)
    parser.add_argument("--y-min", type=float, default=-4.0)
    parser.add_argument("--y-max", type=float, default=4.0)
    parser.add_argument("--sensor-radius", type=float, default=2.8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = []
    x = args.x_min
    while x <= args.x_max + 1e-9:
        xs.append(round(x, 6))
        x += args.resolution
    ys = []
    y = args.y_min
    while y <= args.y_max + 1e-9:
        ys.append(round(y, 6))
        y += args.resolution

    cells = {}
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            cells[(ix, iy)] = {
                "x": x,
                "y": y,
                "state": "unknown",
                "observed_count": 0,
                "is_frontier": False,
            }

    for pose_id, (px, py) in enumerate(PATH):
        for (ix, iy), cell in cells.items():
            x, y = cell["x"], cell["y"]
            d = math.hypot(x - px, y - py)
            if d <= args.sensor_radius and not line_blocked(px, py, x, y, args.resolution):
                cell["observed_count"] += 1
                cell["state"] = "occupied" if occupied_at(x, y) else "known_free"
                cell.setdefault("first_seen_pose_id", pose_id)

    for (ix, iy), cell in cells.items():
        if cell["state"] != "known_free":
            continue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neigh = cells.get((ix + dx, iy + dy))
            if neigh and neigh["state"] == "unknown":
                cell["is_frontier"] = True
                break

    counts = {"known_free": 0, "occupied": 0, "unknown": 0, "frontier": 0}
    for cell in cells.values():
        counts[cell["state"]] += 1
        if cell["is_frontier"]:
            counts["frontier"] += 1

    total = len(cells)
    summary = {
        "resolution": args.resolution,
        "grid_width": len(xs),
        "grid_height": len(ys),
        "total_cells": total,
        "path_pose_count": len(PATH),
        "sensor_radius": args.sensor_radius,
        "counts": counts,
        "known_ratio": (counts["known_free"] + counts["occupied"]) / total,
        "negative_scope": {
            "training": False,
            "rl": False,
            "checkpoint": False,
            "rollout": False,
        },
    }

    with (out_dir / "partial_map_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ix", "iy", "x", "y", "state", "observed_count", "is_frontier", "first_seen_pose_id"])
        writer.writeheader()
        for (ix, iy), cell in sorted(cells.items()):
            row = {"ix": ix, "iy": iy, **cell}
            row.setdefault("first_seen_pose_id", "")
            writer.writerow(row)

    chars = {"unknown": "?", "known_free": ".", "occupied": "#"}
    lines = []
    for iy in reversed(range(len(ys))):
        line = []
        for ix in range(len(xs)):
            cell = cells[(ix, iy)]
            if cell["is_frontier"]:
                line.append("F")
            else:
                line.append(chars[cell["state"]])
        lines.append("".join(line))
    (out_dir / "partial_map_ascii.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "partial_map_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if counts["known_free"] > 0 and counts["occupied"] > 0 and counts["frontier"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
