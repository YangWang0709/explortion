#!/usr/bin/env python3
"""Minimal Isaac scene-load smoke probe for PI_WORKSPACE.

This script intentionally only starts Isaac headless and opens a USD scene.
It does not train, rollout, checkpoint, or run RL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scenes/minimal_indoor_smoke.usda")
    parser.add_argument("--out", default="runs/phase1_scene_load_smoke_summary.json")
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()

    scene_path = Path(args.scene).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "scene": str(scene_path),
        "scene_exists": scene_path.exists(),
        "simulation_app_started": False,
        "open_stage_result": None,
        "stage_available": False,
        "root_prims": [],
        "prim_count": 0,
        "camera_count": 0,
        "cube_count": 0,
        "error": None,
        "negative_scope": {
            "training": False,
            "rl": False,
            "checkpoint": False,
            "rollout": False,
        },
    }

    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
        result["simulation_app_started"] = True

        import omni.usd

        ctx = omni.usd.get_context()
        result["open_stage_result"] = bool(ctx.open_stage(str(scene_path)))
        for _ in range(max(args.updates, 1)):
            app.update()
        stage = ctx.get_stage()
        result["stage_available"] = bool(stage)
        if stage:
            root_children = list(stage.GetPseudoRoot().GetChildren())
            result["root_prims"] = [str(prim.GetPath()) for prim in root_children]
            for prim in stage.Traverse():
                result["prim_count"] += 1
                type_name = prim.GetTypeName()
                if type_name == "Camera":
                    result["camera_count"] += 1
                elif type_name == "Cube":
                    result["cube_count"] += 1
    except Exception as exc:  # pragma: no cover - smoke script diagnostic path
        result["error"] = repr(exc)
    finally:
        if app is not None:
            app.close()
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["scene_exists"] or not result["simulation_app_started"] or not result["stage_available"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
