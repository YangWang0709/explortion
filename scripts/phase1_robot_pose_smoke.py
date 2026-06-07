#!/usr/bin/env python3
"""Minimal robot marker pose/control smoke for the fallback scene.

This synchronous version avoids async stage helpers because repeated headless
Kit startups can occasionally stall before the smoke body runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scenes/minimal_indoor_smoke.usda")
    parser.add_argument("--robot-prim", default="/World/RobotStart")
    parser.add_argument("--out", default="runs/phase1_robot_pose_smoke_summary.json")
    parser.add_argument("--updates-per-step", type=int, default=2)
    args = parser.parse_args()

    scene_path = Path(args.scene).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "scene": str(scene_path),
        "robot_prim": args.robot_prim,
        "scene_exists": scene_path.exists(),
        "simulation_app_started": False,
        "stage_opened": False,
        "robot_prim_found": False,
        "pose_steps": [],
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
        from pxr import Gf, UsdGeom

        ctx = omni.usd.get_context()
        result["stage_opened"] = bool(ctx.open_stage(str(scene_path)))
        for _ in range(8):
            app.update()
        stage = ctx.get_stage()
        prim = stage.GetPrimAtPath(args.robot_prim) if stage else None
        result["robot_prim_found"] = bool(prim and prim.IsValid())
        if not result["robot_prim_found"]:
            raise RuntimeError(f"robot prim not found: {args.robot_prim}")

        xform = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                translate_op = op
                break
        if translate_op is None:
            translate_op = xform.AddTranslateOp()

        waypoints = [
            (-4.5, -2.8, 0.25),
            (-3.2, -2.2, 0.25),
            (-1.0, -1.5, 0.25),
            (1.0, -0.8, 0.25),
            (3.0, -0.2, 0.25),
        ]
        for step_id, xyz in enumerate(waypoints):
            translate_op.Set(Gf.Vec3d(*xyz))
            for _ in range(max(args.updates_per_step, 1)):
                app.update()
            cache = UsdGeom.XformCache()
            world_t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            result["pose_steps"].append(
                {
                    "step_id": step_id,
                    "target_xyz": list(xyz),
                    "observed_world_xyz": [float(world_t[0]), float(world_t[1]), float(world_t[2])],
                }
            )
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if app is not None:
            app.close()
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    ok = (
        result.get("simulation_app_started")
        and result.get("stage_opened")
        and result.get("robot_prim_found")
        and len(result.get("pose_steps", [])) >= 5
        and result.get("error") is None
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
