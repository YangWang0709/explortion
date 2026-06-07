#!/usr/bin/env python3
"""Minimal Isaac RGB-D camera smoke for PI_WORKSPACE fallback scene.

No training, rollout, checkpoint, or RL is performed. The script records only
small metadata summaries, not image arrays.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


def shape_of(value):
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return list(shape)
    if isinstance(value, dict) and "data" in value:
        return shape_of(value["data"])
    try:
        return [len(value)]
    except Exception:
        return str(type(value))


async def run_smoke(args):
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import open_stage_async, update_stage_async
    from isaacsim.sensors.camera import Camera
    import omni.syntheticdata.sensors

    scene_path = Path(args.scene).expanduser().resolve()
    result = {
        "scene": str(scene_path),
        "scene_exists": scene_path.exists(),
        "camera_prim": args.camera_prim,
        "simulation_app_started": True,
        "stage_opened": False,
        "world_initialized": False,
        "camera_initialized": False,
        "render_product_path": None,
        "frame_keys": [],
        "rgb_shape": None,
        "rgba_shape": None,
        "depth_shape": None,
        "distance_to_image_plane_shape": None,
        "distance_to_camera_shape": None,
        "error": None,
        "negative_scope": {
            "training": False,
            "rl": False,
            "checkpoint": False,
            "rollout": False,
        },
    }

    try:
        await open_stage_async(str(scene_path))
        for _ in range(4):
            await update_stage_async()
        result["stage_opened"] = True

        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        await update_stage_async()
        result["world_initialized"] = True

        camera = Camera(
            prim_path=args.camera_prim,
            name="phase1_review_camera",
            resolution=(args.width, args.height),
            frequency=20,
        )
        camera.initialize()
        camera.add_rgb_to_frame()
        camera.add_distance_to_image_plane_to_frame()
        camera.add_distance_to_camera_to_frame()
        result["camera_initialized"] = True
        result["render_product_path"] = str(camera.get_render_product_path())

        await omni.syntheticdata.sensors.next_render_simulation_async(camera.get_render_product_path(), args.render_updates)
        frame = camera.get_current_frame()
        result["frame_keys"] = sorted(frame.keys())
        result["distance_to_image_plane_shape"] = shape_of(frame.get("distance_to_image_plane"))
        result["distance_to_camera_shape"] = shape_of(frame.get("distance_to_camera"))
        result["rgb_shape"] = shape_of(frame.get("rgb"))
        result["rgba_shape"] = shape_of(camera.get_rgba())
        result["depth_shape"] = shape_of(camera.get_depth())
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scenes/minimal_indoor_smoke.usda")
    parser.add_argument("--camera-prim", default="/World/ReviewCamera")
    parser.add_argument("--out", default="runs/phase1_camera_sensor_smoke_summary.json")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--render-updates", type=int, default=12)
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True})
        result = asyncio.get_event_loop().run_until_complete(run_smoke(args))
    except Exception as exc:
        result = {"simulation_app_started": False, "error": repr(exc)}
    finally:
        if app is not None:
            app.close()

    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    ok = (
        result.get("simulation_app_started")
        and result.get("stage_opened")
        and result.get("world_initialized")
        and result.get("camera_initialized")
        and result.get("rgb_shape")
        and (result.get("depth_shape") or result.get("distance_to_image_plane_shape"))
        and result.get("error") is None
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
