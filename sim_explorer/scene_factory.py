#!/usr/bin/env python3
"""Scripted Isaac indoor scene factories for simulator smoke tests.

The metadata builders are pure Python so tests can run without launching Isaac.
When ``spawn=True`` the same specs are also instantiated in Isaac.  The
Stage 4A-6.6c home-like scene uses referenced mesh/USD furniture assets as
the primary furniture representation.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import numpy as np


WALL_COLOR = [0.56, 0.59, 0.63]
FLOOR_COLOR = [0.72, 0.74, 0.70]
WALL_HEIGHT_M = 2.2
WALL_THICKNESS_M = 0.16
MEDIUM_BOUNDS = {"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]}
MINIMAL_BOUNDS = {"x": [-4.0, 4.0], "y": [-4.0, 4.0], "z": [0.0, 3.0]}
LARGE_BOUNDS = {"x": [-12.0, 12.0], "y": [-12.0, 12.0], "z": [0.0, 3.0]}
HOME_LIKE_SCENE_V1_ORIGINAL_STAGED_USD = "/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd"
HOME_LIKE_SCENE_V1_FIXED_USD = "/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
HOME_LIKE_SCENE_V1_STAGED_USD = HOME_LIKE_SCENE_V1_FIXED_USD


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_sim_utils(sim_utils_module: Any | None = None) -> Any:
    if sim_utils_module is not None:
        return sim_utils_module
    import isaaclab.sim as sim_utils  # type: ignore

    return sim_utils


def _spawn_box(sim_utils: Any, path: str, size: list[float], position: list[float], color: list[float]) -> None:
    cfg = sim_utils.CuboidCfg(
        size=tuple(float(v) for v in size),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(v) for v in color)),
    )
    cfg.func(path, cfg, translation=tuple(float(v) for v in position))


def create_floor(
    path: str = "/World/GroundPlane",
    size: tuple[float, float] | list[float] = (12.0, 12.0),
    bounds: dict[str, list[float]] | None = None,
    *,
    spawn: bool = True,
    sim_utils_module: Any | None = None,
) -> dict[str, Any]:
    """Create a ground plane and return its metadata spec."""

    spec = {
        "name": "floor",
        "path": str(path),
        "size": [float(size[0]), float(size[1])],
        "bounds": copy.deepcopy(bounds) if bounds is not None else None,
        "color": list(FLOOR_COLOR),
    }
    if spawn:
        sim_utils = _require_sim_utils(sim_utils_module)
        ground_cfg = sim_utils.GroundPlaneCfg(size=tuple(spec["size"]))
        ground_cfg.func(path, ground_cfg)
    return _jsonable(spec)


def create_wall_box(
    path: str,
    name: str,
    size: tuple[float, float, float] | list[float],
    position: tuple[float, float, float] | list[float],
    color: tuple[float, float, float] | list[float] = WALL_COLOR,
    *,
    spawn: bool = True,
    sim_utils_module: Any | None = None,
) -> dict[str, Any]:
    """Create one wall cuboid and return its metadata spec."""

    spec = {
        "name": str(name),
        "path": str(path),
        "size": [float(v) for v in size],
        "position": [float(v) for v in position],
        "color": [float(v) for v in color],
        "kind": "wall",
    }
    if spawn:
        sim_utils = _require_sim_utils(sim_utils_module)
        _spawn_box(sim_utils, spec["path"], spec["size"], spec["position"], spec["color"])
    return _jsonable(spec)


def create_obstacle_box(
    path: str,
    name: str,
    size: tuple[float, float, float] | list[float],
    position: tuple[float, float, float] | list[float],
    color: tuple[float, float, float] | list[float],
    category: str,
    *,
    spawn: bool = True,
    sim_utils_module: Any | None = None,
) -> dict[str, Any]:
    """Create one furniture/obstacle cuboid and return its metadata spec."""

    spec = {
        "name": str(name),
        "path": str(path),
        "size": [float(v) for v in size],
        "position": [float(v) for v in position],
        "color": [float(v) for v in color],
        "category": str(category),
        "kind": "obstacle",
    }
    if spawn:
        sim_utils = _require_sim_utils(sim_utils_module)
        _spawn_box(sim_utils, spec["path"], spec["size"], spec["position"], spec["color"])
    return _jsonable(spec)


def create_non_cuboid_obstacle(
    path: str,
    name: str,
    primitive_type: str,
    position: tuple[float, float, float] | list[float],
    color: tuple[float, float, float] | list[float],
    category: str,
    *,
    radius: float | None = None,
    height: float | None = None,
    axis: str = "Z",
    spawn: bool = True,
    sim_utils_module: Any | None = None,
) -> dict[str, Any]:
    """Create one non-cuboid primitive obstacle and return its metadata spec."""

    primitive_type = str(primitive_type)
    spec = {
        "name": str(name),
        "path": str(path),
        "primitive_type": primitive_type,
        "position": [float(v) for v in position],
        "color": [float(v) for v in color],
        "category": str(category),
        "kind": "non_cuboid_obstacle",
        "radius": float(radius) if radius is not None else None,
        "height": float(height) if height is not None else None,
        "axis": str(axis),
    }
    if spawn:
        sim_utils = _require_sim_utils(sim_utils_module)
        common = {
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(v) for v in spec["color"])),
        }
        if primitive_type == "sphere":
            cfg = sim_utils.SphereCfg(radius=float(radius), **common)
        elif primitive_type == "cylinder":
            cfg = sim_utils.CylinderCfg(radius=float(radius), height=float(height), axis=str(axis), **common)
        elif primitive_type == "cone":
            cfg = sim_utils.ConeCfg(radius=float(radius), height=float(height), axis=str(axis), **common)
        elif primitive_type == "capsule":
            cfg = sim_utils.CapsuleCfg(radius=float(radius), height=float(height), axis=str(axis), **common)
        else:
            raise ValueError(f"Unsupported non-cuboid primitive_type: {primitive_type}")
        cfg.func(spec["path"], cfg, translation=tuple(float(v) for v in spec["position"]))
    return _jsonable(spec)


def _yaw_quaternion(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(float(yaw_deg)) * 0.5
    return (float(math.cos(half)), 0.0, 0.0, float(math.sin(half)))


def _quat_multiply(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        float(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2),
        float(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2),
        float(w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2),
        float(w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2),
    )


def _furniture_orientation(yaw_deg: float) -> tuple[float, float, float, float]:
    # Kenney GLB models are authored in a glTF-style Y-up orientation.  Rotate
    # them into Isaac's Z-up world, then apply the room-layout yaw.
    q_yaw = _yaw_quaternion(yaw_deg)
    half_x = math.radians(90.0) * 0.5
    q_x90 = (float(math.cos(half_x)), float(math.sin(half_x)), 0.0, 0.0)
    return _quat_multiply(q_yaw, q_x90)


def create_furniture_asset(
    spec: dict[str, Any],
    *,
    spawn: bool = True,
    sim_utils_module: Any | None = None,
    asset_usd_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create one mesh-backed furniture asset and return its metadata spec."""

    out = copy.deepcopy(spec)
    if spawn:
        if not asset_usd_map:
            raise RuntimeError("home_like_scene_v1 requires converted USD mesh assets; procedural furniture fallback is disabled.")
        asset_file = str(out["asset_file"])
        usd_path = asset_usd_map.get(asset_file) or asset_usd_map.get(PathLikeKey(asset_file))
        if not usd_path:
            raise RuntimeError(f"Missing converted USD path for furniture asset {asset_file}")
        out["converted_usd_path"] = str(usd_path)
        sim_utils = _require_sim_utils(sim_utils_module)
        sim_utils.create_prim(
            out["path"],
            "Xform",
            translation=tuple(float(v) for v in out.get("spawn_position", [out["position"][0], out["position"][1], 0.0])),
            orientation=_furniture_orientation(float(out.get("yaw_deg", 0.0))),
            scale=tuple(float(v) for v in out.get("spawn_scale", [1.0, 1.0, 1.0])),
            usd_path=str(usd_path),
            semantic_label=str(out.get("semantic_label", out.get("category", "furniture"))),
            semantic_type="class",
        )
    return _jsonable(out)


def PathLikeKey(value: str) -> str:
    return str(value).replace("\\", "/")


def _vertical_wall_segment(name: str, x: float, y0: float, y1: float, root: str = "/World/Walls") -> dict[str, Any]:
    y0, y1 = sorted((float(y0), float(y1)))
    length = y1 - y0
    return {
        "name": name,
        "path": f"{root}/{name}",
        "size": [WALL_THICKNESS_M, length, WALL_HEIGHT_M],
        "position": [float(x), 0.5 * (y0 + y1), 0.5 * WALL_HEIGHT_M],
        "axis": "vertical",
    }


def _horizontal_wall_segment(name: str, y: float, x0: float, x1: float, root: str = "/World/Walls") -> dict[str, Any]:
    x0, x1 = sorted((float(x0), float(x1)))
    length = x1 - x0
    return {
        "name": name,
        "path": f"{root}/{name}",
        "size": [length, WALL_THICKNESS_M, WALL_HEIGHT_M],
        "position": [0.5 * (x0 + x1), float(y), 0.5 * WALL_HEIGHT_M],
        "axis": "horizontal",
    }


def _door_vertical(name: str, x: float, y: float, width: float, connects: list[str]) -> dict[str, Any]:
    half = 0.5 * float(width)
    return {
        "name": name,
        "center": [float(x), float(y)],
        "width": float(width),
        "orientation": "vertical_wall_opening",
        "connects": list(connects),
        "clear_rect": {
            "x": [float(x) - 0.5 * WALL_THICKNESS_M, float(x) + 0.5 * WALL_THICKNESS_M],
            "y": [float(y) - half, float(y) + half],
        },
    }


def _door_horizontal(name: str, y: float, x: float, width: float, connects: list[str]) -> dict[str, Any]:
    half = 0.5 * float(width)
    return {
        "name": name,
        "center": [float(x), float(y)],
        "width": float(width),
        "orientation": "horizontal_wall_opening",
        "connects": list(connects),
        "clear_rect": {
            "x": [float(x) - half, float(x) + half],
            "y": [float(y) - 0.5 * WALL_THICKNESS_M, float(y) + 0.5 * WALL_THICKNESS_M],
        },
    }


def _pose(index: int, position: list[float], yaw_deg: float, note: str, room: str) -> dict[str, Any]:
    yaw_rad = math.radians(float(yaw_deg))
    return {
        "index": int(index),
        "position": [float(v) for v in position],
        "yaw_rad": float(yaw_rad),
        "yaw_deg": float(yaw_deg),
        "target": [
            float(position[0] + math.cos(yaw_rad)),
            float(position[1] + math.sin(yaw_rad)),
            float(position[2]),
        ],
        "note": str(note),
        "room": str(room),
    }


def _apply_jitter(
    base_obstacles: list[dict[str, Any]],
    seed: int,
    obstacle_jitter_m: float,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    jittered: list[dict[str, Any]] = []
    for obstacle in base_obstacles:
        item = copy.deepcopy(obstacle)
        if float(obstacle_jitter_m) > 0.0:
            jitter = rng.uniform(-float(obstacle_jitter_m), float(obstacle_jitter_m), size=2)
            item["position"][0] = float(item["position"][0] + jitter[0])
            item["position"][1] = float(item["position"][1] + jitter[1])
            item["jitter_xy"] = [float(jitter[0]), float(jitter[1])]
        else:
            item["jitter_xy"] = [0.0, 0.0]
        jittered.append(item)
    return jittered


def _expected_smoke_files(num_poses: int) -> list[str]:
    files = ["camera_info.json", "scene_metadata.json", "observed_summary.json", "observed_state_final.npy"]
    for idx in range(int(num_poses)):
        files.extend(
            [
                f"depth_{idx:03d}.npy",
                f"rgb_{idx:03d}.png",
                f"pose_{idx:03d}.json",
                f"observed_state_step{idx}.npy",
            ]
        )
    return files


def _expected_viz_files() -> list[str]:
    return [
        "scene_overview_rgb.png",
        "scene_overview_depth_color.png",
        "scene_layout_topdown.png",
        "camera_rgb_grid.png",
        "camera_depth_grid.png",
        "observed_topdown_compare.png",
        "free_occupied_voxels_3d_final.png",
        "slices_final.png",
        "scene_viz_summary.json",
    ]


def build_minimal_scene(
    *,
    spawn: bool = False,
    sim_utils_module: Any | None = None,
) -> dict[str, Any]:
    """Return/spawn the Stage 4A-1 minimal scene metadata."""

    if spawn:
        sim_utils = _require_sim_utils(sim_utils_module)
        sim_utils.create_prim("/World/Walls", "Xform")
        sim_utils.create_prim("/World/Obstacles", "Xform")
    else:
        sim_utils = None

    floor = create_floor(size=(8.0, 8.0), bounds=MINIMAL_BOUNDS, spawn=spawn, sim_utils_module=sim_utils)
    walls = [
        create_wall_box("/World/Walls/Left", "left", [8.0, 0.15, 2.0], [0.0, -4.0, 1.0], WALL_COLOR, spawn=spawn, sim_utils_module=sim_utils),
        create_wall_box("/World/Walls/Right", "right", [8.0, 0.15, 2.0], [0.0, 4.0, 1.0], WALL_COLOR, spawn=spawn, sim_utils_module=sim_utils),
        create_wall_box("/World/Walls/Back", "back", [0.15, 8.0, 2.0], [4.0, 0.0, 1.0], [0.50, 0.54, 0.58], spawn=spawn, sim_utils_module=sim_utils),
    ]
    obstacles = [
        create_obstacle_box(
            "/World/Obstacles/Center",
            "center",
            [0.7, 0.7, 1.2],
            [1.7, 0.35, 0.6],
            [0.8, 0.32, 0.24],
            "box_obstacle",
            spawn=spawn,
            sim_utils_module=sim_utils,
        ),
        create_obstacle_box(
            "/World/Obstacles/Back",
            "back",
            [0.8, 1.2, 1.5],
            [3.2, -1.2, 0.75],
            [0.24, 0.50, 0.82],
            "box_obstacle",
            spawn=spawn,
            sim_utils_module=sim_utils,
        ),
    ]
    camera_poses = [
        _pose(0, [0.0, 0.0, 1.2], 0.0, "center, looking +x", "minimal_room"),
        _pose(1, [0.0, 0.0, 1.2], 90.0, "center, looking +y", "minimal_room"),
        _pose(2, [1.0, 0.0, 1.2], 0.0, "shifted forward, looking +x", "minimal_room"),
    ]
    return _jsonable(
        {
            "stage": "Stage 4A-1",
            "scene_id": "minimal_room",
            "variant": "minimal",
            "floor": floor,
            "walls": walls,
            "doors": [],
            "openings": [],
            "obstacles": obstacles,
            "camera_poses": camera_poses,
            "map_bounds": copy.deepcopy(MINIMAL_BOUNDS),
            "voxel_size_recommended": 0.1,
            "prediction_used": False,
            "expert_used": False,
            "rl_or_il_training_used": False,
        }
    )


def build_medium_complex_scene(
    seed: int = 0,
    variant: str = "three_rooms",
    *,
    obstacle_jitter_m: float = 0.0,
    spawn: bool = False,
    sim_utils_module: Any | None = None,
) -> dict[str, Any]:
    """Build a deterministic 12m x 12m multi-room cuboid scene."""

    if variant != "three_rooms":
        raise ValueError("Only variant='three_rooms' is implemented for Stage 4A-3.2")

    if spawn:
        sim_utils = _require_sim_utils(sim_utils_module)
        sim_utils.create_prim("/World/Walls", "Xform")
        sim_utils.create_prim("/World/Obstacles", "Xform")
    else:
        sim_utils = None

    floor = create_floor(size=(12.0, 12.0), bounds=MEDIUM_BOUNDS, spawn=spawn, sim_utils_module=sim_utils)

    doors = [
        _door_vertical("door_room_a_to_corridor", -1.2, -3.55, 1.1, ["room_a", "corridor"]),
        _door_vertical("door_room_b_to_corridor", 1.2, -3.25, 1.1, ["room_b", "corridor"]),
        _door_horizontal("door_corridor_to_room_c", 1.2, 0.0, 1.3, ["corridor", "room_c"]),
    ]
    openings = copy.deepcopy(doors)

    wall_segment_specs = [
        _vertical_wall_segment("outer_west", -6.0, -6.0, 6.0),
        _vertical_wall_segment("outer_east", 6.0, -6.0, 6.0),
        _horizontal_wall_segment("outer_south", -6.0, -6.0, 6.0),
        _horizontal_wall_segment("outer_north", 6.0, -6.0, 6.0),
        _vertical_wall_segment("room_a_corridor_south", -1.2, -6.0, -4.10),
        _vertical_wall_segment("room_a_corridor_north", -1.2, -3.00, 1.2),
        _vertical_wall_segment("room_b_corridor_south", 1.2, -6.0, -3.80),
        _vertical_wall_segment("room_b_corridor_north", 1.2, -2.70, 1.2),
        _horizontal_wall_segment("room_c_split_west", 1.2, -6.0, -0.65),
        _horizontal_wall_segment("room_c_split_east", 1.2, 0.65, 6.0),
        _vertical_wall_segment("room_c_partial_divider", 2.25, 2.05, 5.25),
        _horizontal_wall_segment("room_c_west_stub", 3.65, -5.6, -3.1),
        _horizontal_wall_segment("room_a_stub", -1.00, -5.2, -3.4),
    ]
    walls = [
        create_wall_box(
            spec["path"],
            spec["name"],
            spec["size"],
            spec["position"],
            WALL_COLOR if not spec["name"].startswith("outer") else [0.50, 0.53, 0.57],
            spawn=spawn,
            sim_utils_module=sim_utils,
        )
        | {"axis": spec["axis"]}
        for spec in wall_segment_specs
        if min(spec["size"][0], spec["size"][1]) > 0.0 and max(spec["size"][0], spec["size"][1]) > 0.05
    ]

    base_obstacles = [
        {
            "name": "room_a_table_low",
            "path": "/World/Obstacles/room_a_table_low",
            "size": [1.20, 0.80, 0.45],
            "position": [-4.25, -3.85, 0.225],
            "color": [0.70, 0.42, 0.25],
            "category": "table_like_low_box",
        },
        {
            "name": "room_a_sofa_block",
            "path": "/World/Obstacles/room_a_sofa_block",
            "size": [1.85, 0.72, 0.68],
            "position": [-4.25, 0.35, 0.34],
            "color": [0.28, 0.45, 0.64],
            "category": "sofa_like_block",
        },
        {
            "name": "room_a_tall_cabinet",
            "path": "/World/Obstacles/room_a_tall_cabinet",
            "size": [0.58, 1.20, 1.75],
            "position": [-5.25, -0.80, 0.875],
            "color": [0.36, 0.53, 0.42],
            "category": "cabinet_like_tall_box",
        },
        {
            "name": "room_a_long_shelf",
            "path": "/World/Obstacles/room_a_long_shelf",
            "size": [1.55, 0.42, 1.35],
            "position": [-2.55, -5.10, 0.675],
            "color": [0.62, 0.54, 0.39],
            "category": "shelf_like_long_box",
        },
        {
            "name": "room_b_table_low",
            "path": "/World/Obstacles/room_b_table_low",
            "size": [1.15, 0.82, 0.45],
            "position": [3.10, -3.80, 0.225],
            "color": [0.68, 0.39, 0.24],
            "category": "table_like_low_box",
        },
        {
            "name": "room_b_tall_cabinet",
            "path": "/World/Obstacles/room_b_tall_cabinet",
            "size": [0.62, 1.38, 1.80],
            "position": [5.20, -1.05, 0.90],
            "color": [0.30, 0.50, 0.42],
            "category": "cabinet_like_tall_box",
        },
        {
            "name": "room_b_long_shelf",
            "path": "/World/Obstacles/room_b_long_shelf",
            "size": [1.65, 0.42, 1.45],
            "position": [2.65, 0.62, 0.725],
            "color": [0.58, 0.50, 0.35],
            "category": "shelf_like_long_box",
        },
        {
            "name": "room_b_sofa_block",
            "path": "/World/Obstacles/room_b_sofa_block",
            "size": [1.75, 0.78, 0.70],
            "position": [4.60, -5.05, 0.35],
            "color": [0.33, 0.41, 0.66],
            "category": "sofa_like_block",
        },
        {
            "name": "corridor_low_bench",
            "path": "/World/Obstacles/corridor_low_bench",
            "size": [0.55, 0.80, 0.45],
            "position": [0.00, -0.35, 0.225],
            "color": [0.76, 0.55, 0.25],
            "category": "table_like_low_box",
        },
        {
            "name": "room_c_long_shelf",
            "path": "/World/Obstacles/room_c_long_shelf",
            "size": [2.25, 0.45, 1.45],
            "position": [-3.85, 4.85, 0.725],
            "color": [0.55, 0.48, 0.36],
            "category": "shelf_like_long_box",
        },
        {
            "name": "room_c_table_low",
            "path": "/World/Obstacles/room_c_table_low",
            "size": [1.45, 0.92, 0.45],
            "position": [-1.00, 3.25, 0.225],
            "color": [0.70, 0.43, 0.26],
            "category": "table_like_low_box",
        },
        {
            "name": "room_c_tall_cabinet",
            "path": "/World/Obstacles/room_c_tall_cabinet",
            "size": [0.72, 1.50, 1.82],
            "position": [4.85, 4.25, 0.91],
            "color": [0.34, 0.52, 0.44],
            "category": "cabinet_like_tall_box",
        },
        {
            "name": "room_c_sofa_block",
            "path": "/World/Obstacles/room_c_sofa_block",
            "size": [2.00, 0.82, 0.70],
            "position": [3.75, 2.05, 0.35],
            "color": [0.32, 0.40, 0.65],
            "category": "sofa_like_block",
        },
    ]

    obstacle_specs = _apply_jitter(base_obstacles, seed=seed, obstacle_jitter_m=obstacle_jitter_m)
    obstacles = [
        create_obstacle_box(
            spec["path"],
            spec["name"],
            spec["size"],
            spec["position"],
            spec["color"],
            spec["category"],
            spawn=spawn,
            sim_utils_module=sim_utils,
        )
        | {"jitter_xy": spec.get("jitter_xy", [0.0, 0.0])}
        for spec in obstacle_specs
    ]

    camera_poses = [
        _pose(0, [-4.80, -4.70, 1.20], 20.0, "room A start, yaw toward corridor door", "room_a"),
        _pose(1, [-4.80, -4.70, 1.20], 110.0, "room A, yaw toward side wall and furniture", "room_a"),
        _pose(2, [0.00, -4.45, 1.20], 90.0, "corridor entrance, looking north", "corridor"),
        _pose(3, [0.00, -2.60, 1.20], -18.0, "inside corridor, looking toward room B doorway", "corridor"),
        _pose(4, [0.00, 2.00, 1.20], 105.0, "inside room C near doorway, looking across upper room", "room_c"),
    ]

    metadata = {
        "stage": "Stage 4A-3.2",
        "scene_id": "medium_complex_three_rooms",
        "scene": "medium-complexity scripted Isaac indoor scene",
        "variant": str(variant),
        "seed": int(seed),
        "obstacle_jitter_m": float(obstacle_jitter_m),
        "map_bounds": copy.deepcopy(MEDIUM_BOUNDS),
        "floor": floor,
        "wall_height_m": WALL_HEIGHT_M,
        "wall_thickness_m": WALL_THICKNESS_M,
        "rooms": [
            {"name": "room_a", "label": "room A", "bounds": {"x": [-5.92, -1.28], "y": [-5.92, 1.12]}, "role": "left lower room"},
            {"name": "room_b", "label": "room B", "bounds": {"x": [1.28, 5.92], "y": [-5.92, 1.12]}, "role": "right lower room"},
            {"name": "room_c", "label": "room C", "bounds": {"x": [-5.92, 5.92], "y": [1.28, 5.92]}, "role": "upper room"},
        ],
        "corridors": [
            {"name": "corridor", "label": "corridor", "bounds": {"x": [-1.12, 1.12], "y": [-5.92, 1.12]}, "min_width_m": 2.24}
        ],
        "doors": doors,
        "openings": openings,
        "walls": walls,
        "obstacles": obstacles,
        "camera_poses": camera_poses,
        "overview_pose": {"position": [-7.25, -8.0, 6.7], "target": [0.25, -0.15, 0.65], "note": "high oblique scene overview"},
        "camera_defaults": {
            "height_m": 1.2,
            "width": 160,
            "height": 120,
            "overview_width": 640,
            "overview_height": 480,
            "max_depth_m": 8.0,
            "overview_max_depth_m": 18.0,
        },
        "voxel_size_recommended": 0.1,
        "topology_summary": {
            "room_count": 3,
            "corridor_count": 1,
            "opening_count": len(openings),
            "obstacle_count": len(obstacles),
            "door_widths_m": [door["width"] for door in doors],
            "minimum_passage_width_m": 1.1,
        },
        "expected_output_files": {
            "smoke": _expected_smoke_files(len(camera_poses)),
            "visualization": _expected_viz_files(),
        },
        "leakage_checks": {
            "prediction_used": False,
            "prediction_wrote_observed_map": False,
            "target_lr_used": False,
            "target_hr_used": False,
            "scene_ground_truth_used_for_exploration": False,
            "rl_or_ppo_training": False,
            "behavior_cloning_training": False,
            "imitation_learning_training": False,
            "sscnet_training": False,
        },
        "limitations": [
            "Synthetic cuboid-only scripted scene.",
            "No object mesh assets.",
            "Fixed camera smoke poses only; no physical robot execution.",
            "Observed map updates must remain measured-only from depth.",
        ],
    }
    return _jsonable(metadata)


def build_synthetic_hidden_room_frontier_scene(
    seed: int = 0,
    *,
    spawn: bool = False,
    sim_utils_module: Any | None = None,
) -> dict[str, Any]:
    """Build the Stage 4A-6.5aa controlled hidden-room frontier scene.

    The layout is deliberately small and diagnostic-only: room A is measured,
    a narrow doorway/corridor points into a hidden room B, and a side measured
    frontier gives the measured-only tree a plausible competing branch.
    """

    if spawn:
        sim_utils = _require_sim_utils(sim_utils_module)
        sim_utils.create_prim("/World/Walls", "Xform")
        sim_utils.create_prim("/World/Obstacles", "Xform")
    else:
        sim_utils = None

    floor = create_floor(size=(12.0, 12.0), bounds=MEDIUM_BOUNDS, spawn=spawn, sim_utils_module=sim_utils)
    doors = [
        _door_vertical("door_room_a_to_hidden_corridor", -1.2, 0.0, 1.25, ["room_a", "hidden_corridor"]),
        _door_vertical("hidden_corridor_to_room_b", 0.85, 0.0, 1.00, ["hidden_corridor", "hidden_room_b"]),
    ]
    openings = copy.deepcopy(doors)

    wall_segment_specs = [
        _vertical_wall_segment("room_a_west_wall", -5.65, -6.00, 6.00),
        _horizontal_wall_segment("room_a_south_wall", -2.30, -5.65, -1.20),
        _vertical_wall_segment("room_a_door_wall_south", -1.20, -2.30, -0.62),
        _vertical_wall_segment("room_a_door_wall_north", -1.20, 0.62, 2.85),
        _horizontal_wall_segment("hidden_corridor_south_wall", -0.62, -1.20, 0.85),
        _horizontal_wall_segment("hidden_corridor_north_wall", 0.62, -1.20, 0.85),
        _horizontal_wall_segment("hidden_room_south_wall", -2.35, 0.85, 5.55),
        _horizontal_wall_segment("hidden_room_north_wall", 2.35, 0.85, 5.55),
        _vertical_wall_segment("hidden_room_east_wall", 5.55, -2.35, 2.35),
        _vertical_wall_segment("hidden_room_entry_south_jamb", 0.85, -2.35, -0.50),
        _vertical_wall_segment("hidden_room_entry_north_jamb", 0.85, 0.50, 2.35),
    ]
    walls = [
        create_wall_box(
            spec["path"],
            spec["name"],
            spec["size"],
            spec["position"],
            [0.50, 0.53, 0.57] if "hidden_room" in spec["name"] else WALL_COLOR,
            spawn=spawn,
            sim_utils_module=sim_utils,
        )
        | {"axis": spec["axis"]}
        for spec in wall_segment_specs
        if min(spec["size"][0], spec["size"][1]) > 0.0 and max(spec["size"][0], spec["size"][1]) > 0.05
    ]

    base_obstacles = [
        {
            "name": "room_a_low_table",
            "path": "/World/Obstacles/room_a_low_table",
            "size": [0.85, 0.62, 0.45],
            "position": [-3.30, -0.88, 0.225],
            "color": [0.72, 0.43, 0.24],
            "category": "measured_room_table",
        },
        {
            "name": "room_a_frontier_box",
            "path": "/World/Obstacles/room_a_frontier_box",
            "size": [0.55, 0.55, 0.95],
            "position": [-2.15, 1.72, 0.475],
            "color": [0.30, 0.48, 0.66],
            "category": "measured_frontier_obstacle",
        },
        {
            "name": "hidden_room_inner_wall",
            "path": "/World/Obstacles/hidden_room_inner_wall",
            "size": [0.16, 2.10, 1.70],
            "position": [3.25, 0.42, 0.85],
            "color": [0.45, 0.48, 0.52],
            "category": "oracle_hidden_wall",
        },
        {
            "name": "hidden_room_block",
            "path": "/World/Obstacles/hidden_room_block",
            "size": [0.82, 0.82, 1.20],
            "position": [4.45, -1.10, 0.60],
            "color": [0.64, 0.36, 0.26],
            "category": "oracle_hidden_obstacle",
        },
    ]
    obstacle_specs = _apply_jitter(base_obstacles, seed=seed, obstacle_jitter_m=0.0)
    obstacles = [
        create_obstacle_box(
            spec["path"],
            spec["name"],
            spec["size"],
            spec["position"],
            spec["color"],
            spec["category"],
            spawn=spawn,
            sim_utils_module=sim_utils,
        )
        | {"jitter_xy": spec.get("jitter_xy", [0.0, 0.0])}
        for spec in obstacle_specs
    ]

    camera_poses = [
        _pose(0, [-2.20, 0.00, 1.20], 0.0, "room A near-door fixed frame looking through doorway", "room_a")
    ]
    metadata = {
        "stage": "Stage 4A-6.5aa",
        "scene_id": "synthetic_hidden_room_frontier",
        "scene": "controlled synthetic SC validation scene",
        "variant": "synthetic_hidden_room_frontier",
        "seed": int(seed),
        "map_bounds": copy.deepcopy(MEDIUM_BOUNDS),
        "floor": floor,
        "wall_height_m": WALL_HEIGHT_M,
        "wall_thickness_m": WALL_THICKNESS_M,
        "rooms": [
            {
                "name": "room_a",
                "label": "measured start room A",
                "bounds": {"x": [-5.65, -1.20], "y": [-2.30, 2.85]},
                "role": "start room with measured side frontier",
            },
            {
                "name": "hidden_room_b",
                "label": "hidden room B",
                "bounds": {"x": [0.85, 5.55], "y": [-2.35, 2.35]},
                "role": "oracle-predicted hidden room",
            },
        ],
        "corridors": [
            {
                "name": "hidden_corridor",
                "label": "doorway corridor",
                "bounds": {"x": [-1.20, 0.85], "y": [-0.62, 0.62]},
                "min_width_m": 1.24,
            }
        ],
        "diagnostic_regions": {
            "measured_start_room": {"x": [-5.55, -1.20], "y": [-6.00, 6.00], "z": [0.0, 3.0]},
            "measured_side_frontier": {"x": [-5.20, -1.35], "y": [1.35, 2.70], "z": [0.0, 3.0]},
            "measured_doorway_corridor": {"x": [-1.20, 0.72], "y": [-0.46, 0.46], "z": [0.0, 3.0]},
            "oracle_hidden_room": {"x": [0.72, 5.45], "y": [-2.20, 2.20], "z": [0.0, 2.20]},
        },
        "doors": doors,
        "openings": openings,
        "walls": walls,
        "obstacles": obstacles,
        "camera_poses": camera_poses,
        "camera_defaults": {
            "height_m": 1.2,
            "width": 160,
            "height": 120,
            "max_depth_m": 6.0,
        },
        "voxel_size_recommended": 0.1,
        "topology_summary": {
            "room_count": 2,
            "corridor_count": 1,
            "opening_count": len(openings),
            "obstacle_count": len(obstacles),
            "door_widths_m": [door["width"] for door in doors],
            "minimum_passage_width_m": 1.0,
        },
        "diagnostic_intent": {
            "oracle_should_reward_hidden_room_direction": True,
            "measured_only_has_competing_frontier": True,
            "prediction_writeback_allowed": False,
            "prediction_used_for_collision_traversability_or_ray_blocking": False,
            "coverage_improvement_claimed": False,
        },
        "leakage_checks": {
            "prediction_used": False,
            "prediction_wrote_observed_map": False,
            "target_lr_used": False,
            "target_hr_used": False,
            "scene_ground_truth_used_for_runtime_planning": False,
            "oracle_prediction_diagnostic_only": True,
            "rl_or_ppo_training": False,
            "behavior_cloning_training": False,
            "imitation_learning_training": False,
            "sscnet_training": False,
        },
        "limitations": [
            "Diagnostic synthetic cuboid scene, not a rollout scene.",
            "Exactly one fixed frame is intended.",
            "Oracle prediction is diagnostic-only and must stay out of observed_state.",
        ],
    }
    return _jsonable(metadata)


def _split_vertical_wall_by_gaps(
    prefix: str,
    x: float,
    y0: float,
    y1: float,
    gaps: list[tuple[float, float]],
    root: str = "/World/Walls",
) -> list[dict[str, Any]]:
    y0, y1 = sorted((float(y0), float(y1)))
    intervals = sorted((max(y0, float(a)), min(y1, float(b))) for a, b in gaps if float(b) > y0 and float(a) < y1)
    segments: list[dict[str, Any]] = []
    cursor = y0
    part = 0
    for gap0, gap1 in intervals:
        if gap0 - cursor > 0.05:
            segments.append(_vertical_wall_segment(f"{prefix}_{part:02d}", x, cursor, gap0, root=root))
            part += 1
        cursor = max(cursor, gap1)
    if y1 - cursor > 0.05:
        segments.append(_vertical_wall_segment(f"{prefix}_{part:02d}", x, cursor, y1, root=root))
    return segments


def _split_horizontal_wall_by_gaps(
    prefix: str,
    y: float,
    x0: float,
    x1: float,
    gaps: list[tuple[float, float]],
    root: str = "/World/Walls",
) -> list[dict[str, Any]]:
    x0, x1 = sorted((float(x0), float(x1)))
    intervals = sorted((max(x0, float(a)), min(x1, float(b))) for a, b in gaps if float(b) > x0 and float(a) < x1)
    segments: list[dict[str, Any]] = []
    cursor = x0
    part = 0
    for gap0, gap1 in intervals:
        if gap0 - cursor > 0.05:
            segments.append(_horizontal_wall_segment(f"{prefix}_{part:02d}", y, cursor, gap0, root=root))
            part += 1
        cursor = max(cursor, gap1)
    if x1 - cursor > 0.05:
        segments.append(_horizontal_wall_segment(f"{prefix}_{part:02d}", y, cursor, x1, root=root))
    return segments


def _large_room_wall_specs(
    room_name: str,
    bounds: dict[str, list[float]],
    door_gaps: dict[str, list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    x0, x1 = (float(v) for v in bounds["x"])
    y0, y1 = (float(v) for v in bounds["y"])
    specs: list[dict[str, Any]] = []
    specs.extend(_split_vertical_wall_by_gaps(f"{room_name}_west", x0, y0, y1, door_gaps.get("west", [])))
    specs.extend(_split_vertical_wall_by_gaps(f"{room_name}_east", x1, y0, y1, door_gaps.get("east", [])))
    specs.extend(_split_horizontal_wall_by_gaps(f"{room_name}_south", y0, x0, x1, door_gaps.get("south", [])))
    specs.extend(_split_horizontal_wall_by_gaps(f"{room_name}_north", y1, x0, x1, door_gaps.get("north", [])))
    return specs


def _large_start(
    name: str,
    position: list[float],
    yaw_deg: float,
    source_label: str,
    topology_label: str,
) -> dict[str, Any]:
    yaw_rad = math.radians(float(yaw_deg))
    return {
        "name": str(name),
        "position": [float(v) for v in position],
        "yaw_rad": float(yaw_rad),
        "yaw_deg": float(yaw_deg),
        "source_label": str(source_label),
        "intended_local_topology_label": str(topology_label),
        "inside_bounds": True,
        "not_inside_obstacle": True,
        "not_too_close_to_wall": True,
        "expected_valid_depth_view": True,
    }


def _large_validation_pose(index: int, position: list[float], yaw_deg: float, note: str, region: str) -> dict[str, Any]:
    pose = _pose(index, position, yaw_deg, note, region)
    pose["source_label"] = "stage4a66_fixed_validation_view"
    return pose


HOME_MATERIALS = {
    "warm_wall": [0.74, 0.72, 0.66],
    "white_wall": [0.82, 0.83, 0.80],
    "dark_floor": [0.20, 0.22, 0.22],
    "wood": [0.58, 0.38, 0.20],
    "dark_wood": [0.34, 0.22, 0.13],
    "fabric_blue": [0.22, 0.34, 0.58],
    "fabric_green": [0.36, 0.56, 0.46],
    "fabric_red": [0.62, 0.24, 0.22],
    "fabric_gray": [0.45, 0.48, 0.50],
    "tile": [0.70, 0.76, 0.76],
    "metal": [0.55, 0.57, 0.58],
    "ceramic": [0.86, 0.84, 0.78],
    "accent": [0.74, 0.53, 0.28],
    "plant_green": [0.18, 0.48, 0.25],
}

KENNEY_FURNITURE_SOURCE = "https://kenney.nl/assets/furniture-kit"
KENNEY_FURNITURE_ZIP_URL = "https://kenney.nl/media/pages/assets/furniture-kit/440e0608a4-1677580847/kenney_furniture-kit.zip"
KENNEY_FURNITURE_LICENSE = "Creative Commons Zero, CC0 1.0"


def _home_validation_pose(index: int, position: list[float], yaw_deg: float, note: str, region: str) -> dict[str, Any]:
    pose = _pose(index, position, yaw_deg, note, region)
    pose["source_label"] = "stage4a66c_home_like_fixed_validation_view"
    return pose


def _home_start(name: str, position: list[float], yaw_deg: float, label: str) -> dict[str, Any]:
    return _large_start(name, position, yaw_deg, "home_like_scene_v1", label)


def build_home_like_scene_v1(
    seed: int = 0,
    *,
    spawn: bool = False,
    sim_utils_module: Any | None = None,
    asset_usd_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build Stage 4A-6.6c home_like_scene_v1.

    This replaces the old ``larger_complex_scene_v1`` with a more home-like
    indoor layout.  It is still a deterministic scripted validation scene and
    does not perform rollout, planning, map prediction, expert sampling, or
    training.
    """

    if spawn:
        sim_utils = _require_sim_utils(sim_utils_module)
        sim_utils.create_prim("/World/Walls", "Xform")
        sim_utils.create_prim("/World/FurnitureAssets", "Xform")
        sim_utils.create_prim("/World/StructuralOccluders", "Xform")
        sim_utils.create_prim("/World/DecorativePrimitives", "Xform")
    else:
        sim_utils = None

    floor = create_floor(
        size=(24.0, 24.0),
        bounds=LARGE_BOUNDS,
        spawn=spawn,
        sim_utils_module=sim_utils,
    )

    rooms = [
        {"name": "living_room", "label": "living room", "bounds": {"x": [-11.2, -3.2], "y": [-3.0, 5.0]}, "role": "large living room with sofa cluster"},
        {"name": "kitchen", "label": "kitchen", "bounds": {"x": [-11.2, -3.2], "y": [5.0, 11.2]}, "role": "kitchen with island and cabinets"},
        {"name": "dining_room", "label": "dining room", "bounds": {"x": [-3.2, 3.2], "y": [5.0, 11.2]}, "role": "dining room connecting kitchen and corridor"},
        {"name": "main_bedroom", "label": "main bedroom", "bounds": {"x": [3.2, 11.2], "y": [5.0, 11.2]}, "role": "large bedroom"},
        {"name": "guest_bedroom", "label": "guest bedroom", "bounds": {"x": [3.2, 11.2], "y": [-1.0, 5.0]}, "role": "bedroom with side furniture"},
        {"name": "study", "label": "study", "bounds": {"x": [3.2, 11.2], "y": [-7.0, -1.0]}, "role": "study with desk and shelves"},
        {"name": "bathroom", "label": "bathroom", "bounds": {"x": [-3.2, 3.2], "y": [-7.0, -1.0]}, "role": "bathroom with narrow door"},
        {"name": "laundry_storage", "label": "laundry/storage", "bounds": {"x": [-11.2, -6.2], "y": [-9.2, -3.0]}, "role": "utility room"},
        {"name": "entry_room", "label": "entry room", "bounds": {"x": [-6.2, -3.2], "y": [-9.2, -3.0]}, "role": "entry vestibule"},
    ]
    corridors = [
        {"name": "main_corridor", "label": "main corridor", "bounds": {"x": [-3.2, 3.2], "y": [-9.2, 5.0]}, "min_width_m": 1.55, "role": "long central corridor"},
        {"name": "north_hall", "label": "north hall", "bounds": {"x": [-3.2, 3.2], "y": [4.2, 5.8]}, "min_width_m": 1.60, "role": "loop connector near dining"},
        {"name": "east_hall", "label": "east hall", "bounds": {"x": [2.4, 4.0], "y": [-7.0, 5.0]}, "min_width_m": 1.60, "role": "bedroom/study side hall"},
        {"name": "entry_passage", "label": "entry passage", "bounds": {"x": [-6.2, -3.2], "y": [-5.2, -3.8]}, "min_width_m": 1.40, "role": "entry to main corridor"},
    ]

    openings = [
        _door_vertical("door_living_to_main_corridor", -3.2, 0.6, 1.20, ["living_room", "main_corridor"]),
        _door_vertical("door_kitchen_to_north_hall", -3.2, 7.2, 1.10, ["kitchen", "north_hall"]),
        _door_horizontal("door_living_to_kitchen", 5.0, -7.4, 1.15, ["living_room", "kitchen"]),
        _door_horizontal("door_kitchen_to_dining", 5.0, -1.1, 1.25, ["kitchen", "dining_room"]),
        _door_horizontal("door_dining_to_north_hall", 5.0, 0.8, 1.45, ["dining_room", "north_hall"]),
        _door_vertical("door_dining_to_main_bedroom", 3.2, 8.3, 1.00, ["dining_room", "main_bedroom"]),
        _door_vertical("door_main_bedroom_to_east_hall", 3.2, 5.4, 0.90, ["main_bedroom", "east_hall"]),
        _door_vertical("door_guest_to_east_hall", 3.2, 2.1, 0.95, ["guest_bedroom", "east_hall"]),
        _door_vertical("door_study_to_east_hall", 3.2, -4.2, 0.90, ["study", "east_hall"]),
        _door_horizontal("door_guest_to_study_loop", -1.0, 7.1, 0.85, ["guest_bedroom", "study"]),
        _door_horizontal("door_bathroom_to_main_corridor", -1.0, 0.0, 0.80, ["bathroom", "main_corridor"]),
        _door_vertical("door_entry_to_main_corridor", -3.2, -4.5, 0.90, ["entry_room", "main_corridor"]),
        _door_vertical("door_laundry_to_entry", -6.2, -5.6, 0.85, ["laundry_storage", "entry_room"]),
        _door_horizontal("door_laundry_to_living", -3.0, -8.2, 0.90, ["laundry_storage", "living_room"]),
        _door_horizontal("door_entry_to_living", -3.0, -4.8, 1.05, ["entry_room", "living_room"]),
        _door_horizontal("loop_main_to_north_hall", 4.2, 0.0, 1.50, ["main_corridor", "north_hall"]),
        _door_vertical("loop_north_to_east_hall", 3.2, 4.7, 1.20, ["north_hall", "east_hall"]),
        _door_horizontal("loop_east_to_main_corridor", -7.0, 2.9, 1.10, ["east_hall", "main_corridor"]),
        _door_vertical("narrow_bath_to_laundry_service", -3.2, -6.4, 0.78, ["bathroom", "laundry_storage"]),
    ]
    doors = copy.deepcopy(openings)

    room_door_gaps = {
        "living_room": {"east": [(0.0, 1.2)], "north": [(-8.0, -6.8)], "south": [(-8.65, -7.75), (-5.35, -4.25)]},
        "kitchen": {"east": [(6.65, 7.75)], "south": [(-8.0, -6.8), (-1.72, -0.48)]},
        "dining_room": {"west": [], "east": [(7.8, 8.8)], "south": [(0.08, 1.52), (-1.72, -0.48)]},
        "main_bedroom": {"west": [(7.8, 8.8), (4.95, 5.85)]},
        "guest_bedroom": {"west": [(1.62, 2.58)], "south": [(6.68, 7.52)]},
        "study": {"west": [(-4.65, -3.75)], "north": [(6.68, 7.52)]},
        "bathroom": {"north": [(-0.40, 0.40)], "west": [(-6.79, -6.01)]},
        "laundry_storage": {"east": [(-6.03, -5.17)], "north": [(-8.65, -7.75)]},
        "entry_room": {"east": [(-4.95, -4.05)], "west": [(-6.03, -5.17)], "north": [(-5.35, -4.25)]},
    }
    wall_segment_specs: list[dict[str, Any]] = [
        _vertical_wall_segment("outer_west", -12.0, -12.0, 12.0),
        _vertical_wall_segment("outer_east", 12.0, -12.0, 12.0),
        _horizontal_wall_segment("outer_south", -12.0, -12.0, 12.0),
        _horizontal_wall_segment("outer_north", 12.0, -12.0, 12.0),
    ]
    for room in rooms:
        wall_segment_specs.extend(_large_room_wall_specs(room["name"], room["bounds"], room_door_gaps.get(room["name"], {})))
    wall_segment_specs.extend(
        [
            _vertical_wall_segment("main_corridor_west_partial", -3.2, -8.9, 4.0),
            _vertical_wall_segment("main_corridor_east_partial", 3.2, -8.9, 4.0),
            _horizontal_wall_segment("main_corridor_south_cap", -9.2, -3.2, 3.2),
            _vertical_wall_segment("east_hall_east_partial", 4.0, -6.8, 4.6),
            _horizontal_wall_segment("north_hall_south_partial", 4.2, -3.2, 3.2),
            _horizontal_wall_segment("north_hall_north_partial", 5.8, -3.2, 3.2),
            _horizontal_wall_segment("entry_passage_south", -5.2, -6.2, -3.2),
            _horizontal_wall_segment("entry_passage_north", -3.8, -6.2, -3.2),
        ]
    )
    walls = [
        create_wall_box(
            spec["path"],
            spec["name"],
            spec["size"],
            spec["position"],
            HOME_MATERIALS["white_wall"] if str(spec["name"]).startswith("outer") else HOME_MATERIALS["warm_wall"],
            spawn=spawn,
            sim_utils_module=sim_utils,
        )
        | {"axis": spec["axis"]}
        for spec in wall_segment_specs
        if min(spec["size"][0], spec["size"][1]) > 0.0 and max(spec["size"][0], spec["size"][1]) > 0.05
    ]

    def furniture(
        name: str,
        asset_file: str,
        xy: tuple[float, float],
        size: tuple[float, float, float],
        material: str,
        category: str,
        region: str,
        yaw_deg: float = 0.0,
        scale: tuple[float, float, float] = (2.0, 2.0, 2.0),
    ) -> dict[str, Any]:
        asset_name = str(asset_file).removesuffix(".glb")
        return {
            "name": name,
            "path": f"/World/FurnitureAssets/{name}",
            "asset_id": f"kenney_furniture_kit:{asset_file}",
            "asset_name": asset_name,
            "asset_file": str(asset_file),
            "asset_source": "downloaded_kenney_furniture_kit",
            "asset_source_url": KENNEY_FURNITURE_SOURCE,
            "asset_download_url": KENNEY_FURNITURE_ZIP_URL,
            "license": KENNEY_FURNITURE_LICENSE,
            "size": [float(v) for v in size],
            "position": [float(xy[0]), float(xy[1]), float(size[2]) * 0.5],
            "spawn_position": [float(xy[0]), float(xy[1]), 0.0],
            "yaw_deg": float(yaw_deg),
            "spawn_scale": [float(v) for v in scale],
            "asset_orientation_correction": "gltf_y_up_to_isaac_z_up_x_plus_90_deg",
            "color": HOME_MATERIALS[material],
            "material": material,
            "category": category,
            "region": region,
            "kind": "furniture_asset",
            "shape_source": "downloaded_glb_converted_to_usd",
            "procedural_composite_furniture": False,
            "is_mesh_asset": True,
            "non_cuboid_or_composite_asset": True,
            "footprint_proxy_only": False,
            "semantic_label": f"{region}_{category}",
        }

    furniture_plan: list[tuple[Any, ...]] = [
        ("living_sofa_long", "loungeSofaLong.glb", (-9.6, 2.7), (2.6, 0.95, 0.78), "fabric_blue", "sofa", "living_room", 0.0),
        ("living_sofa_corner", "loungeSofaCorner.glb", (-7.0, 2.5), (1.35, 1.35, 0.78), "fabric_blue", "sofa_corner", "living_room", 90.0),
        ("living_ottoman", "loungeSofaOttoman.glb", (-8.2, 1.1), (0.9, 0.7, 0.42), "fabric_gray", "ottoman", "living_room", 0.0),
        ("living_coffee_table", "tableCoffee.glb", (-8.0, 0.2), (1.35, 0.85, 0.45), "wood", "coffee_table", "living_room", 0.0),
        ("living_glass_table", "tableCoffeeGlassSquare.glb", (-5.1, 0.6), (0.85, 0.85, 0.42), "metal", "side_table", "living_room", 15.0),
        ("living_tv_cabinet", "cabinetTelevision.glb", (-4.6, 3.6), (0.55, 1.8, 1.25), "dark_wood", "media_cabinet", "living_room", 90.0),
        ("living_tv", "televisionModern.glb", (-4.45, 3.6), (0.18, 1.15, 0.75), "metal", "television", "living_room", 90.0),
        ("living_speaker_left", "speaker.glb", (-4.4, 2.3), (0.22, 0.22, 0.95), "metal", "speaker", "living_room", 90.0),
        ("living_speaker_right", "speakerSmall.glb", (-4.4, 4.7), (0.18, 0.18, 0.70), "metal", "speaker", "living_room", 90.0),
        ("living_bookcase_open", "bookcaseOpen.glb", (-10.7, -1.2), (0.45, 2.6, 1.55), "wood", "bookcase", "living_room", 0.0),
        ("living_bookcase_wide", "bookcaseClosedWide.glb", (-10.3, 4.3), (1.4, 0.45, 1.35), "dark_wood", "bookcase", "living_room", 0.0),
        ("living_floor_lamp", "lampRoundFloor.glb", (-5.6, -1.8), (0.35, 0.35, 1.55), "accent", "floor_lamp", "living_room", 0.0),
        ("living_rug", "rugRectangle.glb", (-8.0, 1.2), (3.4, 2.2, 0.05), "fabric_red", "rug", "living_room", 0.0),
        ("living_pillow_blue", "pillowBlue.glb", (-9.2, 2.55), (0.38, 0.28, 0.20), "fabric_blue", "pillow", "living_room", 25.0),
        ("living_pillow_long", "pillowLong.glb", (-7.1, 2.2), (0.55, 0.26, 0.20), "fabric_green", "pillow", "living_room", -20.0),
        ("living_plant_small", "plantSmall1.glb", (-10.2, 3.9), (0.42, 0.42, 0.75), "plant_green", "plant", "living_room", 0.0),
        ("living_potted_plant", "pottedPlant.glb", (-4.8, -1.7), (0.45, 0.45, 0.90), "plant_green", "plant", "living_room", 0.0),
        ("kitchen_counter_west", "kitchenCabinet.glb", (-10.0, 10.4), (1.3, 0.55, 0.95), "wood", "lower_cabinet", "kitchen", 0.0),
        ("kitchen_counter_mid", "kitchenCabinetDrawer.glb", (-8.5, 10.4), (1.1, 0.55, 0.95), "wood", "drawer_cabinet", "kitchen", 0.0),
        ("kitchen_counter_east", "kitchenCabinetCornerRound.glb", (-6.8, 10.4), (1.0, 0.55, 0.95), "wood", "corner_cabinet", "kitchen", 0.0),
        ("kitchen_upper_00", "kitchenCabinetUpper.glb", (-10.0, 10.75), (1.1, 0.35, 1.15), "wood", "upper_cabinet", "kitchen", 0.0),
        ("kitchen_upper_01", "kitchenCabinetUpperDouble.glb", (-8.1, 10.75), (1.25, 0.35, 1.15), "wood", "upper_cabinet", "kitchen", 0.0),
        ("kitchen_fridge", "kitchenFridgeLarge.glb", (-10.6, 7.4), (0.75, 0.95, 1.75), "metal", "refrigerator", "kitchen", 90.0),
        ("kitchen_sink", "kitchenSink.glb", (-5.0, 10.2), (1.0, 0.55, 0.95), "ceramic", "sink_counter", "kitchen", 0.0),
        ("kitchen_stove", "kitchenStoveElectric.glb", (-5.8, 9.1), (0.75, 0.75, 0.95), "metal", "stove", "kitchen", 90.0),
        ("kitchen_hood", "hoodModern.glb", (-5.8, 9.1), (0.75, 0.35, 0.65), "metal", "range_hood", "kitchen", 90.0),
        ("kitchen_island_bar", "kitchenBar.glb", (-7.2, 7.8), (1.8, 0.85, 0.95), "ceramic", "kitchen_island", "kitchen", 0.0),
        ("kitchen_bar_end", "kitchenBarEnd.glb", (-8.4, 7.8), (0.65, 0.85, 0.95), "ceramic", "kitchen_island_end", "kitchen", 0.0),
        ("kitchen_stool_00", "stoolBar.glb", (-7.9, 6.7), (0.42, 0.42, 0.78), "accent", "bar_stool", "kitchen", 0.0),
        ("kitchen_stool_01", "stoolBarSquare.glb", (-7.1, 6.7), (0.42, 0.42, 0.78), "accent", "bar_stool", "kitchen", 0.0),
        ("kitchen_stool_02", "stoolBar.glb", (-6.3, 6.7), (0.42, 0.42, 0.78), "accent", "bar_stool", "kitchen", 0.0),
        ("kitchen_microwave", "kitchenMicrowave.glb", (-4.8, 6.4), (0.45, 0.45, 0.38), "metal", "microwave", "kitchen", 90.0),
        ("kitchen_coffee_machine", "kitchenCoffeeMachine.glb", (-9.4, 5.8), (0.42, 0.38, 0.48), "metal", "coffee_machine", "kitchen", 0.0),
        ("kitchen_toaster", "toaster.glb", (-6.5, 10.0), (0.34, 0.28, 0.26), "metal", "toaster", "kitchen", 0.0),
        ("kitchen_blender", "kitchenBlender.glb", (-9.0, 10.0), (0.28, 0.28, 0.55), "metal", "blender", "kitchen", 0.0),
        ("dining_table", "tableCross.glb", (-0.4, 8.2), (2.2, 1.2, 0.72), "wood", "dining_table", "dining_room", 0.0),
        ("dining_chair_00", "chair.glb", (-1.5, 8.9), (0.48, 0.48, 0.85), "fabric_green", "dining_chair", "dining_room", -90.0),
        ("dining_chair_01", "chairCushion.glb", (0.7, 8.9), (0.48, 0.48, 0.85), "fabric_green", "dining_chair", "dining_room", 90.0),
        ("dining_chair_02", "chairModernCushion.glb", (-1.5, 7.5), (0.50, 0.50, 0.85), "fabric_gray", "dining_chair", "dining_room", -90.0),
        ("dining_chair_03", "chairRounded.glb", (0.7, 7.5), (0.50, 0.50, 0.85), "fabric_gray", "dining_chair", "dining_room", 90.0),
        ("dining_bench_north", "benchCushion.glb", (-0.4, 9.9), (1.4, 0.42, 0.55), "fabric_blue", "bench", "dining_room", 0.0),
        ("dining_sideboard", "bookcaseClosedDoors.glb", (-2.5, 10.2), (0.55, 1.8, 1.35), "dark_wood", "sideboard", "dining_room", 90.0),
        ("dining_china_cabinet", "cabinetBed.glb", (2.2, 10.0), (0.55, 1.4, 1.30), "wood", "cabinet", "dining_room", 90.0),
        ("dining_lamp", "lampSquareCeiling.glb", (-0.4, 8.2), (0.42, 0.42, 0.28), "accent", "ceiling_lamp", "dining_room", 0.0),
        ("dining_rug", "rugRounded.glb", (-0.4, 8.2), (2.9, 2.0, 0.05), "fabric_red", "rug", "dining_room", 0.0),
        ("main_bed", "bedDouble.glb", (7.5, 9.2), (2.7, 1.85, 0.72), "fabric_green", "bed", "main_bedroom", 0.0),
        ("main_pillow_00", "pillow.glb", (6.8, 9.65), (0.45, 0.32, 0.22), "fabric_gray", "pillow", "main_bedroom", 0.0),
        ("main_pillow_01", "pillowBlue.glb", (7.5, 9.65), (0.45, 0.32, 0.22), "fabric_blue", "pillow", "main_bedroom", 0.0),
        ("main_nightstand_l", "sideTableDrawers.glb", (5.8, 6.5), (0.55, 0.55, 0.65), "dark_wood", "nightstand", "main_bedroom", 0.0),
        ("main_nightstand_r", "sideTable.glb", (9.1, 6.5), (0.55, 0.55, 0.65), "dark_wood", "nightstand", "main_bedroom", 0.0),
        ("main_lamp_l", "lampSquareTable.glb", (5.8, 6.95), (0.28, 0.28, 0.65), "accent", "table_lamp", "main_bedroom", 0.0),
        ("main_lamp_r", "lampRoundTable.glb", (9.1, 6.95), (0.28, 0.28, 0.65), "accent", "table_lamp", "main_bedroom", 0.0),
        ("main_wardrobe", "bookcaseClosed.glb", (10.4, 9.2), (0.55, 1.8, 1.65), "wood", "wardrobe", "main_bedroom", 90.0),
        ("main_dresser", "cabinetBedDrawer.glb", (4.2, 10.4), (1.6, 0.48, 1.15), "wood", "dresser", "main_bedroom", 0.0),
        ("main_reading_chair", "loungeChairRelax.glb", (10.2, 6.5), (0.72, 0.88, 0.85), "fabric_blue", "reading_chair", "main_bedroom", 120.0),
        ("main_rug", "rugSquare.glb", (7.5, 7.8), (2.6, 2.0, 0.05), "fabric_gray", "rug", "main_bedroom", 0.0),
        ("main_plant", "plantSmall2.glb", (4.4, 6.6), (0.40, 0.40, 0.75), "plant_green", "plant", "main_bedroom", 0.0),
        ("guest_bed", "bedSingle.glb", (8.1, 3.5), (2.2, 1.35, 0.65), "fabric_blue", "bed", "guest_bedroom", 0.0),
        ("guest_pillow", "pillowBlueLong.glb", (7.7, 3.95), (0.70, 0.28, 0.22), "fabric_blue", "pillow", "guest_bedroom", 0.0),
        ("guest_nightstand", "sideTable.glb", (5.3, 1.4), (0.55, 0.55, 0.65), "dark_wood", "nightstand", "guest_bedroom", 0.0),
        ("guest_lamp", "lampRoundTable.glb", (5.3, 1.85), (0.28, 0.28, 0.65), "accent", "table_lamp", "guest_bedroom", 0.0),
        ("guest_wardrobe", "bookcaseClosed.glb", (10.4, 1.1), (0.55, 1.7, 1.50), "wood", "wardrobe", "guest_bedroom", 90.0),
        ("guest_dresser", "cabinetBedDrawerTable.glb", (4.4, 4.2), (1.6, 0.45, 1.10), "wood", "dresser", "guest_bedroom", 0.0),
        ("guest_chair", "loungeDesignChair.glb", (9.8, 4.6), (0.72, 0.72, 0.85), "fabric_green", "chair", "guest_bedroom", -120.0),
        ("guest_bookshelf", "bookcaseOpenLow.glb", (6.5, 0.2), (1.4, 0.45, 1.05), "wood", "low_shelf", "guest_bedroom", 0.0),
        ("guest_rug", "rugRound.glb", (8.0, 2.3), (1.6, 1.6, 0.05), "fabric_red", "rug", "guest_bedroom", 0.0),
        ("guest_plant", "plantSmall3.glb", (4.5, 0.3), (0.38, 0.38, 0.70), "plant_green", "plant", "guest_bedroom", 0.0),
        ("study_desk", "desk.glb", (8.4, -5.7), (2.2, 0.75, 0.78), "wood", "desk", "study", 180.0),
        ("study_corner_desk", "deskCorner.glb", (4.7, -6.0), (1.4, 1.4, 0.78), "wood", "corner_desk", "study", 0.0),
        ("study_chair", "chairDesk.glb", (6.6, -5.8), (0.65, 0.65, 0.85), "fabric_gray", "desk_chair", "study", 15.0),
        ("study_screen", "computerScreen.glb", (8.2, -5.25), (0.58, 0.20, 0.55), "metal", "monitor", "study", 180.0),
        ("study_keyboard", "computerKeyboard.glb", (8.2, -5.45), (0.55, 0.20, 0.12), "metal", "keyboard", "study", 180.0),
        ("study_mouse", "computerMouse.glb", (8.7, -5.45), (0.18, 0.12, 0.08), "metal", "mouse", "study", 180.0),
        ("study_laptop", "laptop.glb", (4.7, -5.75), (0.55, 0.36, 0.16), "metal", "laptop", "study", 0.0),
        ("study_bookcase", "bookcaseOpen.glb", (10.5, -4.3), (0.55, 2.0, 1.55), "dark_wood", "bookcase", "study", 90.0),
        ("study_low_shelf", "bookcaseOpenLow.glb", (4.4, -2.1), (1.8, 0.42, 1.10), "wood", "low_shelf", "study", 0.0),
        ("study_file_cabinet", "cabinetBedDrawer.glb", (5.8, -3.8), (1.1, 0.42, 1.00), "accent", "file_cabinet", "study", 0.0),
        ("study_radio", "radio.glb", (9.4, -1.8), (0.45, 0.28, 0.35), "metal", "radio", "study", 90.0),
        ("study_lamp", "lampRoundTable.glb", (7.7, -5.25), (0.28, 0.28, 0.65), "accent", "desk_lamp", "study", 0.0),
        ("bath_bathtub", "bathtub.glb", (-2.4, -6.1), (0.85, 1.45, 0.85), "ceramic", "bathtub", "bathroom", 0.0),
        ("bath_shower", "showerRound.glb", (-2.5, -2.2), (0.75, 0.75, 1.85), "ceramic", "shower", "bathroom", 0.0),
        ("bath_toilet", "toilet.glb", (2.4, -5.8), (0.65, 0.65, 0.72), "ceramic", "toilet", "bathroom", 90.0),
        ("bath_sink", "bathroomSink.glb", (-0.4, -6.5), (1.0, 0.55, 0.90), "ceramic", "sink", "bathroom", 0.0),
        ("bath_square_sink", "bathroomSinkSquare.glb", (1.8, -2.2), (0.65, 0.55, 0.90), "ceramic", "sink", "bathroom", 90.0),
        ("bath_cabinet", "bathroomCabinet.glb", (2.5, -3.5), (0.45, 1.20, 1.20), "wood", "linen_cabinet", "bathroom", 90.0),
        ("bath_cabinet_drawer", "bathroomCabinetDrawer.glb", (1.0, -6.7), (0.55, 0.45, 0.90), "wood", "vanity_drawer", "bathroom", 0.0),
        ("bath_mirror", "bathroomMirror.glb", (-0.4, -6.85), (0.70, 0.12, 0.70), "metal", "mirror", "bathroom", 0.0),
        ("bath_trashcan", "trashcan.glb", (0.8, -4.2), (0.35, 0.35, 0.65), "metal", "trashcan", "bathroom", 0.0),
        ("bath_rug", "rugDoormat.glb", (0.0, -4.9), (1.0, 0.65, 0.05), "fabric_gray", "bath_mat", "bathroom", 0.0),
        ("laundry_washer", "washer.glb", (-10.4, -7.8), (0.85, 0.85, 1.05), "metal", "washer", "laundry_storage", 0.0),
        ("laundry_dryer", "dryer.glb", (-9.2, -7.8), (0.85, 0.85, 1.05), "metal", "dryer", "laundry_storage", 0.0),
        ("laundry_stack", "washerDryerStacked.glb", (-7.5, -7.8), (0.85, 0.85, 1.80), "metal", "washer_dryer_stack", "laundry_storage", 0.0),
        ("laundry_shelf", "bookcaseOpenLow.glb", (-10.6, -4.2), (0.45, 1.8, 1.20), "wood", "storage_shelf", "laundry_storage", 90.0),
        ("laundry_box_closed_00", "cardboardBoxClosed.glb", (-8.6, -5.8), (0.55, 0.55, 0.45), "accent", "storage_box", "laundry_storage", 20.0),
        ("laundry_box_closed_01", "cardboardBoxClosed.glb", (-7.8, -5.9), (0.55, 0.55, 0.45), "accent", "storage_box", "laundry_storage", -10.0),
        ("laundry_box_open", "cardboardBoxOpen.glb", (-7.2, -4.2), (0.60, 0.55, 0.45), "accent", "storage_box", "laundry_storage", 0.0),
        ("laundry_utility_table", "sideTable.glb", (-7.2, -8.8), (1.0, 0.55, 0.75), "wood", "utility_table", "laundry_storage", 0.0),
        ("laundry_trashcan", "trashcan.glb", (-9.5, -4.4), (0.35, 0.35, 0.65), "metal", "trashcan", "laundry_storage", 0.0),
        ("entry_coat_rack", "coatRackStanding.glb", (-5.5, -8.4), (0.55, 0.55, 1.45), "dark_wood", "coat_rack", "entry_room", 0.0),
        ("entry_wall_coat_rack", "coatRack.glb", (-5.9, -4.2), (0.30, 0.95, 0.80), "dark_wood", "wall_coat_rack", "entry_room", 90.0),
        ("entry_bench", "benchCushionLow.glb", (-4.1, -8.5), (1.0, 0.45, 0.48), "fabric_green", "shoe_bench", "entry_room", 0.0),
        ("entry_console", "sideTableDrawers.glb", (-5.0, -4.0), (0.55, 1.35, 0.95), "accent", "entry_console", "entry_room", 90.0),
        ("entry_rug", "rugDoormat.glb", (-4.9, -6.4), (1.2, 0.75, 0.05), "fabric_red", "entry_rug", "entry_room", 0.0),
        ("entry_plant", "plantSmall1.glb", (-5.8, -6.2), (0.42, 0.42, 0.72), "plant_green", "plant", "entry_room", 0.0),
        ("corridor_console", "sideTable.glb", (-2.45, -2.9), (0.35, 1.1, 0.78), "accent", "corridor_console", "main_corridor", 90.0),
        ("corridor_planter", "pottedPlant.glb", (-2.45, -0.9), (0.42, 0.42, 0.85), "plant_green", "plant", "main_corridor", 0.0),
        ("east_hall_plant", "plantSmall2.glb", (3.25, 4.0), (0.38, 0.38, 0.70), "plant_green", "plant", "east_hall", 0.0),
        ("north_hall_low_shelf", "bookcaseOpenLow.glb", (-1.4, 4.95), (1.4, 0.35, 0.85), "wood", "low_shelf", "north_hall", 0.0),
    ]
    furniture_asset_specs = [furniture(*row) for row in furniture_plan]
    furniture_assets = [
        create_furniture_asset(spec, spawn=spawn, sim_utils_module=sim_utils, asset_usd_map=asset_usd_map)
        for spec in furniture_asset_specs
    ]

    def structural_obstacle(name: str, xy: tuple[float, float], size: tuple[float, float, float], material: str, category: str, region: str) -> dict[str, Any]:
        return {
            "name": name,
            "path": f"/World/StructuralOccluders/{name}",
            "size": [float(v) for v in size],
            "position": [float(xy[0]), float(xy[1]), float(size[2]) * 0.5],
            "color": HOME_MATERIALS[material],
            "material": material,
            "category": category,
            "region": region,
            "kind": "structural_primitive_obstacle",
            "primitive_role": "passage shaping or occlusion; not procedural furniture main solution",
        }

    corridor_specs = [
        ("main_corridor_halfwall_00", (-2.45, -7.3), (0.35, 1.4, 1.05), "white_wall", "half_wall_occluder", "main_corridor"),
        ("main_corridor_shelf_01", (2.45, -5.3), (0.35, 1.6, 1.20), "wood", "corridor_shelf", "main_corridor"),
        ("main_corridor_console_02", (-2.45, -2.9), (0.35, 1.4, 0.85), "accent", "console", "main_corridor"),
        ("main_corridor_pillar_03", (2.5, 1.1), (0.45, 0.45, 1.65), "metal", "pillar", "main_corridor"),
        ("north_hall_low_shelf_00", (-1.4, 4.95), (1.4, 0.35, 0.85), "wood", "low_shelf", "north_hall"),
        ("north_hall_halfwall_01", (1.7, 4.95), (1.2, 0.35, 0.95), "white_wall", "half_wall_occluder", "north_hall"),
        ("east_hall_pillar_00", (3.2, -6.0), (0.45, 0.45, 1.6), "metal", "pillar", "east_hall"),
        ("east_hall_console_01", (3.2, -2.5), (0.42, 1.25, 0.85), "accent", "console", "east_hall"),
        ("east_hall_halfwall_02", (3.2, 3.4), (0.35, 1.4, 0.95), "white_wall", "half_wall_occluder", "east_hall"),
        ("entry_passage_shoe_rack", (-4.8, -4.5), (1.2, 0.35, 0.85), "wood", "shoe_rack", "entry_passage"),
        ("entry_passage_planter", (-5.9, -4.5), (0.45, 0.45, 0.65), "ceramic", "planter_box", "entry_passage"),
        ("bath_narrow_door_occluder", (-2.2, -1.35), (0.42, 0.65, 0.95), "white_wall", "narrow_passage_occluder", "main_corridor"),
    ]
    structural_specs = [structural_obstacle(name, xy, size, material, category, region) for name, xy, size, material, category, region in corridor_specs]
    structural_obstacles = [
        create_obstacle_box(
            spec["path"],
            spec["name"],
            spec["size"],
            spec["position"],
            spec["color"],
            spec["category"],
            spawn=spawn,
            sim_utils_module=sim_utils,
        )
        | {
            "region": spec.get("region", ""),
            "material": spec.get("material", ""),
            "kind": "structural_primitive_obstacle",
            "primitive_role": spec.get("primitive_role"),
        }
        for spec in structural_specs
    ]
    non_cuboid_primitives: list[dict[str, Any]] = []
    all_obstacles = furniture_assets + structural_obstacles

    start_variants = [
        _home_start("start_living_room", [-8.8, 1.4, 1.20], 0.0, "living room near sofa facing corridor"),
        _home_start("start_kitchen", [-8.8, 8.8, 1.20], -35.0, "kitchen near island"),
        _home_start("start_dining_room", [0.0, 8.4, 1.20], -90.0, "dining room facing north hall"),
        _home_start("start_main_bedroom", [7.4, 7.0, 1.20], 180.0, "main bedroom facing hall"),
        _home_start("start_guest_bedroom", [7.0, 2.2, 1.20], 180.0, "guest bedroom facing east hall"),
        _home_start("start_study", [7.4, -4.8, 1.20], 180.0, "study near desk"),
        _home_start("start_bathroom", [0.4, -4.2, 1.20], 90.0, "bathroom narrow door context"),
        _home_start("start_entry", [-4.8, -7.4, 1.20], 65.0, "entry vestibule"),
        _home_start("start_main_corridor", [0.0, -5.4, 1.20], 90.0, "long central corridor"),
        _home_start("start_loop_hall", [2.6, 4.4, 1.20], -90.0, "loop connector near east hall"),
    ]

    validation_camera_poses = [
        _home_validation_pose(0, [-8.8, 1.4, 1.20], 0.0, "living room toward main corridor", "living_room"),
        _home_validation_pose(1, [-10.0, 3.6, 1.20], -35.0, "living room furniture cluster", "living_room"),
        _home_validation_pose(2, [-8.8, 8.8, 1.20], -35.0, "kitchen island and cabinets", "kitchen"),
        _home_validation_pose(3, [-5.0, 7.0, 1.20], 150.0, "kitchen back toward living loop", "kitchen"),
        _home_validation_pose(4, [0.0, 8.4, 1.20], -90.0, "dining room toward north hall", "dining_room"),
        _home_validation_pose(5, [7.4, 8.2, 1.20], 180.0, "main bedroom toward dining/loop", "main_bedroom"),
        _home_validation_pose(6, [10.0, 6.4, 1.20], 130.0, "main bedroom wardrobe occlusion", "main_bedroom"),
        _home_validation_pose(7, [7.0, 2.2, 1.20], 180.0, "guest bedroom toward east hall", "guest_bedroom"),
        _home_validation_pose(8, [9.8, 3.8, 1.20], -135.0, "guest bedroom far corner", "guest_bedroom"),
        _home_validation_pose(9, [7.4, -4.8, 1.20], 180.0, "study desk and shelves", "study"),
        _home_validation_pose(10, [4.6, -2.4, 1.20], -60.0, "study/hall narrow context", "study"),
        _home_validation_pose(11, [0.4, -4.2, 1.20], 90.0, "bathroom narrow door and fixtures", "bathroom"),
        _home_validation_pose(12, [-8.8, -6.8, 1.20], 20.0, "laundry storage utility objects", "laundry_storage"),
        _home_validation_pose(13, [-4.8, -7.4, 1.20], 65.0, "entry room to corridor", "entry_room"),
        _home_validation_pose(14, [0.0, -5.4, 1.20], 90.0, "long main corridor north", "main_corridor"),
        _home_validation_pose(15, [0.0, 2.8, 1.20], -90.0, "main corridor south", "main_corridor"),
        _home_validation_pose(16, [0.6, 5.35, 1.20], 0.0, "north/east hall loop", "north_hall"),
        _home_validation_pose(17, [3.55, -4.8, 1.20], 90.0, "east hall side rooms", "east_hall"),
        _home_validation_pose(18, [-5.0, -4.4, 1.20], -170.0, "entry passage toward living/laundry loop", "entry_room"),
        _home_validation_pose(19, [-9.8, -4.6, 1.20], -90.0, "laundry storage shelves and boxes", "laundry_storage"),
        _home_validation_pose(20, [-2.2, -6.0, 1.20], 20.0, "bathroom service opening and fixtures", "bathroom"),
        _home_validation_pose(21, [2.7, 2.0, 1.20], 95.0, "east hall bedroom loop connector", "east_hall"),
    ]

    inspection_camera_poses = list(validation_camera_poses)
    extra_inspection = [
        ("inspection_living_kitchen_loop", [-7.5, 4.8, 1.20], 90.0, "living/kitchen loop doorway", "living_room"),
        ("inspection_kitchen_dining_pass", [-2.4, 5.4, 1.20], 35.0, "kitchen dining pass-through", "kitchen"),
        ("inspection_bathroom_narrow_passage", [-0.2, -1.4, 1.20], -90.0, "bathroom narrow passage", "main_corridor"),
        ("inspection_entry_laundry_narrow", [-6.0, -5.8, 1.20], 180.0, "entry/laundry narrow connection", "entry_room"),
        ("inspection_loop_east_to_main", [3.55, -5.2, 1.20], 90.0, "east hall loop to main corridor", "east_hall"),
        ("inspection_living_sofa_occlusion", [-9.6, 2.7, 1.20], -20.0, "living room sofa occlusion", "living_room"),
        ("inspection_kitchen_island_occlusion", [-7.3, 7.0, 1.20], 35.0, "kitchen island occlusion", "kitchen"),
        ("inspection_dining_table_close", [-0.2, 7.2, 1.20], 90.0, "dining table close view", "dining_room"),
        ("inspection_main_bed_close", [6.8, 7.4, 1.20], 35.0, "main bed close view", "main_bedroom"),
        ("inspection_guest_study_loop", [7.1, -0.8, 1.20], -90.0, "guest/study loop door", "guest_bedroom"),
        ("inspection_study_shelf_close", [10.0, -4.2, 1.20], 180.0, "study shelf occlusion", "study"),
        ("inspection_long_corridor_occluders", [0.0, -2.2, 1.20], 90.0, "main corridor occluders", "main_corridor"),
        ("inspection_laundry_boxes_close", [-8.1, -5.7, 1.20], 180.0, "laundry storage boxes close view", "laundry_storage"),
        ("inspection_entry_console_close", [-5.2, -4.2, 1.20], -10.0, "entry console and coat rack close view", "entry_room"),
        ("inspection_main_bedroom_dresser", [4.6, 9.8, 1.20], -35.0, "main bedroom dresser and bed occlusion", "main_bedroom"),
        ("inspection_kitchen_counter_appliances", [-9.6, 9.8, 1.20], -20.0, "kitchen counter appliances close view", "kitchen"),
    ]
    for offset, (name, position, yaw_deg, note, region) in enumerate(extra_inspection, start=len(inspection_camera_poses)):
        pose = _home_validation_pose(offset, position, yaw_deg, note, region)
        pose["name"] = name
        pose["source_label"] = "stage4a66c_home_like_extra_inspection_view"
        inspection_camera_poses.append(pose)

    graph_nodes = [
        *[{"id": room["name"], "kind": "room", "label": room["label"]} for room in rooms],
        *[{"id": corridor["name"], "kind": "corridor", "label": corridor["label"]} for corridor in corridors],
    ]
    graph_edges = [
        {"id": opening["name"], "source": opening["connects"][0], "target": opening["connects"][1], "width_m": opening["width"]}
        for opening in openings
    ]
    room_areas = [
        (float(room["bounds"]["x"][1]) - float(room["bounds"]["x"][0]))
        * (float(room["bounds"]["y"][1]) - float(room["bounds"]["y"][0]))
        for room in rooms
    ]
    distinct_asset_files = sorted({str(item["asset_file"]) for item in furniture_assets})
    material_color_inventory = []
    for material_name, color in HOME_MATERIALS.items():
        usage_count = sum(1 for item in furniture_assets + structural_obstacles if item.get("material") == material_name)
        if material_name in ("warm_wall", "white_wall", "dark_floor"):
            usage_count += 1
        material_color_inventory.append(
            {
                "material": material_name,
                "color": list(color),
                "usage_count": int(usage_count),
            }
        )
    primitive_inventory = [
        {
            "name": "floor",
            "kind": "floor_primitive",
            "path": floor["path"],
            "size": floor["size"],
            "role": "home floor base; not furniture",
        },
        *[
            {
                "name": wall["name"],
                "kind": "wall_primitive",
                "path": wall["path"],
                "size": wall["size"],
                "position": wall["position"],
                "axis": wall.get("axis"),
                "role": "room boundary and doorway shaping; not furniture",
            }
            for wall in walls
        ],
        *[
            {
                "name": item["name"],
                "kind": "structural_primitive_obstacle",
                "path": item["path"],
                "size": item["size"],
                "position": item["position"],
                "category": item["category"],
                "region": item["region"],
                "role": item.get("primitive_role"),
            }
            for item in structural_obstacles
        ],
    ]
    topology_summary = {
        "room_count": len(rooms),
        "corridor_count": len(corridors),
        "opening_count": len(openings),
        "wall_count": len(walls),
        "furniture_object_count": len(furniture_assets),
        "mesh_asset_instance_count": len(furniture_assets),
        "distinct_mesh_asset_count": len(distinct_asset_files),
        "downloaded_mesh_asset_instance_count": len(furniture_assets),
        "local_mesh_asset_instance_count": 0,
        "non_cuboid_or_composite_asset_count": sum(1 for item in furniture_assets if item.get("non_cuboid_or_composite_asset")),
        "structural_primitive_obstacle_count": len(structural_obstacles),
        "cuboid_obstacle_count": len(structural_obstacles),
        "non_cuboid_primitive_count": len(non_cuboid_primitives),
        "obstacle_count": len(all_obstacles),
        "material_color_count": len(HOME_MATERIALS),
        "start_variant_count": len(start_variants),
        "validation_pose_count": len(validation_camera_poses),
        "inspection_pose_count": len(inspection_camera_poses),
        "door_widths_m": [opening["width"] for opening in openings],
        "minimum_passage_width_m": min(float(opening["width"]) for opening in openings),
        "narrow_passage_count": sum(1 for opening in openings if float(opening["width"]) <= 0.90),
        "loop_closure_count": 4,
        "dead_end_or_service_branch_count": 2,
        "room_area_min_m2": float(min(room_areas)),
        "room_area_max_m2": float(max(room_areas)),
        "room_area_diversity_ratio": float(max(room_areas) / min(room_areas)),
    }
    complexity_targets = {
        "home_like_domains_present": all(
            name in {room["name"] for room in rooms}
            for name in ("living_room", "kitchen", "main_bedroom", "study", "bathroom")
        ),
        "rooms_ge_8": len(rooms) >= 8,
        "semantic_rooms_ge_8": len(rooms) >= 8,
        "furniture_objects_ge_80": len(furniture_assets) >= 80,
        "non_cuboid_or_composite_assets_ge_25": topology_summary["non_cuboid_or_composite_asset_count"] >= 25,
        "procedural_composite_furniture_disabled_as_main": True,
        "materials_ge_10": len(HOME_MATERIALS) >= 10,
        "starts_ge_10": len(start_variants) >= 10,
        "validation_poses_ge_20": len(validation_camera_poses) >= 20,
        "inspection_poses_ge_36": len(inspection_camera_poses) >= 36,
        "narrow_passages_present": topology_summary["narrow_passage_count"] >= 4,
        "loops_present": topology_summary["loop_closure_count"] >= 2,
        "formal_expert_sampling_ready": False,
    }
    metadata = {
        "stage": "Stage 4A-6.6c",
        "scene_id": "home_like_scene_v1",
        "scene": "home-like deterministic Isaac indoor scene",
        "variant": "home_like_scene_v1",
        "seed": int(seed),
        "scene_seed": int(seed),
        "map_bounds": copy.deepcopy(LARGE_BOUNDS),
        "floor": floor,
        "wall_height_m": WALL_HEIGHT_M,
        "wall_thickness_m": WALL_THICKNESS_M,
        "materials": copy.deepcopy(HOME_MATERIALS),
        "rooms": rooms,
        "corridors": corridors,
        "doors": doors,
        "openings": openings,
        "walls": walls,
        "furniture_assets": furniture_assets,
        "furniture_inventory": furniture_assets,
        "structural_primitive_obstacles": structural_obstacles,
        "primitive_inventory": primitive_inventory,
        "material_color_inventory": material_color_inventory,
        "cuboid_obstacles": structural_obstacles,
        "non_cuboid_primitives": non_cuboid_primitives,
        "obstacles": all_obstacles,
        "start_variants": start_variants,
        "validation_camera_poses": validation_camera_poses,
        "inspection_camera_poses": inspection_camera_poses,
        "camera_poses": validation_camera_poses,
        "overview_pose": {"position": [0.0, -17.5, 15.0], "target": [0.0, 0.0, 0.6], "note": "high oblique home-like overview"},
        "camera_defaults": {
            "height_m": 1.2,
            "width": 320,
            "height": 240,
            "overview_width": 1024,
            "overview_height": 768,
            "max_depth_m": 22.0,
            "overview_max_depth_m": 36.0,
        },
        "voxel_size_recommended": 0.1,
        "expected_observed_state_shape": [240, 240, 30],
        "topology_graph": {"nodes": graph_nodes, "edges": graph_edges},
        "topology_summary": topology_summary,
        "complexity_targets": complexity_targets,
        "asset_policy": {
            "primary_furniture_representation": "downloaded CC0 mesh GLB converted/imported to USD for Isaac rendering",
            "local_mesh_assets_found": 0,
            "downloaded_mesh_assets_used": True,
            "procedural_composite_furniture_as_main_solution": False,
            "fallback_to_cuboid_furniture_allowed": False,
            "block_if_asset_download_or_license_invalid": True,
        },
        "asset_sources": [
            {
                "name": "Kenney Furniture Kit",
                "source_url": KENNEY_FURNITURE_SOURCE,
                "download_url": KENNEY_FURNITURE_ZIP_URL,
                "license": KENNEY_FURNITURE_LICENSE,
                "asset_files_used": distinct_asset_files,
            }
        ],
        "old_scene_removed": True,
        "old_larger_complex_scene_v1_disabled": True,
        "leakage_checks": {
            "prediction_used": False,
            "prediction_wrote_observed_map": False,
            "map_predict_called": False,
            "sscnet_inference_called": False,
            "target_lr_used": False,
            "target_hr_used": False,
            "scene_ground_truth_used_for_exploration": False,
            "rollout_run": False,
            "selected_expert_action_executed": False,
            "formal_expert_sampling": False,
            "rl_or_ppo_training": False,
            "gdpo_training": False,
            "behavior_cloning_training": False,
            "imitation_learning_training": False,
            "sscnet_training": False,
        },
        "manual_review_gate": {
            "human_visual_inspection_done": False,
            "formal_expert_sampling_ready": False,
            "next_stage": "Stage 4A-6.6d review + human visual confirmation",
        },
    }
    return _jsonable(metadata)


def build_home_like_scene_v1_from_usd(
    seed: int = 0,
    *,
    spawn: bool = False,
    sim_utils_module: Any | None = None,
    staged_usd_path: str = HOME_LIKE_SCENE_V1_STAGED_USD,
    prim_path: str = "/World/HomeLikeSceneV1",
) -> dict[str, Any]:
    """Load the user-provided Stage 4A-6.6c home_like_scene_v1 USD.

    This is the active Stage 4A-6.6c scene entry point.  It intentionally
    references the staged USD file and does not construct procedural rooms,
    cuboid furniture, or generated fallback geometry.
    """

    staged_usd_path = str(staged_usd_path)
    metadata = {
        "stage": "Stage 4A-6.6c-usd-import",
        "scene_id": "home_like_scene_v1",
        "variant": "home_like_scene_v1",
        "scene_seed": int(seed),
        "source": "user_provided_staged_usd",
        "staged_usd_path": staged_usd_path,
        "spawn_prim_path": str(prim_path),
        "map_bounds": copy.deepcopy(LARGE_BOUNDS),
        "voxel_size_recommended": 0.1,
        "expected_observed_state_shape": [240, 240, 30],
        "procedural_scene_generated": False,
        "procedural_composite_furniture_fallback_used": False,
        "cuboid_furniture_fallback_used": False,
        "downloaded_furniture_assets_used": False,
        "old_larger_complex_scene_v1_disabled": True,
        "leakage_checks": {
            "prediction_used": False,
            "prediction_wrote_observed_map": False,
            "map_predict_called": False,
            "sscnet_inference_called": False,
            "rollout_run": False,
            "selected_expert_action_executed": False,
            "formal_expert_sampling": False,
            "rl_or_ppo_training": False,
            "gdpo_training": False,
            "behavior_cloning_training": False,
            "imitation_learning_training": False,
        },
        "manual_review_gate": {
            "human_visual_inspection_done": False,
            "formal_expert_sampling_ready": False,
            "full_expert_dataset_ready": False,
            "next_stage": "Stage 4A-6.6d USD scene audit + human visual review",
        },
    }
    if spawn:
        if not os.path.isfile(staged_usd_path):
            raise RuntimeError(f"home_like_scene_v1 staged USD is missing: {staged_usd_path}")
        try:
            from pxr import Usd  # type: ignore

            stage = Usd.Stage.Open(staged_usd_path)
            default_prim = stage.GetDefaultPrim() if stage is not None else None
            if stage is None or default_prim is None or not default_prim.IsValid():
                raise RuntimeError("USD stage has no valid defaultPrim")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"home_like_scene_v1 fixed USD is invalid or missing defaultPrim: {staged_usd_path}; {exc}") from exc
        sim_utils = _require_sim_utils(sim_utils_module)
        sim_utils.create_prim(str(prim_path), "Xform", usd_path=staged_usd_path)
        metadata["spawned"] = True
    else:
        metadata["spawned"] = False
        metadata["staged_usd_exists"] = os.path.isfile(staged_usd_path)
    return _jsonable(metadata)


def build_home_like_scene_v1(
    seed: int = 0,
    *,
    spawn: bool = False,
    sim_utils_module: Any | None = None,
    staged_usd_path: str = HOME_LIKE_SCENE_V1_STAGED_USD,
) -> dict[str, Any]:
    """Active home_like_scene_v1 builder: load the staged user USD only."""

    return build_home_like_scene_v1_from_usd(
        seed=seed,
        spawn=spawn,
        sim_utils_module=sim_utils_module,
        staged_usd_path=staged_usd_path,
    )


def build_home_like_scene_v1_metadata(seed: int = 0) -> dict[str, Any]:
    """Return active Stage 4A-6.6c USD-import metadata without spawning."""

    return build_home_like_scene_v1(seed=seed, spawn=False)


def build_larger_complex_scene_v1(
    seed: int = 0,
    *,
    spawn: bool = False,
    sim_utils_module: Any | None = None,
) -> dict[str, Any]:
    """Disabled legacy Stage 4A-6.6 scene builder.

    ``larger_complex_scene_v1`` was intentionally removed from the active
    scene catalog in Stage 4A-6.6c.  Keep this stub so stale callers fail
    loudly instead of silently constructing the retired scene.
    """

    raise RuntimeError(
        "larger_complex_scene_v1 was removed/disabled in Stage 4A-6.6c. "
        "Use build_home_like_scene_v1(seed=0, ...) instead."
    )


def build_larger_complex_scene_v1_metadata(seed: int = 0) -> dict[str, Any]:
    """Disabled legacy metadata helper for the retired larger scene."""

    return build_larger_complex_scene_v1(seed=seed, spawn=False)
