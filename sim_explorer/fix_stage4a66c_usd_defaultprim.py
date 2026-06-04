#!/usr/bin/env python3
"""Fix localized USD defaultPrim and run one gated Isaac validation retry.

This is Stage 4A-6.6c-usd-defaultprim-fix.  It only repairs the localized USD
spawn target.  It does not download assets, regenerate a scene, run rollout,
run expert sampling, call map_predict/SSCNet, or train policies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
CONDA_PYTHON = Path("/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python")
CONDA_LIB = Path("/home/ubuntu22/miniconda3/envs/env_isaaclab/lib")
PXR_USD_LIB = Path(
    "/home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
]
PREVIOUS_REQUIRED_FILES = [
    "stage4a66c_usd_download_official_isaac_deps_summary.json",
    "isaac_retry_gate_report.json",
    "localized_package_manifest.json",
    "localized_patch_report.json",
    "unresolved_after_patch.json",
    "official_dependency_download_manifest.json",
    "downloaded_file_hash_manifest.json",
]
OFFICIAL_BASE_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/"
USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
ASSET_TOKEN_RE = re.compile(r"@([^@\r\n]+)@")
REMOTE_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(jsonable(value), sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        if isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(jsonable(value), sort_keys=True)
            if len(rendered) > 1600:
                rendered = rendered[:1600] + "..."
            value_text = f"`{rendered}`"
        else:
            value_text = f"`{value}`"
        lines.append(f"| {key} | {value_text} |")
    return "\n".join(lines)


def markdown_list(title: str, rows: list[str]) -> str:
    body = rows or ["No rows."]
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in body]])


def is_usd_path(path_or_url: str | Path) -> bool:
    suffix = Path(urllib.parse.urlparse(str(path_or_url)).path).suffix.lower()
    return suffix in USD_EXTENSIONS


def canonical_asset(value: str) -> str:
    text = str(value or "").strip().strip("@").strip()
    text = text.split("?", 1)[0].split("#", 1)[0]
    text = text.replace("\\", "/")
    if text.startswith("https:/") and not text.startswith("https://"):
        text = "https://" + text[len("https:/") :]
    if text.startswith("http:/") and not text.startswith("http://"):
        text = "http://" + text[len("http:/") :]
    return text


def is_remote(value: str) -> bool:
    text = canonical_asset(value).lower()
    return bool(REMOTE_RE.match(text)) or text.startswith("omniverse:")


def is_official(value: str) -> bool:
    return canonical_asset(value).startswith(OFFICIAL_BASE_URL)


def extract_asset_paths_from_text(text: str) -> list[str]:
    out: list[str] = []
    for match in ASSET_TOKEN_RE.finditer(text):
        value = canonical_asset(match.group(1))
        if value and value not in out:
            out.append(value)
    return out


def pxr_child_env() -> dict[str, str]:
    env = os.environ.copy()
    py_parts = [str(PXR_USD_LIB)]
    if env.get("PYTHONPATH"):
        py_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_parts)
    ld_parts = [str(CONDA_LIB), str(PXR_USD_LIB / "bin")]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


def run_pxr_child(operation: str, payload: dict[str, Any], timeout: int = 240) -> dict[str, Any]:
    python = CONDA_PYTHON if CONDA_PYTHON.is_file() else Path(sys.executable)
    result = subprocess.run(
        [str(python), str(Path(__file__).resolve()), "__pxr_child", operation],
        input=json.dumps(jsonable(payload)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        env=pxr_child_env(),
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "operation": operation,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "operation": operation,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "error": repr(exc),
        }


def _finite_bound(bound: list[list[float]] | None) -> bool:
    if not bound:
        return False
    values = [float(v) for row in bound for v in row]
    return all(abs(v) < 1.0e20 for v in values) and all(values[i] <= values[i + 3] for i in range(3))


def pxr_child_main(operation: str) -> None:
    from pxr import Sdf, Usd, UsdGeom  # type: ignore

    payload = json.loads(sys.stdin.read() or "{}")
    path = Path(payload.get("path", ""))

    if operation == "scan":
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            print(json.dumps({"ok": False, "path": str(path), "asset_paths": [], "error": "Sdf.Layer.FindOrOpen returned None"}))
            return
        text = layer.ExportToString()
        print(
            json.dumps(
                {
                    "ok": True,
                    "path": str(path),
                    "identifier": str(layer.identifier),
                    "file_format": str(layer.GetFileFormat().formatId),
                    "asset_paths": extract_asset_paths_from_text(text),
                }
            )
        )
        return

    if operation == "inspect_root":
        layer = Sdf.Layer.FindOrOpen(str(path))
        stage = Usd.Stage.Open(str(path))
        if layer is None or stage is None:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "path": str(path),
                        "can_open_layer": layer is not None,
                        "can_open_stage": stage is not None,
                    }
                )
            )
            return
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )

        def prim_record(prim: Any) -> dict[str, Any]:
            subtree_count = sum(1 for _ in Usd.PrimRange(prim))
            bound = None
            try:
                box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                bound = [list(box.GetMin()), list(box.GetMax())]
            except Exception as exc:  # noqa: BLE001
                bound = None
                bound_error = repr(exc)
            else:
                bound_error = None
            return {
                "path": str(prim.GetPath()),
                "name": str(prim.GetName()),
                "typeName": str(prim.GetTypeName()),
                "active": bool(prim.IsActive()),
                "defined": bool(prim.IsDefined()),
                "child_count": len(list(prim.GetChildren())),
                "has_references": bool(prim.HasAuthoredReferences()),
                "has_payloads": bool(prim.HasAuthoredPayloads()),
                "approx_subtree_prim_count": int(subtree_count),
                "approx_bound": bound,
                "approx_bound_computable": _finite_bound(bound),
                "bound_error": bound_error,
            }

        top = [prim_record(prim) for prim in stage.GetPseudoRoot().GetChildren()]
        world = stage.GetPrimAtPath("/World")
        default = stage.GetDefaultPrim()
        print(
            json.dumps(
                {
                    "ok": True,
                    "path": str(path),
                    "can_open_layer": True,
                    "can_open_stage": True,
                    "existing_layer_defaultPrim": str(layer.defaultPrim or ""),
                    "existing_stage_defaultPrim_path": str(default.GetPath()) if default and default.IsValid() else "",
                    "top_level_prims": top,
                    "root_layer_file_format": str(layer.GetFileFormat().formatId),
                    "root_layer_extension": path.suffix.lower(),
                    "edit_save_possible": bool(layer.permissionToEdit and layer.permissionToSave),
                    "world_exists": bool(world and world.IsValid()),
                    "world_defined": bool(world and world.IsValid() and world.IsDefined()),
                    "world_active": bool(world and world.IsValid() and world.IsActive()),
                    "world_subtree_prim_count": int(sum(1 for _ in Usd.PrimRange(world))) if world and world.IsValid() else 0,
                    "world_contains_scene_content": bool(world and world.IsValid() and sum(1 for _ in Usd.PrimRange(world)) > 10),
                }
            )
        )
        return

    if operation == "set_defaultprim":
        prim_name = str(payload["prim_name"])
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            print(json.dumps({"ok": False, "path": str(path), "error": "Sdf.Layer.FindOrOpen returned None"}))
            return
        layer.defaultPrim = prim_name
        layer.Save()
        stage = Usd.Stage.Open(str(path))
        default = stage.GetDefaultPrim() if stage is not None else None
        print(
            json.dumps(
                {
                    "ok": bool(stage is not None and default and default.IsValid()),
                    "path": str(path),
                    "set_defaultPrim": prim_name,
                    "stage_defaultPrim_path": str(default.GetPath()) if default and default.IsValid() else "",
                    "stage_defaultPrim_valid": bool(default and default.IsValid()),
                    "root_layer_file_format": str(layer.GetFileFormat().formatId),
                }
            )
        )
        return

    if operation == "validate_fixed":
        layer = Sdf.Layer.FindOrOpen(str(path))
        stage = Usd.Stage.Open(str(path))
        if layer is None or stage is None:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "path": str(path),
                        "can_open_layer": layer is not None,
                        "can_open_stage": stage is not None,
                    }
                )
            )
            return
        default = stage.GetDefaultPrim()
        subtree_count = int(sum(1 for _ in Usd.PrimRange(default))) if default and default.IsValid() else 0
        print(
            json.dumps(
                {
                    "ok": bool(default and default.IsValid() and subtree_count > 0),
                    "path": str(path),
                    "can_open_layer": True,
                    "can_open_stage": True,
                    "layer_defaultPrim": str(layer.defaultPrim or ""),
                    "stage_defaultPrim_path": str(default.GetPath()) if default and default.IsValid() else "",
                    "stage_defaultPrim_valid": bool(default and default.IsValid()),
                    "defaultPrim_is_concrete": bool(default and default.IsValid() and "<defaultPrim>" not in str(default.GetPath())),
                    "defaultPrim_subtree_prim_count": subtree_count,
                    "root_layer_file_format": str(layer.GetFileFormat().formatId),
                }
            )
        )
        return

    print(json.dumps({"ok": False, "operation": operation, "error": "unknown_pxr_child_operation"}))


def load_context_and_previous(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    previous_dir = Path(args.previous_download_output_dir).resolve()
    entries = []
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        entries.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "contains_dependency_closure": "remote refs 0" in text.lower() or "remote official refs `0`" in text.lower(),
                "contains_missing_defaultprim": "defaultprim" in text.lower(),
                "contains_stage4a66d_blocked": "stage4a66d_executed=false" in text.lower(),
                "contains_stage4a67_blocked": "stage4a67_executed=false" in text.lower(),
            }
        )
    previous_files = {}
    for name in PREVIOUS_REQUIRED_FILES:
        path = previous_dir / name
        previous_files[name] = {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)}
        if not path.is_file():
            raise RuntimeError(f"required previous output is missing: {path}")
    previous_summary = read_json(previous_dir / "stage4a66c_usd_download_official_isaac_deps_summary.json")
    previous_gate = read_json(previous_dir / "isaac_retry_gate_report.json")
    previous_unresolved = read_json(previous_dir / "unresolved_after_patch.json")
    loaded = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "created_at_utc": utc_now(),
        "context_files": entries,
        "previous_output_dir": str(previous_dir),
        "previous_files": previous_files,
        "dependencies_complete_confirmed": bool(previous_summary.get("dependency_closure_complete") is True),
        "remote_official_refs_remaining_confirmed": int(previous_unresolved.get("remote_official_ref_count", -1)) == 0,
        "omniverse_refs_remaining_confirmed": int(previous_unresolved.get("omniverse_ref_count", -1)) == 0,
        "unresolved_local_deps_remaining_confirmed": int(previous_unresolved.get("unresolved_local_dependency_count", -1)) == 0,
        "previous_retry_failed_only_missing_defaultPrim": previous_gate.get("retry_result") == "failed_missing_defaultPrim",
        "previous_no_rgb_depth": not (previous_dir / "rgb_validation_grid.png").exists() and not (previous_dir / "depth_validation_grid.png").exists(),
        "previous_no_observed_state_final": not (previous_dir / "observed_state_final.npy").exists(),
        "previous_no_visual_html": not (previous_dir / "visual_inspection_index.html").exists(),
        "previous_no_mp4": not (previous_dir / "usd_scene_flythrough.mp4").exists(),
        "stage4a66d_blocked": True,
        "stage4a67_blocked": True,
        "no_new_asset_download_this_stage": True,
        "no_scene_regeneration_this_stage": True,
    }
    save_json(output_dir / "loaded_context_manifest.json", loaded)
    write_text(
        output_dir / "loaded_context_manifest.md",
        markdown_table(
            "Loaded Context Manifest",
            {
                "dependencies_complete_confirmed": loaded["dependencies_complete_confirmed"],
                "remote_official_refs_remaining_confirmed": loaded["remote_official_refs_remaining_confirmed"],
                "omniverse_refs_remaining_confirmed": loaded["omniverse_refs_remaining_confirmed"],
                "unresolved_local_deps_remaining_confirmed": loaded["unresolved_local_deps_remaining_confirmed"],
                "previous_retry_failed_only_missing_defaultPrim": loaded["previous_retry_failed_only_missing_defaultPrim"],
                "previous_no_rgb_depth": loaded["previous_no_rgb_depth"],
                "previous_no_observed_state_final": loaded["previous_no_observed_state_final"],
                "previous_no_visual_html": loaded["previous_no_visual_html"],
                "previous_no_mp4": loaded["previous_no_mp4"],
                "stage4a66d_blocked": loaded["stage4a66d_blocked"],
                "stage4a67_blocked": loaded["stage4a67_blocked"],
            },
        ),
    )
    previous = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "previous_stage": previous_summary.get("stage"),
        "localized_package": previous_summary.get("localized_root"),
        "localized_usd": previous_summary.get("localized_usd"),
        "initial_requested_urls": previous_summary.get("initial_requested_urls"),
        "dependency_file_count": read_json(previous_dir / "localized_package_manifest.json").get("package_file_count"),
        "dependency_total_size_bytes": read_json(previous_dir / "localized_package_manifest.json").get("package_total_size_bytes"),
        "remote_official_refs_remaining": previous_summary.get("remote_official_refs_remaining"),
        "omniverse_refs_remaining": previous_summary.get("omniverse_refs_remaining"),
        "unresolved_local_deps_remaining": previous_summary.get("unresolved_local_deps_remaining"),
        "previous_retry_allowed": previous_summary.get("retry_allowed"),
        "previous_retry_attempted": previous_summary.get("retry_attempted"),
        "previous_retry_attempt_count": previous_summary.get("retry_attempt_count"),
        "previous_retry_result": previous_summary.get("retry_result"),
        "previous_isaac_error": previous_summary.get("isaac_retry_error"),
        "source_sha256": previous_summary.get("source_sha256_after"),
        "original_staged_sha256": previous_summary.get("staged_sha256_after"),
        "procedural_fallback": previous_summary.get("procedural_fallback"),
        "cuboid_fallback": False,
        "no_rollout": True,
        "no_expert_sampling": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
    }
    save_json(output_dir / "previous_download_dependency_summary.json", previous)
    write_text(output_dir / "previous_download_dependency_summary.md", markdown_table("Previous Download Dependency Summary", previous))
    return loaded, previous


def write_no_scope_reports(output_dir: Path) -> None:
    reports = {
        "no_rollout_report": {
            "rollout_run": False,
            "open_ended_loop_run": False,
            "selected_action_executed": False,
            "transitions_jsonl_created": False,
        },
        "no_expert_sampling_report": {
            "formal_expert_sampling_run": False,
            "expert_sampling_run": False,
            "expert_dataset_generated": False,
            "expert_dataset_manifest_created": False,
            "formal_expert_sampling_ready": False,
            "full_expert_dataset_ready": False,
        },
        "no_map_predict_report": {
            "map_predict_called": False,
            "sscnet_inference_called": False,
            "prediction_npz_created": False,
            "prediction_written_to_observed_state": False,
        },
        "no_rl_gdpo_report": {
            "rl_run": False,
            "gdpo_run": False,
            "ppo_run": False,
            "behavior_cloning_run": False,
            "imitation_learning_run": False,
            "replay_buffer_created": False,
            "policy_checkpoint_created": False,
            "checkpoint_modified": False,
        },
    }
    for stem, report in reports.items():
        save_json(output_dir / f"{stem}.json", report)
        write_text(output_dir / f"{stem}.md", markdown_table(stem.replace("_", " ").title(), report))


def write_manual_review_gate(output_dir: Path, *, attempted: bool = False, succeeded: bool = False) -> None:
    checklist = {
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": bool(succeeded),
        "visual_approval_required_before_6_7": True,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
        "isaac_validation_attempted": bool(attempted),
        "isaac_validation_succeeded": bool(succeeded),
        "items": [
            {"id": 1, "question": "Review fixed localized USD RGB/depth validation views.", "status": "pending_user_review"},
            {"id": 2, "question": "Confirm no broken references, missing materials, or scale issues.", "status": "pending_user_review"},
            {"id": 3, "question": "Keep Stage 4A-6.7 blocked until Stage 4A-6.6d approval exists.", "status": "pending_user_review"},
        ],
    }
    save_json(output_dir / "human_visual_review_checklist.json", checklist)
    write_text(
        output_dir / "human_visual_review_checklist.md",
        "\n".join(
            [
                "# Human Visual Review Checklist",
                "",
                *[f"{item['id']}. {item['question']} Status: `{item['status']}`." for item in checklist["items"]],
                "",
                f"human_visual_inspection_done: `{checklist['human_visual_inspection_done']}`",
                f"formal_expert_sampling_ready: `{checklist['formal_expert_sampling_ready']}`",
            ]
        ),
    )
    gate = {
        "human_visual_inspection_done": False,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
        "user_needs_to_review_visuals": bool(succeeded),
        "visual_approval_required_before_6_7": True,
    }
    save_json(output_dir / "manual_review_gate.json", gate)
    write_text(output_dir / "manual_review_gate.md", markdown_table("Manual Review Gate", gate))


def inspect_localized_usd(output_dir: Path, localized_usd: Path) -> dict[str, Any]:
    inspection = run_pxr_child("inspect_root", {"path": str(localized_usd)}, timeout=240)
    rows = inspection.get("top_level_prims", []) if inspection.get("ok") else []
    save_json(output_dir / "localized_usd_root_inspection.json", inspection)
    write_text(
        output_dir / "localized_usd_root_inspection.md",
        markdown_table(
            "Localized USD Root Inspection",
            {
                "localized_usd": str(localized_usd),
                "can_open_layer": inspection.get("can_open_layer"),
                "can_open_stage": inspection.get("can_open_stage"),
                "existing defaultPrim": inspection.get("existing_layer_defaultPrim"),
                "top_level_prim_count": len(rows),
                "world_exists": inspection.get("world_exists"),
                "world_contains_scene_content": inspection.get("world_contains_scene_content"),
                "root_layer_file_format": inspection.get("root_layer_file_format"),
                "edit_save_possible": inspection.get("edit_save_possible"),
            },
        ),
    )
    write_csv(output_dir / "top_level_prim_inventory.csv", rows)
    save_json(output_dir / "top_level_prim_inventory.json", rows)
    return inspection


def meaningful_top_prims(inspection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in inspection.get("top_level_prims", []):
        subtree = int(row.get("approx_subtree_prim_count") or 0)
        child_count = int(row.get("child_count") or 0)
        meaningful = bool(row.get("active") and row.get("defined") and (subtree > 1 or child_count > 0 or row.get("has_references") or row.get("has_payloads") or row.get("approx_bound_computable")))
        if meaningful:
            rows.append(row | {"meaningful_reason": f"subtree={subtree}, children={child_count}, refs={row.get('has_references')}, bound={row.get('approx_bound_computable')}"})
    return rows


def choose_defaultprim(output_dir: Path, inspection: dict[str, Any], preferred: str) -> dict[str, Any]:
    meaningful = meaningful_top_prims(inspection)
    world = next((row for row in inspection.get("top_level_prims", []) if row.get("path") == f"/{preferred}"), None)
    if (
        world
        and world.get("active")
        and world.get("defined")
        and int(world.get("approx_subtree_prim_count") or 0) > 10
        and inspection.get("world_contains_scene_content")
    ):
        decision = {
            "case": "A",
            "blocked": False,
            "chosen_defaultPrim": preferred,
            "chosen_defaultPrim_path": f"/{preferred}",
            "wrapper_required": False,
            "meaningful_top_level_prims": meaningful,
            "reason": f"/{preferred} exists, is active/defined, and contains the main scene subtree.",
        }
    elif len(meaningful) == 1:
        chosen = meaningful[0]
        decision = {
            "case": "B",
            "blocked": False,
            "chosen_defaultPrim": chosen["name"],
            "chosen_defaultPrim_path": chosen["path"],
            "wrapper_required": False,
            "meaningful_top_level_prims": meaningful,
            "reason": "Exactly one meaningful top-level prim exists.",
        }
    elif len(meaningful) > 1:
        decision = {
            "case": "C",
            "blocked": False,
            "chosen_defaultPrim": "World",
            "chosen_defaultPrim_path": "/World",
            "wrapper_required": True,
            "meaningful_top_level_prims": meaningful,
            "reason": "Multiple meaningful top-level prims exist; wrapper is required if /World cannot safely be chosen.",
        }
    else:
        decision = {
            "case": "D",
            "blocked": True,
            "chosen_defaultPrim": "",
            "chosen_defaultPrim_path": "",
            "wrapper_required": False,
            "meaningful_top_level_prims": meaningful,
            "reason": "No meaningful root prim can be determined.",
        }
    save_json(output_dir / "defaultprim_candidate_report.json", decision)
    write_text(output_dir / "defaultprim_candidate_report.md", markdown_table("DefaultPrim Candidate Report", decision))
    return decision


def safe_copy_localized_package(localized_root: Path, fixed_root: Path) -> None:
    if localized_root.resolve() == fixed_root.resolve():
        raise RuntimeError("fixed_root must differ from localized_root")
    if fixed_root.exists():
        if fixed_root.name != "current_environment_localized_defaultprim":
            raise RuntimeError(f"refusing to replace unexpected fixed_root: {fixed_root}")
        shutil.rmtree(fixed_root)
    fixed_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(localized_root, fixed_root, symlinks=False)


def create_wrapper_usd(wrapper_path: Path, fixed_usd: Path, prims: list[dict[str, Any]]) -> dict[str, Any]:
    used_names: set[str] = set()

    def safe_name(name: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_]", "_", name.strip("/") or "Part")
        if not base or base[0].isdigit():
            base = "Part_" + base
        candidate = base
        idx = 1
        while candidate in used_names:
            idx += 1
            candidate = f"{base}_{idx}"
        used_names.add(candidate)
        return candidate

    rel = os.path.relpath(fixed_usd, wrapper_path.parent).replace(os.sep, "/")
    if not rel.startswith("."):
        rel = "./" + rel
    referenced = []
    lines = ['#usda 1.0', '(', '    defaultPrim = "World"', ')', '', 'def Xform "World"', '{']
    for prim in prims:
        name = safe_name(str(prim.get("name") or prim.get("path", "Part")).strip("/"))
        source_path = str(prim["path"])
        lines.extend(
            [
                f'    def Xform "{name}" (',
                f"        references = @{rel}@<{source_path}>",
                "    )",
                "    {",
                "    }",
                "",
            ]
        )
        referenced.append({"wrapper_prim": f"/World/{name}", "source_prim_path": source_path})
    lines.append("}")
    write_text(wrapper_path, "\n".join(lines))
    return {
        "wrapper_used": True,
        "wrapper_usd": str(wrapper_path),
        "fixed_usd": str(fixed_usd),
        "referenced_prims": referenced,
        "reference_style": "explicit_prim_paths",
    }


def apply_defaultprim_fix(args: argparse.Namespace, output_dir: Path, inspection: dict[str, Any], decision: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    localized_root = Path(args.localized_root).resolve()
    localized_usd = Path(args.localized_usd).resolve()
    fixed_root = Path(args.fixed_root).resolve()
    fixed_usd = fixed_root / localized_usd.name
    wrapper_usd = fixed_root / "home_like_scene_v1_wrapper.usda"

    plan = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "case": decision["case"],
        "localized_root": str(localized_root),
        "localized_usd": str(localized_usd),
        "fixed_root": str(fixed_root),
        "fixed_usd": str(fixed_usd),
        "wrapper_usd": str(wrapper_usd),
        "chosen_defaultPrim": decision.get("chosen_defaultPrim"),
        "wrapper_required": decision.get("wrapper_required"),
        "source_usd_will_be_modified": False,
        "original_staged_usd_will_be_modified": False,
        "original_localized_package_will_be_modified": False,
        "no_new_assets_downloaded": True,
        "no_scene_regenerated": True,
    }
    save_json(output_dir / "defaultprim_fix_plan.json", plan)
    write_text(output_dir / "defaultprim_fix_plan.md", markdown_table("DefaultPrim Fix Plan", plan))

    if decision.get("blocked"):
        blocker = {
            "blocked": True,
            "reason": decision.get("reason"),
            "localized_usd": str(localized_usd),
            "recommended_next": "Ask teammate to provide a root USD with valid defaultPrim or a clear root prim.",
        }
        save_json(output_dir / "defaultprim_root_selection_blocker.json", blocker)
        write_text(output_dir / "defaultprim_root_selection_blocker.md", markdown_table("DefaultPrim Root Selection Blocker", blocker))
        save_json(output_dir / "wrapper_usd_report.json", {"wrapper_used": False, "reason": "root_selection_blocked"})
        write_text(output_dir / "wrapper_usd_report.md", markdown_table("Wrapper USD Report", {"wrapper_used": False, "reason": "root_selection_blocked"}))
        return {"blocked": True, "patch_ok": False, "reason": "root_selection_blocked"}, fixed_usd

    safe_copy_localized_package(localized_root, fixed_root)
    if decision.get("wrapper_required") and decision.get("case") == "C" and not inspection.get("world_contains_scene_content"):
        wrapper_report = create_wrapper_usd(wrapper_usd, fixed_usd, decision["meaningful_top_level_prims"])
        target_usd = wrapper_usd
        patch_result = {"ok": True, "method": "wrapper_usd", "defaultPrim": "World"}
    else:
        wrapper_report = {
            "wrapper_used": False,
            "wrapper_usd": str(wrapper_usd),
            "reason": "direct defaultPrim repair selected",
            "referenced_prims": [],
        }
        patch_result = run_pxr_child("set_defaultprim", {"path": str(fixed_usd), "prim_name": decision["chosen_defaultPrim"]}, timeout=240)
        target_usd = fixed_usd
    save_json(output_dir / "wrapper_usd_report.json", wrapper_report)
    write_text(output_dir / "wrapper_usd_report.md", markdown_table("Wrapper USD Report", wrapper_report))
    patch_report = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "blocked": False,
        "patch_ok": bool(patch_result.get("ok")),
        "patch_result": patch_result,
        "localized_usd": str(localized_usd),
        "fixed_usd": str(fixed_usd),
        "target_usd": str(target_usd),
        "chosen_defaultPrim": decision.get("chosen_defaultPrim"),
        "source_usd_modified": False,
        "original_staged_usd_modified": False,
        "original_localized_package_modified": False,
        "no_new_assets_downloaded": True,
        "no_scene_regenerated": True,
    }
    save_json(output_dir / "defaultprim_patch_report.json", patch_report)
    write_text(output_dir / "defaultprim_patch_report.md", markdown_table("DefaultPrim Patch Report", patch_report))
    return patch_report, target_usd


def write_fixed_hash_manifest(output_dir: Path, fixed_root: Path, source_usd: Path, original_staged_usd: Path, localized_usd: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(fixed_root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path),
                    "relative_to_fixed_root": path.relative_to(fixed_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "is_usd": is_usd_path(path),
                }
            )
    manifest = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "created_at_utc": utc_now(),
        "fixed_root": str(fixed_root),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "source_usd": str(source_usd),
        "source_sha256": sha256_file(source_usd),
        "original_staged_usd": str(original_staged_usd),
        "original_staged_sha256": sha256_file(original_staged_usd),
        "original_localized_usd": str(localized_usd),
        "original_localized_sha256": sha256_file(localized_usd),
        "files": rows,
    }
    write_csv(output_dir / "fixed_usd_hash_manifest.csv", rows)
    save_json(output_dir / "fixed_usd_hash_manifest.json", manifest)
    write_text(
        output_dir / "fixed_usd_hash_manifest.md",
        markdown_table(
            "Fixed USD Hash Manifest",
            {
                "fixed_root": str(fixed_root),
                "file_count": len(rows),
                "total_size_bytes": manifest["total_size_bytes"],
                "source_sha256": manifest["source_sha256"],
                "original_staged_sha256": manifest["original_staged_sha256"],
                "original_localized_sha256": manifest["original_localized_sha256"],
            },
        ),
    )
    return manifest


def scan_unresolved(output_dir: Path, target_usd: Path, fixed_root: Path) -> dict[str, Any]:
    usd_files = []
    if target_usd.is_file():
        usd_files.append(target_usd)
    for path in sorted(fixed_root.rglob("*")):
        if path.is_file() and is_usd_path(path) and path not in usd_files:
            usd_files.append(path)
    file_reports = []
    remote_official: list[dict[str, Any]] = []
    omniverse: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    unresolved_local: list[dict[str, Any]] = []
    for path in usd_files:
        scan = run_pxr_child("scan", {"path": str(path)}, timeout=240)
        assets = scan.get("asset_paths", []) if scan.get("ok") else []
        file_reports.append({"path": str(path), "scan_ok": bool(scan.get("ok")), "asset_path_count": len(assets)})
        if not scan.get("ok"):
            unresolved_local.append({"path": str(path), "reason": "pxr_scan_failed", "scan": scan})
            continue
        for raw in assets:
            asset = canonical_asset(raw)
            if not asset or asset.startswith("<") or asset.startswith("#"):
                continue
            if is_official(asset):
                remote_official.append({"path": str(path), "asset_path": asset})
                continue
            if asset.lower().startswith("omniverse://") or asset.lower().startswith("omniverse:"):
                omniverse.append({"path": str(path), "asset_path": asset})
                continue
            if is_remote(asset):
                unsupported.append({"path": str(path), "asset_path": asset})
                continue
            candidate = Path(asset) if Path(asset).is_absolute() else (path.parent / asset).resolve()
            if not candidate.exists():
                unresolved_local.append({"path": str(path), "asset_path": asset, "resolved_candidate": str(candidate), "reason": "local_dependency_missing"})
    report = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "target_usd": str(target_usd),
        "fixed_root": str(fixed_root),
        "usd_file_count_scanned": len(usd_files),
        "file_reports": file_reports,
        "remote_official_refs": remote_official,
        "remote_official_ref_count": len(remote_official),
        "omniverse_refs": omniverse,
        "omniverse_ref_count": len(omniverse),
        "unsupported_external_dependencies": unsupported,
        "unsupported_external_dependency_count": len(unsupported),
        "unresolved_local_dependencies": unresolved_local,
        "unresolved_local_dependency_count": len(unresolved_local),
        "dependency_closure_complete": not remote_official and not omniverse and not unsupported and not unresolved_local,
    }
    save_json(output_dir / "unresolved_after_defaultprim_fix.json", report)
    write_text(
        output_dir / "unresolved_after_defaultprim_fix.md",
        markdown_table(
            "Unresolved After DefaultPrim Fix",
            {
                "remote_official_ref_count": report["remote_official_ref_count"],
                "omniverse_ref_count": report["omniverse_ref_count"],
                "unsupported_external_dependency_count": report["unsupported_external_dependency_count"],
                "unresolved_local_dependency_count": report["unresolved_local_dependency_count"],
                "dependency_closure_complete": report["dependency_closure_complete"],
            },
        ),
    )
    return report


def validate_fixed_usd(
    output_dir: Path,
    target_usd: Path,
    source_usd: Path,
    original_staged_usd: Path,
    localized_usd: Path,
    source_hash_before: str | None,
    staged_hash_before: str | None,
    previous_package: dict[str, Any],
    unresolved: dict[str, Any],
) -> dict[str, Any]:
    pxr = run_pxr_child("validate_fixed", {"path": str(target_usd)}, timeout=240)
    source_hash_after = sha256_file(source_usd)
    staged_hash_after = sha256_file(original_staged_usd)
    localized_hash_after = sha256_file(localized_usd)
    validation = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "fixed_usd": str(target_usd),
        "fixed_usd_exists": target_usd.is_file(),
        "pxr_validation": pxr,
        "can_open_layer": bool(pxr.get("can_open_layer")),
        "can_open_stage": bool(pxr.get("can_open_stage")),
        "defaultPrim_valid": bool(pxr.get("stage_defaultPrim_valid")),
        "defaultPrim_path": pxr.get("stage_defaultPrim_path"),
        "defaultPrim_is_concrete": bool(pxr.get("defaultPrim_is_concrete")),
        "defaultPrim_subtree_prim_count": int(pxr.get("defaultPrim_subtree_prim_count") or 0),
        "remote_official_refs_remaining": unresolved.get("remote_official_ref_count"),
        "omniverse_refs_remaining": unresolved.get("omniverse_ref_count"),
        "unsupported_external_dependencies_remaining": unresolved.get("unsupported_external_dependency_count"),
        "unresolved_local_deps_remaining": unresolved.get("unresolved_local_dependency_count"),
        "dependency_closure_complete": bool(unresolved.get("dependency_closure_complete")),
        "source_usd_hash_unchanged": source_hash_before == source_hash_after,
        "original_staged_usd_hash_unchanged": staged_hash_before == staged_hash_after,
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "original_staged_sha256_before": staged_hash_before,
        "original_staged_sha256_after": staged_hash_after,
        "original_localized_package_hash_manifest_preserved": previous_package.get("localized_usd_sha256") == localized_hash_after,
        "original_localized_usd_sha256_after": localized_hash_after,
        "no_procedural_fallback_files_created": True,
    }
    validation["offline_validation_passed"] = bool(
        validation["fixed_usd_exists"]
        and validation["can_open_layer"]
        and validation["can_open_stage"]
        and validation["defaultPrim_valid"]
        and validation["defaultPrim_is_concrete"]
        and validation["defaultPrim_subtree_prim_count"] > 0
        and validation["dependency_closure_complete"]
        and validation["source_usd_hash_unchanged"]
        and validation["original_staged_usd_hash_unchanged"]
        and validation["original_localized_package_hash_manifest_preserved"]
    )
    save_json(output_dir / "fixed_usd_validation_report.json", validation)
    write_text(
        output_dir / "fixed_usd_validation_report.md",
        markdown_table(
            "Fixed USD Validation Report",
            {
                "fixed_usd": str(target_usd),
                "defaultPrim_valid": validation["defaultPrim_valid"],
                "defaultPrim_path": validation["defaultPrim_path"],
                "defaultPrim_subtree_prim_count": validation["defaultPrim_subtree_prim_count"],
                "dependency_closure_complete": validation["dependency_closure_complete"],
                "source_usd_hash_unchanged": validation["source_usd_hash_unchanged"],
                "original_staged_usd_hash_unchanged": validation["original_staged_usd_hash_unchanged"],
                "offline_validation_passed": validation["offline_validation_passed"],
            },
        ),
    )
    return validation


def scene_factory_registration_report(output_dir: Path, target_usd: Path) -> dict[str, Any]:
    import importlib

    import scene_factory

    importlib.reload(scene_factory)
    text = (WORKSPACE / "sim_explorer/scene_factory.py").read_text(encoding="utf-8")
    try:
        larger_disabled = False
        scene_factory.build_larger_complex_scene_v1(seed=0, spawn=False)
    except RuntimeError:
        larger_disabled = True
    metadata = scene_factory.build_home_like_scene_v1(seed=0, spawn=False)
    report = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "scene_factory_path": str(WORKSPACE / "sim_explorer/scene_factory.py"),
        "scene_factory_updated": str(target_usd) in text and "HOME_LIKE_SCENE_V1_FIXED_USD" in text,
        "home_like_scene_v1_target": metadata.get("staged_usd_path"),
        "home_like_scene_v1_points_to_fixed_usd": metadata.get("staged_usd_path") == str(target_usd),
        "fixed_usd_constant_present": "HOME_LIKE_SCENE_V1_FIXED_USD" in text,
        "larger_complex_scene_v1_disabled": larger_disabled,
        "procedural_fallback": False,
        "medium_three_rooms_behavior_preserved": "def build_medium_three_rooms" in text,
    }
    save_json(output_dir / "scene_factory_defaultprim_registration_report.json", report)
    write_text(output_dir / "scene_factory_defaultprim_registration_report.md", markdown_table("Scene Factory DefaultPrim Registration Report", report))
    return report


def build_validation_gate(
    output_dir: Path,
    target_usd: Path,
    validation: dict[str, Any],
    scene_report: dict[str, Any],
    patch_report: dict[str, Any],
) -> dict[str, Any]:
    blockers = []
    if not target_usd.is_file():
        blockers.append("fixed_usd_missing")
    if not validation.get("offline_validation_passed"):
        blockers.append("offline_validation_failed")
    if validation.get("remote_official_refs_remaining") != 0:
        blockers.append("remote_official_refs_remaining")
    if validation.get("omniverse_refs_remaining") != 0:
        blockers.append("omniverse_refs_remaining")
    if validation.get("unresolved_local_deps_remaining") != 0:
        blockers.append("unresolved_local_deps_remaining")
    if not validation.get("source_usd_hash_unchanged"):
        blockers.append("source_usd_hash_changed")
    if not validation.get("original_staged_usd_hash_unchanged"):
        blockers.append("original_staged_usd_hash_changed")
    if not scene_report.get("home_like_scene_v1_points_to_fixed_usd"):
        blockers.append("scene_factory_not_pointing_to_fixed_usd")
    if scene_report.get("procedural_fallback"):
        blockers.append("procedural_fallback_detected")
    if not patch_report.get("patch_ok"):
        blockers.append("defaultprim_patch_failed")
    gate = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "validation_gate_passed": not blockers,
        "retry_allowed": not blockers,
        "reason": "offline_defaultprim_and_dependency_checks_passed" if not blockers else ";".join(blockers),
        "blockers": blockers,
        "fixed_usd": str(target_usd),
        "defaultPrim_valid": validation.get("defaultPrim_valid"),
        "dependency_closure_complete": validation.get("dependency_closure_complete"),
        "source_usd_hash_unchanged": validation.get("source_usd_hash_unchanged"),
        "original_staged_usd_hash_unchanged": validation.get("original_staged_usd_hash_unchanged"),
        "scene_factory_points_to_fixed_usd": scene_report.get("home_like_scene_v1_points_to_fixed_usd"),
        "procedural_fallback": False,
        "retry_attempted": False,
        "retry_attempt_count": 0,
        "retry_result": "not_attempted",
        "isaac_headless_startup_count": 0,
        "exactly_one_attempt": False,
        "no_rollout": True,
        "no_expert_sampling": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
    }
    save_json(output_dir / "isaac_validation_gate_report.json", gate)
    write_text(output_dir / "isaac_validation_gate_report.md", markdown_table("Isaac Validation Gate Report", gate))
    return gate


def run_isaac_validation_once(args: argparse.Namespace, output_dir: Path, target_usd: Path, gate: dict[str, Any]) -> dict[str, Any]:
    gate.update(
        {
            "retry_attempted": True,
            "retry_attempt_count": 1,
            "isaac_headless_startup_count": 1,
            "exactly_one_attempt": True,
        }
    )
    simulation_app = None
    try:
        from isaaclab.app import AppLauncher  # type: ignore

        import import_stage4a66c_usd_home_like_scene as usd_import

        retry_args = argparse.Namespace()
        retry_args.source_usd = str(Path(args.source_usd).resolve())
        retry_args.scene_seed = 0
        retry_args.x_min = -12.0
        retry_args.x_max = 12.0
        retry_args.y_min = -12.0
        retry_args.y_max = 12.0
        retry_args.z_min = 0.0
        retry_args.z_max = 3.0
        retry_args.voxel_size = 0.1
        retry_args.min_validation_poses = int(args.min_validation_poses)
        retry_args.min_inspection_poses = int(args.min_inspection_poses)
        retry_args.camera_width = 320
        retry_args.camera_height = 240
        retry_args.max_depth = 26.0
        retry_args.settle_steps = 12
        retry_args.pixel_stride = 5
        retry_args.make_flythrough = bool(args.make_flythrough)
        retry_args.max_workers = int(args.max_workers)
        retry_args.save_viz = bool(args.save_viz)
        retry_args.no_rollout = True
        retry_args.no_formal_expert_sampling = True
        retry_args.no_map_predict = True
        retry_args.no_rl_gdpo = True
        retry_args.device = getattr(args, "device", "cuda:0")
        retry_args.headless = True
        retry_args.enable_cameras = True
        retry_args.kit_args = ""
        retry_args.experience = ""
        retry_args.livestream = -1
        retry_args.offscreen_render = True
        retry_args.rendering = True
        retry_args.width = 1280
        retry_args.height = 720
        retry_args.display_options = 3286

        validation, simulation_app = usd_import.capture_with_isaac(retry_args, AppLauncher, output_dir, target_usd)
        usd_import.save_image_grids(output_dir, "validation", validation["fixed_capture_validation"]["records"])
        usd_import.save_image_grids(output_dir, "inspection", validation["visual_inspection_capture_validation"]["records"])
        video_report = (
            usd_import.make_flythrough(output_dir, validation["visual_inspection_capture_validation"]["records"])
            if args.make_flythrough
            else {"mp4_created": False, "video_path": None, "frame_count": 0}
        )
        write_success_retry_reports(output_dir, validation, video_report, target_usd)
        gate.update(
            {
                "retry_result": "succeeded",
                "result": "succeeded",
                "scene_loaded": True,
                "validation_rgb_count": validation["fixed_capture_validation"]["nonblank_rgb_count"],
                "validation_depth_count": validation["fixed_capture_validation"]["finite_positive_depth_count"],
                "inspection_rgb_count": validation["visual_inspection_capture_validation"]["nonblank_rgb_count"],
                "inspection_depth_count": validation["visual_inspection_capture_validation"]["finite_positive_depth_count"],
                "observed_state_final": str(output_dir / "observed_state_final.npy"),
                "visual_html": str(output_dir / "visual_inspection_index.html"),
                "mp4": str(output_dir / "usd_scene_flythrough.mp4") if (output_dir / "usd_scene_flythrough.mp4").is_file() else None,
                "flythrough_frames": str(output_dir / "usd_scene_flythrough_frames"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        excerpt = repr(exc)
        gate.update({"retry_result": "failed", "result": "failed", "scene_loaded": False, "isaac_retry_error": excerpt})
        for name in [
            "scene_load_validation_defaultprim_retry.json",
            "scene_load_validation_defaultprim_retry.md",
            "fixed_capture_validation_defaultprim_retry.json",
            "fixed_capture_validation_defaultprim_retry.md",
            "visual_inspection_capture_validation_defaultprim_retry.json",
            "visual_inspection_capture_validation_defaultprim_retry.md",
            "observed_state_validation_summary_defaultprim_retry.json",
            "observed_state_validation_summary_defaultprim_retry.md",
            "observed_state_final.npy",
            "visual_inspection_index.html",
            "usd_scene_flythrough.mp4",
            "rgb_validation_grid.png",
            "depth_validation_grid.png",
            "rgb_inspection_grid.png",
            "depth_inspection_grid.png",
            "observed_topdown_final.png",
        ]:
            path = output_dir / name
            if path.is_file():
                path.unlink()
        blocker = {
            "blocked": True,
            "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
            "reason": "isaac_validation_retry_failed",
            "fixed_usd": str(target_usd),
            "attempt_count": 1,
            "error_excerpt": excerpt,
            "no_success_outputs_fabricated": True,
        }
        save_json(output_dir / "isaac_defaultprim_retry_blocker.json", blocker)
        write_text(output_dir / "isaac_defaultprim_retry_blocker.md", markdown_table("Isaac DefaultPrim Retry Blocker", blocker))
        write_text(output_dir / "kit_error_excerpt.txt", excerpt)
    finally:
        if simulation_app is not None:
            try:
                simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
            except BaseException as exc:  # noqa: BLE001
                gate["simulation_app_close_error"] = repr(exc)
    save_json(output_dir / "isaac_validation_gate_report.json", gate)
    write_text(output_dir / "isaac_validation_gate_report.md", markdown_table("Isaac Validation Gate Report", gate))
    return gate


def write_success_retry_reports(output_dir: Path, validation: dict[str, Any], video_report: dict[str, Any], target_usd: Path) -> None:
    pairs = [
        ("scene_load_validation", "scene_load_validation_defaultprim_retry"),
        ("fixed_capture_validation", "fixed_capture_validation_defaultprim_retry"),
        ("visual_inspection_capture_validation", "visual_inspection_capture_validation_defaultprim_retry"),
    ]
    for src, dst in pairs:
        src_path = output_dir / f"{src}.json"
        if src_path.is_file():
            data = read_json(src_path)
        else:
            data = validation.get(src, {})
        data["retry_stage"] = "Stage 4A-6.6c-usd-defaultprim-fix"
        data["fixed_usd"] = str(target_usd)
        save_json(output_dir / f"{dst}.json", data)
        write_text(output_dir / f"{dst}.md", markdown_table(dst.replace("_", " ").title(), data))
    observed = validation.get("observed_summary", {})
    observed["retry_stage"] = "Stage 4A-6.6c-usd-defaultprim-fix"
    observed["observed_state_final"] = str(output_dir / "observed_state_final.npy")
    save_json(output_dir / "observed_state_validation_summary_defaultprim_retry.json", observed)
    write_text(output_dir / "observed_state_validation_summary_defaultprim_retry.md", markdown_table("Observed State Validation Summary DefaultPrim Retry", observed))
    write_manual_review_gate(output_dir, attempted=True, succeeded=True)
    write_success_html(output_dir, target_usd, video_report)


def write_success_html(output_dir: Path, target_usd: Path, video_report: dict[str, Any]) -> None:
    image_names = [
        "usd_scene_layout_topdown.png",
        "start_variants_topdown.png",
        "validation_camera_poses_topdown.png",
        "inspection_camera_poses_topdown.png",
        "rgb_validation_grid.png",
        "depth_validation_grid.png",
        "rgb_inspection_grid.png",
        "depth_inspection_grid.png",
        "observed_topdown_final.png",
        "living_area_closeup.png",
        "kitchen_or_counter_closeup.png",
        "bedroom_or_private_room_closeup.png",
        "bathroom_or_small_room_closeup.png",
        "hallway_or_junction_closeup.png",
        "clutter_or_storage_closeup.png",
    ]
    figures = "\n".join(
        f'<figure><img src="{html.escape(name)}" width="360"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in image_names
        if (output_dir / name).is_file()
    )
    if video_report.get("mp4_created"):
        video = '<video controls width="720" src="usd_scene_flythrough.mp4"></video>'
    else:
        video = '<p><a href="usd_scene_flythrough_frames/">Fallback flythrough frames</a></p>'
    body = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Stage 4A-6.6c USD defaultPrim retry</title></head>
<body>
<h1>Stage 4A-6.6c USD defaultPrim retry</h1>
<p>Fixed USD: <code>{html.escape(str(target_usd))}</code></p>
<p>human_visual_inspection_done: <code>false</code>; formal_expert_sampling_ready: <code>false</code>; full_expert_dataset_ready: <code>false</code>.</p>
{figures}
<h2>Flythrough</h2>
{video}
</body>
</html>"""
    write_text(output_dir / "visual_inspection_index.html", body)


def write_summary(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    previous: dict[str, Any],
    inspection: dict[str, Any],
    decision: dict[str, Any],
    patch_report: dict[str, Any],
    target_usd: Path,
    validation: dict[str, Any] | None,
    scene_report: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    gate = gate or {}
    validation = validation or {}
    scene_report = scene_report or {}
    retry_succeeded = gate.get("retry_result") == "succeeded"
    blocked = not retry_succeeded
    summary = {
        "stage": "Stage 4A-6.6c-usd-defaultprim-fix",
        "completed": bool(retry_succeeded),
        "blocked": bool(blocked),
        "main_blocker": None if retry_succeeded else gate.get("reason") or gate.get("isaac_retry_error") or patch_report.get("reason"),
        "elapsed_seconds": float(elapsed_seconds),
        "source_usd": str(Path(args.source_usd).resolve()),
        "original_staged_usd": str(Path(args.original_staged_usd).resolve()),
        "localized_root": str(Path(args.localized_root).resolve()),
        "localized_usd": str(Path(args.localized_usd).resolve()),
        "fixed_root": str(Path(args.fixed_root).resolve()),
        "fixed_usd": str(Path(args.fixed_root).resolve() / Path(args.localized_usd).name),
        "wrapper_usd": str(Path(args.fixed_root).resolve() / "home_like_scene_v1_wrapper.usda"),
        "target_usd": str(target_usd),
        "previous_dependency_files": previous.get("dependency_file_count"),
        "remote_official_refs_remaining": validation.get("remote_official_refs_remaining"),
        "omniverse_refs_remaining": validation.get("omniverse_refs_remaining"),
        "unresolved_local_deps_remaining": validation.get("unresolved_local_deps_remaining"),
        "dependency_closure_complete": validation.get("dependency_closure_complete"),
        "existing_defaultPrim": inspection.get("existing_layer_defaultPrim"),
        "top_level_prims": [row.get("path") for row in inspection.get("top_level_prims", [])],
        "world_exists": inspection.get("world_exists"),
        "chosen_defaultPrim": decision.get("chosen_defaultPrim"),
        "chosen_defaultPrim_path": decision.get("chosen_defaultPrim_path"),
        "wrapper_used": bool(read_json(output_dir / "wrapper_usd_report.json").get("wrapper_used")) if (output_dir / "wrapper_usd_report.json").is_file() else False,
        "defaultPrim_valid": validation.get("defaultPrim_valid"),
        "source_usd_modified": False,
        "original_staged_usd_modified": False,
        "scene_factory_updated": scene_report.get("scene_factory_updated"),
        "home_like_scene_v1_target": scene_report.get("home_like_scene_v1_target"),
        "larger_complex_scene_v1_disabled": scene_report.get("larger_complex_scene_v1_disabled"),
        "procedural_fallback": False,
        "validation_gate_passed": gate.get("validation_gate_passed"),
        "isaac_attempted": gate.get("retry_attempted", False),
        "attempt_count": gate.get("retry_attempt_count", 0),
        "isaac_result": gate.get("retry_result", "not_attempted"),
        "rgb_depth": {
            "validation_rgb_count": gate.get("validation_rgb_count"),
            "validation_depth_count": gate.get("validation_depth_count"),
            "inspection_rgb_count": gate.get("inspection_rgb_count"),
            "inspection_depth_count": gate.get("inspection_depth_count"),
        },
        "observed_state_final": gate.get("observed_state_final"),
        "visual_html": gate.get("visual_html"),
        "mp4": gate.get("mp4"),
        "human_visual_inspection_done": False,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
        "rollout": False,
        "selected_action": False,
        "expert_sampling": False,
        "map_predict": False,
        "sscnet_inference": False,
        "prediction_npz": False,
        "training_rl_gdpo_ppo_bc_il": False,
        "checkpoint_modified": False,
        "test_log": str(WORKSPACE / "logs/stage4a66c_usd_defaultprim_fix_test.log"),
        "py_compile_log": str(WORKSPACE / "logs/stage4a66c_usd_defaultprim_fix_py_compile.log"),
    }
    save_json(output_dir / "stage4a66c_usd_defaultprim_fix_summary.json", summary)
    write_text(output_dir / "stage4a66c_usd_defaultprim_fix_summary.md", markdown_table("Stage 4A-6.6c USD DefaultPrim Fix Summary", summary))
    next_text = (
        "Proceed to Stage 4A-6.6d USD scene audit + human visual review. Keep formal expert sampling disabled."
        if retry_succeeded
        else "Fix the remaining USD load issue or ask teammate for a root USD with valid defaultPrim. Keep Stage 4A-6.6d and Stage 4A-6.7 blocked."
    )
    write_text(output_dir / "recommended_next_faithful_step.md", "# Recommended Next Faithful Step\n\n" + next_text)
    return summary


def update_project_context(output_dir: Path, summary: dict[str, Any]) -> None:
    section = "\n".join(
        [
            "",
            "Stage 4A-6.6c-usd-defaultprim-fix result:",
            "",
            f"- Output: `{output_dir}`.",
            "- Previous dependency closure remains complete: 67 initial URLs, 278 package files, remote refs 0, omniverse refs 0, unresolved local deps 0.",
            "- Previous Isaac blocker was the localized USD missing `defaultPrim` / concrete spawn target.",
            f"- Chosen fix: `defaultPrim={summary.get('chosen_defaultPrim')}` on `{summary.get('fixed_usd')}`; wrapper used `{summary.get('wrapper_used')}`.",
            f"- scene_factory points to `{summary.get('home_like_scene_v1_target')}`; larger_complex_scene_v1 remains disabled.",
            f"- Isaac validation attempted: `{summary.get('isaac_attempted')}`; attempt_count `{summary.get('attempt_count')}`; result `{summary.get('isaac_result')}`.",
            f"- RGB/depth summary: `{summary.get('rgb_depth')}`.",
            f"- observed_state_final: `{summary.get('observed_state_final')}`.",
            f"- visual HTML: `{summary.get('visual_html')}`.",
            f"- MP4/flythrough: `{summary.get('mp4') or (output_dir / 'usd_scene_flythrough_frames')}`.",
            "- Gates remain closed: `human_visual_inspection_done=false`, `formal_expert_sampling_ready=false`, `full_expert_dataset_ready=false`, `stage4a66d_executed=false`, `stage4a67_executed=false`.",
            "- No rollout, selected action, expert sampling, map_predict, SSCNet inference, prediction NPZ, checkpoint change, or RL/GDPO/PPO/BC/IL was run.",
            f"- Next: `{('Stage 4A-6.6d USD scene audit + human visual review' if summary.get('isaac_result') == 'succeeded' else 'fix remaining USD load issue or ask teammate for root USD with valid defaultPrim')}`.",
            "",
        ]
    )
    for path in CONTEXT_FILES:
        original = path.read_text(encoding="utf-8")
        path.write_text(original.rstrip() + "\n" + section, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4A-6.6c USD defaultPrim repair.")
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--original_staged_usd", required=True)
    parser.add_argument("--localized_usd", required=True)
    parser.add_argument("--localized_root", required=True)
    parser.add_argument("--fixed_root", required=True)
    parser.add_argument("--previous_download_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_variant", default="home_like_scene_v1")
    parser.add_argument("--preferred_default_prim", default="World")
    parser.add_argument("--create_wrapper_if_multiple_roots", action="store_true")
    parser.add_argument("--update_scene_factory_to_fixed_usd", action="store_true")
    parser.add_argument("--retry_isaac_if_defaultprim_fixed", action="store_true")
    parser.add_argument("--min_validation_poses", type=int, default=20)
    parser.add_argument("--min_inspection_poses", type=int, default=36)
    parser.add_argument("--make_flythrough", action="store_true")
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_formal_expert_sampling", action="store_true")
    parser.add_argument("--no_map_predict", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "__pxr_child":
        pxr_child_main(sys.argv[2])
        return

    started = time.perf_counter()
    args = parse_args()
    if args.scene_variant != "home_like_scene_v1":
        raise ValueError("This stage only supports --scene_variant home_like_scene_v1")
    if not (args.no_rollout and args.no_formal_expert_sampling and args.no_map_predict and args.no_rl_gdpo):
        raise ValueError("All negative-scope flags are required.")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_no_scope_reports(output_dir)
    write_manual_review_gate(output_dir, attempted=False, succeeded=False)

    source_usd = Path(args.source_usd).resolve()
    original_staged_usd = Path(args.original_staged_usd).resolve()
    localized_usd = Path(args.localized_usd).resolve()
    fixed_root = Path(args.fixed_root).resolve()
    source_hash_before = sha256_file(source_usd)
    staged_hash_before = sha256_file(original_staged_usd)

    loaded, previous = load_context_and_previous(args, output_dir)
    if not (
        loaded["dependencies_complete_confirmed"]
        and loaded["remote_official_refs_remaining_confirmed"]
        and loaded["omniverse_refs_remaining_confirmed"]
        and loaded["unresolved_local_deps_remaining_confirmed"]
        and loaded["previous_retry_failed_only_missing_defaultPrim"]
    ):
        raise RuntimeError("Previous stage evidence does not match the authorized defaultPrim-fix preconditions.")

    previous_package = read_json(Path(args.previous_download_output_dir).resolve() / "localized_package_manifest.json")
    inspection = inspect_localized_usd(output_dir, localized_usd)
    decision = choose_defaultprim(output_dir, inspection, str(args.preferred_default_prim))
    patch_report, target_usd = apply_defaultprim_fix(args, output_dir, inspection, decision)
    if patch_report.get("blocked"):
        write_fixed_hash_manifest(output_dir, fixed_root, source_usd, original_staged_usd, localized_usd) if fixed_root.exists() else None
        blocker = read_json(output_dir / "defaultprim_root_selection_blocker.json")
        summary = write_summary(
            args,
            output_dir,
            previous=previous,
            inspection=inspection,
            decision=decision,
            patch_report=blocker,
            target_usd=target_usd,
            validation=None,
            scene_report=None,
            gate=None,
            elapsed_seconds=time.perf_counter() - started,
        )
        update_project_context(output_dir, summary)
        print(json.dumps({"completed": False, "blocked": True, "reason": blocker["reason"], "output_dir": str(output_dir)}, indent=2, sort_keys=True))
        return

    hash_manifest = write_fixed_hash_manifest(output_dir, fixed_root, source_usd, original_staged_usd, localized_usd)
    unresolved = scan_unresolved(output_dir, target_usd, fixed_root)
    validation = validate_fixed_usd(
        output_dir,
        target_usd,
        source_usd,
        original_staged_usd,
        localized_usd,
        source_hash_before,
        staged_hash_before,
        previous_package,
        unresolved,
    )
    scene_report = scene_factory_registration_report(output_dir, target_usd)
    gate = build_validation_gate(output_dir, target_usd, validation, scene_report, patch_report)
    gate["fixed_usd_hash_manifest_file_count"] = hash_manifest["file_count"]
    save_json(output_dir / "isaac_validation_gate_report.json", gate)
    write_text(output_dir / "isaac_validation_gate_report.md", markdown_table("Isaac Validation Gate Report", gate))

    if gate["retry_allowed"] and args.retry_isaac_if_defaultprim_fixed:
        gate = run_isaac_validation_once(args, output_dir, target_usd, gate)
    elif not gate["retry_allowed"]:
        blocker = {
            "blocked": True,
            "reason": gate["reason"],
            "fixed_usd": str(target_usd),
            "isaac_validation_attempted": False,
            "recommended_next": "Fix offline USD validation blockers before starting Isaac.",
        }
        save_json(output_dir / "defaultprim_root_selection_blocker.json", blocker)
        write_text(output_dir / "defaultprim_root_selection_blocker.md", markdown_table("DefaultPrim Root Selection Blocker", blocker))

    summary = write_summary(
        args,
        output_dir,
        previous=previous,
        inspection=inspection,
        decision=decision,
        patch_report=patch_report,
        target_usd=target_usd,
        validation=validation,
        scene_report=scene_report,
        gate=gate,
        elapsed_seconds=time.perf_counter() - started,
    )
    update_project_context(output_dir, summary)
    print(
        json.dumps(
            {
                "completed": summary["completed"],
                "blocked": summary["blocked"],
                "main_blocker": summary["main_blocker"],
                "fixed_usd": summary["fixed_usd"],
                "defaultPrim_valid": summary["defaultPrim_valid"],
                "isaac_attempted": summary["isaac_attempted"],
                "isaac_result": summary["isaac_result"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
