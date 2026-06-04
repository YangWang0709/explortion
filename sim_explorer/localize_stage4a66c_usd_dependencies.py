#!/usr/bin/env python3
"""Stage 4A-6.6c USD dependency localization audit.

This script only analyzes and, when fully justified, localizes dependencies
for the staged user-provided USD. It does not download assets, generate a
fallback scene, start Isaac, run rollout, run expert sampling, call
map_predict/SSCNet, create predictions, or train policies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_SOURCE_USD = WORKSPACE / "building_scene.usd"
DEFAULT_STAGED_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a66c_usd_home_like_scene_validation"
DEFAULT_DEPENDENCY_DIR = DEFAULT_STAGED_USD.parent / "dependencies"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
]
PREVIOUS_OUTPUT_FILES = [
    "isaac_headless_blocker_report.md",
    "usd_dependency_report.json",
    "usd_prim_inventory.json",
    "stage4a66c_usd_home_like_scene_summary.json",
    "scene_load_validation.json",
    "fixed_capture_validation.json",
    "observed_state_validation_summary.json",
]
REQUESTED_SEARCH_ROOTS = [
    Path("/home/ubuntu22"),
    Path("/home/ubuntu22/sc_explorer_ws"),
    Path("/home/ubuntu22/sc_explorer_ws/assets"),
    Path("/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1"),
    Path("/home/ubuntu22/IsaacLab"),
    Path("/home/ubuntu22/miniconda3/envs/env_isaaclab"),
    Path("/isaac-sim"),
    Path("/opt"),
    Path("/omniverse"),
    Path("/tmp"),
    Path("/var/tmp"),
]
SEARCH_EXTENSIONS = {
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".obj",
    ".fbx",
    ".glb",
    ".gltf",
    ".mtl",
    ".png",
    ".jpg",
    ".jpeg",
    ".exr",
    ".hdr",
    ".tga",
}
DEPENDENCY_COLUMNS = [
    "original_asset_path",
    "dependency_type",
    "is_remote",
    "is_absolute",
    "is_relative",
    "resolved_before",
    "local_candidate_found",
    "local_candidate_path",
    "copied_to_package",
    "patched_path",
    "still_missing",
    "source_prim",
    "reason_if_unresolved",
]
PXR_USD_LIB = Path(
    "/home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
PXR_ENV_LIB = Path("/home/ubuntu22/miniconda3/envs/env_isaaclab/lib")


@dataclass(frozen=True)
class FileCandidate:
    path: Path
    basename: str
    lower_basename: str
    normalized_basename: str
    lower_stem: str
    suffix: str
    size_bytes: int
    search_root: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "path") and hasattr(value, "resolvedPath"):
        return {"path": str(value.path), "resolvedPath": str(value.resolvedPath)}
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
            text = json.dumps(to_jsonable(value), sort_keys=True)
            if len(text) > 1800:
                text = text[:1800] + "..."
            value_text = f"`{text}`"
        else:
            value_text = f"`{value}`"
        lines.append(f"| {key} | {value_text} |")
    return "\n".join(lines)


def list_md(title: str, rows: list[str]) -> str:
    body = rows or ["No rows."]
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in body]])


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def canonicalize_asset_path(asset_path: str) -> str:
    text = str(asset_path or "").strip()
    if text.startswith("@") and text.endswith("@") and len(text) >= 2:
        text = text[1:-1]
    if text.startswith("https:/") and not text.startswith("https://"):
        text = "https://" + text[len("https:/") :]
    if text.startswith("http:/") and not text.startswith("http://"):
        text = "http://" + text[len("http:/") :]
    return text


def is_remote_asset(asset_path: str) -> bool:
    text = canonicalize_asset_path(asset_path).lower()
    return bool(
        re.match(r"^[a-z][a-z0-9+.-]*://", text)
        or text.startswith("omniverse://")
        or "omniverse-content-production" in text
        or text.startswith("omniverse:")
    )


def basename_from_asset(asset_path: str) -> str:
    text = canonicalize_asset_path(asset_path)
    text = text.split("?", 1)[0].split("#", 1)[0]
    return Path(text).name


def asset_relative_tail(asset_path: str) -> Path:
    text = canonicalize_asset_path(asset_path)
    marker = "/Assets/"
    if marker in text:
        return Path("Assets") / Path(text.split(marker, 1)[1])
    return Path(basename_from_asset(text))


def local_resolved_path(asset_path: str, staged_usd: Path) -> Path | None:
    if not asset_path or is_remote_asset(asset_path):
        return None
    candidate = Path(asset_path)
    if not candidate.is_absolute():
        candidate = staged_usd.parent / candidate
    return candidate


def prepare_pxr_import() -> None:
    if PXR_USD_LIB.is_dir() and str(PXR_USD_LIB) not in sys.path:
        sys.path.insert(0, str(PXR_USD_LIB))


def import_pxr() -> tuple[Any, Any, Any]:
    prepare_pxr_import()
    try:
        from pxr import Sdf, UsdUtils, Ar  # type: ignore
    except Exception as exc:
        hint = (
            "PXR import failed. Re-run with: "
            f"PYTHONPATH={PXR_USD_LIB}:$PYTHONPATH "
            f"LD_LIBRARY_PATH={PXR_ENV_LIB}:{PXR_USD_LIB / 'bin'}:$LD_LIBRARY_PATH "
            "/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python "
            "sim_explorer/localize_stage4a66c_usd_dependencies.py"
        )
        raise RuntimeError(f"{hint}\nOriginal error: {exc}") from exc
    return Sdf, UsdUtils, Ar


def list_editor_items(editor: Any) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for field in ("explicitItems", "addedItems", "prependedItems", "appendedItems"):
        try:
            rows.extend((field, item) for item in list(getattr(editor, field)))
        except Exception:
            pass
    return rows


def iter_prim_specs(layer: Any) -> list[Any]:
    out: list[Any] = []

    def walk(prim_spec: Any) -> None:
        out.append(prim_spec)
        try:
            children = list(prim_spec.nameChildren)
        except Exception:
            children = []
        for child in children:
            walk(child)

    for root in list(layer.rootPrims):
        walk(root)
    return out


def asset_path_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "path"):
        return str(value.path)
    if isinstance(value, str):
        return value
    return None


def collect_asset_values_from_property(prop: Any, Sdf: Any) -> list[str]:
    values: list[str] = []
    raw_values: list[Any] = []
    try:
        if hasattr(prop, "default") and prop.default is not None:
            raw_values.append(prop.default)
    except Exception:
        pass
    try:
        samples = getattr(prop, "timeSamples", {})
        if isinstance(samples, dict):
            raw_values.extend(samples.values())
    except Exception:
        pass
    for value in raw_values:
        if isinstance(value, Sdf.AssetPath):
            text = asset_path_to_string(value)
            if text:
                values.append(text)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Sdf.AssetPath):
                    text = asset_path_to_string(item)
                    if text:
                        values.append(text)
    return values


def dependency_record_base(
    *,
    asset_path: str,
    dependency_type: str,
    source_prim: str,
    staged_usd: Path,
) -> dict[str, Any]:
    canonical = canonicalize_asset_path(asset_path)
    remote = is_remote_asset(canonical)
    local_path = local_resolved_path(canonical, staged_usd)
    resolved_before = bool(local_path and local_path.exists())
    return {
        "original_asset_path": canonical,
        "dependency_type": dependency_type,
        "is_remote": remote,
        "is_absolute": bool(not remote and Path(canonical).is_absolute()),
        "is_relative": bool(not remote and not Path(canonical).is_absolute()),
        "resolved_before": resolved_before,
        "local_candidate_found": False,
        "local_candidate_path": None,
        "copied_to_package": False,
        "patched_path": None,
        "still_missing": not resolved_before,
        "source_prim": source_prim,
        "reason_if_unresolved": None if resolved_before else ("remote_dependency_not_localized" if remote else "local_path_missing"),
        "basename": basename_from_asset(canonical),
        "canonical_asset_path": canonical,
    }


def inspect_staged_usd(staged_usd: Path) -> dict[str, Any]:
    Sdf, UsdUtils, _Ar = import_pxr()
    layer = Sdf.Layer.FindOrOpen(str(staged_usd))
    if layer is None:
        raise RuntimeError(f"Sdf.Layer.FindOrOpen failed for {staged_usd}")

    records: list[dict[str, Any]] = []
    sublayers = list(layer.subLayerPaths)
    for sublayer in sublayers:
        records.append(
            dependency_record_base(
                asset_path=str(sublayer),
                dependency_type="sublayer",
                source_prim="<rootLayer>",
                staged_usd=staged_usd,
            )
        )

    reference_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    asset_attribute_rows: list[dict[str, Any]] = []
    material_asset_rows: list[dict[str, Any]] = []
    texture_rows: list[dict[str, Any]] = []
    for prim_spec in iter_prim_specs(layer):
        prim_path = str(prim_spec.path)
        for list_name, ref in list_editor_items(prim_spec.referenceList):
            asset_path = str(getattr(ref, "assetPath", "") or "")
            if not asset_path:
                continue
            rec = dependency_record_base(
                asset_path=asset_path,
                dependency_type="reference",
                source_prim=prim_path,
                staged_usd=staged_usd,
            )
            rec["list_op"] = list_name
            rec["prim_path_in_reference"] = str(getattr(ref, "primPath", "") or "")
            records.append(rec)
            reference_rows.append(rec)
        for list_name, payload in list_editor_items(prim_spec.payloadList):
            asset_path = str(getattr(payload, "assetPath", "") or "")
            if not asset_path:
                continue
            rec = dependency_record_base(
                asset_path=asset_path,
                dependency_type="payload",
                source_prim=prim_path,
                staged_usd=staged_usd,
            )
            rec["list_op"] = list_name
            rec["prim_path_in_payload"] = str(getattr(payload, "primPath", "") or "")
            records.append(rec)
            payload_rows.append(rec)
        try:
            properties = list(prim_spec.properties)
        except Exception:
            properties = []
        for prop in properties:
            for asset_path in collect_asset_values_from_property(prop, Sdf):
                attr_name = str(getattr(prop, "name", ""))
                dep_type = "texture" if Path(basename_from_asset(asset_path)).suffix.lower() in {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tga"} else "material"
                rec = dependency_record_base(
                    asset_path=asset_path,
                    dependency_type=dep_type,
                    source_prim=prim_path,
                    staged_usd=staged_usd,
                )
                rec["attribute"] = attr_name
                records.append(rec)
                asset_attribute_rows.append(rec)
                if dep_type == "texture":
                    texture_rows.append(rec)
                else:
                    material_asset_rows.append(rec)

    compute_all_dependencies: dict[str, Any] = {
        "all_dependency_layers": [],
        "all_dependency_assets": [],
        "unresolved_paths": [],
        "error": None,
    }
    try:
        all_layers, all_assets, unresolved = UsdUtils.ComputeAllDependencies(str(staged_usd))
        compute_all_dependencies["all_dependency_layers"] = [str(layer_item.identifier) for layer_item in all_layers]
        compute_all_dependencies["all_dependency_assets"] = [asset_path_to_string(asset) or str(asset) for asset in all_assets]
        compute_all_dependencies["unresolved_paths"] = [canonicalize_asset_path(asset_path_to_string(path) or str(path)) for path in unresolved]
    except Exception as exc:
        compute_all_dependencies["error"] = str(exc)

    return {
        "layer_identifier": str(layer.identifier),
        "sublayers": sublayers,
        "dependency_records": records,
        "reference_rows": reference_rows,
        "payload_rows": payload_rows,
        "asset_attribute_rows": asset_attribute_rows,
        "material_asset_rows": material_asset_rows,
        "texture_rows": texture_rows,
        "compute_all_dependencies": compute_all_dependencies,
    }


def effective_search_roots(roots: list[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    existing = [root.resolve() for root in roots if root.exists()]
    effective: list[Path] = []
    notes: list[dict[str, Any]] = []
    for root in roots:
        resolved = root.resolve() if root.exists() else root
        if not root.exists():
            notes.append({"requested_root": str(root), "exists": False, "effective": False, "reason": "missing"})
            continue
        covered_by = None
        for prior in effective:
            try:
                resolved.relative_to(prior)
                covered_by = prior
                break
            except ValueError:
                pass
        if covered_by is not None:
            notes.append(
                {
                    "requested_root": str(root),
                    "exists": True,
                    "effective": False,
                    "reason": "covered_by_parent_root",
                    "covered_by": str(covered_by),
                }
            )
            continue
        effective.append(resolved)
        notes.append({"requested_root": str(root), "exists": True, "effective": True, "reason": "searched"})
    return effective, notes


def rg_available() -> bool:
    return shutil.which("rg") is not None


def list_files_with_rg(roots: list[Path]) -> list[Path]:
    cmd = ["rg", "--files", "--hidden", "--no-ignore"]
    for ext in sorted(SEARCH_EXTENSIONS):
        cmd.extend(["-g", f"*{ext}"])
        cmd.extend(["-g", f"*{ext.upper()}"])
    cmd.extend(str(root) for root in roots)
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    files = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return files


def list_files_with_find(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        find_expr: list[str] = []
        for idx, ext in enumerate(sorted(SEARCH_EXTENSIONS)):
            if idx > 0:
                find_expr.append("-o")
            find_expr.extend(["-iname", f"*{ext}"])
        cmd = ["find", str(root), "-type", "f", "("] + find_expr + [")"]
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
        )
        files.extend(Path(line.strip()) for line in result.stdout.splitlines() if line.strip())
    return files


def file_search_index(search_roots: list[Path]) -> tuple[list[FileCandidate], dict[str, Any]]:
    effective_roots, root_notes = effective_search_roots(search_roots)
    raw_files: list[Path] = []
    search_backend = "rg --files --hidden --no-ignore"
    if rg_available():
        try:
            raw_files = list_files_with_rg(effective_roots)
        except Exception:
            raw_files = []
    if not raw_files:
        search_backend = "find"
        raw_files = list_files_with_find(effective_roots)

    seen: set[Path] = set()
    candidates: list[FileCandidate] = []
    for raw_path in raw_files:
        try:
            path = raw_path.resolve()
        except Exception:
            path = raw_path
        if path in seen:
            continue
        seen.add(path)
        suffix = path.suffix.lower()
        if suffix not in SEARCH_EXTENSIONS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        root_hit = ""
        for root in effective_roots:
            try:
                path.relative_to(root)
                root_hit = str(root)
                break
            except ValueError:
                pass
        candidates.append(
            FileCandidate(
                path=path,
                basename=path.name,
                lower_basename=path.name.lower(),
                normalized_basename=normalize_name(path.name),
                lower_stem=path.stem.lower(),
                suffix=suffix,
                size_bytes=int(stat.st_size),
                search_root=root_hit,
            )
        )
    metadata = {
        "search_backend": search_backend,
        "requested_search_roots": [str(root) for root in search_roots],
        "search_root_notes": root_notes,
        "effective_search_roots": [str(root) for root in effective_roots],
        "search_extensions": sorted(SEARCH_EXTENSIONS),
        "indexed_file_count": len(candidates),
    }
    return candidates, metadata


def build_candidate_maps(candidates: list[FileCandidate]) -> dict[str, dict[str, list[FileCandidate]]]:
    maps: dict[str, dict[str, list[FileCandidate]]] = {
        "basename": defaultdict(list),
        "lower_basename": defaultdict(list),
        "normalized_basename": defaultdict(list),
        "lower_stem": defaultdict(list),
    }
    for candidate in candidates:
        maps["basename"][candidate.basename].append(candidate)
        maps["lower_basename"][candidate.lower_basename].append(candidate)
        maps["normalized_basename"][candidate.normalized_basename].append(candidate)
        maps["lower_stem"][candidate.lower_stem].append(candidate)
    return maps


def candidate_row(dep: dict[str, Any], candidate: FileCandidate, strategy: str, score: int) -> dict[str, Any]:
    return {
        "original_asset_path": dep["original_asset_path"],
        "dependency_type": dep["dependency_type"],
        "source_prim": dep["source_prim"],
        "target_basename": dep["basename"],
        "match_strategy": strategy,
        "score": score,
        "candidate_path": str(candidate.path),
        "candidate_basename": candidate.basename,
        "candidate_extension": candidate.suffix,
        "candidate_size_bytes": candidate.size_bytes,
        "candidate_search_root": candidate.search_root,
    }


def find_candidates_for_dependency(dep: dict[str, Any], maps: dict[str, dict[str, list[FileCandidate]]]) -> list[dict[str, Any]]:
    basename = str(dep["basename"])
    lower_basename = basename.lower()
    normalized_basename = normalize_name(basename)
    lower_stem = Path(basename).stem.lower()
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    def add(strategy: str, score: int, items: list[FileCandidate]) -> None:
        for item in sorted(items, key=lambda cand: (cand.path.as_posix(), cand.size_bytes)):
            key = str(item.path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            rows.append(candidate_row(dep, item, strategy, score))

    add("exact_basename", 100, maps["basename"].get(basename, []))
    add("case_insensitive_basename", 90, maps["lower_basename"].get(lower_basename, []))
    add("normalized_basename", 75, maps["normalized_basename"].get(normalized_basename, []))
    same_stem = [
        item
        for item in maps["lower_stem"].get(lower_stem, [])
        if item.lower_basename != lower_basename
    ]
    add("same_stem_different_extension", 55, same_stem)
    if Path(basename).suffix.lower() in {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tga"}:
        add("texture_basename_search", 95, maps["lower_basename"].get(lower_basename, []))
    return rows[:30]


def annotate_with_candidates(
    dependency_records: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> None:
    best_by_asset: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        key = str(row["original_asset_path"])
        current = best_by_asset.get(key)
        if current is None or int(row["score"]) > int(current["score"]):
            best_by_asset[key] = row
    for dep in dependency_records:
        row = best_by_asset.get(str(dep["original_asset_path"]))
        if row is None:
            continue
        dep["local_candidate_found"] = True
        dep["local_candidate_path"] = row["candidate_path"]
        if dep["still_missing"]:
            dep["reason_if_unresolved"] = "local_candidate_found_but_not_applied_without_complete_trusted_package"


def trusted_complete_localization_possible(
    unique_missing: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> bool:
    exact_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if row.get("match_strategy") == "exact_basename":
            exact_candidates[str(row["original_asset_path"])].append(row)
    for dep in unique_missing:
        rows = exact_candidates.get(str(dep["original_asset_path"]), [])
        if len(rows) != 1:
            return False
    return bool(unique_missing)


def copy_and_patch_if_complete(
    *,
    staged_usd: Path,
    dependency_dir: Path,
    dependency_records: list[dict[str, Any]],
    unique_missing: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    apply_patches: bool,
) -> dict[str, Any]:
    if not apply_patches:
        return {
            "patch_attempted": False,
            "patch_applied": False,
            "reason": "patch disabled; analysis/request only",
            "patch_count": 0,
            "copied_file_count": 0,
            "staged_sha256_after": sha256_file(staged_usd),
        }
    if not trusted_complete_localization_possible(unique_missing, candidate_rows):
        return {
            "patch_attempted": False,
            "patch_applied": False,
            "reason": "not all unique missing dependencies had exactly one exact-basename local candidate",
            "patch_count": 0,
            "copied_file_count": 0,
            "staged_sha256_after": sha256_file(staged_usd),
        }

    Sdf, _UsdUtils, _Ar = import_pxr()
    best: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        key = str(row["original_asset_path"])
        if row.get("match_strategy") == "exact_basename" and key not in best:
            best[key] = row

    local_paths: dict[str, str] = {}
    copied = 0
    dependency_dir.mkdir(parents=True, exist_ok=True)
    for dep in unique_missing:
        asset_path = str(dep["original_asset_path"])
        src = Path(best[asset_path]["candidate_path"])
        rel_tail = asset_relative_tail(asset_path)
        dst = dependency_dir / rel_tail
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256_file(dst) != sha256_file(src):
            shutil.copy2(src, dst)
            copied += 1
        local_paths[asset_path] = os.path.relpath(dst, staged_usd.parent)

    layer = Sdf.Layer.FindOrOpen(str(staged_usd))
    if layer is None:
        raise RuntimeError(f"Sdf.Layer.FindOrOpen failed for patching {staged_usd}")
    patch_count = 0
    for prim_spec in iter_prim_specs(layer):
        for field in ("explicitItems", "addedItems", "prependedItems", "appendedItems"):
            editor = prim_spec.referenceList
            try:
                items = list(getattr(editor, field))
            except Exception:
                continue
            changed = False
            new_items = []
            for ref in items:
                canonical = canonicalize_asset_path(str(getattr(ref, "assetPath", "") or ""))
                replacement = local_paths.get(canonical)
                if replacement:
                    new_items.append(Sdf.Reference(replacement, ref.primPath, ref.layerOffset, ref.customData))
                    changed = True
                    patch_count += 1
                else:
                    new_items.append(ref)
            if changed:
                setattr(editor, field, new_items)
    layer.Save()
    for dep in dependency_records:
        replacement = local_paths.get(str(dep["original_asset_path"]))
        if replacement:
            dep["copied_to_package"] = True
            dep["patched_path"] = replacement
            dep["still_missing"] = False
            dep["reason_if_unresolved"] = None
    return {
        "patch_attempted": True,
        "patch_applied": patch_count > 0,
        "reason": "all unique missing dependencies had a unique exact local candidate" if patch_count else "no reference items required patching",
        "patch_count": patch_count,
        "copied_file_count": copied,
        "dependency_dir": str(dependency_dir),
        "staged_sha256_after": sha256_file(staged_usd),
    }


def unique_missing_records(dependency_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_asset: dict[str, dict[str, Any]] = {}
    source_prims: dict[str, list[str]] = defaultdict(list)
    for dep in dependency_records:
        if not dep.get("still_missing"):
            continue
        key = str(dep["original_asset_path"])
        source = str(dep.get("source_prim") or "")
        if source and source not in source_prims[key]:
            source_prims[key].append(source)
        if key not in by_asset:
            by_asset[key] = dict(dep)
    rows: list[dict[str, Any]] = []
    for key, dep in sorted(by_asset.items()):
        item = dict(dep)
        item["source_prim"] = ";".join(source_prims.get(key, []))
        item["occurrence_count"] = sum(1 for row in dependency_records if str(row["original_asset_path"]) == key)
        rows.append(item)
    return rows


def load_input_context(output_dir: Path, source_usd: Path, staged_usd: Path) -> dict[str, Any]:
    context_entries = []
    for path in CONTEXT_FILES:
        context_entries.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    previous = []
    for name in PREVIOUS_OUTPUT_FILES:
        path = output_dir / name
        entry: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
        if path.suffix == ".json" and path.is_file():
            try:
                data = read_json(path)
                if name == "stage4a66c_usd_home_like_scene_summary.json":
                    entry["blocked"] = data.get("blocked")
                    entry["isaac_headless_loaded_usd"] = data.get("isaac_headless_loaded_usd")
                    entry["validation_rgb_depth_valid"] = data.get("validation_rgb_depth_valid")
                    entry["observed_state_valid"] = data.get("observed_state_valid")
                    entry["mp4"] = data.get("mp4")
                    entry["stage4a67_executed"] = data.get("stage4a67_executed")
                elif name == "usd_dependency_report.json":
                    entry["dependencies_complete"] = data.get("dependencies_complete")
                    entry["missing_dependency_count"] = len(data.get("missing_dependencies", []))
                    entry["reference_count"] = data.get("reference_count")
                    entry["payload_count"] = data.get("payload_count")
                elif name == "scene_load_validation.json":
                    entry["scene_loaded"] = data.get("scene_loaded")
                    entry["blocked"] = data.get("blocked")
                    entry["failure_reason"] = data.get("failure_reason")
                elif name == "fixed_capture_validation.json":
                    entry["nonblank_rgb_count"] = data.get("nonblank_rgb_count")
                    entry["finite_positive_depth_count"] = data.get("finite_positive_depth_count")
                    entry["blocked"] = data.get("blocked")
                elif name == "observed_state_validation_summary.json":
                    entry["observed_state_final_created"] = data.get("observed_state_final_created")
                    entry["blocked"] = data.get("blocked")
            except Exception as exc:
                entry["json_load_error"] = str(exc)
        previous.append(entry)
    forbidden_absent = {
        "validation_rgb_png_count": len(list(output_dir.glob("validation_rgb_*.png"))),
        "validation_depth_npy_count": len(list(output_dir.glob("validation_depth_*.npy"))),
        "inspection_rgb_png_count": len(list(output_dir.glob("inspection_rgb_*.png"))),
        "inspection_depth_npy_count": len(list(output_dir.glob("inspection_depth_*.npy"))),
        "observed_state_final_exists": (output_dir / "observed_state_final.npy").exists(),
        "mp4_exists": (output_dir / "usd_scene_flythrough.mp4").exists(),
    }
    summary = {
        "stage": "Stage 4A-6.6c-usd-dependency-fix",
        "created_at_utc": utc_now(),
        "source_usd": str(source_usd),
        "staged_usd": str(staged_usd),
        "source_exists": source_usd.is_file(),
        "staged_exists": staged_usd.is_file(),
        "source_sha256": sha256_file(source_usd),
        "staged_sha256_before": sha256_file(staged_usd),
        "source_modified": False,
        "context_files": context_entries,
        "previous_output_files": previous,
        "previous_run_confirmed_blocked_by_isaac_load": True,
        "previous_run_confirmed_no_rgb_depth": forbidden_absent["validation_rgb_png_count"] == 0
        and forbidden_absent["validation_depth_npy_count"] == 0
        and forbidden_absent["inspection_rgb_png_count"] == 0
        and forbidden_absent["inspection_depth_npy_count"] == 0,
        "previous_run_confirmed_no_observed_state_final": not forbidden_absent["observed_state_final_exists"],
        "previous_run_confirmed_no_mp4": not forbidden_absent["mp4_exists"],
        "cannot_enter_stage4a66d": True,
        "cannot_enter_stage4a67": True,
        "forbidden_artifact_presence": forbidden_absent,
    }
    return summary


def write_package_request(
    output_dir: Path,
    unique_missing: list[dict[str, Any]],
    expanded: dict[str, Any],
) -> dict[str, Any]:
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for dep in unique_missing:
        asset = str(dep["original_asset_path"])
        canonical = canonicalize_asset_path(asset)
        if "/Assets/Isaac/4.5/" in canonical:
            prefix = canonical.split("/Assets/Isaac/4.5/", 1)[1].rsplit("/", 1)[0]
        else:
            prefix = "other"
        by_prefix[prefix].append(canonical)
    request = {
        "stage": "Stage 4A-6.6c-usd-dependency-fix",
        "request_type": "dependency_package_request",
        "reason": "staged USD still has missing remote Omniverse/S3 dependencies; Isaac retry is not allowed until these are local-only",
        "source_usd": expanded["source_usd"],
        "staged_usd": expanded["staged_usd"],
        "dependency_dir_expected": str(DEFAULT_DEPENDENCY_DIR),
        "missing_unique_dependency_count": len(unique_missing),
        "missing_occurrence_count": expanded["still_missing_occurrence_count"],
        "requested_package_shape": {
            "preferred": "Provide a local package preserving the Assets/Isaac/4.5/Isaac/... directory tree under current_environment/dependencies, including transitive USD/material/texture dependencies.",
            "acceptable": "Provide a lighter fully local USD that keeps the scene home-like and has no remote references.",
            "not_allowed": [
                "random asset downloads",
                "unknown-license substitutions",
                "procedural/cuboid fallback scene",
                "old larger_complex_scene_v1",
            ],
        },
        "missing_dependencies": [
            {
                "asset_path": dep["original_asset_path"],
                "basename": dep["basename"],
                "occurrence_count": dep.get("occurrence_count"),
                "source_prims": str(dep.get("source_prim", "")).split(";") if dep.get("source_prim") else [],
            }
            for dep in unique_missing
        ],
        "missing_grouped_by_remote_directory": {key: sorted(values) for key, values in sorted(by_prefix.items())},
        "isaac_retry_executed": False,
        "isaac_retry_allowed": False,
    }
    save_json(output_dir / "dependency_package_request.json", request)
    lines = [
        "# Dependency Package Request",
        "",
        "Stage 4A-6.6c-usd-dependency-fix could not localize the staged USD.",
        "",
        f"- source_usd: `{request['source_usd']}`",
        f"- staged_usd: `{request['staged_usd']}`",
        f"- expected local dependency root: `{request['dependency_dir_expected']}`",
        f"- missing unique dependencies: `{request['missing_unique_dependency_count']}`",
        f"- missing reference occurrences: `{request['missing_occurrence_count']}`",
        "- Isaac retry executed: `False`",
        "",
        "Please provide either:",
        "",
        "1. A local package preserving `Assets/Isaac/4.5/Isaac/...` under `current_environment/dependencies`, including all transitive USD/material/texture files.",
        "2. A lighter fully local USD with no remote references.",
        "",
        "Do not provide random downloads, unknown-license substitutions, procedural/cuboid fallback, or the old larger scene.",
        "",
        "## Missing Assets",
        "",
    ]
    for dep in unique_missing:
        lines.append(f"- `{dep['original_asset_path']}` occurrences `{dep.get('occurrence_count')}`")
    write_text(output_dir / "dependency_package_request.md", "\n".join(lines))
    return request


def write_reports(
    *,
    output_dir: Path,
    source_usd: Path,
    staged_usd: Path,
    input_summary: dict[str, Any],
    usd_info: dict[str, Any],
    search_metadata: dict[str, Any],
    dependency_records: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    patch_report: dict[str, Any],
) -> dict[str, Any]:
    unique_missing = unique_missing_records(dependency_records)
    remote_records = [dep for dep in dependency_records if dep.get("is_remote")]
    unresolved_tokens = [
        dep["original_asset_path"]
        for dep in dependency_records
        if any(token in str(dep["original_asset_path"]) for token in ("${", "$(", "<", ">"))
    ]
    duplicate_counts = Counter(str(dep["original_asset_path"]) for dep in dependency_records)
    duplicated_references = {
        asset: count
        for asset, count in sorted(duplicate_counts.items())
        if count > 1
    }
    compute_unresolved = usd_info["compute_all_dependencies"].get("unresolved_paths", [])
    expanded = {
        "stage": "Stage 4A-6.6c-usd-dependency-fix",
        "created_at_utc": utc_now(),
        "source_usd": str(source_usd),
        "staged_usd": str(staged_usd),
        "source_sha256": sha256_file(source_usd),
        "staged_sha256_before": input_summary.get("staged_sha256_before"),
        "staged_sha256_after": sha256_file(staged_usd),
        "source_modified": False,
        "dependency_dir": str(DEFAULT_DEPENDENCY_DIR),
        "sublayer_count": len(usd_info["sublayers"]),
        "reference_occurrence_count": len(usd_info["reference_rows"]),
        "reference_unique_asset_count": len({row["original_asset_path"] for row in usd_info["reference_rows"]}),
        "payload_occurrence_count": len(usd_info["payload_rows"]),
        "asset_attribute_count": len(usd_info["asset_attribute_rows"]),
        "material_asset_path_count": len(usd_info["material_asset_rows"]),
        "texture_asset_path_count": len(usd_info["texture_rows"]),
        "remote_dependency_occurrence_count": len(remote_records),
        "remote_unique_dependency_count": len({row["original_asset_path"] for row in remote_records}),
        "still_missing_occurrence_count": len([dep for dep in dependency_records if dep.get("still_missing")]),
        "still_missing_unique_count": len(unique_missing),
        "absolute_path_count": len([dep for dep in dependency_records if dep.get("is_absolute")]),
        "relative_path_count": len([dep for dep in dependency_records if dep.get("is_relative")]),
        "unresolved_token_count": len(unresolved_tokens),
        "unresolved_tokens": sorted(set(unresolved_tokens)),
        "duplicated_reference_asset_counts": duplicated_references,
        "compute_all_dependencies": usd_info["compute_all_dependencies"],
        "compute_unresolved_unique_count": len(set(compute_unresolved)),
        "local_search": search_metadata,
        "local_candidate_row_count": len(candidate_rows),
        "local_candidate_unique_dependency_count": len({row["original_asset_path"] for row in candidate_rows}),
        "patch_report": patch_report,
        "dependencies_complete_after_localization": len(unique_missing) == 0,
        "isaac_retry_allowed": len(unique_missing) == 0,
        "isaac_retry_executed": False,
        "dependency_records": dependency_records,
    }
    save_json(output_dir / "dependency_localization_input_summary.json", input_summary)
    write_text(
        output_dir / "dependency_localization_input_summary.md",
        markdown_table(
            "Dependency Localization Input Summary",
            {
                "source_usd": input_summary["source_usd"],
                "staged_usd": input_summary["staged_usd"],
                "source_sha256": input_summary["source_sha256"],
                "staged_sha256_before": input_summary["staged_sha256_before"],
                "previous_run_confirmed_blocked_by_isaac_load": input_summary["previous_run_confirmed_blocked_by_isaac_load"],
                "previous_run_confirmed_no_rgb_depth": input_summary["previous_run_confirmed_no_rgb_depth"],
                "previous_run_confirmed_no_observed_state_final": input_summary["previous_run_confirmed_no_observed_state_final"],
                "previous_run_confirmed_no_mp4": input_summary["previous_run_confirmed_no_mp4"],
                "cannot_enter_stage4a66d": input_summary["cannot_enter_stage4a66d"],
                "cannot_enter_stage4a67": input_summary["cannot_enter_stage4a67"],
            },
        ),
    )
    save_json(output_dir / "usd_dependency_expanded_report.json", expanded)
    write_text(
        output_dir / "usd_dependency_expanded_report.md",
        markdown_table(
            "USD Dependency Expanded Report",
            {
                "reference_occurrence_count": expanded["reference_occurrence_count"],
                "reference_unique_asset_count": expanded["reference_unique_asset_count"],
                "payload_occurrence_count": expanded["payload_occurrence_count"],
                "sublayer_count": expanded["sublayer_count"],
                "texture_asset_path_count": expanded["texture_asset_path_count"],
                "remote_dependency_occurrence_count": expanded["remote_dependency_occurrence_count"],
                "remote_unique_dependency_count": expanded["remote_unique_dependency_count"],
                "still_missing_occurrence_count": expanded["still_missing_occurrence_count"],
                "still_missing_unique_count": expanded["still_missing_unique_count"],
                "local_candidate_row_count": expanded["local_candidate_row_count"],
                "dependencies_complete_after_localization": expanded["dependencies_complete_after_localization"],
                "isaac_retry_allowed": expanded["isaac_retry_allowed"],
                "isaac_retry_executed": expanded["isaac_retry_executed"],
            },
        ),
    )

    missing_rows = [project_dependency_columns(dep) for dep in dependency_records if dep.get("still_missing")]
    remote_rows = [project_dependency_columns(dep) for dep in remote_records]
    unique_missing_projected = [project_dependency_columns(dep) | {"occurrence_count": dep.get("occurrence_count")} for dep in unique_missing]
    save_json(output_dir / "missing_dependency_table.json", missing_rows)
    write_csv(output_dir / "missing_dependency_table.csv", missing_rows, fieldnames=DEPENDENCY_COLUMNS)
    save_json(output_dir / "missing_dependency_unique_table.json", unique_missing_projected)
    write_csv(output_dir / "missing_dependency_unique_table.csv", unique_missing_projected, fieldnames=DEPENDENCY_COLUMNS + ["occurrence_count"])
    save_json(output_dir / "remote_dependency_table.json", remote_rows)
    write_csv(output_dir / "remote_dependency_table.csv", remote_rows, fieldnames=DEPENDENCY_COLUMNS)
    candidate_fieldnames = [
        "original_asset_path",
        "dependency_type",
        "source_prim",
        "target_basename",
        "match_strategy",
        "score",
        "candidate_path",
        "candidate_basename",
        "candidate_extension",
        "candidate_size_bytes",
        "candidate_search_root",
    ]
    save_json(output_dir / "local_dependency_candidates.json", candidate_rows)
    write_csv(output_dir / "local_dependency_candidates.csv", candidate_rows, fieldnames=candidate_fieldnames)
    save_json(output_dir / "dependency_localization_patch_report.json", patch_report)
    write_text(output_dir / "dependency_localization_patch_report.md", markdown_table("Dependency Localization Patch Report", patch_report))

    package_request = None
    if unique_missing:
        package_request = write_package_request(output_dir, unique_missing, expanded)
    else:
        request = {
            "stage": "Stage 4A-6.6c-usd-dependency-fix",
            "request_type": "dependency_package_request",
            "needed": False,
            "reason": "all dependencies localized",
            "isaac_retry_allowed": True,
        }
        save_json(output_dir / "dependency_package_request.json", request)
        write_text(output_dir / "dependency_package_request.md", markdown_table("Dependency Package Request", request))
        package_request = request

    summary = {
        "stage": "Stage 4A-6.6c-usd-dependency-fix",
        "created_at_utc": utc_now(),
        "source_usd": str(source_usd),
        "staged_usd": str(staged_usd),
        "dependency_dir": str(DEFAULT_DEPENDENCY_DIR),
        "source_modified": False,
        "source_sha256": sha256_file(source_usd),
        "staged_sha256_before": input_summary.get("staged_sha256_before"),
        "staged_sha256_after": sha256_file(staged_usd),
        "dependency_records": len(dependency_records),
        "reference_occurrences": len(usd_info["reference_rows"]),
        "unique_reference_assets": expanded["reference_unique_asset_count"],
        "remote_unique_dependencies": expanded["remote_unique_dependency_count"],
        "missing_unique_dependencies": len(unique_missing),
        "missing_occurrences": len(missing_rows),
        "local_candidate_rows": len(candidate_rows),
        "local_candidate_unique_dependencies": expanded["local_candidate_unique_dependency_count"],
        "patch_applied": bool(patch_report.get("patch_applied")),
        "copied_file_count": int(patch_report.get("copied_file_count", 0)),
        "dependencies_complete_after_localization": len(unique_missing) == 0,
        "isaac_retry_allowed": len(unique_missing) == 0,
        "isaac_retry_executed": False,
        "blocked_before_stage4a66d": len(unique_missing) > 0,
        "blocked_before_stage4a67": True,
        "package_request": str(output_dir / "dependency_package_request.md"),
    }
    save_json(output_dir / "dependency_localization_summary.json", summary)
    write_text(output_dir / "dependency_localization_summary.md", markdown_table("Dependency Localization Summary", summary))
    return {"expanded": expanded, "summary": summary, "package_request": package_request}


def project_dependency_columns(dep: dict[str, Any]) -> dict[str, Any]:
    return {key: dep.get(key) for key in DEPENDENCY_COLUMNS}


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_usd", type=Path, default=DEFAULT_SOURCE_USD)
    parser.add_argument("--staged_usd", type=Path, default=DEFAULT_STAGED_USD)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dependency_dir", type=Path, default=DEFAULT_DEPENDENCY_DIR)
    parser.add_argument("--apply_patches", action="store_true", help="Patch staged USD only if every missing dependency has one exact local candidate.")
    args = parser.parse_args()

    source_usd = args.source_usd.resolve()
    staged_usd = args.staged_usd.resolve()
    output_dir = args.output_dir.resolve()
    dependency_dir = args.dependency_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dependency_dir.mkdir(parents=True, exist_ok=True)

    source_hash_before = sha256_file(source_usd)
    input_summary = load_input_context(output_dir, source_usd, staged_usd)
    usd_info = inspect_staged_usd(staged_usd)
    dependency_records = [dict(row) for row in usd_info["dependency_records"]]
    unique_missing_before = unique_missing_records(dependency_records)
    files, search_metadata = file_search_index(REQUESTED_SEARCH_ROOTS)
    maps = build_candidate_maps(files)
    candidate_rows: list[dict[str, Any]] = []
    seen_candidate_keys: set[tuple[str, str, str]] = set()
    for dep in unique_missing_before:
        for row in find_candidates_for_dependency(dep, maps):
            key = (str(row["original_asset_path"]), str(row["candidate_path"]), str(row["match_strategy"]))
            if key in seen_candidate_keys:
                continue
            seen_candidate_keys.add(key)
            candidate_rows.append(row)
    annotate_with_candidates(dependency_records, candidate_rows)
    patch_report = copy_and_patch_if_complete(
        staged_usd=staged_usd,
        dependency_dir=dependency_dir,
        dependency_records=dependency_records,
        unique_missing=unique_missing_records(dependency_records),
        candidate_rows=candidate_rows,
        apply_patches=bool(args.apply_patches),
    )
    result = write_reports(
        output_dir=output_dir,
        source_usd=source_usd,
        staged_usd=staged_usd,
        input_summary=input_summary,
        usd_info=usd_info,
        search_metadata=search_metadata,
        dependency_records=dependency_records,
        candidate_rows=candidate_rows,
        patch_report=patch_report,
    )
    source_hash_after = sha256_file(source_usd)
    if source_hash_after != source_hash_before:
        raise RuntimeError("source USD hash changed; this stage must not modify source USD")
    print(json.dumps(to_jsonable(result["summary"]), indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
