#!/usr/bin/env python3
"""Stage 4A-6.6c USD dependency re-search in the corrected conda env.

The task namespace still says env_isaacsim, but the user corrected the actual
conda environment to env_isaaclab. This script records that distinction and
keeps all scene/RL boundaries closed unless every dependency is localized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
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
CONDA_BASE = Path("/home/ubuntu22/miniconda3")
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
]
PREVIOUS_REQUIRED_FILES = [
    "dependency_package_request.md",
    "dependency_localization_summary.json",
    "usd_dependency_expanded_report.json",
    "remote_dependency_table.json",
    "missing_dependency_table.json",
    "missing_dependency_unique_table.json",
]
USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
REMOTE_PREFIX = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
FULL_ASSET_MARKER = "Assets/Isaac/4.5/Isaac/"


@dataclass(frozen=True)
class Candidate:
    path: Path
    basename: str
    lower_basename: str
    stem: str
    suffix: str
    size_bytes: int
    search_root: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
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
        text = json.dumps(to_jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(value)
        if len(text) > 1800:
            text = text[:1800] + "..."
        lines.append(f"| {key} | `{text}` |")
    return "\n".join(lines)


def markdown_list(title: str, rows: list[str]) -> str:
    body = rows or ["No rows."]
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in body]])


def file_manifest_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def normalize_path_text(path: Path) -> str:
    return path.as_posix()


def canonical_url(text: str) -> str:
    value = str(text or "").strip().strip("@")
    if value.startswith("https:/") and not value.startswith("https://"):
        value = "https://" + value[len("https:/") :]
    return value


def relative_full_asset_path(url: str) -> str:
    text = canonical_url(url)
    if "Assets/Isaac/4.5/Isaac/" in text:
        return "Assets/Isaac/4.5/Isaac/" + text.split("Assets/Isaac/4.5/Isaac/", 1)[1]
    if "Assets/" in text:
        return "Assets/" + text.split("Assets/", 1)[1]
    return Path(text).name


def relative_after_isaac_root(url: str) -> str:
    full = relative_full_asset_path(url)
    if full.startswith(FULL_ASSET_MARKER):
        return full[len(FULL_ASSET_MARKER) :]
    return full


def basename_from_url(url: str) -> str:
    return Path(canonical_url(url).split("?", 1)[0].split("#", 1)[0]).name


def category_path_from_url(url: str) -> str:
    rel = relative_after_isaac_root(url)
    return str(Path(rel).parent)


def run_short_command(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int = 20) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=env,
        )
        return {
            "cmd": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": repr(exc), "ok": False}


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {"ok": True, "file": getattr(module, "__file__", None), "repr": repr(module)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def find_omni_usd_libs(conda_prefix: Path) -> list[Path]:
    roots = list((conda_prefix / "lib").glob("python*/site-packages/isaacsim/extscache/omni.usd.libs-*"))
    return sorted([root for root in roots if root.is_dir()])


def probe_environment(conda_env_name: str) -> dict[str, Any]:
    conda_prefix = Path(os.environ.get("CONDA_PREFIX") or CONDA_BASE / "envs" / conda_env_name)
    py_compile_probe = {
        "which_python": shutil.which("python"),
        "sys_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "sys_path_head": sys.path[:20],
        "conda_prefix_env": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    imports = {name: import_status(name) for name in ("isaacsim", "omni", "pxr")}
    omni_usd_libs = find_omni_usd_libs(conda_prefix)
    pxr_with_libs = {"ok": False, "reason": "omni.usd.libs not found"}
    if omni_usd_libs:
        lib_root = omni_usd_libs[-1]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{lib_root}:{env.get('PYTHONPATH', '')}"
        ld_parts = [str(conda_prefix / "lib"), str(lib_root / "bin")]
        if env.get("LD_LIBRARY_PATH"):
            ld_parts.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
        pxr_with_libs = run_short_command(
            [sys.executable, "-c", "from pxr import Sdf, UsdUtils; print('pxr ok', Sdf.Layer)"],
            env=env,
            timeout=20,
        )
    isaacsim_file = imports.get("isaacsim", {}).get("file")
    isaacsim_root = None
    if isaacsim_file:
        isaacsim_root = str(Path(str(isaacsim_file)).resolve().parents[0])
    return {
        "stage_namespace": "env_isaacsim_dependency_fix",
        "user_corrected_conda_env_name": "env_isaaclab",
        "effective_conda_env_name": conda_env_name,
        "conda_prefix": str(conda_prefix),
        "python": py_compile_probe,
        "imports": imports,
        "pxr_import_with_omni_usd_libs": pxr_with_libs,
        "omni_usd_libs_roots": [str(path) for path in omni_usd_libs],
        "isaacsim_python_package_root": isaacsim_root,
        "probe_success": conda_prefix.exists() and imports["isaacsim"]["ok"],
        "created_at_utc": utc_now(),
    }


def common_search_roots(conda_prefix: Path, probe: dict[str, Any], staged_usd: Path) -> list[Path]:
    roots = [
        conda_prefix,
        staged_usd.parent / "dependencies",
        Path("/home/ubuntu22/.cache"),
        Path("/home/ubuntu22/.local/share/ov"),
        Path("/home/ubuntu22/.nvidia-omniverse"),
        Path("/home/ubuntu22/ov"),
        Path("/home/ubuntu22/isaacsim"),
        Path("/home/ubuntu22/IsaacSim"),
        Path("/home/ubuntu22/isaac-sim"),
        Path("/home/ubuntu22/IsaacLab"),
        Path("/isaac-sim"),
        Path("/opt"),
        Path("/omniverse"),
    ]
    isaac_pkg = probe.get("isaacsim_python_package_root")
    if isaac_pkg:
        roots.append(Path(str(isaac_pkg)).resolve().parents[0])
    for lib_root in probe.get("omni_usd_libs_roots", []):
        roots.append(Path(lib_root))
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def effective_roots(roots: list[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    existing = []
    notes = []
    for root in roots:
        if root.exists():
            existing.append(root)
        else:
            notes.append({"requested_root": str(root), "exists": False, "effective": False, "reason": "missing"})
    effective: list[Path] = []
    for root in existing:
        covered_by = None
        for prior in effective:
            try:
                root.relative_to(prior)
                covered_by = prior
                break
            except ValueError:
                pass
        if covered_by:
            notes.append(
                {
                    "requested_root": str(root),
                    "exists": True,
                    "effective": False,
                    "reason": "covered_by_parent_root",
                    "covered_by": str(covered_by),
                }
            )
        else:
            effective.append(root)
            notes.append({"requested_root": str(root), "exists": True, "effective": True, "reason": "searched"})
    return effective, notes


def rg_list_usd_files(roots: list[Path], max_workers: int) -> tuple[list[Path], dict[str, Any]]:
    effective, notes = effective_roots(roots)
    cmd = ["rg", "--files", "--hidden", "--no-ignore"]
    for ext in sorted(USD_EXTENSIONS):
        cmd.extend(["-g", f"*{ext}", "-g", f"*{ext.upper()}"])
    cmd.extend(str(root) for root in effective)
    metadata = {
        "search_backend": "rg --files",
        "requested_roots": [str(root) for root in roots],
        "effective_roots": [str(root) for root in effective],
        "root_notes": notes,
        "max_workers_requested": max_workers,
        "usd_extensions": sorted(USD_EXTENSIONS),
    }
    if not shutil.which("rg"):
        files: list[Path] = []
        for root in effective:
            for current, _dirs, names in os.walk(root):
                for name in names:
                    path = Path(current) / name
                    if path.suffix.lower() in USD_EXTENSIONS:
                        files.append(path)
        metadata["search_backend"] = "os.walk fallback"
        metadata["returncode"] = 0
        metadata["stderr"] = ""
        return files, metadata
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=360,
    )
    metadata["returncode"] = result.returncode
    metadata["stderr"] = result.stderr.strip()[:5000]
    files = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    return files, metadata


def build_candidates(roots: list[Path], max_workers: int) -> tuple[list[Candidate], dict[str, Any]]:
    raw_files, metadata = rg_list_usd_files(roots, max_workers)
    effective = [Path(root) for root in metadata.get("effective_roots", [])]
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for raw in raw_files:
        try:
            path = raw.resolve()
        except Exception:
            path = raw
        key = str(path)
        if key in seen or path.suffix.lower() not in USD_EXTENSIONS:
            continue
        seen.add(key)
        try:
            stat = path.stat()
        except OSError:
            continue
        root_hit = ""
        for root in effective:
            try:
                path.relative_to(root)
                root_hit = str(root)
                break
            except ValueError:
                pass
        candidates.append(
            Candidate(
                path=path,
                basename=path.name,
                lower_basename=path.name.lower(),
                stem=path.stem.lower(),
                suffix=path.suffix.lower(),
                size_bytes=int(stat.st_size),
                search_root=root_hit,
            )
        )
    metadata["indexed_usd_file_count"] = len(candidates)
    metadata["candidate_basename_counts"] = dict(Counter(item.basename for item in candidates).most_common(30))
    return candidates, metadata


def discover_asset_roots(roots: list[Path]) -> dict[str, Any]:
    target_names = {"Assets", "Isaac", "Environments", "Props", "Robots", "NVIDIA"}
    rows: list[dict[str, Any]] = []
    cache_roots = []
    install_roots = []
    for root in roots:
        if not root.exists():
            continue
        if any(token in str(root) for token in (".cache", ".local/share/ov", ".nvidia-omniverse")):
            cache_roots.append(str(root))
        if "isaacsim" in str(root).lower() or "isaac-sim" in str(root).lower() or root.name == "env_isaaclab":
            install_roots.append(str(root))
        root_parts = len(root.parts)
        visited = 0
        for current, dirs, _files in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.parts) - root_parts
            if depth > 8:
                dirs[:] = []
                continue
            if any(part in {".git", "__pycache__", "node_modules"} for part in current_path.parts):
                dirs[:] = []
                continue
            visited += 1
            if visited > 30000:
                break
            if current_path.name in target_names or "Assets/Isaac" in current_path.as_posix():
                rows.append(
                    {
                        "path": str(current_path),
                        "name": current_path.name,
                        "root": str(root),
                        "looks_like_required_isaac_45_root": FULL_ASSET_MARKER.rstrip("/").lower()
                        in current_path.as_posix().lower(),
                    }
                )
            if len(rows) >= 800:
                break
    required_roots = [row for row in rows if row["looks_like_required_isaac_45_root"]]
    return {
        "created_at_utc": utc_now(),
        "isaacsim_install_roots": sorted(set(install_roots)),
        "omniverse_cache_roots": sorted(set(cache_roots)),
        "asset_like_directories": rows,
        "required_isaac_45_asset_roots": required_roots,
        "required_isaac_45_asset_root_count": len(required_roots),
    }


def load_previous_context(previous_output_dir: Path, source_usd: Path, staged_usd: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context_entries = [file_manifest_entry(path) for path in CONTEXT_FILES]
    previous_entries = [file_manifest_entry(previous_output_dir / name) for name in PREVIOUS_REQUIRED_FILES]
    previous_summary = read_json(previous_output_dir / "dependency_localization_summary.json")
    expanded = read_json(previous_output_dir / "usd_dependency_expanded_report.json")
    package_request_md = previous_output_dir / "dependency_package_request.md"
    unique_path = previous_output_dir / "missing_dependency_unique_table.json"
    unique_rows = read_json(unique_path)
    missing_table = read_json(previous_output_dir / "missing_dependency_table.json")
    remote_table = read_json(previous_output_dir / "remote_dependency_table.json")
    forbidden_previous = {
        "validation_rgb_png_count": len(list(previous_output_dir.glob("validation_rgb_*.png"))),
        "validation_depth_npy_count": len(list(previous_output_dir.glob("validation_depth_*.npy"))),
        "inspection_rgb_png_count": len(list(previous_output_dir.glob("inspection_rgb_*.png"))),
        "inspection_depth_npy_count": len(list(previous_output_dir.glob("inspection_depth_*.npy"))),
        "observed_state_final_exists": (previous_output_dir / "observed_state_final.npy").exists(),
        "mp4_exists": (previous_output_dir / "usd_scene_flythrough.mp4").exists(),
    }
    loaded = {
        "created_at_utc": utc_now(),
        "stage": "Stage 4A-6.6c-usd-dependency-fix-env-corrected",
        "task_namespace_from_prompt": "env_isaacsim_dependency_fix",
        "user_corrected_conda_env_name": "env_isaaclab",
        "source_usd": str(source_usd),
        "staged_usd": str(staged_usd),
        "source_sha256_current": sha256_file(source_usd),
        "staged_sha256_current": sha256_file(staged_usd),
        "context_files": context_entries,
        "previous_output_dir": str(previous_output_dir),
        "previous_required_files": previous_entries,
        "dependency_package_request_loaded": package_request_md.is_file(),
        "previous_dependency_blocked": True,
        "previous_scene_validation_success": False,
        "previous_no_rgb_depth": sum(
            forbidden_previous[key]
            for key in (
                "validation_rgb_png_count",
                "validation_depth_npy_count",
                "inspection_rgb_png_count",
                "inspection_depth_npy_count",
            )
        )
        == 0,
        "previous_no_observed_state_final": not forbidden_previous["observed_state_final_exists"],
        "previous_no_mp4": not forbidden_previous["mp4_exists"],
        "stage4a66d_allowed": False,
        "stage4a67_allowed": False,
        "previous_forbidden_artifacts": forbidden_previous,
    }
    prev = {
        "created_at_utc": utc_now(),
        "previous_output_dir": str(previous_output_dir),
        "source_usd": previous_summary.get("source_usd"),
        "staged_usd": previous_summary.get("staged_usd"),
        "source_sha256": previous_summary.get("source_sha256"),
        "staged_sha256_before": previous_summary.get("staged_sha256_before"),
        "staged_sha256_after": previous_summary.get("staged_sha256_after"),
        "reference_occurrences": expanded.get("reference_occurrence_count"),
        "remote_reference_occurrences": expanded.get("remote_dependency_occurrence_count"),
        "remote_unique_dependencies": expanded.get("remote_unique_dependency_count"),
        "missing_unique_dependencies": previous_summary.get("missing_unique_dependencies"),
        "missing_occurrences": previous_summary.get("missing_occurrences"),
        "local_candidate_rows": previous_summary.get("local_candidate_rows"),
        "copied_file_count": previous_summary.get("copied_file_count"),
        "patch_applied": previous_summary.get("patch_applied"),
        "isaac_retry_allowed": previous_summary.get("isaac_retry_allowed"),
        "isaac_retry_executed": previous_summary.get("isaac_retry_executed"),
        "blocked_before_stage4a66d": previous_summary.get("blocked_before_stage4a66d", True),
        "blocked_before_stage4a67": previous_summary.get("blocked_before_stage4a67", True),
        "remote_dependency_table_rows": len(remote_table),
        "missing_dependency_table_rows": len(missing_table),
        "dependency_package_request_loaded": package_request_md.is_file(),
    }
    return loaded, unique_rows


def maps_for_candidates(candidates: list[Candidate]) -> dict[str, dict[str, list[Candidate]]]:
    maps: dict[str, dict[str, list[Candidate]]] = {
        "basename": defaultdict(list),
        "lower_basename": defaultdict(list),
        "stem": defaultdict(list),
    }
    for item in candidates:
        maps["basename"][item.basename].append(item)
        maps["lower_basename"][item.lower_basename].append(item)
        maps["stem"][item.stem].append(item)
    return maps


def candidate_to_json(item: Candidate) -> dict[str, Any]:
    return {
        "path": str(item.path),
        "basename": item.basename,
        "suffix": item.suffix,
        "size_bytes": item.size_bytes,
        "search_root": item.search_root,
    }


def best_match_for_dependency(
    dep: dict[str, Any],
    candidates: list[Candidate],
    maps: dict[str, dict[str, list[Candidate]]],
    url_log_hits: dict[str, list[str]],
) -> dict[str, Any]:
    url = canonical_url(str(dep["original_asset_path"]))
    full_rel = relative_full_asset_path(url)
    after_root = relative_after_isaac_root(url)
    basename = basename_from_url(url)
    full_rel_norm = full_rel.replace("\\", "/")
    after_norm = after_root.replace("\\", "/")
    exact_full = [
        item
        for item in candidates
        if normalize_path_text(item.path).endswith(full_rel_norm)
        or f"/{full_rel_norm}" in normalize_path_text(item.path)
    ]
    asset_root_rel = [
        item
        for item in candidates
        if normalize_path_text(item.path).endswith(f"Isaac/{after_norm}")
        or f"/Isaac/{after_norm}" in normalize_path_text(item.path)
    ]
    exact_base = maps["basename"].get(basename, [])
    ci_base = [item for item in maps["lower_basename"].get(basename.lower(), []) if item.basename != basename]
    stem_matches = [
        item
        for item in maps["stem"].get(Path(basename).stem.lower(), [])
        if item.lower_basename != basename.lower() and item.suffix in {".usd", ".usda", ".usdc"}
    ]
    all_candidate_rows: list[dict[str, Any]] = []
    for match_type, confidence, items in [
        ("exact_full_relative_path", "exact", exact_full),
        ("isaac_asset_root_relative_match", "medium", asset_root_rel),
        ("exact_basename", "medium", exact_base),
        ("case_insensitive_basename", "low", ci_base),
        ("same_stem_usd_variant", "low", stem_matches),
    ]:
        for item in items[:20]:
            row = candidate_to_json(item)
            row["match_type"] = match_type
            row["confidence"] = confidence
            all_candidate_rows.append(row)
    local_match_found = bool(exact_full or asset_root_rel or exact_base or ci_base or stem_matches)
    chosen: Candidate | None = None
    match_type = "none"
    confidence = "none"
    eligible_to_copy = False
    if exact_full:
        chosen = sorted(exact_full, key=lambda item: str(item.path))[0]
        match_type = "exact_full_relative_path"
        confidence = "exact"
        eligible_to_copy = True
    elif asset_root_rel:
        chosen = sorted(asset_root_rel, key=lambda item: str(item.path))[0]
        match_type = "isaac_asset_root_relative_match"
        confidence = "medium"
    elif exact_base:
        chosen = sorted(exact_base, key=lambda item: str(item.path))[0]
        match_type = "exact_basename"
        confidence = "medium"
    elif ci_base:
        chosen = sorted(ci_base, key=lambda item: str(item.path))[0]
        match_type = "case_insensitive_basename"
        confidence = "low"
    elif stem_matches:
        chosen = sorted(stem_matches, key=lambda item: str(item.path))[0]
        match_type = "same_stem_usd_variant"
        confidence = "low"
    log_hits = url_log_hits.get(url, [])
    if not local_match_found and log_hits:
        match_type = "url_seen_in_isaac_log_only"
        confidence = "none"
    reason = "no local file matched full path, basename, case-insensitive basename, stem, or trusted cache mapping"
    if local_match_found and not eligible_to_copy:
        reason = "only non-exact local candidate found; not copied because exact Assets/Isaac/4.5 path was not proven"
    if eligible_to_copy:
        reason = "exact full relative path local file found"
    if log_hits and not eligible_to_copy:
        reason += "; URL appears in IsaacSim logs but no URL-to-cache-file mapping was available"
    return {
        "original_url": url,
        "original_asset_path": url,
        "relative_isaac_path": full_rel,
        "path_after_isaac_root": after_root,
        "category_path": category_path_from_url(url),
        "basename": basename,
        "occurrence_count": int(dep.get("occurrence_count") or 1),
        "source_prims": str(dep.get("source_prim", "")).split(";") if dep.get("source_prim") else [],
        "local_match_found": local_match_found,
        "local_match_path": str(chosen.path) if chosen else None,
        "match_type": match_type,
        "confidence": confidence,
        "copy_eligible": eligible_to_copy,
        "copied": False,
        "copied_to": None,
        "patched": False,
        "patched_path": None,
        "still_missing": not eligible_to_copy,
        "reason": reason,
        "log_or_cache_observations": log_hits,
        "candidate_rows": all_candidate_rows,
    }


def collect_url_log_hits(search_roots: list[Path], urls: list[str]) -> dict[str, list[str]]:
    log_paths: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if "isaacsim" not in str(root).lower() and ".cache" not in str(root):
            continue
        for current, dirs, names in os.walk(root):
            current_path = Path(current)
            if len(current_path.parts) - len(root.parts) > 8:
                dirs[:] = []
                continue
            for name in names:
                if name.endswith((".log", ".txt", ".json", ".yaml", ".yml")):
                    log_paths.append(current_path / name)
            if len(log_paths) > 5000:
                break
    hits: dict[str, list[str]] = {url: [] for url in urls}
    url_set = set(urls)
    for path in log_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for url in url_set:
            if url in text:
                hits[url].append(str(path))
    return {url: paths[:20] for url, paths in hits.items() if paths}


def copy_exact_matches(results: list[dict[str, Any]], staged_usd: Path, create_package: bool) -> dict[str, Any]:
    dependency_root = staged_usd.parent / "dependencies"
    copied_rows = []
    if not create_package:
        return {
            "dependency_root": str(dependency_root),
            "copy_attempted": False,
            "copied_count": 0,
            "copied_rows": [],
            "reason": "create_localized_package_if_found flag not set",
        }
    for row in results:
        if not row.get("copy_eligible") or not row.get("local_match_path"):
            continue
        src = Path(str(row["local_match_path"]))
        dst = dependency_root / str(row["relative_isaac_path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256_file(dst) != sha256_file(src):
            shutil.copy2(src, dst)
        row["copied"] = True
        row["copied_to"] = str(dst)
        row["still_missing"] = False
        copied_rows.append(
            {
                "original_url": row["original_url"],
                "src": str(src),
                "dst": str(dst),
                "sha256": sha256_file(dst),
                "size_bytes": dst.stat().st_size if dst.is_file() else None,
            }
        )
    return {
        "dependency_root": str(dependency_root),
        "copy_attempted": True,
        "copied_count": len(copied_rows),
        "copied_rows": copied_rows,
        "reason": "copied exact full-relative-path matches only",
    }


def create_localized_usd_if_complete(
    results: list[dict[str, Any]],
    staged_usd: Path,
    localized_root: Path,
) -> dict[str, Any]:
    unresolved = [row for row in results if row.get("still_missing")]
    localized_usd = localized_root / staged_usd.name
    if unresolved:
        return {
            "localized_usd_created": False,
            "localized_usd": str(localized_usd),
            "patch_attempted": False,
            "patch_applied": False,
            "patch_count": 0,
            "reason": "unresolved dependencies remain; localized USD not created",
            "unresolved_count": len(unresolved),
        }
    localized_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staged_usd, localized_usd)
    patch_map = {
        row["original_url"]: "./dependencies/" + row["relative_isaac_path"]
        for row in results
        if row.get("copied")
    }
    patch_map_path = localized_root / "stage4a66c_path_patch_map.json"
    save_json(patch_map_path, patch_map)
    return {
        "localized_usd_created": True,
        "localized_usd": str(localized_usd),
        "localized_usd_sha256_before_patch": sha256_file(localized_usd),
        "patch_attempted": False,
        "patch_applied": False,
        "patch_count": 0,
        "reason": "all dependencies copied, but patching is intentionally gated to a separate Isaac/PXR retry path",
        "patch_map": str(patch_map_path),
        "unresolved_count": 0,
    }


def hash_manifest_rows(
    source_usd: Path,
    staged_usd: Path,
    localized_report: dict[str, Any],
    copy_report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {"role": "source_usd", "path": str(source_usd), "exists": source_usd.is_file(), "sha256": sha256_file(source_usd)},
        {"role": "staged_usd", "path": str(staged_usd), "exists": staged_usd.is_file(), "sha256": sha256_file(staged_usd)},
    ]
    loc = Path(str(localized_report.get("localized_usd", "")))
    rows.append({"role": "localized_usd", "path": str(loc), "exists": loc.is_file(), "sha256": sha256_file(loc)})
    for item in copy_report.get("copied_rows", []):
        path = Path(item["dst"])
        rows.append({"role": "copied_dependency", "path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)})
    return rows


def negative_scope_report(report_type: str) -> dict[str, Any]:
    base = {
        "created_at_utc": utc_now(),
        "stage": "Stage 4A-6.6c-usd-env-corrected-dependency-fix",
        "report_type": report_type,
        "procedural_fallback_used": False,
        "cuboid_fallback_used": False,
        "larger_complex_scene_v1_restored": False,
        "selected_action_executed": False,
        "rollout_run": False,
        "formal_expert_sampling_run": False,
        "expert_dataset_generated": False,
        "map_predict_called": False,
        "sscnet_inference_called": False,
        "prediction_npz_created": False,
        "training_run": False,
        "rl_run": False,
        "gdpo_run": False,
        "ppo_run": False,
        "bc_run": False,
        "il_run": False,
        "checkpoint_modified": False,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
    }
    if report_type == "no_rollout":
        base["rollout_run"] = False
    elif report_type == "no_formal_expert_sampling":
        base["formal_expert_sampling_run"] = False
        base["expert_dataset_generated"] = False
    elif report_type == "no_map_predict":
        base["map_predict_called"] = False
        base["sscnet_inference_called"] = False
    elif report_type == "no_rl_gdpo":
        base["rl_run"] = False
        base["gdpo_run"] = False
    return base


def write_negative_scope(output_dir: Path) -> None:
    for stem, title in [
        ("no_rollout_report", "No Rollout Report"),
        ("no_expert_sampling_report", "No Expert Sampling Report"),
        ("no_map_predict_report", "No Map Predict Report"),
        ("no_rl_gdpo_report", "No RL GDPO Report"),
    ]:
        report_type = stem.replace("_report", "")
        payload = negative_scope_report(report_type)
        save_json(output_dir / f"{stem}.json", payload)
        write_text(output_dir / f"{stem}.md", markdown_table(title, payload))


def write_dependency_package_request_updated(output_dir: Path, results: list[dict[str, Any]], conda_env_name: str) -> None:
    unresolved = [row for row in results if row.get("still_missing")]
    lines = [
        "# Updated Dependency Package Request",
        "",
        f"- User-corrected conda environment searched: `{conda_env_name}`",
        "- The prompt namespace/report filenames still use `env_isaacsim`, but the effective environment is `env_isaaclab`.",
        f"- Still missing unique remote Isaac assets: `{len(unresolved)}`",
        "- Current USD remains not self-contained.",
        "- Stage 4A-6.6d remains blocked.",
        "- Stage 4A-6.7 remains blocked.",
        "",
        "Please provide a complete local package preserving `Assets/Isaac/4.5/Isaac/...` under `current_environment/dependencies`, including transitive USD/material/texture dependencies, or allow downloading these exact URLs.",
        "",
        "## Still Missing",
        "",
    ]
    for row in unresolved:
        lines.append(f"- `{row['original_url']}` occurrences `{row['occurrence_count']}`")
    write_text(output_dir / "dependency_package_request_updated.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-search USD dependencies in the corrected Isaac conda environment.")
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--staged_usd", required=True)
    parser.add_argument("--previous_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--localized_root", required=True)
    parser.add_argument("--conda_env_name", default="env_isaaclab")
    parser.add_argument("--search_env_isaacsim", action="store_true")
    parser.add_argument("--search_isaacsim_install_roots", action="store_true")
    parser.add_argument("--search_omniverse_cache", action="store_true")
    parser.add_argument("--create_localized_package_if_found", action="store_true")
    parser.add_argument("--retry_isaac_if_dependencies_resolved", action="store_true")
    parser.add_argument("--min_validation_poses", type=int, default=20)
    parser.add_argument("--min_inspection_poses", type=int, default=36)
    parser.add_argument("--make_flythrough", action="store_true")
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_formal_expert_sampling", action="store_true")
    parser.add_argument("--no_map_predict", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    args = parser.parse_args()

    source_usd = Path(args.source_usd).resolve()
    staged_usd = Path(args.staged_usd).resolve()
    previous_output_dir = Path(args.previous_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    localized_root = Path(args.localized_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_context, unique_deps = load_previous_context(previous_output_dir, source_usd, staged_usd)
    save_json(output_dir / "loaded_context_manifest.json", loaded_context)
    write_text(output_dir / "loaded_context_manifest.md", markdown_table("Loaded Context Manifest", loaded_context))

    previous_summary = {
        "created_at_utc": utc_now(),
        "previous_blocker_loaded": True,
        "dependency_package_request_loaded": loaded_context["dependency_package_request_loaded"],
        "previous_remote_unique_dependencies": len(unique_deps),
        "previous_remote_reference_occurrences": sum(int(row.get("occurrence_count") or 1) for row in unique_deps),
        "previous_local_candidate_rows": 0,
        "previous_copied_dependencies": 0,
        "previous_staged_usd_patches": 0,
        "previous_no_isaac_retry": True,
        "previous_no_rgb_depth": loaded_context["previous_no_rgb_depth"],
        "previous_no_observed_state_final": loaded_context["previous_no_observed_state_final"],
        "previous_no_mp4": loaded_context["previous_no_mp4"],
        "stage4a66d_blocked": True,
        "stage4a67_blocked": True,
    }
    prior_loc = read_json(previous_output_dir / "dependency_localization_summary.json")
    previous_summary.update(
        {
            "previous_source_sha256": prior_loc.get("source_sha256"),
            "previous_staged_sha256_before": prior_loc.get("staged_sha256_before"),
            "previous_staged_sha256_after": prior_loc.get("staged_sha256_after"),
            "previous_missing_unique_dependencies": prior_loc.get("missing_unique_dependencies"),
            "previous_missing_occurrences": prior_loc.get("missing_occurrences"),
            "previous_isaac_retry_allowed": prior_loc.get("isaac_retry_allowed"),
            "previous_isaac_retry_executed": prior_loc.get("isaac_retry_executed"),
        }
    )
    save_json(output_dir / "previous_dependency_fix_summary.json", previous_summary)
    write_text(output_dir / "previous_dependency_fix_summary.md", markdown_table("Previous Dependency Fix Summary", previous_summary))

    probe = probe_environment(args.conda_env_name)
    save_json(output_dir / "env_isaacsim_probe_report.json", probe)
    write_text(output_dir / "env_isaacsim_probe_report.md", markdown_table("env_isaacsim Probe Report (effective env_isaaclab)", probe))

    conda_prefix = Path(probe["conda_prefix"])
    search_roots = common_search_roots(conda_prefix, probe, staged_usd)
    asset_roots = discover_asset_roots(search_roots)
    save_json(output_dir / "env_isaacsim_asset_roots.json", asset_roots)
    write_text(output_dir / "env_isaacsim_asset_roots.md", markdown_table("env_isaacsim Asset Roots (effective env_isaaclab)", asset_roots))

    candidates, search_metadata = build_candidates(search_roots, args.max_workers)
    candidate_maps = maps_for_candidates(candidates)
    urls = [canonical_url(str(row["original_asset_path"])) for row in unique_deps]
    url_log_hits = collect_url_log_hits(search_roots, urls)
    search_results = [
        best_match_for_dependency(dep, candidates, candidate_maps, url_log_hits)
        for dep in sorted(unique_deps, key=lambda row: str(row["original_asset_path"]))
    ]
    copy_report = copy_exact_matches(search_results, staged_usd, args.create_localized_package_if_found)
    localized_report = create_localized_usd_if_complete(search_results, staged_usd, localized_root)

    flat_results = []
    for row in search_results:
        flat = dict(row)
        flat.pop("candidate_rows", None)
        flat_results.append(flat)
    result_fields = [
        "original_url",
        "relative_isaac_path",
        "path_after_isaac_root",
        "category_path",
        "basename",
        "occurrence_count",
        "local_match_found",
        "local_match_path",
        "match_type",
        "confidence",
        "copy_eligible",
        "copied",
        "copied_to",
        "patched",
        "patched_path",
        "still_missing",
        "reason",
    ]
    save_json(output_dir / "env_isaacsim_asset_search_results.json", search_results)
    write_csv(output_dir / "env_isaacsim_asset_search_results.csv", flat_results, result_fields)
    search_md_rows = [
        f"`{row['basename']}` `{row['confidence']}` `{row['match_type']}` missing=`{row['still_missing']}` path=`{row['local_match_path']}`"
        for row in flat_results
    ]
    write_text(output_dir / "env_isaacsim_asset_search_results.md", markdown_list("env_isaacsim Asset Search Results (effective env_isaaclab)", search_md_rows))

    localized_manifest = {
        "created_at_utc": utc_now(),
        "source_usd": str(source_usd),
        "staged_usd": str(staged_usd),
        "localized_root": str(localized_root),
        "localized_usd": localized_report.get("localized_usd"),
        "localized_usd_created": localized_report.get("localized_usd_created"),
        "dependency_root": copy_report.get("dependency_root"),
        "copied_dependency_count": copy_report.get("copied_count"),
        "all_dependencies_resolved": len([row for row in search_results if row.get("still_missing")]) == 0,
        "copy_report": copy_report,
        "localized_report": localized_report,
        "no_procedural_fallback": True,
    }
    save_json(output_dir / "localized_package_manifest.json", localized_manifest)
    write_text(output_dir / "localized_package_manifest.md", markdown_table("Localized Package Manifest", localized_manifest))

    hash_rows = hash_manifest_rows(source_usd, staged_usd, localized_report, copy_report)
    hash_manifest = {
        "created_at_utc": utc_now(),
        "rows": hash_rows,
        "source_usd_hash_unchanged": sha256_file(source_usd) == prior_loc.get("source_sha256"),
        "staged_usd_hash_unchanged": sha256_file(staged_usd) == prior_loc.get("staged_sha256_after"),
    }
    save_json(output_dir / "localized_hash_manifest.json", hash_manifest)
    write_csv(output_dir / "localized_hash_manifest.csv", hash_rows, ["role", "path", "exists", "sha256"])
    write_text(output_dir / "localized_hash_manifest.md", markdown_table("Localized Hash Manifest", hash_manifest))

    unresolved = [row for row in search_results if row.get("still_missing")]
    patch_report = {
        "created_at_utc": utc_now(),
        "source_usd_modified": False,
        "staged_usd_modified": False,
        "localized_usd_created": localized_report.get("localized_usd_created"),
        "patch_attempted": localized_report.get("patch_attempted", False),
        "patch_applied": localized_report.get("patch_applied", False),
        "patch_count": localized_report.get("patch_count", 0),
        "remote_url_replacement": {
            "from": REMOTE_PREFIX + FULL_ASSET_MARKER + "...",
            "to": "./dependencies/" + FULL_ASSET_MARKER + "...",
        },
        "reason": localized_report.get("reason"),
        "unresolved_dependency_count": len(unresolved),
    }
    save_json(output_dir / "env_isaacsim_path_patch_report.json", patch_report)
    write_text(output_dir / "env_isaacsim_path_patch_report.md", markdown_table("env_isaacsim Path Patch Report (effective env_isaaclab)", patch_report))

    unresolved_report = {
        "created_at_utc": utc_now(),
        "unresolved_unique_remote_dependency_count": len(unresolved),
        "unresolved_reference_occurrence_count": sum(int(row.get("occurrence_count") or 1) for row in unresolved),
        "unresolved_dependencies": unresolved,
        "stage4a66d_blocked": len(unresolved) > 0,
        "stage4a67_blocked": len(unresolved) > 0,
    }
    save_json(output_dir / "unresolved_after_env_isaacsim_search.json", unresolved_report)
    write_text(output_dir / "unresolved_after_env_isaacsim_search.md", markdown_table("Unresolved After env_isaacsim Search (effective env_isaaclab)", unresolved_report))

    retry_gate = {
        "created_at_utc": utc_now(),
        "retry_requested_by_flag": bool(args.retry_isaac_if_dependencies_resolved),
        "retry_allowed": False,
        "retry_executed": False,
        "isaac_headless_startup_count": 0,
        "exactly_one_retry": False,
        "gate_checks": {
            "unresolved_unique_remote_dependencies_zero": len(unresolved) == 0,
            "localized_usd_exists": bool(localized_manifest["localized_usd_created"]),
            "localized_usd_hash_recorded": any(row["role"] == "localized_usd" and row["sha256"] for row in hash_rows),
            "no_remote_https_references_remain": False if unresolved else None,
            "no_omniverse_references_remain": False if unresolved else None,
            "no_missing_local_files": len(unresolved) == 0,
            "no_procedural_fallback": True,
        },
        "blocked_reason": "unresolved remote dependencies remain" if unresolved else "localized USD path patch did not run in this dependency-only script",
        "min_validation_poses": args.min_validation_poses,
        "min_inspection_poses": args.min_inspection_poses,
        "make_flythrough_requested": bool(args.make_flythrough),
    }
    save_json(output_dir / "isaac_retry_gate_report.json", retry_gate)
    write_text(output_dir / "isaac_retry_gate_report.md", markdown_table("Isaac Retry Gate Report", retry_gate))

    blocker = {
        "created_at_utc": utc_now(),
        "blocked": len(unresolved) > 0,
        "main_blocker": "env_isaaclab does not contain these local Isaac assets in a trusted Assets/Isaac/4.5/Isaac path",
        "user_confirmed_conda_env": "env_isaaclab",
        "searched_conda_prefix": str(conda_prefix),
        "searched_isaacsim_install_roots": asset_roots["isaacsim_install_roots"],
        "searched_omniverse_cache_roots": asset_roots["omniverse_cache_roots"],
        "still_missing_unique_remote_assets": len(unresolved),
        "current_usd_self_contained": False,
        "need_complete_dependency_package_or_exact_url_download_permission": True,
        "stage4a66d_blocked": True,
        "stage4a67_blocked": True,
    }
    save_json(output_dir / "env_isaacsim_dependency_blocker.json", blocker)
    write_text(
        output_dir / "env_isaacsim_dependency_blocker.md",
        "\n".join(
            [
                "# env_isaacsim Dependency Blocker (effective env_isaaclab)",
                "",
                "- User corrected the actual Isaac conda environment to `env_isaaclab`.",
                "- Searched `env_isaaclab` CONDA_PREFIX plus IsaacSim install/cache/Omniverse roots.",
                f"- Still missing `{len(unresolved)}` remote Isaac assets as trusted local files.",
                "- Current USD is still not self-contained.",
                "- Need a complete dependency package, or permission to download these exact URLs.",
                "- Stage 4A-6.6d and Stage 4A-6.7 remain blocked.",
            ]
        ),
    )
    write_dependency_package_request_updated(output_dir, search_results, args.conda_env_name)
    write_negative_scope(output_dir)

    summary = {
        "created_at_utc": utc_now(),
        "stage": "Stage 4A-6.6c-usd-env-corrected-dependency-fix",
        "task_namespace_from_prompt": "Stage 4A-6.6c-usd-dependency-fix-env-isaacsim",
        "completed": True,
        "blocked": len(unresolved) > 0,
        "main_blocker": blocker["main_blocker"] if unresolved else None,
        "conda_env": args.conda_env_name,
        "python": probe["python"]["sys_executable"],
        "isaacsim_import_ok": probe["imports"]["isaacsim"]["ok"],
        "omni_import_ok": probe["imports"]["omni"]["ok"],
        "pxr_import_ok": probe["imports"]["pxr"]["ok"],
        "pxr_import_with_omni_usd_libs_ok": probe["pxr_import_with_omni_usd_libs"].get("ok"),
        "isaacsim_install_roots": asset_roots["isaacsim_install_roots"],
        "isaac_asset_roots": asset_roots["required_isaac_45_asset_roots"],
        "omniverse_cache_roots": asset_roots["omniverse_cache_roots"],
        "previous_unique_remote_dependencies": len(unique_deps),
        "searched_unique_remote_dependencies": len(search_results),
        "local_match_found_count": len([row for row in search_results if row["local_match_found"]]),
        "exact_local_match_count": len([row for row in search_results if row["confidence"] == "exact"]),
        "copied_dependency_count": copy_report["copied_count"],
        "patched_dependency_count": patch_report["patch_count"],
        "unresolved_unique_remote_dependency_count": len(unresolved),
        "isaac_retry_allowed": retry_gate["retry_allowed"],
        "isaac_retry_executed": retry_gate["retry_executed"],
        "source_usd_hash_unchanged": hash_manifest["source_usd_hash_unchanged"],
        "staged_usd_hash_unchanged": hash_manifest["staged_usd_hash_unchanged"],
        "no_rollout": True,
        "no_formal_expert_sampling": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "stage4a66c_usd_env_isaacsim_dependency_fix_summary.json", summary)
    write_text(output_dir / "stage4a66c_usd_env_isaacsim_dependency_fix_summary.md", markdown_table("Stage 4A-6.6c USD env_isaacsim Dependency Fix Summary", summary))
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                "Provide a self-contained dependency package preserving `Assets/Isaac/4.5/Isaac/...` under `assets/home_like_scene_v1/current_environment/dependencies`, including transitive USD/material/texture dependencies, or allow downloading the exact listed URLs. Do not enter Stage 4A-6.6d or 6.7 until the USD is local-only and a single Isaac validation retry succeeds.",
            ]
        ),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
