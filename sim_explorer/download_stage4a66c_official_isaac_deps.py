#!/usr/bin/env python3
"""Download and localize exact official Isaac USD dependencies.

Stage 4A-6.6c-usd-download-official-isaac-deps is a dependency repair stage.
It downloads only exact official URLs from the previous dependency request and
transitive dependencies discovered inside those files.  It does not generate a
fallback scene, run rollout, run expert sampling, call map_predict/SSCNet, or
train policies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
CONDA_PYTHON = Path("/home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python")
PXR_USD_LIB = Path(
    "/home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/"
    "isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
CONDA_LIB = Path("/home/ubuntu22/miniconda3/envs/env_isaaclab/lib")
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
]
USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}
ALLOWED_EXTENSIONS = USD_EXTENSIONS | {
    ".mdl",
    ".mtl",
    ".obj",
    ".fbx",
    ".glb",
    ".gltf",
    ".png",
    ".jpg",
    ".jpeg",
    ".exr",
    ".hdr",
    ".tga",
    ".bmp",
    ".tif",
    ".tiff",
    ".normal",
    ".dds",
}
ASSET_TOKEN_RE = re.compile(r"@([^@\r\n]+)@")
QUOTED_ASSET_RE = re.compile(
    r"""["']([^"']+\.(?:usd|usda|usdc|usdz|mdl|mtl|obj|fbx|glb|gltf|png|jpg|jpeg|exr|hdr|tga|bmp|tif|tiff|normal)(?:[?#][^"']*)?)["']""",
    re.IGNORECASE,
)
OFFICIAL_URL_RE = re.compile(
    r"https:/+omniverse-content-production\.s3-us-west-2\.amazonaws\.com/Assets/Isaac/4\.5/Isaac/[^\s`)'\"<>]+",
    re.IGNORECASE,
)
USER_AGENT = "sc-explorer-stage4a66c-official-isaac-deps/1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
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
            text = json.dumps(jsonable(value), sort_keys=True)
            if len(text) > 1600:
                text = text[:1600] + "..."
            value_text = f"`{text}`"
        else:
            value_text = f"`{value}`"
        lines.append(f"| {key} | {value_text} |")
    return "\n".join(lines)


def markdown_list(title: str, rows: list[str]) -> str:
    body = rows or ["No rows."]
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in body]])


def canonical_url(value: str) -> str:
    text = str(value or "").strip().strip("@").rstrip(".,)")
    text = text.replace("\\", "/")
    if text.startswith("https:/") and not text.startswith("https://"):
        text = "https://" + text[len("https:/") :]
    if text.startswith("http:/") and not text.startswith("http://"):
        text = "http://" + text[len("http:/") :]
    return text


def normalize_base_url(base_url: str) -> str:
    base = canonical_url(base_url)
    return base if base.endswith("/") else base + "/"


def normalize_preserve_root(root: str) -> str:
    text = str(root).strip().strip("/")
    return text if text else "Assets/Isaac/4.5/Isaac"


def official_tail(url: str, base_url: str) -> str:
    canonical = canonical_url(url)
    base = normalize_base_url(base_url)
    if canonical.startswith(base):
        return canonical[len(base) :]
    marker = "/Assets/Isaac/4.5/Isaac/"
    if marker in canonical:
        return canonical.split(marker, 1)[1]
    raise ValueError(f"URL is outside official base: {url}")


def local_path_for_url(url: str, dependency_local_root: Path, base_url: str, preserve_root: str) -> Path:
    tail = official_tail(url, base_url)
    return dependency_local_root / normalize_preserve_root(preserve_root) / Path(tail)


def official_url_for_tail(tail: str, base_url: str) -> str:
    return normalize_base_url(base_url) + tail.lstrip("/")


def official_url_for_local(path: Path, dependency_local_root: Path, preserve_root: str, base_url: str) -> str | None:
    try:
        rel = path.resolve().relative_to((dependency_local_root / normalize_preserve_root(preserve_root)).resolve())
    except Exception:
        return None
    return official_url_for_tail(rel.as_posix(), base_url)


def make_relative_asset_path(target: Path, start_dir: Path) -> str:
    rel = os.path.relpath(target, start_dir).replace(os.sep, "/")
    if not rel.startswith("."):
        rel = "./" + rel
    return rel


def extension_allowed(path_or_url: str) -> bool:
    suffix = Path(urllib.parse.urlparse(str(path_or_url)).path).suffix.lower()
    return bool(suffix and suffix in ALLOWED_EXTENSIONS)


def is_usd_path(path_or_url: str | Path) -> bool:
    suffix = Path(urllib.parse.urlparse(str(path_or_url)).path).suffix.lower()
    return suffix in USD_EXTENSIONS


def clean_asset_path(asset: str) -> str:
    text = str(asset or "").strip().strip("@").strip()
    text = text.split("?", 1)[0].split("#", 1)[0]
    if text.startswith("./"):
        return text
    return canonical_url(text)


def is_remote(value: str) -> bool:
    text = canonical_url(value).lower()
    return bool(re.match(r"^[a-z][a-z0-9+.-]*://", text)) or text.startswith("omniverse:")


def is_official_url(value: str, base_url: str) -> bool:
    return canonical_url(value).startswith(normalize_base_url(base_url))


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


def extract_asset_paths_from_text(text: str) -> list[str]:
    out: list[str] = []
    for match in ASSET_TOKEN_RE.finditer(text):
        value = clean_asset_path(match.group(1))
        if value and value not in out:
            out.append(value)
    return out


def run_pxr_child(operation: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    python = CONDA_PYTHON if CONDA_PYTHON.is_file() else Path(sys.executable)
    cmd = [str(python), str(Path(__file__).resolve()), "__pxr_child", operation]
    result = subprocess.run(
        cmd,
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
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "operation": operation,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "error": repr(exc),
        }
    return data


def pxr_validate(path: Path) -> dict[str, Any]:
    if not is_usd_path(path):
        return {"ok": True, "skipped": True, "reason": "not_usd"}
    return run_pxr_child("validate", {"path": str(path)}, timeout=90)


def pxr_scan(path: Path) -> dict[str, Any]:
    if not is_usd_path(path):
        return {"ok": True, "skipped": True, "asset_paths": []}
    return run_pxr_child("scan", {"path": str(path)}, timeout=180)


def pxr_patch(input_path: Path, output_path: Path, replacements: dict[str, str]) -> dict[str, Any]:
    return run_pxr_child(
        "patch",
        {"input_path": str(input_path), "output_path": str(output_path), "replacements": replacements},
        timeout=240,
    )


def pxr_child_main(operation: str) -> None:
    from pxr import Sdf  # type: ignore

    payload = json.loads(sys.stdin.read() or "{}")
    if operation == "validate":
        path = Path(payload["path"])
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            print(json.dumps({"ok": False, "path": str(path), "error": "Sdf.Layer.FindOrOpen returned None"}))
            return
        text_ok = False
        text_len = None
        try:
            text_len = len(layer.ExportToString())
            text_ok = True
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "ok": False,
                        "path": str(path),
                        "identifier": str(layer.identifier),
                        "error": f"ExportToString failed: {exc}",
                    }
                )
            )
            return
        print(
            json.dumps(
                {
                    "ok": True,
                    "path": str(path),
                    "identifier": str(layer.identifier),
                    "file_format": str(layer.GetFileFormat().formatId),
                    "export_to_string_ok": text_ok,
                    "exported_text_length": text_len,
                    "root_prim_count": len(layer.rootPrims),
                }
            )
        )
        return
    if operation == "scan":
        path = Path(payload["path"])
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            print(json.dumps({"ok": False, "path": str(path), "asset_paths": [], "error": "Sdf.Layer.FindOrOpen returned None"}))
            return
        text = layer.ExportToString()
        asset_paths = extract_asset_paths_from_text(text)
        print(
            json.dumps(
                {
                    "ok": True,
                    "path": str(path),
                    "identifier": str(layer.identifier),
                    "file_format": str(layer.GetFileFormat().formatId),
                    "asset_paths": asset_paths,
                    "asset_path_count": len(asset_paths),
                    "official_like_url_count": len([p for p in asset_paths if "omniverse-content-production" in p]),
                    "omniverse_url_count": len([p for p in asset_paths if canonical_url(p).lower().startswith("omniverse://")]),
                }
            )
        )
        return
    if operation == "patch":
        input_path = Path(payload["input_path"])
        output_path = Path(payload["output_path"])
        replacements = {str(k): str(v) for k, v in payload.get("replacements", {}).items()}
        layer = Sdf.Layer.FindOrOpen(str(input_path))
        if layer is None:
            print(json.dumps({"ok": False, "error": "Sdf.Layer.FindOrOpen returned None", "input_path": str(input_path)}))
            return
        before_text = layer.ExportToString()
        after_text = before_text
        applied: list[dict[str, Any]] = []
        for old, new in replacements.items():
            variants = OrderedDict()
            variants[old] = new
            canonical = canonical_url(old)
            variants[canonical] = new
            if canonical.startswith("https://"):
                variants[canonical.replace("https://", "https:/", 1)] = new
            for variant, replacement in variants.items():
                count = after_text.count(variant)
                if count:
                    after_text = after_text.replace(variant, replacement)
                    applied.append({"old": variant, "new": replacement, "count": count})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(after_text, encoding="utf-8")
        reopened = Sdf.Layer.FindOrOpen(str(output_path))
        print(
            json.dumps(
                {
                    "ok": reopened is not None,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "input_file_format": str(layer.GetFileFormat().formatId),
                    "output_encoding": "usda_text",
                    "replacement_count": int(sum(item["count"] for item in applied)),
                    "replacement_items": applied[:200],
                    "output_opens_with_sdf": reopened is not None,
                    "output_root_prim_count": len(reopened.rootPrims) if reopened is not None else None,
                    "before_text_length": len(before_text),
                    "after_text_length": len(after_text),
                }
            )
        )
        return
    print(json.dumps({"ok": False, "error": f"unknown operation {operation}"}))


def load_context(output_dir: Path, previous_output_dir: Path, source_usd: Path, staged_usd: Path) -> dict[str, Any]:
    entries = []
    all_text = ""
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        all_text += "\n" + text
        entries.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "contains_larger_complex_scene_v1": "larger_complex_scene_v1" in text,
                "contains_rejected_or_disabled": ("rejected" in text.lower()) or ("disabled" in text.lower()),
                "contains_no_rgb_depth": "No validation RGB/depth" in text or "no validation RGB/depth" in text.lower(),
                "contains_no_observed_state_final": "no observed_state_final.npy" in text.lower(),
                "contains_no_mp4": "no MP4" in text or "no mp4" in text.lower(),
            }
        )
    previous_paths = {name: previous_output_dir / name for name in PREVIOUS_REQUIRED_FILES}
    previous_summary = read_json(previous_paths["dependency_localization_summary.json"])
    scene_load = read_json(previous_output_dir / "scene_load_validation.json")
    fixed = read_json(previous_output_dir / "fixed_capture_validation.json")
    observed = read_json(previous_output_dir / "observed_state_validation_summary.json")
    previous_blocked = bool(previous_summary.get("dependencies_complete_after_localization") is False)
    loaded = {
        "stage": "Stage 4A-6.6c-usd-download-official-isaac-deps",
        "created_at_utc": utc_now(),
        "loaded_context_files": entries,
        "previous_required_files": [
            {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in previous_paths.values()
        ],
        "dependency_package_request_loaded": previous_paths["dependency_package_request.md"].is_file(),
        "previous_dependency_blocked": previous_blocked,
        "previous_scene_validation_success": bool(scene_load.get("scene_loaded") is True and fixed.get("nonblank_rgb_count", 0) > 0),
        "previous_no_rgb_depth": int(fixed.get("nonblank_rgb_count", 0)) == 0 and int(fixed.get("finite_positive_depth_count", 0)) == 0,
        "previous_no_observed_state_final": observed.get("observed_state_final_created") is False,
        "previous_no_mp4": not (previous_output_dir / "usd_scene_flythrough.mp4").exists(),
        "current_source_usd": str(source_usd),
        "current_staged_usd": str(staged_usd),
        "current_usd_is_building_scene_usd": str(source_usd) == str(WORKSPACE / "building_scene.usd"),
        "current_staged_usd_is_expected": str(staged_usd)
        == str(WORKSPACE / "assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd"),
        "larger_complex_scene_v1_rejected_or_disabled": "larger_complex_scene_v1" in all_text
        and (("rejected" in all_text.lower()) or ("disabled" in all_text.lower())),
        "stage4a66d_allowed": False,
        "stage4a67_allowed": False,
        "allowed_download_scope": "exact dependency_package_request.md URLs plus recursive dependencies discovered from downloaded USDs under the same official base or relative paths",
    }
    save_json(output_dir / "loaded_context_manifest.json", loaded)
    write_text(
        output_dir / "loaded_context_manifest.md",
        markdown_table(
            "Loaded Context Manifest",
            {
                "previous_dependency_blocked": loaded["previous_dependency_blocked"],
                "previous_scene_validation_success": loaded["previous_scene_validation_success"],
                "previous_no_rgb_depth": loaded["previous_no_rgb_depth"],
                "previous_no_observed_state_final": loaded["previous_no_observed_state_final"],
                "previous_no_mp4": loaded["previous_no_mp4"],
                "source_usd": str(source_usd),
                "staged_usd": str(staged_usd),
                "stage4a66d_allowed": False,
                "stage4a67_allowed": False,
            },
        ),
    )
    return loaded


def previous_dependency_summary(output_dir: Path, previous_output_dir: Path) -> dict[str, Any]:
    summary = read_json(previous_output_dir / "dependency_localization_summary.json")
    expanded = read_json(previous_output_dir / "usd_dependency_expanded_report.json")
    package = read_json(previous_output_dir / "dependency_package_request.json")
    scene_load = read_json(previous_output_dir / "scene_load_validation.json")
    fixed = read_json(previous_output_dir / "fixed_capture_validation.json")
    observed = read_json(previous_output_dir / "observed_state_validation_summary.json")
    out = {
        "stage": "Stage 4A-6.6c-usd-download-official-isaac-deps",
        "previous_stage": summary.get("stage"),
        "previous_blocker_loaded": True,
        "dependency_package_request_loaded": True,
        "previous_remote_unique_dependencies": int(summary.get("remote_unique_dependencies", 0)),
        "previous_remote_reference_occurrences": int(summary.get("reference_occurrences", 0)),
        "previous_local_candidates": int(summary.get("local_candidate_rows", 0)),
        "previous_copied_dependencies": int(summary.get("copied_file_count", 0)),
        "previous_patch_applied": bool(summary.get("patch_applied")),
        "previous_dependencies_complete": bool(summary.get("dependencies_complete_after_localization")),
        "previous_no_isaac_retry": bool(summary.get("isaac_retry_executed") is False),
        "previous_scene_load_blocked": bool(scene_load.get("blocked") is True),
        "previous_scene_loaded": bool(scene_load.get("scene_loaded")),
        "previous_validation_rgb_depth": {
            "nonblank_rgb_count": int(fixed.get("nonblank_rgb_count", 0)),
            "finite_positive_depth_count": int(fixed.get("finite_positive_depth_count", 0)),
        },
        "previous_observed_state_final_created": bool(observed.get("observed_state_final_created")),
        "source_usd": summary.get("source_usd"),
        "staged_usd": summary.get("staged_usd"),
        "source_sha256": summary.get("source_sha256"),
        "staged_sha256_after": summary.get("staged_sha256_after"),
        "dependency_package_request_missing_unique": package.get("missing_unique_dependency_count"),
        "expanded_still_missing_unique": expanded.get("still_missing_unique_count"),
    }
    save_json(output_dir / "previous_dependency_fix_summary.json", out)
    write_text(output_dir / "previous_dependency_fix_summary.md", markdown_table("Previous Dependency Fix Summary", out))
    return out


def parse_official_urls_from_request(path: Path, base_url: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    urls = []
    for match in OFFICIAL_URL_RE.finditer(text):
        url = canonical_url(match.group(0))
        if is_official_url(url, base_url) and url not in urls:
            urls.append(url)
    return sorted(urls)


def write_url_list(output_dir: Path, urls: list[str], base_url: str, dependency_local_root: Path, preserve_root: str) -> list[dict[str, Any]]:
    rows = []
    for idx, url in enumerate(urls):
        local_path = local_path_for_url(url, dependency_local_root, base_url, preserve_root)
        rows.append(
            {
                "index": idx,
                "url": url,
                "local_path": str(local_path),
                "relative_tail": official_tail(url, base_url),
                "basename": Path(urllib.parse.urlparse(url).path).name,
                "source": "dependency_package_request.md",
            }
        )
    write_csv(output_dir / "official_dependency_url_list.csv", rows)
    save_json(output_dir / "official_dependency_url_list.json", rows)
    return rows


def validate_local_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"valid": False, "reason": "missing"}
    size = path.stat().st_size
    if size <= 0:
        return {"valid": False, "reason": "empty", "size_bytes": size}
    if not extension_allowed(path.as_posix()):
        return {"valid": False, "reason": "extension_not_allowed", "size_bytes": size}
    pxr = pxr_validate(path) if is_usd_path(path) else {"ok": True, "skipped": True}
    return {
        "valid": bool(pxr.get("ok")),
        "reason": "ok" if pxr.get("ok") else "pxr_validation_failed",
        "size_bytes": size,
        "sha256": sha256_file(path),
        "pxr_validation": pxr,
    }


def download_one_once(
    url: str,
    local_path: Path,
    *,
    max_single_file_bytes: int,
    existing_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "url": url,
        "local_path": str(local_path),
        "status": "not_started",
        "http_status": None,
        "content_length": None,
        "download_time_seconds": None,
        "size_bytes": None,
        "sha256": None,
        "error": None,
        "already_existed": False,
        "validation": None,
        "source": "official_isaac_s3",
    }
    if local_path.exists():
        validation = validate_local_file(local_path)
        manifest_ok = False
        if existing_manifest:
            previous = existing_manifest.get(str(local_path)) or existing_manifest.get(local_path.as_posix())
            manifest_ok = bool(
                previous
                and previous.get("sha256") == validation.get("sha256")
                and int(previous.get("size_bytes") or -1) == int(validation.get("size_bytes") or -2)
            )
        if validation.get("valid") or manifest_ok:
            row.update(
                {
                    "status": "already_exists_valid",
                    "already_existed": True,
                    "download_time_seconds": 0.0,
                    "size_bytes": validation.get("size_bytes"),
                    "sha256": validation.get("sha256"),
                    "validation": validation,
                }
            )
            return row
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        request = urllib.request.Request(canonical_url(url), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=90) as response:
            row["http_status"] = int(getattr(response, "status", 0) or 0)
            row["content_length"] = int(response.headers.get("Content-Length") or 0) or None
            if row["content_length"] and int(row["content_length"]) > max_single_file_bytes:
                row.update(
                    {
                        "status": "failed",
                        "error": f"content_length_exceeds_max_single_file_bytes:{row['content_length']}>{max_single_file_bytes}",
                    }
                )
                return row
            bytes_written = 0
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_single_file_bytes:
                        row.update(
                            {
                                "status": "failed",
                                "error": f"download_exceeds_max_single_file_bytes:{bytes_written}>{max_single_file_bytes}",
                            }
                        )
                        handle.close()
                        tmp_path.unlink(missing_ok=True)
                        return row
                    handle.write(chunk)
        tmp_path.replace(local_path)
        validation = validate_local_file(local_path)
        row.update(
            {
                "status": "downloaded" if validation.get("valid") else "downloaded_invalid",
                "download_time_seconds": round(time.perf_counter() - started, 6),
                "size_bytes": local_path.stat().st_size if local_path.is_file() else None,
                "sha256": sha256_file(local_path),
                "validation": validation,
                "error": None if validation.get("valid") else validation.get("reason"),
            }
        )
        return row
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        row.update({"status": "failed", "download_time_seconds": round(time.perf_counter() - started, 6), "error": repr(exc)})
        if isinstance(exc, urllib.error.HTTPError):
            row["http_status"] = int(exc.code)
        return row


def download_one(
    url: str,
    local_path: Path,
    *,
    max_single_file_bytes: int,
    existing_manifest: dict[str, Any] | None = None,
    attempts: int = 3,
) -> dict[str, Any]:
    attempt_rows: list[dict[str, Any]] = []
    final_row: dict[str, Any] | None = None
    for attempt_index in range(1, max(int(attempts), 1) + 1):
        row = download_one_once(
            url,
            local_path,
            max_single_file_bytes=max_single_file_bytes,
            existing_manifest=existing_manifest,
        )
        row["attempt_index"] = attempt_index
        attempt_rows.append({key: row.get(key) for key in ("attempt_index", "status", "http_status", "error", "size_bytes")})
        final_row = row
        if row["status"] in {"downloaded", "already_exists_valid"}:
            break
        if row.get("error") in {"extension_not_allowed", "pxr_validation_failed"}:
            break
        time.sleep(min(2.0 * attempt_index, 6.0))
    assert final_row is not None
    final_row["attempt_count"] = len(attempt_rows)
    final_row["attempts"] = attempt_rows
    return final_row


def classify_discovered_asset(
    asset_path: str,
    current_url: str,
    current_local_path: Path,
    *,
    base_url: str,
    dependency_local_root: Path,
    preserve_root: str,
) -> dict[str, Any]:
    asset = clean_asset_path(asset_path)
    out: dict[str, Any] = {
        "asset_path": asset,
        "current_url": current_url,
        "current_local_path": str(current_local_path),
        "dependency_kind": "unknown",
        "official_url": None,
        "local_candidate": None,
        "supported": True,
        "reason": None,
    }
    if not asset or asset.startswith("<") or asset.startswith("#"):
        out.update({"dependency_kind": "ignored", "supported": True, "reason": "empty_or_prim_path"})
        return out
    if is_official_url(asset, base_url):
        official = canonical_url(asset)
        out.update(
            {
                "dependency_kind": "official_remote",
                "official_url": official,
                "local_candidate": str(local_path_for_url(official, dependency_local_root, base_url, preserve_root)),
                "supported": extension_allowed(official),
                "reason": None if extension_allowed(official) else "extension_not_allowed",
            }
        )
        return out
    if is_remote(asset):
        out.update({"dependency_kind": "unsupported_external_dependency", "supported": False, "reason": "remote_not_official_base"})
        return out
    if Path(asset).is_absolute():
        out.update({"dependency_kind": "unsupported_absolute_dependency", "supported": False, "reason": "absolute_local_path"})
        return out

    resolved_local = (current_local_path.parent / asset).resolve()
    out["local_candidate"] = str(resolved_local)
    if resolved_local.exists():
        out.update({"dependency_kind": "relative_local_existing", "supported": True, "reason": "already_exists"})
        return out

    current_tail = official_tail(current_url, base_url)
    joined_tail = posixpath.normpath(posixpath.join(posixpath.dirname(current_tail), asset.replace("\\", "/")))
    if joined_tail.startswith("../") or joined_tail == "..":
        out.update({"dependency_kind": "unsupported_relative_dependency", "supported": False, "reason": "relative_path_escapes_official_root"})
        return out
    official = official_url_for_tail(joined_tail, base_url)
    out.update(
        {
            "dependency_kind": "relative_official_constructed",
            "official_url": official,
            "local_candidate": str(local_path_for_url(official, dependency_local_root, base_url, preserve_root)),
            "supported": extension_allowed(official),
            "reason": None if extension_allowed(official) else "extension_not_allowed",
        }
    )
    return out


def load_existing_download_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "official_dependency_download_manifest.json"
    if not path.is_file():
        return {}
    try:
        rows = read_json(path)
    except Exception:
        return {}
    out = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("local_path"):
                out[str(row["local_path"])] = row
    return out


def download_dependency_closure(args: argparse.Namespace, initial_urls: list[str], output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_url = normalize_base_url(args.official_base_url)
    dependency_local_root = Path(args.dependency_local_root).resolve()
    preserve_root = normalize_preserve_root(args.preserve_relative_root)
    max_single = int(float(args.max_single_file_gb) * 1024**3)
    max_total = int(float(args.max_total_download_gb) * 1024**3)
    max_files = int(args.max_files)
    max_depth = int(args.max_dependency_depth)
    existing_manifest = load_existing_download_manifest(output_dir)

    queue: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for url in initial_urls:
        queue[url] = {"url": url, "depth": 0, "discovered_from": "dependency_package_request.md", "initial_requested": True}
    processed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    discovered_edges: list[dict[str, Any]] = []
    unsupported_external: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    limit_blockers: list[str] = []
    total_local_size = 0
    worker_count = max(1, min(int(args.max_workers), 8))

    while queue:
        if len(processed) >= max_files:
            limit_blockers.append(f"max_files_exceeded:{len(processed)}>={max_files}")
            break
        current_batch: list[tuple[str, dict[str, Any]]] = []
        while queue and len(current_batch) < max(1, max_files - len(processed)):
            url, item = queue.popitem(last=False)
            if url not in processed:
                current_batch.append((url, item))
        if not current_batch:
            continue
        print(
            f"[INFO]: downloading batch size={len(current_batch)} processed={len(processed)} "
            f"queued_after_pop={len(queue)} workers={worker_count}",
            flush=True,
        )
        downloaded_batch: list[tuple[str, dict[str, Any], Path]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {}
            for url, item in current_batch:
                depth = int(item.get("depth", 0))
                if depth > max_depth:
                    limit_blockers.append(f"max_dependency_depth_exceeded:{url}:depth={depth}:max={max_depth}")
                    continue
                local_path = local_path_for_url(url, dependency_local_root, base_url, preserve_root)
                future = executor.submit(download_one, url, local_path, max_single_file_bytes=max_single, existing_manifest=existing_manifest)
                future_map[future] = (url, item, local_path)
            for future, (url, item, local_path) in list(future_map.items()):
                row = future.result()
                depth = int(item.get("depth", 0))
                row.update(
                    {
                        "depth": depth,
                        "initial_requested": bool(item.get("initial_requested")),
                        "discovered_from": item.get("discovered_from"),
                        "official_relative_tail": official_tail(url, base_url),
                    }
                )
                processed[url] = row
                downloaded_batch.append((url, row, local_path))
                if row.get("size_bytes"):
                    total_local_size += int(row["size_bytes"])
                    if total_local_size > max_total:
                        limit_blockers.append(f"max_total_download_bytes_exceeded:{total_local_size}>{max_total}")
        print(
            f"[INFO]: batch complete processed={len(processed)} "
            f"ok={len([r for _, r, _ in downloaded_batch if r['status'] in {'downloaded', 'already_exists_valid'}])} "
            f"failed={len([r for _, r, _ in downloaded_batch if r['status'] not in {'downloaded', 'already_exists_valid'}])}",
            flush=True,
        )
        if limit_blockers:
            break
        for url, row, local_path in downloaded_batch:
            depth = int(row.get("depth", 0))
            if row["status"] not in {"downloaded", "already_exists_valid"}:
                continue
            if not is_usd_path(local_path):
                continue
            scan = pxr_scan(local_path)
            row["recursive_scan"] = scan
            if not scan.get("ok"):
                unsupported_external.append(
                    {
                        "asset_path": url,
                        "current_url": url,
                        "current_local_path": str(local_path),
                        "dependency_kind": "scan_failed",
                        "supported": False,
                        "reason": scan.get("error") or scan.get("stderr"),
                    }
                )
                continue
            for asset_path in scan.get("asset_paths", []):
                dep = classify_discovered_asset(
                    asset_path,
                    url,
                    local_path,
                    base_url=base_url,
                    dependency_local_root=dependency_local_root,
                    preserve_root=preserve_root,
                )
                discovered_edges.append(dep | {"parent_depth": depth, "child_depth": depth + 1})
                if dep["dependency_kind"] in {"ignored", "relative_local_existing"}:
                    continue
                if dep["dependency_kind"] == "unsupported_external_dependency" or not dep.get("supported"):
                    unsupported_external.append(dep)
                    continue
                child_url = dep.get("official_url")
                if child_url and child_url not in processed and child_url not in queue:
                    child_depth = depth + 1
                    if child_depth > max_depth:
                        limit_blockers.append(f"max_dependency_depth_exceeded:{child_url}:depth={child_depth}:max={max_depth}")
                    else:
                        queue[child_url] = {
                            "url": child_url,
                            "depth": child_depth,
                            "discovered_from": url,
                            "initial_requested": False,
                        }
        if len(processed) + len(queue) > max_files:
            limit_blockers.append(f"max_files_queue_exceeded:{len(processed) + len(queue)}>{max_files}")
            break

    rows = list(processed.values())
    initial_set = set(initial_urls)
    successful = [row for row in rows if row["status"] in {"downloaded", "already_exists_valid"}]
    failed = [row for row in rows if row["status"] not in {"downloaded", "already_exists_valid"}]
    transitive_urls = [url for url in processed if url not in initial_set]
    recursive = {
        "stage": "Stage 4A-6.6c-usd-download-official-isaac-deps",
        "official_base_url": base_url,
        "preserve_relative_root": preserve_root,
        "initial_requested_url_count": len(initial_urls),
        "processed_url_count": len(rows),
        "successful_url_count": len(successful),
        "failed_url_count": len(failed),
        "transitive_dependency_url_count": len(transitive_urls),
        "recursive_dependencies_discovered": len([edge for edge in discovered_edges if edge.get("official_url") and edge.get("official_url") not in initial_set]),
        "unsupported_external_dependency_count": len(unsupported_external),
        "downloaded_or_existing_size_bytes": total_local_size,
        "limit_blockers": limit_blockers,
        "max_dependency_depth": max_depth,
        "max_total_download_gb": args.max_total_download_gb,
        "max_single_file_gb": args.max_single_file_gb,
        "max_files": max_files,
        "closure_download_complete": not failed and not unsupported_external and not limit_blockers,
        "discovered_edges": discovered_edges,
        "unsupported_external_dependencies": unsupported_external,
        "skipped_dependencies": skipped,
    }
    return rows, recursive


def write_download_reports(output_dir: Path, rows: list[dict[str, Any]], recursive: dict[str, Any]) -> None:
    write_csv(output_dir / "official_dependency_download_manifest.csv", rows)
    save_json(output_dir / "official_dependency_download_manifest.json", rows)
    write_text(
        output_dir / "official_dependency_download_manifest.md",
        markdown_table(
            "Official Dependency Download Manifest",
            {
                "processed_url_count": len(rows),
                "downloaded": len([r for r in rows if r["status"] == "downloaded"]),
                "already_exists_valid": len([r for r in rows if r["status"] == "already_exists_valid"]),
                "failed": len([r for r in rows if r["status"] not in {"downloaded", "already_exists_valid"}]),
                "size_bytes": sum(int(r.get("size_bytes") or 0) for r in rows),
            },
        ),
    )
    errors = [row for row in rows if row["status"] not in {"downloaded", "already_exists_valid"}]
    write_csv(output_dir / "official_dependency_download_errors.csv", errors)
    save_json(output_dir / "official_dependency_download_errors.json", errors)
    save_json(output_dir / "recursive_dependency_report.json", recursive)
    write_text(
        output_dir / "recursive_dependency_report.md",
        markdown_table(
            "Recursive Dependency Report",
            {
                "processed_url_count": recursive["processed_url_count"],
                "transitive_dependency_url_count": recursive["transitive_dependency_url_count"],
                "recursive_dependencies_discovered": recursive["recursive_dependencies_discovered"],
                "unsupported_external_dependency_count": recursive["unsupported_external_dependency_count"],
                "limit_blockers": recursive["limit_blockers"],
                "closure_download_complete": recursive["closure_download_complete"],
            },
        ),
    )


def write_hash_and_source_manifests(output_dir: Path, rows: list[dict[str, Any]], source_usd: Path, staged_usd: Path) -> dict[str, Any]:
    file_rows = []
    for row in rows:
        path = Path(row["local_path"])
        if path.is_file():
            file_rows.append(
                {
                    "url": row["url"],
                    "local_path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "status": row["status"],
                    "initial_requested": row.get("initial_requested"),
                    "source": "NVIDIA Isaac official public S3 content",
                    "source_url": row["url"],
                    "license_note": "Official Isaac asset source; no substituted or random assets were used.",
                }
            )
    manifest = {
        "stage": "Stage 4A-6.6c-usd-download-official-isaac-deps",
        "created_at_utc": utc_now(),
        "file_count": len(file_rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in file_rows),
        "source_usd": str(source_usd),
        "staged_usd": str(staged_usd),
        "source_sha256": sha256_file(source_usd),
        "staged_sha256": sha256_file(staged_usd),
        "files": file_rows,
    }
    write_csv(output_dir / "downloaded_file_hash_manifest.csv", file_rows)
    save_json(output_dir / "downloaded_file_hash_manifest.json", manifest)
    write_text(
        output_dir / "downloaded_file_hash_manifest.md",
        markdown_table(
            "Downloaded File Hash Manifest",
            {"file_count": manifest["file_count"], "total_size_bytes": manifest["total_size_bytes"], "source": "official Isaac S3 URLs only"},
        ),
    )
    source_rows = [
        {
            "local_path": row["local_path"],
            "source_url": row["url"],
            "source": "NVIDIA Isaac official public S3 content",
            "license_note": "Official Isaac asset source; exact URL download, no replacement asset.",
        }
        for row in file_rows
    ]
    write_csv(output_dir / "official_dependency_source_license_manifest.csv", source_rows)
    save_json(output_dir / "official_dependency_source_license_manifest.json", source_rows)
    write_text(
        output_dir / "official_dependency_source_license_manifest.md",
        markdown_table("Official Dependency Source/License Manifest", {"row_count": len(source_rows), "substitutions_used": False}),
    )
    return manifest


def ensure_localized_dependencies_link(localized_root: Path, dependency_local_root: Path) -> dict[str, Any]:
    localized_root.mkdir(parents=True, exist_ok=True)
    link_path = localized_root / "dependencies"
    if link_path.is_symlink():
        if link_path.resolve() != dependency_local_root.resolve():
            link_path.unlink()
            link_path.symlink_to(dependency_local_root, target_is_directory=True)
    elif link_path.exists():
        # Keep a real directory if it already exists; copy missing files lazily.
        for source in dependency_local_root.rglob("*"):
            if source.is_file():
                target = link_path / source.relative_to(dependency_local_root)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    else:
        link_path.symlink_to(dependency_local_root, target_is_directory=True)
    return {
        "localized_dependencies_path": str(link_path),
        "exists": link_path.exists(),
        "is_symlink": link_path.is_symlink(),
        "symlink_target": str(link_path.resolve()) if link_path.exists() else None,
    }


def replacement_map_for_file(asset_paths: list[str], current_file: Path, dependency_root_for_resolution: Path, base_url: str, preserve_root: str) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for asset in asset_paths:
        if not is_official_url(asset, base_url):
            continue
        url = canonical_url(asset)
        target = local_path_for_url(url, dependency_root_for_resolution, base_url, preserve_root)
        replacements[asset] = make_relative_asset_path(target, current_file.parent)
        replacements[url] = make_relative_asset_path(target, current_file.parent)
    return replacements


def create_localized_package(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    output_dir: Path,
    source_usd: Path,
    staged_usd: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_url = normalize_base_url(args.official_base_url)
    dependency_local_root = Path(args.dependency_local_root).resolve()
    localized_root = Path(args.localized_root).resolve()
    localized_usd = localized_root / staged_usd.name
    preserve_root = normalize_preserve_root(args.preserve_relative_root)
    dep_link = ensure_localized_dependencies_link(localized_root, dependency_local_root)

    patch_records: list[dict[str, Any]] = []
    root_scan = pxr_scan(staged_usd)
    root_replacements = {}
    if root_scan.get("ok"):
        root_replacements = replacement_map_for_file(
            root_scan.get("asset_paths", []),
            localized_usd,
            localized_root / "dependencies",
            base_url,
            preserve_root,
        )
    if root_replacements:
        root_patch = pxr_patch(staged_usd, localized_usd, root_replacements)
    else:
        localized_usd.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_usd, localized_usd)
        root_patch = {"ok": True, "replacement_count": 0, "reason": "no_official_remote_refs_in_root_scan"}
    patch_records.append({"file_role": "root_localized_usd", "path": str(localized_usd), "scan": root_scan, "patch": root_patch})

    for row in rows:
        path = Path(row["local_path"])
        if row.get("status") not in {"downloaded", "already_exists_valid"} or not path.is_file() or not is_usd_path(path):
            continue
        scan = pxr_scan(path)
        if not scan.get("ok"):
            patch_records.append({"file_role": "dependency_usd", "path": str(path), "scan": scan, "patch": {"ok": False, "reason": "scan_failed"}})
            continue
        replacements = replacement_map_for_file(scan.get("asset_paths", []), path, dependency_local_root, base_url, preserve_root)
        if replacements:
            patch = pxr_patch(path, path, replacements)
        else:
            patch = {"ok": True, "replacement_count": 0, "reason": "no_official_remote_refs"}
        patch_records.append({"file_role": "dependency_usd", "path": str(path), "scan": scan, "patch": patch})

    package_files = []
    for path in sorted(dependency_local_root.rglob("*")):
        if path.is_file():
            package_files.append(
                {
                    "path": str(path),
                    "relative_to_dependency_local_root": path.relative_to(dependency_local_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "is_usd": is_usd_path(path),
                }
            )
    package_manifest = {
        "stage": "Stage 4A-6.6c-usd-download-official-isaac-deps",
        "dependency_local_root": str(dependency_local_root),
        "localized_root": str(localized_root),
        "localized_usd": str(localized_usd),
        "localized_usd_exists": localized_usd.is_file(),
        "localized_usd_sha256": sha256_file(localized_usd),
        "dependency_link": dep_link,
        "package_file_count": len(package_files),
        "package_total_size_bytes": sum(int(row["size_bytes"]) for row in package_files),
        "files": package_files,
        "source_usd_hash_after": sha256_file(source_usd),
        "staged_usd_hash_after": sha256_file(staged_usd),
    }
    save_json(output_dir / "localized_package_manifest.json", package_manifest)
    write_text(
        output_dir / "localized_package_manifest.md",
        markdown_table(
            "Localized Package Manifest",
            {
                "dependency_local_root": str(dependency_local_root),
                "localized_root": str(localized_root),
                "localized_usd": str(localized_usd),
                "package_file_count": len(package_files),
                "package_total_size_bytes": package_manifest["package_total_size_bytes"],
                "dependency_link": dep_link,
            },
        ),
    )
    patch_report = {
        "stage": "Stage 4A-6.6c-usd-download-official-isaac-deps",
        "patch_method": "PXR Sdf.Layer.ExportToString to usda text, string-replace exact official asset URLs, reopen with Sdf.Layer",
        "source_usd_modified": False,
        "original_staged_usd_modified": False,
        "localized_usd": str(localized_usd),
        "root_usd_format_changed": True,
        "dependency_usd_format_changed_when_patched": True,
        "patch_records": patch_records,
        "root_replacement_count": int(root_patch.get("replacement_count") or 0),
        "total_replacement_count": sum(int((record.get("patch") or {}).get("replacement_count") or 0) for record in patch_records),
        "all_patch_operations_ok": all(bool((record.get("patch") or {}).get("ok")) for record in patch_records),
    }
    save_json(output_dir / "localized_patch_report.json", patch_report)
    write_text(
        output_dir / "localized_patch_report.md",
        markdown_table(
            "Localized Patch Report",
            {
                "patch_method": patch_report["patch_method"],
                "localized_usd": str(localized_usd),
                "root_replacement_count": patch_report["root_replacement_count"],
                "total_replacement_count": patch_report["total_replacement_count"],
                "all_patch_operations_ok": patch_report["all_patch_operations_ok"],
                "source_usd_modified": False,
                "original_staged_usd_modified": False,
            },
        ),
    )
    unresolved = scan_unresolved_after_patch(args, localized_usd, rows)
    save_json(output_dir / "unresolved_after_patch.json", unresolved)
    write_text(
        output_dir / "unresolved_after_patch.md",
        markdown_table(
            "Unresolved After Patch",
            {
                "remote_official_ref_count": unresolved["remote_official_ref_count"],
                "omniverse_ref_count": unresolved["omniverse_ref_count"],
                "unsupported_external_dependency_count": unresolved["unsupported_external_dependency_count"],
                "unresolved_local_dependency_count": unresolved["unresolved_local_dependency_count"],
                "dependency_closure_complete": unresolved["dependency_closure_complete"],
            },
        ),
    )
    return package_manifest, patch_report, unresolved


def scan_unresolved_after_patch(args: argparse.Namespace, localized_usd: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_url = normalize_base_url(args.official_base_url)
    files = [localized_usd]
    for row in rows:
        path = Path(row["local_path"])
        if path.is_file() and is_usd_path(path):
            files.append(path)
    file_reports = []
    remote_official: list[dict[str, Any]] = []
    omniverse: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    unresolved_local: list[dict[str, Any]] = []
    for path in files:
        scan = pxr_scan(path)
        report = {"path": str(path), "scan_ok": bool(scan.get("ok")), "asset_path_count": len(scan.get("asset_paths", []))}
        file_reports.append(report)
        if not scan.get("ok"):
            unresolved_local.append({"path": str(path), "asset_path": None, "reason": "pxr_scan_failed", "scan": scan})
            continue
        for asset in scan.get("asset_paths", []):
            asset = clean_asset_path(asset)
            if not asset or asset.startswith("<") or asset.startswith("#"):
                continue
            if is_official_url(asset, base_url):
                remote_official.append({"path": str(path), "asset_path": asset})
                continue
            canonical = canonical_url(asset)
            if canonical.lower().startswith("omniverse://") or canonical.lower().startswith("omniverse:"):
                omniverse.append({"path": str(path), "asset_path": asset})
                continue
            if is_remote(asset):
                unsupported.append({"path": str(path), "asset_path": asset, "reason": "remote_not_official_base"})
                continue
            if Path(asset).is_absolute():
                candidate = Path(asset)
            else:
                candidate = (path.parent / asset).resolve()
            if not candidate.exists():
                unresolved_local.append({"path": str(path), "asset_path": asset, "resolved_candidate": str(candidate), "reason": "local_dependency_missing"})
    return {
        "stage": "Stage 4A-6.6c-usd-download-official-isaac-deps",
        "localized_usd": str(localized_usd),
        "usd_file_count_scanned": len(files),
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


def build_gate_report(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    source_hash_before: str | None,
    staged_hash_before: str | None,
    source_usd: Path,
    staged_usd: Path,
    initial_urls: list[str],
    rows: list[dict[str, Any]],
    recursive: dict[str, Any],
    package_manifest: dict[str, Any],
    patch_report: dict[str, Any],
    unresolved: dict[str, Any],
) -> dict[str, Any]:
    successful_initial = [
        row for row in rows if row.get("initial_requested") and row.get("status") in {"downloaded", "already_exists_valid"}
    ]
    failed_initial = [
        row for row in rows if row.get("initial_requested") and row.get("status") not in {"downloaded", "already_exists_valid"}
    ]
    localized_usd = Path(package_manifest["localized_usd"])
    localized_validate = pxr_validate(localized_usd) if localized_usd.is_file() else {"ok": False, "reason": "localized_usd_missing"}
    source_hash_after = sha256_file(source_usd)
    staged_hash_after = sha256_file(staged_usd)
    blockers = []
    if len(successful_initial) != len(initial_urls):
        blockers.append("not_all_initial_requested_urls_downloaded_or_valid")
    if failed_initial:
        blockers.append("initial_requested_download_failures_present")
    if recursive.get("failed_url_count"):
        blockers.append("download_failures_present")
    if recursive.get("unsupported_external_dependency_count"):
        blockers.append("unsupported_external_dependencies_present")
    if recursive.get("limit_blockers"):
        blockers.append("download_limit_blocker_present")
    if not localized_usd.is_file() or not localized_validate.get("ok"):
        blockers.append("localized_usd_missing_or_unreadable")
    if unresolved.get("remote_official_ref_count") != 0:
        blockers.append("remote_official_refs_remaining")
    if unresolved.get("omniverse_ref_count") != 0:
        blockers.append("omniverse_refs_remaining")
    if unresolved.get("unsupported_external_dependency_count") != 0:
        blockers.append("unsupported_external_dependencies_remaining_after_patch")
    if unresolved.get("unresolved_local_dependency_count") != 0:
        blockers.append("unresolved_local_dependencies_remaining")
    if source_hash_before != source_hash_after:
        blockers.append("source_usd_hash_changed")
    if staged_hash_before != staged_hash_after:
        blockers.append("original_staged_usd_hash_changed")
    if not patch_report.get("all_patch_operations_ok"):
        blockers.append("patch_operation_failed")
    retry_allowed = not blockers
    gate = {
        "stage": "Stage 4A-6.6c-usd-download-official-isaac-deps",
        "retry_allowed": retry_allowed,
        "reason": "dependency_closure_complete" if retry_allowed else ";".join(blockers),
        "blockers": blockers,
        "retry_attempted": False,
        "retry_attempt_count": 0,
        "retry_result": "not_attempted",
        "isaac_headless_startup_count": 0,
        "exactly_one_retry": False,
        "initial_requested_urls": len(initial_urls),
        "initial_requested_downloaded_or_valid": len(successful_initial),
        "initial_requested_failed": len(failed_initial),
        "recursive_dependencies_discovered": recursive.get("recursive_dependencies_discovered"),
        "transitive_dependency_url_count": recursive.get("transitive_dependency_url_count"),
        "localized_usd": str(localized_usd),
        "localized_usd_pxr_readable": bool(localized_validate.get("ok")),
        "localized_usd_validation": localized_validate,
        "remote_official_refs_remaining": unresolved.get("remote_official_ref_count"),
        "omniverse_refs_remaining": unresolved.get("omniverse_ref_count"),
        "unsupported_external_dependencies_remaining": unresolved.get("unsupported_external_dependency_count"),
        "unresolved_local_dependencies_remaining": unresolved.get("unresolved_local_dependency_count"),
        "source_usd_hash_unchanged": source_hash_before == source_hash_after,
        "original_staged_usd_hash_unchanged": staged_hash_before == staged_hash_after,
        "procedural_fallback_used": False,
        "cuboid_fallback_used": False,
        "no_rollout": True,
        "no_expert_sampling": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
    }
    save_json(output_dir / "isaac_retry_gate_report.json", gate)
    write_text(output_dir / "isaac_retry_gate_report.md", markdown_table("Isaac Retry Gate Report", gate))
    return gate


def run_isaac_retry_once(args: argparse.Namespace, output_dir: Path, localized_usd: Path, gate: dict[str, Any]) -> dict[str, Any]:
    try:
        from isaaclab.app import AppLauncher  # type: ignore

        import import_stage4a66c_usd_home_like_scene as usd_import
    except Exception as exc:  # noqa: BLE001
        gate.update(
            {
                "retry_attempted": True,
                "retry_attempt_count": 1,
                "retry_result": "failed_to_import_isaac_retry_helpers",
                "isaac_retry_error": repr(exc),
                "isaac_headless_startup_count": 0,
                "exactly_one_retry": True,
            }
        )
        return gate

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

    gate.update({"retry_attempted": True, "retry_attempt_count": 1, "isaac_headless_startup_count": 1, "exactly_one_retry": True})
    simulation_app = None
    try:
        validation, simulation_app = usd_import.capture_with_isaac(retry_args, AppLauncher, output_dir, localized_usd)
        usd_import.save_image_grids(output_dir, "validation", validation["fixed_capture_validation"]["records"])
        usd_import.save_image_grids(output_dir, "inspection", validation["visual_inspection_capture_validation"]["records"])
        video_report = (
            usd_import.make_flythrough(output_dir, validation["visual_inspection_capture_validation"]["records"])
            if args.make_flythrough
            else {"mp4_created": False, "video_path": None, "frame_count": 0}
        )
        write_retry_reports(output_dir, validation, video_report, localized_usd)
        gate.update(
            {
                "retry_result": "succeeded",
                "scene_loaded": True,
                "validation_rgb_count": validation["fixed_capture_validation"]["nonblank_rgb_count"],
                "validation_depth_count": validation["fixed_capture_validation"]["finite_positive_depth_count"],
                "inspection_rgb_count": validation["visual_inspection_capture_validation"]["nonblank_rgb_count"],
                "inspection_depth_count": validation["visual_inspection_capture_validation"]["finite_positive_depth_count"],
                "observed_state_final": str(output_dir / "observed_state_final.npy"),
                "visual_package": str(output_dir / "visual_inspection_index.html"),
                "mp4": str(output_dir / "usd_scene_flythrough.mp4"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        gate.update({"retry_result": "failed", "scene_loaded": False, "isaac_retry_error": repr(exc)})
    finally:
        if simulation_app is not None:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    return gate


def write_retry_reports(output_dir: Path, validation: dict[str, Any], video_report: dict[str, Any], localized_usd: Path) -> None:
    pairs = [
        ("scene_load_validation", "scene_load_validation_retry"),
        ("fixed_capture_validation", "fixed_capture_validation_retry"),
        ("visual_inspection_capture_validation", "visual_inspection_capture_validation_retry"),
        ("observed_state_validation_summary", "observed_state_validation_summary_retry"),
    ]
    for src, dst in pairs:
        json_src = output_dir / f"{src}.json"
        md_src = output_dir / f"{src}.md"
        if json_src.is_file():
            data = read_json(json_src)
            data["retry_stage"] = "Stage 4A-6.6c-usd-download-official-isaac-deps"
            save_json(output_dir / f"{dst}.json", data)
            write_text(output_dir / f"{dst}.md", markdown_table(dst.replace("_", " ").title(), data))
        elif src == "observed_state_validation_summary":
            data = validation.get("observed_summary", {})
            data["retry_stage"] = "Stage 4A-6.6c-usd-download-official-isaac-deps"
            save_json(output_dir / f"{dst}.json", data)
            write_text(output_dir / f"{dst}.md", markdown_table(dst.replace("_", " ").title(), data))
        elif md_src.is_file():
            shutil.copy2(md_src, output_dir / f"{dst}.md")
    checklist = {
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "visual_approval_required_before_6_7": True,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "items": [
            {"id": 1, "question": "Review localized USD RGB/depth validation views.", "status": "pending_user_review"},
            {"id": 2, "question": "Confirm scene quality before Stage 4A-6.6d.", "status": "pending_user_review"},
        ],
    }
    save_json(output_dir / "human_visual_review_checklist.json", checklist)
    write_text(output_dir / "human_visual_review_checklist.md", markdown_list("Human Visual Review Checklist", [item["question"] for item in checklist["items"]]))
    manual = {
        "human_visual_inspection_done": False,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
        "localized_usd": str(localized_usd),
    }
    save_json(output_dir / "manual_review_gate.json", manual)
    write_text(output_dir / "manual_review_gate.md", markdown_table("Manual Review Gate", manual))
    write_retry_html(output_dir, localized_usd, video_report)


def write_retry_html(output_dir: Path, localized_usd: Path, video_report: dict[str, Any]) -> None:
    images = [
        "rgb_validation_grid.png",
        "depth_validation_grid.png",
        "rgb_inspection_grid.png",
        "depth_inspection_grid.png",
        "observed_topdown_final.png",
    ]
    figures = "\n".join(
        f'<figure><img src="{html.escape(name)}" width="360"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in images
        if (output_dir / name).is_file()
    )
    video = '<video controls width="720" src="usd_scene_flythrough.mp4"></video>' if video_report.get("mp4_created") else ""
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 4A-6.6c localized USD retry</title></head>
<body>
<h1>Stage 4A-6.6c localized USD retry</h1>
<p>Localized USD: <code>{html.escape(str(localized_usd))}</code></p>
<p>human_visual_inspection_done: <code>false</code>; formal_expert_sampling_ready: <code>false</code>.</p>
{figures}
{video}
</body></html>"""
    write_text(output_dir / "visual_inspection_index.html", body)


def write_blocker_and_summary(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    previous: dict[str, Any],
    initial_urls: list[str],
    rows: list[dict[str, Any]],
    recursive: dict[str, Any],
    package_manifest: dict[str, Any],
    patch_report: dict[str, Any],
    unresolved: dict[str, Any],
    gate: dict[str, Any],
    source_hash_before: str | None,
    staged_hash_before: str | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    source_usd = Path(args.source_usd).resolve()
    staged_usd = Path(args.staged_usd).resolve()
    downloaded = [row for row in rows if row["status"] == "downloaded"]
    existing = [row for row in rows if row["status"] == "already_exists_valid"]
    failed = [row for row in rows if row["status"] not in {"downloaded", "already_exists_valid"}]
    initial_set = set(initial_urls)
    transitive_processed = [row for row in rows if row["url"] not in initial_set]
    transitive_downloaded = [row for row in transitive_processed if row["status"] in {"downloaded", "already_exists_valid"}]
    blocked = gate.get("retry_result") != "succeeded"
    summary = {
        "stage": "Stage 4A-6.6c-usd-download-official-isaac-deps",
        "completed": not blocked,
        "blocked": blocked,
        "main_blocker": None if not blocked else gate.get("reason") or gate.get("retry_result"),
        "source_usd": str(source_usd),
        "staged_usd": str(staged_usd),
        "source_sha256_before": source_hash_before,
        "staged_sha256_before": staged_hash_before,
        "source_sha256_after": sha256_file(source_usd),
        "staged_sha256_after": sha256_file(staged_usd),
        "source_usd_hash_unchanged": source_hash_before == sha256_file(source_usd),
        "staged_usd_hash_unchanged": staged_hash_before == sha256_file(staged_usd),
        "previous_missing_unique_dependencies": previous.get("previous_remote_unique_dependencies"),
        "previous_remote_occurrences": previous.get("previous_remote_reference_occurrences"),
        "previous_local_candidates": previous.get("previous_local_candidates"),
        "initial_requested_urls": len(initial_urls),
        "downloaded_successfully": len(downloaded),
        "already_existed_and_valid": len(existing),
        "failed_downloads": len(failed),
        "recursive_dependencies_discovered": recursive.get("recursive_dependencies_discovered"),
        "recursive_dependencies_downloaded": len(transitive_downloaded),
        "unsupported_external_dependencies": recursive.get("unsupported_external_dependency_count"),
        "total_downloaded_size_bytes": sum(int(row.get("size_bytes") or 0) for row in rows),
        "dependency_local_root": str(Path(args.dependency_local_root).resolve()),
        "localized_root": str(Path(args.localized_root).resolve()),
        "localized_usd": package_manifest.get("localized_usd"),
        "patch_method": patch_report.get("patch_method"),
        "remote_official_refs_remaining": unresolved.get("remote_official_ref_count"),
        "omniverse_refs_remaining": unresolved.get("omniverse_ref_count"),
        "unresolved_local_deps_remaining": unresolved.get("unresolved_local_dependency_count"),
        "retry_allowed": gate.get("retry_allowed"),
        "retry_attempted": gate.get("retry_attempted"),
        "retry_attempt_count": gate.get("retry_attempt_count"),
        "retry_result": gate.get("retry_result"),
        "human_visual_inspection_done": False,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
        "procedural_fallback": False,
        "larger_complex_scene_restored": False,
        "rollout": False,
        "selected_action": False,
        "expert_sampling": False,
        "map_predict": False,
        "sscnet_inference": False,
        "prediction_npz": False,
        "training_rl_gdpo_ppo_bc_il": False,
        "checkpoint_modified": False,
        "elapsed_seconds": elapsed_seconds,
        "download_manifest": str(output_dir / "official_dependency_download_manifest.json"),
        "test_log": str(WORKSPACE / "logs/stage4a66c_download_official_isaac_deps_test.log"),
    }
    save_json(output_dir / "stage4a66c_usd_download_official_isaac_deps_summary.json", summary)
    write_text(output_dir / "stage4a66c_usd_download_official_isaac_deps_summary.md", markdown_table("Stage 4A-6.6c USD Download Official Isaac Deps Summary", summary))
    if blocked:
        blocker = {
            "stage": summary["stage"],
            "blocked": True,
            "reason": summary["main_blocker"],
            "failed_downloads": failed,
            "unresolved_after_patch": unresolved,
            "retry_gate": gate,
            "next": "continue downloading missing exact official URLs or ask teammate for complete package; do not enter Stage 4A-6.6d/6.7",
            "no_isaac_retry_artifacts_fabricated": gate.get("retry_result") != "succeeded",
        }
        save_json(output_dir / "official_dependency_download_blocker.json", blocker)
        write_text(output_dir / "official_dependency_download_blocker.md", markdown_table("Official Dependency Download Blocker", blocker))
        needed_lines = [
            "# Dependency Package Request Still Needed",
            "",
            f"- Retry allowed: `{gate.get('retry_allowed')}`",
            f"- Retry attempted: `{gate.get('retry_attempted')}`",
            f"- Blocker: `{summary['main_blocker']}`",
            f"- Failed downloads: `{len(failed)}`",
            f"- Remote official refs remaining: `{unresolved.get('remote_official_ref_count')}`",
            f"- Unresolved local deps remaining: `{unresolved.get('unresolved_local_dependency_count')}`",
            "",
            "Only exact official Isaac dependency URLs and their recursive same-base dependencies are allowed.",
        ]
        write_text(output_dir / "dependency_package_request_still_needed.md", "\n".join(needed_lines))
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\n"
        + (
            "Run Stage 4A-6.6d USD scene audit and human visual review on the localized validation package. Keep Stage 4A-6.7 blocked until review passes."
            if not blocked
            else "Resolve the remaining exact official Isaac dependency closure, then rerun this same dependency-localization gate. Do not run Stage 4A-6.6d or Stage 4A-6.7 yet."
        ),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download exact official Isaac USD dependencies and localize a staged USD.")
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--staged_usd", required=True)
    parser.add_argument("--dependency_request", required=True)
    parser.add_argument("--previous_output_dir", required=True)
    parser.add_argument("--dependency_local_root", required=True)
    parser.add_argument("--localized_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--official_base_url", required=True)
    parser.add_argument("--preserve_relative_root", required=True)
    parser.add_argument("--download_exact_request_urls", action="store_true")
    parser.add_argument("--download_recursive_official_dependencies", action="store_true")
    parser.add_argument("--max_dependency_depth", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=2000)
    parser.add_argument("--max_total_download_gb", type=float, default=20.0)
    parser.add_argument("--max_single_file_gb", type=float, default=3.0)
    parser.add_argument("--create_localized_package", action="store_true")
    parser.add_argument("--patch_remote_refs_to_local", action="store_true")
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
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "__pxr_child":
        pxr_child_main(sys.argv[2])
        return
    started = time.perf_counter()
    args = parse_args()
    if not args.download_exact_request_urls:
        raise ValueError("--download_exact_request_urls is required")
    if not args.download_recursive_official_dependencies:
        raise ValueError("--download_recursive_official_dependencies is required")
    if not args.create_localized_package or not args.patch_remote_refs_to_local:
        raise ValueError("--create_localized_package and --patch_remote_refs_to_local are required")
    if not (args.no_rollout and args.no_formal_expert_sampling and args.no_map_predict and args.no_rl_gdpo):
        raise ValueError("All negative-scope flags are required")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_usd = Path(args.source_usd).resolve()
    staged_usd = Path(args.staged_usd).resolve()
    previous_output_dir = Path(args.previous_output_dir).resolve()
    source_hash_before = sha256_file(source_usd)
    staged_hash_before = sha256_file(staged_usd)

    loaded = load_context(output_dir, previous_output_dir, source_usd, staged_usd)
    previous = previous_dependency_summary(output_dir, previous_output_dir)
    if not loaded["previous_dependency_blocked"]:
        raise RuntimeError("Previous run was not dependency blocked; refusing to run this repair stage.")

    initial_urls = parse_official_urls_from_request(Path(args.dependency_request).resolve(), args.official_base_url)
    write_url_list(output_dir, initial_urls, args.official_base_url, Path(args.dependency_local_root).resolve(), args.preserve_relative_root)
    rows, recursive = download_dependency_closure(args, initial_urls, output_dir)
    write_download_reports(output_dir, rows, recursive)

    package_manifest, patch_report, unresolved = create_localized_package(args, rows, output_dir, source_usd, staged_usd)
    hash_manifest = write_hash_and_source_manifests(output_dir, rows, source_usd, staged_usd)
    gate = build_gate_report(
        args,
        output_dir,
        source_hash_before=source_hash_before,
        staged_hash_before=staged_hash_before,
        source_usd=source_usd,
        staged_usd=staged_usd,
        initial_urls=initial_urls,
        rows=rows,
        recursive=recursive,
        package_manifest=package_manifest,
        patch_report=patch_report,
        unresolved=unresolved,
    )
    write_no_scope_reports(output_dir)
    if gate["retry_allowed"] and args.retry_isaac_if_dependencies_resolved:
        gate = run_isaac_retry_once(args, output_dir, Path(package_manifest["localized_usd"]), gate)
        save_json(output_dir / "isaac_retry_gate_report.json", gate)
        write_text(output_dir / "isaac_retry_gate_report.md", markdown_table("Isaac Retry Gate Report", gate))
    summary = write_blocker_and_summary(
        args,
        output_dir,
        previous=previous,
        initial_urls=initial_urls,
        rows=rows,
        recursive=recursive,
        package_manifest=package_manifest,
        patch_report=patch_report,
        unresolved=unresolved,
        gate=gate,
        source_hash_before=source_hash_before,
        staged_hash_before=staged_hash_before,
        elapsed_seconds=time.perf_counter() - started,
    )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "completed": summary["completed"],
                "blocked": summary["blocked"],
                "initial_requested_urls": summary["initial_requested_urls"],
                "downloaded_successfully": summary["downloaded_successfully"],
                "already_existed_and_valid": summary["already_existed_and_valid"],
                "failed_downloads": summary["failed_downloads"],
                "recursive_dependencies_downloaded": summary["recursive_dependencies_downloaded"],
                "retry_allowed": summary["retry_allowed"],
                "retry_attempted": summary["retry_attempted"],
                "retry_result": summary["retry_result"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
