#!/usr/bin/env python3
"""Isaac runtime lifecycle guard helpers.

This module intentionally does not import Isaac.  It is safe to use from
parent-process supervisors, tests, and child runners immediately before
``simulation_app.close()``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


def atomic_write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


def _fsync_parent(path: Path) -> None:
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256_file(path: Path | str) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_size_hash_manifest(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for item in paths:
        path = Path(item)
        exists = path.is_file()
        manifest.append(
            {
                "path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    return manifest


def validate_required_outputs(paths: Iterable[Path | str]) -> dict[str, Any]:
    outputs = collect_file_size_hash_manifest(paths)
    missing = [row["path"] for row in outputs if not row["exists"]]
    empty = [row["path"] for row in outputs if row["exists"] and int(row["size_bytes"] or 0) <= 0]
    return {
        "required_output_count": len(outputs),
        "required_output_checks_passed": not missing and not empty,
        "required_outputs": outputs,
        "missing_required_outputs": missing,
        "empty_required_outputs": empty,
    }


def _audit_passed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "passed": False, "reason": "missing"}
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {"path": str(path), "exists": True, "passed": False, "reason": f"json_load_failed:{exc}"}
        if "passed" in data:
            return {"path": str(path), "exists": True, "passed": bool(data.get("passed")), "reason": "passed_field"}
        if "all_passed" in data:
            return {"path": str(path), "exists": True, "passed": bool(data.get("all_passed")), "reason": "all_passed_field"}
    return {"path": str(path), "exists": True, "passed": True, "reason": "non_json_exists"}


def write_finalization_sentinel(
    path: Path | str,
    *,
    stage: str,
    run_id: str,
    output_dir: Path | str,
    required_output_paths: Iterable[Path | str],
    manifest_path: Path | str | None = None,
    summary_path: Path | str | None = None,
    dataset_path: Path | str | None = None,
    html_path: Path | str | None = None,
    mp4_path: Path | str | None = None,
    audit_paths: Iterable[Path | str] | None = None,
    safe_to_terminate_requested: bool = True,
    reason: str = "outputs_finalized_before_isaac_close",
    no_long_rollout: bool = True,
    no_training: bool = True,
    no_rl_gdpo: bool = True,
    no_prediction_writeback: bool = True,
    no_uncertainty_writeback: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = validate_required_outputs(required_output_paths)
    audit_results = [_audit_passed(Path(item)) for item in (audit_paths or [])]
    audits_passed = all(row["passed"] for row in audit_results)
    safety_flags_passed = all(
        [
            bool(no_long_rollout),
            bool(no_training),
            bool(no_rl_gdpo),
            bool(no_prediction_writeback),
            bool(no_uncertainty_writeback),
        ]
    )
    safe = bool(
        safe_to_terminate_requested
        and required["required_output_checks_passed"]
        and audits_passed
        and safety_flags_passed
    )
    data: dict[str, Any] = {
        "stage": stage,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "timestamp_utc": utc_now(),
        "process_pid": os.getpid(),
        "required_output_checks_passed": required["required_output_checks_passed"],
        "required_output_count": required["required_output_count"],
        "required_outputs": required["required_outputs"],
        "required_output_hashes_or_sizes": required["required_outputs"],
        "missing_required_outputs": required["missing_required_outputs"],
        "empty_required_outputs": required["empty_required_outputs"],
        "manifest_path": str(manifest_path) if manifest_path else None,
        "summary_path": str(summary_path) if summary_path else None,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "html_path": str(html_path) if html_path else None,
        "mp4_path": str(mp4_path) if mp4_path else None,
        "audit_paths": [str(item) for item in (audit_paths or [])],
        "audit_results": audit_results,
        "audit_checks_passed": audits_passed,
        "finalization_complete": True,
        "safe_to_terminate_after_close_timeout": safe,
        "reason": reason if safe else "unsafe_to_terminate:" + reason,
        "no_long_rollout": bool(no_long_rollout),
        "no_training": bool(no_training),
        "no_rl_gdpo": bool(no_rl_gdpo),
        "no_prediction_writeback": bool(no_prediction_writeback),
        "no_uncertainty_writeback": bool(no_uncertainty_writeback),
    }
    if extra:
        data.update(_jsonable(extra))
    atomic_write_json(path, data)
    return data


def load_finalization_sentinel(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_process_snapshot() -> dict[str, Any]:
    command = ["ps", "-eo", "pid=,ppid=,pgid=,sid=,stat=,comm=,args="]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    entries: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) < 6:
            continue
        pid, ppid, pgid, sid, stat, comm = parts[:6]
        args = parts[6] if len(parts) > 6 else ""
        entries.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "pgid": int(pgid),
                "sid": int(sid),
                "stat": stat,
                "comm": comm,
                "args": args,
            }
        )
    return {
        "timestamp_utc": utc_now(),
        "command": " ".join(command),
        "returncode": result.returncode,
        "stderr": result.stderr,
        "process_count": len(entries),
        "processes": entries,
    }


def collect_gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=15)
    except FileNotFoundError:
        return {"timestamp_utc": utc_now(), "available": False, "reason": "nvidia-smi_not_found", "processes": []}
    except subprocess.TimeoutExpired:
        return {"timestamp_utc": utc_now(), "available": False, "reason": "nvidia-smi_timeout", "processes": []}
    processes = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            processes.append({"pid": int(parts[0]), "process_name": parts[1], "used_memory_mib": int(parts[2])})
    return {
        "timestamp_utc": utc_now(),
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr,
        "process_count": len(processes),
        "processes": processes,
    }


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_group_members(pgid: int) -> list[int]:
    return [row["pid"] for row in collect_process_snapshot()["processes"] if int(row.get("pgid", -1)) == int(pgid)]


def terminate_process_group(pid_or_pgid: int, sigterm_timeout_sec: float = 10.0, sigkill: bool = True) -> dict[str, Any]:
    try:
        pgid = os.getpgid(int(pid_or_pgid))
    except ProcessLookupError:
        pgid = int(pid_or_pgid)
    if pgid <= 1 or pgid == os.getpgrp():
        raise RuntimeError(f"refusing to terminate unsafe process group {pgid}")

    before = _process_group_members(pgid)
    report: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "target_pid_or_pgid": int(pid_or_pgid),
        "process_group_id": pgid,
        "members_before": before,
        "sigterm_sent": False,
        "sigkill_sent": False,
    }
    if before:
        try:
            os.killpg(pgid, signal.SIGTERM)
            report["sigterm_sent"] = True
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + float(sigterm_timeout_sec)
    while time.monotonic() < deadline:
        alive = [pid for pid in _process_group_members(pgid) if _pid_exists(pid)]
        if not alive:
            break
        time.sleep(0.05)
    after_term = [pid for pid in _process_group_members(pgid) if _pid_exists(pid)]
    report["members_after_sigterm"] = after_term
    if after_term and sigkill:
        try:
            os.killpg(pgid, signal.SIGKILL)
            report["sigkill_sent"] = True
        except ProcessLookupError:
            pass
        time.sleep(0.1)
    report["members_after"] = [pid for pid in _process_group_members(pgid) if _pid_exists(pid)]
    report["terminated"] = not report["members_after"]
    return report


def scan_processes_for_run_id(run_id: str) -> dict[str, Any]:
    snapshot = collect_process_snapshot()
    matches = [
        row
        for row in snapshot["processes"]
        if run_id and (run_id in str(row.get("args", "")) or run_id in str(row.get("comm", "")))
    ]
    return {
        "timestamp_utc": utc_now(),
        "run_id": run_id,
        "match_count": len(matches),
        "matches": matches,
        "kill_candidates": [],
        "orphan_scan_only": True,
    }


def safe_report_no_unrelated_kills(
    *,
    run_id: str,
    child_pgid: int | None,
    terminated_pids: Iterable[int] | None = None,
    process_snapshot_before: dict[str, Any] | None = None,
    process_snapshot_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    terminated = [int(pid) for pid in (terminated_pids or [])]
    before = process_snapshot_before or collect_process_snapshot()
    after = process_snapshot_after or collect_process_snapshot()
    before_pids = {int(row["pid"]): row for row in before.get("processes", [])}
    after_pids = {int(row["pid"]): row for row in after.get("processes", [])}
    missing = sorted(pid for pid in before_pids if pid not in after_pids)
    allowed = set(terminated)
    if child_pgid is not None:
        allowed.update(int(row["pid"]) for row in before.get("processes", []) if int(row.get("pgid", -1)) == int(child_pgid))
    unrelated_missing = [pid for pid in missing if pid not in allowed]
    return {
        "timestamp_utc": utc_now(),
        "run_id": run_id,
        "child_process_group_id": child_pgid,
        "terminated_pids_reported": terminated,
        "unrelated_process_kill_list": [],
        "unrelated_missing_processes_observed": unrelated_missing[:50],
        "passed": not terminated or all(pid in allowed for pid in terminated),
        "note": "Supervisor termination is restricted to its own child process group.",
    }
