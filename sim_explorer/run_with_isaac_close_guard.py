#!/usr/bin/env python3
"""Run an Isaac child command under a close-timeout lifecycle guard.

The wrapper intentionally does not import Isaac.  It supervises a child process
group, watches a finalization sentinel, and only treats forced termination as
success after verified output finalization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from isaac_lifecycle_guard import (
    atomic_write_json,
    atomic_write_text,
    collect_gpu_snapshot,
    collect_process_snapshot,
    load_finalization_sentinel,
    safe_report_no_unrelated_kills,
    scan_processes_for_run_id,
    terminate_process_group,
    utc_now,
    validate_required_outputs,
)


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, sort_keys=True)
        else:
            text = str(value)
        if len(text) > 1800:
            text = text[:1800] + "..."
        lines.append(f"| `{key}` | `{text}` |")
    return "\n".join(lines)


def _reader_thread(pipe: Any, log_path: Path, mirror: bool) -> threading.Thread:
    def _run() -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            for line in iter(pipe.readline, ""):
                handle.write(line)
                handle.flush()
                if mirror:
                    print(line, end="")
            pipe.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run_id", default="")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--finalization_sentinel_name", default="stage_finalized_before_isaac_close.json")
    parser.add_argument("--finalization_sentinel_path", type=Path, default=None)
    parser.add_argument("--close_timeout_sec", type=float, default=90.0)
    parser.add_argument("--terminate_grace_sec", type=float, default=20.0)
    parser.add_argument("--total_timeout_sec", type=float, default=7200.0)
    parser.add_argument("--poll_interval_sec", type=float, default=0.1)
    parser.add_argument("--required_output", action="append", type=Path, default=[])
    parser.add_argument("--required_outputs_json", type=Path, default=None)
    parser.add_argument("--require_safe_finalization_for_success", action="store_true")
    parser.add_argument("--kill_matching_run_id_orphans", action="store_true")
    parser.add_argument("--mirror_child_output", action="store_true")
    parser.add_argument("--child_cwd", type=Path, default=None)
    parser.add_argument("child_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.child_command and args.child_command[0] == "--":
        args.child_command = args.child_command[1:]
    if not args.child_command:
        parser.error("child command is required after --")
    if not args.run_id:
        args.run_id = f"{args.stage.replace(' ', '_').replace('/', '_')}_{uuid.uuid4().hex[:12]}"
    return args


def _load_required_outputs(args: argparse.Namespace) -> list[Path]:
    paths = list(args.required_output or [])
    if args.required_outputs_json:
        data = json.loads(args.required_outputs_json.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("required_outputs", data.get("paths", []))
        paths.extend(Path(item) for item in data)
    return paths


def run_supervised(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = (args.finalization_sentinel_path or (output_dir / args.finalization_sentinel_name)).resolve()
    required_outputs = _load_required_outputs(args)

    process_before = collect_process_snapshot()
    gpu_before = collect_gpu_snapshot()
    atomic_write_json(output_dir / "process_snapshot_before.json", process_before)
    atomic_write_json(output_dir / "gpu_snapshot_before.json", gpu_before)

    env = os.environ.copy()
    env["ISAAC_CLOSE_GUARD_RUN_ID"] = args.run_id
    env["ISAAC_CLOSE_GUARD_SENTINEL"] = str(sentinel_path)
    env["PYTHONUNBUFFERED"] = "1"

    started_at = time.monotonic()
    started_at_utc = utc_now()
    proc = subprocess.Popen(
        [str(part) for part in args.child_command],
        cwd=str(args.child_cwd) if args.child_cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    child_pgid = os.getpgid(proc.pid)
    stdout_thread = _reader_thread(proc.stdout, output_dir / "child_stdout.log", args.mirror_child_output)
    stderr_thread = _reader_thread(proc.stderr, output_dir / "child_stderr.log", args.mirror_child_output)

    sentinel: dict[str, Any] | None = None
    sentinel_seen_at: float | None = None
    close_status = "running"
    termination_report: dict[str, Any] | None = None
    child_returncode: int | None = None
    success_with_close_hang = False
    success = False
    failure_reason = ""

    while True:
        child_returncode = proc.poll()
        loaded = load_finalization_sentinel(sentinel_path)
        if loaded and sentinel is None:
            sentinel = loaded
            sentinel_seen_at = time.monotonic()

        if child_returncode is not None:
            close_status = "clean_exit" if child_returncode == 0 else "child_exit_nonzero"
            break

        now = time.monotonic()
        sentinel_safe = bool(sentinel and sentinel.get("safe_to_terminate_after_close_timeout"))
        if sentinel_safe and sentinel_seen_at is not None and now - sentinel_seen_at >= float(args.close_timeout_sec):
            termination_report = terminate_process_group(child_pgid, args.terminate_grace_sec, sigkill=True)
            try:
                child_returncode = proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child_returncode = proc.poll()
            close_status = "forced_terminated_after_finalization"
            success_with_close_hang = True
            break

        if now - started_at >= float(args.total_timeout_sec):
            termination_report = terminate_process_group(child_pgid, args.terminate_grace_sec, sigkill=True)
            try:
                child_returncode = proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child_returncode = proc.poll()
            close_status = "failed_before_finalization" if not sentinel_safe else "failed_after_finalization_timeout"
            failure_reason = "total_timeout"
            break

        time.sleep(float(args.poll_interval_sec))

    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)

    output_validation = validate_required_outputs(required_outputs)
    sentinel_safe = bool(sentinel and sentinel.get("safe_to_terminate_after_close_timeout"))
    if close_status == "clean_exit":
        success = child_returncode == 0 and bool(output_validation["required_output_checks_passed"])
        if args.require_safe_finalization_for_success:
            success = success and sentinel_safe
            if not sentinel_safe:
                failure_reason = "missing_or_unsafe_finalization_sentinel"
    elif close_status == "forced_terminated_after_finalization":
        success = sentinel_safe and bool(output_validation["required_output_checks_passed"])
        success_with_close_hang = success
        if not success:
            failure_reason = "forced_termination_without_safe_revalidation"
    else:
        success = False
        if not failure_reason:
            failure_reason = close_status

    orphan_scan = scan_processes_for_run_id(args.run_id)
    killed_pids = termination_report.get("members_before", []) if termination_report else []
    process_after = collect_process_snapshot()
    gpu_after = collect_gpu_snapshot()
    no_unrelated = safe_report_no_unrelated_kills(
        run_id=args.run_id,
        child_pgid=child_pgid,
        terminated_pids=killed_pids,
        process_snapshot_before=process_before,
        process_snapshot_after=process_after,
    )
    atomic_write_json(output_dir / "process_snapshot_after.json", process_after)
    atomic_write_json(output_dir / "gpu_snapshot_after.json", gpu_after)

    orphan_cleanup_performed = False
    if args.kill_matching_run_id_orphans and orphan_scan.get("match_count"):
        orphan_cleanup_performed = False

    report: dict[str, Any] = {
        "stage": args.stage,
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started_at,
        "child_command": [str(part) for part in args.child_command],
        "child_pid": proc.pid,
        "child_process_group_id": child_pgid,
        "child_returncode": child_returncode,
        "sentinel_path": str(sentinel_path),
        "sentinel_seen": sentinel is not None,
        "sentinel_safe": sentinel_safe,
        "sentinel": sentinel,
        "required_outputs": [str(path) for path in required_outputs],
        "output_revalidation": output_validation,
        "close_status": close_status,
        "termination_report": termination_report,
        "success_with_close_hang": success_with_close_hang,
        "success": success,
        "failure_reason": failure_reason,
        "orphan_scan": orphan_scan,
        "orphan_cleanup_performed": orphan_cleanup_performed,
        "no_unrelated_process_kill_report": no_unrelated,
        "process_snapshot_before": str(output_dir / "process_snapshot_before.json"),
        "process_snapshot_after": str(output_dir / "process_snapshot_after.json"),
        "gpu_snapshot_before": str(output_dir / "gpu_snapshot_before.json"),
        "gpu_snapshot_after": str(output_dir / "gpu_snapshot_after.json"),
    }
    atomic_write_json(output_dir / "supervisor_report.json", report)
    atomic_write_text(output_dir / "supervisor_report.md", markdown_table("Isaac Close Guard Supervisor Report", report))
    return (0 if success else 2), report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exit_code, report = run_supervised(args)
    print(json.dumps({"success": report["success"], "close_status": report["close_status"], "output_dir": report["output_dir"]}, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
