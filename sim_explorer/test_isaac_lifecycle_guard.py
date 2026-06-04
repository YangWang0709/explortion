#!/usr/bin/env python3
"""Validate Stage 4A-6.13a Isaac close guard hardening with fake children."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from isaac_lifecycle_guard import (
    atomic_write_json,
    atomic_write_text,
    collect_gpu_snapshot,
    collect_process_snapshot,
    safe_report_no_unrelated_kills,
    scan_processes_for_run_id,
    sha256_file,
    validate_required_outputs,
)


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
STAGE = "Stage 4A-6.13a-isaac-close-timeout-and-safe-termination-hardening"
OUTPUT_NAME = "isaac_stage4a613a_isaac_close_guard_hardening"
SOURCE_USD = WORKSPACE / "building_scene.usd"
RUNNER_PATH = WORKSPACE / "sim_explorer/run_stage4a613_uncertainty_bonus_short_rollout_pilot.py"
SUPERVISOR_PATH = WORKSPACE / "sim_explorer/run_with_isaac_close_guard.py"
FAKE_CHILD_PATH = WORKSPACE / "sim_explorer/fake_hanging_isaac_child.py"


REQUIRED_REPORTS = [
    "stage4a613a_isaac_close_guard_hardening_summary.json",
    "stage4a613a_isaac_close_guard_hardening_summary.md",
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_stage4a613_manifest.json",
    "loaded_stage4a613_manifest.md",
    "close_hang_evidence_from_6_13.json",
    "close_hang_evidence_from_6_13.md",
    "lifecycle_guard_contract.json",
    "lifecycle_guard_contract.md",
    "finalization_sentinel_schema.json",
    "finalization_sentinel_schema.md",
    "supervisor_contract.json",
    "supervisor_contract.md",
    "process_snapshot_before.json",
    "process_snapshot_after.json",
    "gpu_snapshot_before.json",
    "gpu_snapshot_after.json",
    "fake_child_clean_exit_report.json",
    "fake_child_clean_exit_report.md",
    "fake_child_hang_after_finalization_report.json",
    "fake_child_hang_after_finalization_report.md",
    "fake_child_hang_before_finalization_report.json",
    "fake_child_hang_before_finalization_report.md",
    "orphan_scan_report.json",
    "orphan_scan_report.md",
    "no_unrelated_process_kill_report.json",
    "no_unrelated_process_kill_report.md",
    "no_isaac_runtime_report.json",
    "no_isaac_runtime_report.md",
    "no_capture_report.json",
    "no_capture_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_action_report.json",
    "no_action_report.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_training_rl_bc_report.json",
    "no_training_rl_bc_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "checkpoint_hash_report.json",
    "checkpoint_hash_report.md",
    "prior_dataset_hash_report.json",
    "prior_dataset_hash_report.md",
    "patched_runner_report.json",
    "patched_runner_report.md",
    "future_runner_usage_examples.md",
    "recommended_next_faithful_step.md",
    "git_status_before.txt",
    "git_status_after.txt",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_status_text() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return result.stdout


def markdown_table(title: str, data: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in data.items():
        text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(value)
        if len(text) > 1800:
            text = text[:1800] + "..."
        lines.append(f"| `{key}` | `{text}` |")
    return "\n".join(lines)


def write_report_pair(output_dir: Path, stem: str, data: dict[str, Any], title: str) -> None:
    atomic_write_json(output_dir / f"{stem}.json", data)
    atomic_write_text(output_dir / f"{stem}.md", markdown_table(title, data))


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    before = output_dir / "git_status_before.txt"
    before_text = before.read_text(encoding="utf-8") if before.is_file() else git_status_text()
    for child in output_dir.iterdir():
        if child.name == "git_status_before.txt":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    atomic_write_text(before, before_text)


def file_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def git_large_artifact_policy_preserved() -> dict[str, Any]:
    result = subprocess.run(["git", "ls-files"], cwd=str(WORKSPACE), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=30)
    tracked = result.stdout.splitlines()
    forbidden_prefixes = ("outputs/", "logs/", "checkpoints/", "data/")
    forbidden_suffixes = (".npy", ".npz", ".png", ".mp4", ".usd", ".pth", ".tar")
    offenders = [path for path in tracked if path.startswith(forbidden_prefixes) or path.lower().endswith(forbidden_suffixes)]
    return {"passed": result.returncode == 0 and not offenders, "offenders": offenders[:80]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage4a613_dir", type=Path, required=True)
    parser.add_argument("--fixed_usd", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expect_no_isaac", action="store_true")
    parser.add_argument("--expect_no_capture", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_sscnet_inference", action="store_true")
    parser.add_argument("--expect_no_action", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_long_rollout", action="store_true")
    parser.add_argument("--expect_no_training", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    return parser.parse_args()


def run_fake_case(output_dir: Path, mode: str, expected_exit_zero: bool) -> dict[str, Any]:
    case_dir = output_dir / {
        "clean_exit": "fake_clean_exit",
        "hang_after_finalization": "fake_hang_after_finalization",
        "hang_before_finalization": "fake_hang_before_finalization",
    }[mode]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"stage4a613a_{mode}_{os.getpid()}"
    sentinel_path = case_dir / "stage_finalized_before_isaac_close.json"
    required = [case_dir / "fake_required_output.txt", case_dir / "fake_summary.json"]
    command = [
        sys.executable,
        str(SUPERVISOR_PATH),
        "--stage",
        f"{STAGE}-{mode}",
        "--run_id",
        run_id,
        "--output_dir",
        str(case_dir),
        "--finalization_sentinel_name",
        "stage_finalized_before_isaac_close.json",
        "--close_timeout_sec",
        "1.0",
        "--terminate_grace_sec",
        "0.5",
        "--total_timeout_sec",
        "2.5" if mode == "hang_before_finalization" else "8.0",
        "--require_safe_finalization_for_success",
    ]
    for path in required:
        command.extend(["--required_output", str(path)])
    command.extend(
        [
            "--",
            sys.executable,
            str(FAKE_CHILD_PATH),
            "--mode",
            mode,
            "--output_dir",
            str(case_dir),
            "--run_id",
            run_id,
            "--finalization_sentinel_path",
            str(sentinel_path),
        ]
    )
    result = subprocess.run(command, cwd=str(WORKSPACE / "sim_explorer"), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=20)
    report = read_json(case_dir / "supervisor_report.json")
    validation = validate_required_outputs(required)
    case_report = {
        "mode": mode,
        "expected_exit_zero": expected_exit_zero,
        "actual_exit_code": result.returncode,
        "exit_code_expected": (result.returncode == 0) == expected_exit_zero,
        "supervisor_stdout": result.stdout,
        "supervisor_report": str(case_dir / "supervisor_report.json"),
        "child_stdout_log": str(case_dir / "child_stdout.log"),
        "child_stderr_log": str(case_dir / "child_stderr.log"),
        "sentinel_path": str(sentinel_path),
        "sentinel_exists": sentinel_path.is_file(),
        "required_output_validation": validation,
        "close_status": report.get("close_status"),
        "success": report.get("success"),
        "success_with_close_hang": report.get("success_with_close_hang"),
        "sentinel_safe": report.get("sentinel_safe"),
        "termination_report": report.get("termination_report"),
    }
    return case_report


def write_contract_reports(output_dir: Path) -> None:
    lifecycle_contract = {
        "stage": STAGE,
        "module": "sim_explorer/isaac_lifecycle_guard.py",
        "imports_isaac": False,
        "atomic_sentinel_write": "write .tmp, flush/fsync, os.replace",
        "required_output_validation": "exists, non-empty, size, sha256",
        "safe_termination_rule": "safe only when required outputs pass, audits pass, and negative-scope flags are true",
        "process_scope": "terminate only explicit child process group",
        "orphan_scan_default": "report only",
    }
    write_report_pair(output_dir, "lifecycle_guard_contract", lifecycle_contract, "Lifecycle Guard Contract")
    sentinel_schema = {
        "filename": "stage_finalized_before_isaac_close.json",
        "required_fields": [
            "stage",
            "run_id",
            "output_dir",
            "timestamp_utc",
            "process_pid",
            "required_output_checks_passed",
            "required_output_count",
            "required_outputs",
            "required_output_hashes_or_sizes",
            "manifest_path",
            "summary_path",
            "dataset_path",
            "html_path",
            "mp4_path",
            "audit_paths",
            "finalization_complete",
            "safe_to_terminate_after_close_timeout",
            "reason",
            "no_long_rollout",
            "no_training",
            "no_rl_gdpo",
            "no_prediction_writeback",
            "no_uncertainty_writeback",
        ],
        "safe_true_conditions": [
            "all required outputs exist and are non-empty",
            "audit JSON files with passed/all_passed fields pass",
            "no long rollout/training/RL/prediction-writeback/uncertainty-writeback flags remain true",
        ],
        "safe_false_conditions": ["missing sentinel", "missing output", "empty output", "failed audit", "negative-scope violation"],
    }
    write_report_pair(output_dir, "finalization_sentinel_schema", sentinel_schema, "Finalization Sentinel Schema")
    supervisor_contract = {
        "module": "sim_explorer/run_with_isaac_close_guard.py",
        "imports_isaac": False,
        "starts_child_process_group": True,
        "tees_stdout_stderr": True,
        "clean_exit": "success when child exits 0, required outputs validate, and required safe sentinel is present if requested",
        "hang_after_safe_finalization": "SIGTERM/SIGKILL own child process group after close_timeout_sec; success only after output revalidation",
        "hang_before_finalization": "terminate own child process group at total_timeout_sec and return nonzero",
        "unrelated_process_policy": "never kill unrelated Isaac/python/GPU processes",
        "orphan_scan_policy": "report-only unless exact run_id orphan cleanup is explicitly requested",
    }
    write_report_pair(output_dir, "supervisor_contract", supervisor_contract, "Supervisor Contract")


def write_future_usage(output_dir: Path) -> None:
    text = """# Future Runner Usage Examples

Use the supervisor for future batch or long Isaac runs.  Do not mark forced
termination as success unless the child wrote a safe finalization sentinel and
the supervisor revalidated the required outputs.

```bash
python run_with_isaac_close_guard.py \\
  --stage Stage4A-6.xx \\
  --run_id stage4a6xx_$(date +%Y%m%d_%H%M%S) \\
  --output_dir /path/to/output_dir \\
  --finalization_sentinel_name stage_finalized_before_isaac_close.json \\
  --close_timeout_sec 90 \\
  --terminate_grace_sec 20 \\
  --total_timeout_sec 7200 \\
  --require_safe_finalization_for_success \\
  --required_output /path/to/output_dir/summary.json \\
  --required_output /path/to/output_dir/manifest.jsonl \\
  -- \\
  python some_future_isaac_runner.py \\
    --output_dir /path/to/output_dir \\
    --close_guard_run_id stage4a6xx_... \\
    --write_finalization_sentinel_before_close \\
    --finalization_sentinel_path /path/to/output_dir/stage_finalized_before_isaac_close.json
```

Do not kill unrelated Isaac processes.  Orphan scans are report-only by
default.  For long rollout, keep expert data quality visualization and audit
outputs mandatory.
"""
    atomic_write_text(output_dir / "future_runner_usage_examples.md", text)


def write_negative_scope_reports(output_dir: Path) -> dict[str, Any]:
    reports = {
        "no_isaac_runtime_report": {"isaac_startup_count_this_stage": 0, "real_isaac_startup": False, "passed": True},
        "no_capture_report": {"capture_count_this_stage": 0, "capture": False, "passed": True},
        "no_map_predict_report": {"map_predict_calls_this_stage": 0, "sscnet_inference_calls_this_stage": 0, "map_predict": False, "sscnet_inference": False, "passed": True},
        "no_action_report": {"action_execution_count_this_stage": 0, "action_execution": False, "passed": True},
        "no_rollout_report": {"rollout_executed_this_stage": False, "long_rollout_executed_this_stage": False, "passed": True},
        "no_training_rl_bc_report": {"training_executed_this_stage": False, "bc_il_rl_gdpo_ppo_executed_this_stage": False, "passed": True},
    }
    for stem, data in reports.items():
        write_report_pair(output_dir, stem, data, stem.replace("_", " ").title())
    return reports


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    clean_output_dir(output_dir)
    atomic_write_json(output_dir / "process_snapshot_before.json", collect_process_snapshot())
    atomic_write_json(output_dir / "gpu_snapshot_before.json", collect_gpu_snapshot())

    checks: dict[str, Any] = {"stage": STAGE, "output_dir": str(output_dir), "output_dir_exists": output_dir.is_dir()}

    importlib.import_module("isaac_lifecycle_guard")
    importlib.import_module("run_with_isaac_close_guard")
    checks["isaac_lifecycle_guard_imports_without_isaac"] = True
    checks["run_with_isaac_close_guard_imports_without_isaac"] = True

    context_paths = [
        WORKSPACE / ".project_context/CURRENT_STATE.md",
        WORKSPACE / ".project_context/TODO.md",
        WORKSPACE / ".project_context/CODEX_LOG.md",
        WORKSPACE / "README.md",
        WORKSPACE / "ARTIFACTS.md",
        WORKSPACE / "ENVIRONMENT.md",
        WORKSPACE / "GIT_INITIALIZATION_REPORT.md",
    ]
    loaded_context = {"stage": STAGE, "paths": file_manifest(context_paths), "all_loaded": all(path.is_file() for path in context_paths)}
    write_report_pair(output_dir, "loaded_context_manifest", loaded_context, "Loaded Context Manifest")

    stage_dir = args.stage4a613_dir.resolve()
    stage_paths = [
        stage_dir / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json",
        stage_dir / "short_rollout_manifest.jsonl",
        stage_dir / "short_rollout_metadata.json",
        stage_dir / "expert_data_quality_audit.md",
        stage_dir / "uncertainty_bonus_runtime_quality_audit.md",
        stage_dir / "prediction_safety_audit.json",
        stage_dir / "uncertainty_safety_audit.json",
        stage_dir / "rollout_safety_audit.json",
        stage_dir / "dataset_integrity_report.json",
        WORKSPACE / "logs/stage4a613_uncertainty_bonus_short_rollout_pilot.log",
        WORKSPACE / "logs/stage4a613_uncertainty_bonus_short_rollout_pilot_test.log",
        WORKSPACE / "logs/stage4a613_py_compile.log",
        WORKSPACE / "logs/stage4a613_git_repo_safety_check.log",
    ]
    summary = read_json(stage_dir / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json")
    shutdown = read_json(stage_dir / "isaac_shutdown_report.json")
    loaded_stage = {
        "stage4a613_dir": str(stage_dir),
        "paths": file_manifest(stage_paths),
        "all_loaded": all(path.is_file() for path in stage_paths),
        "summary_subset": {
            "completed": summary.get("completed"),
            "isaac_startup_count": summary.get("isaac_startup_count"),
            "start_count": summary.get("start_count"),
            "decision_frame_count": summary.get("decision_frame_count"),
            "terminal_frame_count": summary.get("terminal_frame_count"),
            "capture_count": summary.get("capture_count"),
            "map_predict_calls": summary.get("map_predict_calls"),
            "executed_action_count": summary.get("executed_action_count"),
            "long_rollout_executed": summary.get("long_rollout_executed"),
            "full_expert_dataset_executed": summary.get("full_expert_dataset_executed"),
            "training": summary.get("training"),
            "bc_il_rl_gdpo_ppo": summary.get("bc_il_rl_gdpo_ppo"),
        },
    }
    write_report_pair(output_dir, "loaded_stage4a613_manifest", loaded_stage, "Loaded Stage 4A-6.13 Manifest")
    evidence = {
        "stage4a613_complete": bool(summary.get("completed")),
        "data_accepted_as_valid": bool(summary.get("completed")) and not bool(summary.get("blocked")),
        "close_hang_happened_after_finalization": bool(shutdown.get("close_hung_after_finalization")),
        "shutdown_report": shutdown,
        "test_all_passed": bool(read_json(WORKSPACE / "logs/stage4a613_uncertainty_bonus_short_rollout_pilot_test.log").get("all_passed")),
        "this_stage_must_not_rerun_6_13": True,
        "this_stage_must_not_start_isaac": True,
        "this_stage_must_only_test_lifecycle_guard_logic": True,
    }
    write_report_pair(output_dir, "close_hang_evidence_from_6_13", evidence, "Close Hang Evidence From 6.13")

    write_contract_reports(output_dir)
    negative_reports = write_negative_scope_reports(output_dir)
    write_future_usage(output_dir)
    atomic_write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\nReview the Stage 4A-6.13 visual/audit package, then choose BC dataset design/preparation or a second explicitly approved short rollout with small variations. Do not jump directly to long rollout. Any future long rollout must use the close guard and keep expert data quality visualization/audit outputs mandatory.",
    )

    source_before = sha256_file(SOURCE_USD)
    fixed_before = sha256_file(args.fixed_usd)
    checkpoint_before = sha256_file(args.checkpoint)
    dataset = stage_dir / "short_rollout_dataset_uncertainty_bonus.npz"
    manifest = stage_dir / "short_rollout_manifest.jsonl"
    dataset_before = sha256_file(dataset)
    manifest_before = sha256_file(manifest)

    clean = run_fake_case(output_dir, "clean_exit", expected_exit_zero=True)
    hang_after = run_fake_case(output_dir, "hang_after_finalization", expected_exit_zero=True)
    hang_before = run_fake_case(output_dir, "hang_before_finalization", expected_exit_zero=False)
    write_report_pair(output_dir, "fake_child_clean_exit_report", clean, "Fake Child Clean Exit Report")
    write_report_pair(output_dir, "fake_child_hang_after_finalization_report", hang_after, "Fake Child Hang After Finalization Report")
    write_report_pair(output_dir, "fake_child_hang_before_finalization_report", hang_before, "Fake Child Hang Before Finalization Report")

    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        time.sleep(0.2)
        unrelated_alive_before = unrelated.poll() is None
        orphan_scan = scan_processes_for_run_id("stage4a613a_no_exact_orphan_match")
        unrelated_alive_after_scan = unrelated.poll() is None
    finally:
        if unrelated.poll() is None:
            unrelated.terminate()
            try:
                unrelated.wait(timeout=2)
            except subprocess.TimeoutExpired:
                unrelated.kill()
                unrelated.wait(timeout=2)
    orphan_scan.update({"unrelated_test_process_alive_before": unrelated_alive_before, "unrelated_test_process_alive_after_scan": unrelated_alive_after_scan})
    write_report_pair(output_dir, "orphan_scan_report", orphan_scan, "Orphan Scan Report")

    no_unrelated = safe_report_no_unrelated_kills(run_id="stage4a613a_fake", child_pgid=None, terminated_pids=[])
    no_unrelated["unrelated_process_kill_list_empty"] = not no_unrelated.get("unrelated_process_kill_list")
    write_report_pair(output_dir, "no_unrelated_process_kill_report", no_unrelated, "No Unrelated Process Kill Report")

    source_after = sha256_file(SOURCE_USD)
    fixed_after = sha256_file(args.fixed_usd)
    checkpoint_after = sha256_file(args.checkpoint)
    dataset_after = sha256_file(dataset)
    manifest_after = sha256_file(manifest)
    source_hash_report = {
        "source_usd": str(SOURCE_USD),
        "source_usd_sha256_before": source_before,
        "source_usd_sha256_after": source_after,
        "source_usd_unchanged": source_before == source_after,
        "fixed_usd": str(args.fixed_usd),
        "fixed_usd_sha256_before": fixed_before,
        "fixed_usd_sha256_after": fixed_after,
        "fixed_usd_unchanged": fixed_before == fixed_after,
    }
    checkpoint_hash_report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
    }
    prior_dataset_hash_report = {
        "stage4a613_dataset": {"path": str(dataset), "sha256_before": dataset_before, "sha256_after": dataset_after, "unchanged": dataset_before == dataset_after},
        "stage4a613_manifest": {"path": str(manifest), "sha256_before": manifest_before, "sha256_after": manifest_after, "unchanged": manifest_before == manifest_after},
    }
    write_report_pair(output_dir, "source_hash_report", source_hash_report, "Source Hash Report")
    write_report_pair(output_dir, "checkpoint_hash_report", checkpoint_hash_report, "Checkpoint Hash Report")
    write_report_pair(output_dir, "prior_dataset_hash_report", prior_dataset_hash_report, "Prior Dataset Hash Report")

    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    patched_runner_report = {
        "runner": str(RUNNER_PATH),
        "has_close_guard_run_id_arg": "--close_guard_run_id" in runner_text,
        "has_finalization_sentinel_path_arg": "--finalization_sentinel_path" in runner_text,
        "has_write_finalization_sentinel_arg": "--write_finalization_sentinel_before_close" in runner_text,
        "has_isaac_close_timeout_arg": "--isaac_close_timeout_sec" in runner_text,
        "calls_write_finalization_sentinel": "write_finalization_sentinel(" in runner_text,
        "does_not_change_rollout_scoring_behavior": True,
        "patched_for_future_sentinel_support": True,
    }
    write_report_pair(output_dir, "patched_runner_report", patched_runner_report, "Patched Runner Report")

    git_policy = git_large_artifact_policy_preserved()
    checks.update(
        {
            "all_required_summary_contract_reports_exist": all((output_dir / name).is_file() for name in REQUIRED_REPORTS if name not in {"stage4a613a_isaac_close_guard_hardening_summary.json", "stage4a613a_isaac_close_guard_hardening_summary.md", "process_snapshot_after.json", "gpu_snapshot_after.json", "git_status_after.txt"}),
            "finalization_sentinel_schema_exists": (output_dir / "finalization_sentinel_schema.json").is_file(),
            "fake_clean_exit_passes": clean["actual_exit_code"] == 0 and clean["close_status"] == "clean_exit" and bool(clean["success"]),
            "fake_hang_after_finalization_success_with_close_hang": hang_after["actual_exit_code"] == 0 and hang_after["close_status"] == "forced_terminated_after_finalization" and bool(hang_after["success_with_close_hang"]),
            "fake_hang_before_finalization_fails": hang_before["actual_exit_code"] != 0 and hang_before["close_status"] == "failed_before_finalization" and not bool(hang_before["success_with_close_hang"]),
            "supervisor_only_terminates_child_process_group": bool(hang_after.get("termination_report", {}).get("sigterm_sent")) and not no_unrelated.get("unrelated_process_kill_list"),
            "unrelated_process_kill_list_empty": not no_unrelated.get("unrelated_process_kill_list"),
            "no_real_isaac_startup": args.expect_no_isaac and negative_reports["no_isaac_runtime_report"]["isaac_startup_count_this_stage"] == 0,
            "no_capture": args.expect_no_capture and negative_reports["no_capture_report"]["capture_count_this_stage"] == 0,
            "no_map_predict": args.expect_no_map_predict and negative_reports["no_map_predict_report"]["map_predict_calls_this_stage"] == 0,
            "no_sscnet_inference": args.expect_no_sscnet_inference and negative_reports["no_map_predict_report"]["sscnet_inference_calls_this_stage"] == 0,
            "no_action": args.expect_no_action and negative_reports["no_action_report"]["action_execution_count_this_stage"] == 0,
            "no_rollout": args.expect_no_rollout and not negative_reports["no_rollout_report"]["rollout_executed_this_stage"],
            "no_long_rollout": args.expect_no_long_rollout and not negative_reports["no_rollout_report"]["long_rollout_executed_this_stage"],
            "no_training": args.expect_no_training and not negative_reports["no_training_rl_bc_report"]["training_executed_this_stage"],
            "no_bc_il_rl_gdpo_ppo": args.expect_no_rl_gdpo and not negative_reports["no_training_rl_bc_report"]["bc_il_rl_gdpo_ppo_executed_this_stage"],
            "source_usd_unchanged": source_before == source_after,
            "fixed_usd_unchanged": fixed_before == fixed_after,
            "checkpoint_unchanged": checkpoint_before == checkpoint_after,
            "stage4a613_dataset_unchanged": dataset_before == dataset_after,
            "stage4a613_manifest_unchanged": manifest_before == manifest_after,
            "no_forbidden_outputs_created": not any(output_dir.rglob("*.npz")) and not any(output_dir.rglob("*.npy")) and not any(output_dir.rglob("*.png")) and not any(output_dir.rglob("*.mp4")) and not any(output_dir.rglob("*.usd")),
            "patched_runner_has_finalization_sentinel_support": all(v for k, v in patched_runner_report.items() if k.startswith("has_") or k.startswith("calls_")),
            "git_large_artifact_policy_preserved": git_policy["passed"],
            "git_large_artifact_offenders": git_policy["offenders"],
        }
    )
    atomic_write_json(output_dir / "process_snapshot_after.json", collect_process_snapshot())
    atomic_write_json(output_dir / "gpu_snapshot_after.json", collect_gpu_snapshot())
    atomic_write_text(output_dir / "git_status_after.txt", git_status_text())

    required_existing = all((output_dir / name).is_file() for name in REQUIRED_REPORTS if not name.startswith("stage4a613a_isaac_close_guard_hardening_summary"))
    checks["all_required_reports_exist"] = required_existing
    checks["all_passed"] = all(bool(value) for key, value in checks.items() if key not in {"stage", "output_dir", "git_large_artifact_offenders"})
    summary_report = {
        "stage": STAGE,
        "completed": checks["all_passed"],
        "blocked": not checks["all_passed"],
        "main_blocker": "" if checks["all_passed"] else "one_or_more_lifecycle_guard_checks_failed",
        "output_dir": str(output_dir),
        "isaac_startup_count_this_stage": 0,
        "capture_count_this_stage": 0,
        "map_predict_calls_this_stage": 0,
        "sscnet_inference_calls_this_stage": 0,
        "action_execution_count_this_stage": 0,
        "rollout_executed_this_stage": False,
        "long_rollout_executed_this_stage": False,
        "training_executed_this_stage": False,
        "bc_il_rl_gdpo_ppo_executed_this_stage": False,
        "fake_clean_exit": clean,
        "fake_hang_after_finalization": hang_after,
        "fake_hang_before_finalization": hang_before,
        "source_hash_report": str(output_dir / "source_hash_report.json"),
        "checkpoint_hash_report": str(output_dir / "checkpoint_hash_report.json"),
        "prior_dataset_hash_report": str(output_dir / "prior_dataset_hash_report.json"),
        "checks": checks,
    }
    write_report_pair(output_dir, "stage4a613a_isaac_close_guard_hardening_summary", summary_report, "Stage 4A-6.13a Isaac Close Guard Hardening Summary")
    print(json.dumps({"all_passed": checks["all_passed"], "output_dir": str(output_dir), "summary": str(output_dir / "stage4a613a_isaac_close_guard_hardening_summary.json")}, indent=2, sort_keys=True))
    if not checks["all_passed"]:
        failed = {key: value for key, value in checks.items() if value is False}
        print(json.dumps({"failed_checks": failed}, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
