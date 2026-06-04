#!/usr/bin/env python3
"""Fake Isaac child for lifecycle guard tests.

This helper deliberately does not import Isaac.  It only writes lightweight
files and optionally sleeps forever to simulate ``simulation_app.close()``
hanging after or before output finalization.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from isaac_lifecycle_guard import write_finalization_sentinel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["clean_exit", "hang_after_finalization", "hang_before_finalization"], required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--finalization_sentinel_path", type=Path, required=True)
    return parser.parse_args()


def _write_required_outputs(output_dir: Path, mode: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [output_dir / "fake_required_output.txt", output_dir / "fake_summary.json"]
    required[0].write_text(f"fake required output for {mode}\n", encoding="utf-8")
    required[1].write_text(json.dumps({"mode": mode, "completed": True}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return required


def main() -> int:
    args = parse_args()
    print(f"fake child start mode={args.mode} pid={os.getpid()} run_id={args.run_id}", flush=True)
    if args.mode == "hang_before_finalization":
        while True:
            time.sleep(1)

    required = _write_required_outputs(args.output_dir, args.mode)
    write_finalization_sentinel(
        args.finalization_sentinel_path,
        stage="Fake Isaac Lifecycle Child",
        run_id=args.run_id,
        output_dir=args.output_dir,
        required_output_paths=required,
        manifest_path=required[0],
        summary_path=required[1],
        audit_paths=[],
        reason=f"fake_{args.mode}_outputs_finalized",
        no_long_rollout=True,
        no_training=True,
        no_rl_gdpo=True,
        no_prediction_writeback=True,
        no_uncertainty_writeback=True,
    )
    print("fake child wrote finalization sentinel", flush=True)
    if args.mode == "hang_after_finalization":
        while True:
            time.sleep(1)
    print("fake child clean exit", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
