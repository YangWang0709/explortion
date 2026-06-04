#!/usr/bin/env python3
"""Inspect external active_3d_planning source for tree utility evidence.

This is a static source-inspection helper for Stage 4A-6.5g. It reads local
SC-Explorer dependency/config files, optionally shallow-clones the external
active_3d_planning repository, searches the external source, and writes a
structured evidence report. It intentionally does not build ROS packages,
launch nodes, start Isaac, run map_predict, train, or execute planners.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TARGET_SYMBOLS = [
    "RRTStar",
    "RRTStarEvaluatorAdapter",
    "SegmentTime",
    "GlobalNormalizedGain",
    "SubsequentBest",
    "ContinuousYawPlanningEvaluator",
    "TrajectorySegment",
    "TrajectoryEvaluator",
    "TrajectoryGenerator",
    "ValueComputer",
    "CostComputer",
    "NextSelector",
]

KEYWORDS = TARGET_SYMBOLS + [
    "active_3d_planning",
    "mav_active_3d_planning",
    "rrt_star",
    "trajectory_generator",
    "trajectory_evaluator",
    "value_computer",
    "cost_computer",
    "next_selector",
    "gain",
    "value",
    "utility",
    "cost",
    "segment",
    "accumulate",
    "accumulated",
    "parent",
    "children",
    "branch",
    "root",
    "tree",
    "node",
    "select",
    "selectNext",
    "best",
    "bestNode",
    "bestBranch",
    "next_best",
    "yaw",
    "path",
    "trajectory",
    "visible_voxels",
    "computeGain",
    "computeCost",
    "computeValue",
]

SOURCE_SUFFIXES = {
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".yaml",
    ".yml",
    ".xml",
    ".md",
    ".txt",
    ".rosinstall",
}

SPECIAL_NAMES = {"CMakeLists.txt", "package.xml", "README", "README.md"}

LOCAL_SC_PATTERNS = [
    "SSCExplorationEvaluator",
    "SSCVoxbloxOccupancyMap",
    "computeGainFromVisibleVoxels",
    "getVoxelType",
    "getVoxelSSCState",
    "use_ssc_planning",
    "use_ssc_information_planning",
    "use_voxblox_planning",
    "use_voxblox_information_planning",
    "predicted_occ_weight",
    "predicted_free_weight",
    "unobserved_weight",
    "IterativeRayCaster",
]


@dataclass
class RosinstallRepo:
    local_name: str
    uri: str
    version: str | None
    source_file: str
    active3d_relevant: bool


@dataclass
class PackageDependency:
    package_file: str
    package_name: str
    tag: str
    dependency: str


@dataclass
class SourceHit:
    source_scope: str
    repo_name: str
    file_path: str
    line_number: int
    keyword: str
    line_text: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_lines(path: Path) -> list[str]:
    text = read_text(path)
    if not text:
        return []
    return text.splitlines()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def is_source_file(path: Path) -> bool:
    return path.name in SPECIAL_NAMES or path.name.startswith("README") or path.suffix in SOURCE_SUFFIXES


def iter_source_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if ".git" in path.parts:
                continue
            if is_source_file(path):
                yield path


def parse_rosinstall(repo_root: Path) -> list[RosinstallRepo]:
    rosinstall = repo_root / ".rosinstall"
    repos: list[RosinstallRepo] = []
    current: dict[str, str] = {}
    for raw_line in read_lines(rosinstall):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- git:"):
            if current:
                repos.append(make_rosinstall_repo(current, rosinstall))
            current = {}
            continue
        match = re.match(r"(local-name|uri|version):\s*(.+)", line)
        if match:
            current[match.group(1)] = match.group(2).strip().strip("'\"")
    if current:
        repos.append(make_rosinstall_repo(current, rosinstall))
    return repos


def make_rosinstall_repo(current: dict[str, str], rosinstall: Path) -> RosinstallRepo:
    local_name = current.get("local-name", "")
    uri = current.get("uri", "")
    haystack = f"{local_name} {uri}".lower()
    active = "active_3d_planning" in haystack or "mav_active_3d_planning" in haystack
    return RosinstallRepo(
        local_name=local_name,
        uri=uri,
        version=current.get("version"),
        source_file=rosinstall.as_posix(),
        active3d_relevant=active,
    )


def parse_package_xmls(repo_root: Path) -> list[PackageDependency]:
    results: list[PackageDependency] = []
    tags = [
        "depend",
        "build_depend",
        "buildtool_depend",
        "exec_depend",
        "run_depend",
        "test_depend",
    ]
    for package_xml in sorted(repo_root.rglob("package.xml")):
        text = read_text(package_xml)
        name_match = re.search(r"<name>\s*([^<]+)\s*</name>", text)
        package_name = name_match.group(1).strip() if name_match else ""
        for tag in tags:
            for dep in re.findall(rf"<{tag}(?:\s+[^>]*)?>\s*([^<]+)\s*</{tag}>", text):
                results.append(
                    PackageDependency(
                        package_file=package_xml.as_posix(),
                        package_name=package_name,
                        tag=tag,
                        dependency=dep.strip(),
                    )
                )
    return results


def collect_config_references(repo_root: Path) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = []
    patterns = TARGET_SYMBOLS + ["trajectory_generator", "trajectory_evaluator", "cost_computer", "value_computer", "next_selector"]
    for path in iter_source_files([repo_root]):
        rel = safe_rel(path, repo_root)
        if not (path.suffix in {".yaml", ".yml", ".xml", ".launch", ".rosinstall"} or path.name in {"package.xml", "README.md"}):
            continue
        for line_number, line in enumerate(read_lines(path), 1):
            if any(pattern in line for pattern in patterns):
                refs.append(
                    {
                        "file_path": rel,
                        "line_number": line_number,
                        "line_text": line.strip(),
                    }
                )
    return refs


def normalize_clone_url(uri: str) -> str:
    if uri.startswith("git@github.com:"):
        return "https://github.com/" + uri.split(":", 1)[1]
    if uri.startswith("ssh://git@github.com/"):
        return "https://github.com/" + uri.split("ssh://git@github.com/", 1)[1]
    return uri


def run_command(args: list[str], cwd: Path | None = None, timeout: int = 120) -> dict[str, object]:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": args,
            "returncode": None,
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }
    except FileNotFoundError as exc:
        return {
            "command": args,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
        }


def git_commit(path: Path) -> str:
    if not (path / ".git").exists():
        return ""
    result = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=20)
    if result["returncode"] == 0:
        return str(result["stdout"]).strip()
    return ""


def find_packages(source_root: Path) -> list[str]:
    packages: list[str] = []
    for package_xml in sorted(source_root.rglob("package.xml")):
        text = read_text(package_xml)
        match = re.search(r"<name>\s*([^<]+)\s*</name>", text)
        if match:
            packages.append(match.group(1).strip())
    return packages


def inspect_or_clone_repositories(
    repos: list[RosinstallRepo],
    repo_root: Path,
    external_root: Path,
    clone_missing: bool,
) -> tuple[list[dict[str, object]], dict[str, object], list[Path]]:
    external_root.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    clone_status: dict[str, object] = {}
    active_source_roots: list[Path] = []
    local_scan_root = repo_root.parent

    for repo in repos:
        target = external_root / repo.local_name
        status = "not_cloned_out_of_scope"
        clone_result: dict[str, object] | None = None

        discovered_paths = []
        for candidate in [target, local_scan_root / repo.local_name]:
            if candidate.exists():
                discovered_paths.append(candidate)

        if repo.active3d_relevant:
            if target.exists() and (target / ".git").exists():
                status = "found_existing"
            elif target.exists() and not (target / ".git").exists():
                status = "exists_non_git_or_partial"
            elif clone_missing:
                clone_url = normalize_clone_url(repo.uri)
                clone_result = run_command(
                    ["git", "clone", "--depth", "1", clone_url, str(target)],
                    cwd=external_root,
                    timeout=180,
                )
                if clone_result["returncode"] == 0:
                    status = "cloned"
                else:
                    status = "clone_failed"
            else:
                status = "missing_clone_not_requested"

            if target.exists() and target not in discovered_paths:
                discovered_paths.insert(0, target)
            for path in discovered_paths:
                if (path / ".git").exists() and path not in active_source_roots:
                    active_source_roots.append(path)

        local_path = ""
        commit = ""
        packages: list[str] = []
        if discovered_paths:
            local_path = discovered_paths[0].as_posix()
            commit = git_commit(discovered_paths[0])
            packages = find_packages(discovered_paths[0])
        elif target.exists():
            local_path = target.as_posix()
            commit = git_commit(target)
            packages = find_packages(target)

        row = {
            "repo_name": repo.local_name,
            "url": repo.uri,
            "clone_url_used": normalize_clone_url(repo.uri),
            "active3d_relevant": repo.active3d_relevant,
            "local_path": local_path,
            "clone_status": status,
            "commit_hash": commit,
            "packages_found": packages,
        }
        inventory.append(row)
        clone_status[repo.local_name] = {
            **row,
            "clone_result": clone_result,
        }

    return inventory, clone_status, active_source_roots


def collect_hits(source_roots: list[Path], workspace_root: Path, scope: str) -> list[SourceHit]:
    hits: list[SourceHit] = []
    lowered_keywords = [(keyword, keyword.lower()) for keyword in KEYWORDS]
    for source_root in source_roots:
        repo_name = source_root.name
        for path in iter_source_files([source_root]):
            lines = read_lines(path)
            if not lines:
                continue
            file_path = safe_rel(path, workspace_root)
            for line_number, line in enumerate(lines, 1):
                lower = line.lower()
                for keyword, keyword_lower in lowered_keywords:
                    if keyword_lower in lower:
                        hits.append(
                            SourceHit(
                                source_scope=scope,
                                repo_name=repo_name,
                                file_path=file_path,
                                line_number=line_number,
                                keyword=keyword,
                                line_text=line.strip()[:700],
                            )
                        )
    return hits


def collect_local_sc_hits(repo_root: Path, workspace_root: Path) -> list[SourceHit]:
    hits: list[SourceHit] = []
    for path in iter_source_files([repo_root]):
        if not any(part in path.parts for part in ["ssc_planning", "ssc_mapping", "ssc_msgs"]):
            continue
        file_path = safe_rel(path, workspace_root)
        for line_number, line in enumerate(read_lines(path), 1):
            for pattern in LOCAL_SC_PATTERNS:
                if pattern in line:
                    hits.append(
                        SourceHit(
                            source_scope="local_sc_explorer",
                            repo_name=repo_root.name,
                            file_path=file_path,
                            line_number=line_number,
                            keyword=pattern,
                            line_text=line.strip()[:700],
                        )
                    )
    return hits


def context_symbol(lines: list[str], index: int) -> str:
    class_re = re.compile(r"\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_:]*)")
    func_re = re.compile(r"^\s*(?:[A-Za-z_][\w:<>,&*\s~]*\s+)?([A-Za-z_][\w:~]*)\s*\([^;]*\)\s*(?:const\s*)?(?:override\s*)?\{?\s*$")
    for j in range(index, max(-1, index - 80), -1):
        text = lines[j].strip()
        match = class_re.search(text)
        if match:
            return match.group(2)
        match = func_re.search(text)
        if match and not text.startswith("if ") and not text.startswith("for ") and not text.startswith("while "):
            return match.group(1)
    return ""


def build_symbol_index(source_roots: list[Path], workspace_root: Path) -> dict[str, list[dict[str, object]]]:
    index: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in TARGET_SYMBOLS}
    patterns = {symbol: re.compile(rf"\b{re.escape(symbol)}\b") for symbol in TARGET_SYMBOLS}
    for path in iter_source_files(source_roots):
        lines = read_lines(path)
        if not lines:
            continue
        file_path = safe_rel(path, workspace_root)
        for line_number, line in enumerate(lines, 1):
            for symbol, pattern in patterns.items():
                if pattern.search(line):
                    index[symbol].append(
                        {
                            "file_path": file_path,
                            "line_number": line_number,
                            "function_or_class": context_symbol(lines, line_number - 1),
                            "line_text": line.strip()[:700],
                        }
                    )
    return index


def find_first_file(source_roots: list[Path], suffix: str) -> Path | None:
    matches = []
    for root in source_roots:
        matches.extend(root.rglob(suffix))
    return sorted(matches)[0] if matches else None


def evidence_line(path: Path | None, line_number: int, workspace_root: Path) -> dict[str, object]:
    if path is None:
        return {"file_path": "", "line_number": 0, "line_text": ""}
    lines = read_lines(path)
    text = lines[line_number - 1].strip() if 0 < line_number <= len(lines) else ""
    return {
        "file_path": safe_rel(path, workspace_root),
        "line_number": line_number,
        "line_text": text,
    }


def collect_named_evidence(source_roots: list[Path], repo_root: Path, workspace_root: Path) -> dict[str, object]:
    files = {
        "trajectory_segment_h": find_first_file(source_roots, "trajectory_segment.h"),
        "trajectory_segment_cpp": find_first_file(source_roots, "trajectory_segment.cpp"),
        "rrt_star_h": find_first_file(source_roots, "rrt_star.h"),
        "rrt_star_cpp": find_first_file(source_roots, "rrt_star.cpp"),
        "rrt_cpp": find_first_file(source_roots, "rrt.cpp"),
        "segment_time_h": find_first_file(source_roots, "segment_time.h"),
        "segment_time_cpp": find_first_file(source_roots, "segment_time.cpp"),
        "global_normalized_gain_h": find_first_file(source_roots, "global_normalized_gain.h"),
        "global_normalized_gain_cpp": find_first_file(source_roots, "global_normalized_gain.cpp"),
        "subsequent_best_h": find_first_file(source_roots, "subsequent_best.h"),
        "subsequent_best_cpp": find_first_file(source_roots, "subsequent_best.cpp"),
        "continuous_yaw_h": find_first_file(source_roots, "continuous_yaw_planning_evaluator.h"),
        "continuous_yaw_cpp": find_first_file(source_roots, "continuous_yaw_planning_evaluator.cpp"),
        "yaw_planning_h": find_first_file(source_roots, "yaw_planning_evaluator.h"),
        "yaw_planning_cpp": find_first_file(source_roots, "yaw_planning_evaluator.cpp"),
        "simulated_sensor_h": find_first_file(source_roots, "simulated_sensor_evaluator.h"),
        "simulated_sensor_cpp": find_first_file(source_roots, "simulated_sensor_evaluator.cpp"),
        "online_planner_cpp": find_first_file(source_roots, "online_planner.cpp"),
    }

    local_files = {
        "ssc_exploration_evaluator_cpp": repo_root / "ssc_planning/src/trajectory_evaluator/ssc_exploration_evaluator.cpp",
        "ssc_exploration_evaluator_h": repo_root / "ssc_planning/include/ssc_planning/trajectory_evaluator/ssc_exploration_evaluator.h",
        "ssc_voxblox_map_cpp": repo_root / "ssc_planning/src/map/ssc_voxblox_map.cpp",
        "ssc_voxblox_map_h": repo_root / "ssc_planning/include/ssc_planning/map/ssc_voxblox_map.h",
        "baseline_yaml": repo_root / "ssc_planning/cfg/planners/baseline.yaml",
        "sc_explorer_yaml": repo_root / "ssc_planning/cfg/planners/sc_explorer.yaml",
    }

    line_evidence = {
        "trajectory_segment_fields": [
            evidence_line(files["trajectory_segment_h"], n, workspace_root) for n in [21, 25, 26, 27, 30, 34, 37, 40]
        ],
        "trajectory_segment_tree_methods": [
            evidence_line(files["trajectory_segment_cpp"], n, workspace_root) for n in [19, 22, 32, 42, 59, 62, 63, 64, 66]
        ],
        "rrtstar_structure": [
            evidence_line(files["rrt_star_h"], n, workspace_root) for n in [20, 21, 22, 29, 32, 35, 37, 73, 85, 92, 94, 96, 98]
        ],
        "rrtstar_expansion_rewire": [
            evidence_line(files["rrt_star_cpp"], n, workspace_root) for n in [109, 125, 127, 130, 133, 151, 211, 221, 240, 254, 277, 335, 353, 354, 356, 371, 388, 392]
        ],
        "rrt_sampling_connect": [
            evidence_line(files["rrt_cpp"], n, workspace_root) for n in [103, 122, 136, 233, 247, 267, 274, 281, 310]
        ],
        "segment_time": [
            evidence_line(files["segment_time_h"], n, workspace_root) for n in [9, 22]
        ]
        + [evidence_line(files["segment_time_cpp"], n, workspace_root) for n in [12, 13, 16, 21, 22, 23, 24, 25, 26]],
        "global_normalized_gain": [
            evidence_line(files["global_normalized_gain_h"], n, workspace_root) for n in [10, 16, 24]
        ]
        + [evidence_line(files["global_normalized_gain_cpp"], n, workspace_root) for n in [17, 18, 20, 21, 23, 24, 27, 31, 33, 36, 37, 38, 39, 41, 42]],
        "subsequent_best": [
            evidence_line(files["subsequent_best_h"], n, workspace_root) for n in [9, 15, 23]
        ]
        + [evidence_line(files["subsequent_best_cpp"], n, workspace_root) for n in [17, 18, 19, 20, 21, 22, 30, 31, 35, 36, 37, 38, 40, 41, 43]],
        "rrtstar_evaluator_adapter": [
            evidence_line(files["rrt_star_cpp"], n, workspace_root) for n in [408, 416, 420, 424, 428, 429, 430, 443, 444, 451]
        ],
        "continuous_yaw": [
            evidence_line(files["continuous_yaw_h"], n, workspace_root) for n in [12, 14, 22, 43, 45, 48]
        ]
        + [evidence_line(files["continuous_yaw_cpp"], n, workspace_root) for n in [26, 33, 36, 38, 40, 44, 51, 58, 70, 82, 85, 86, 88, 90, 91, 102, 105, 106]],
        "yaw_planning_base": [
            evidence_line(files["yaw_planning_cpp"], n, workspace_root) for n in [48, 57, 58, 61, 62, 63, 64, 68, 73, 103, 104, 111, 115, 119]
        ],
        "simulated_sensor_flow": [
            evidence_line(files["simulated_sensor_cpp"], n, workspace_root) for n in [45, 47, 59, 78, 79, 80, 84, 87, 88]
        ],
        "online_planner_selection": [
            evidence_line(files["online_planner_cpp"], n, workspace_root) for n in [245, 292, 293, 294, 296, 297, 298, 309, 310, 314, 332, 333, 339, 413, 415, 423, 425, 433, 435]
        ],
        "local_sc_gain": [
            evidence_line(local_files["ssc_exploration_evaluator_cpp"], n, workspace_root) for n in [15, 17, 21, 25, 26, 34, 46, 48, 55, 57, 65, 69, 74, 76, 81, 82, 84, 87]
        ],
        "local_sc_map": [
            evidence_line(local_files["ssc_voxblox_map_cpp"], n, workspace_root) for n in [55, 56, 58, 60, 171, 175, 185, 196, 210, 216, 249, 253, 265, 273, 281, 287, 289]
        ],
        "local_config": [
            evidence_line(local_files["baseline_yaml"], n, workspace_root) for n in [19, 31, 33, 38, 43, 53, 54, 57, 58, 78, 79, 81, 82, 84, 85]
        ]
        + [evidence_line(local_files["sc_explorer_yaml"], n, workspace_root) for n in [1, 17, 18, 20, 21, 22, 24, 27, 28, 29, 30, 32, 33]],
    }
    file_index = {
        name: safe_rel(path, workspace_root) if path else "" for name, path in {**files, **local_files}.items()
    }
    return {"files": file_index, "line_evidence": line_evidence}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evidence_refs(lines: list[dict[str, object]], limit: int = 8) -> list[str]:
    refs = []
    for item in lines[:limit]:
        if not item.get("file_path"):
            continue
        refs.append(f"{item['file_path']}:{item['line_number']} `{item['line_text']}`")
    return refs


def build_summary(
    inventory: list[dict[str, object]],
    dependency_json: dict[str, object],
    symbol_index: dict[str, list[dict[str, object]]],
    formula_evidence: dict[str, object],
    output_dir: Path,
    external_missing: bool,
) -> dict[str, object]:
    files = formula_evidence["files"]
    ev = formula_evidence["line_evidence"]
    active_repos = [row for row in inventory if row.get("active3d_relevant")]
    found_repos = [row for row in inventory if row.get("local_path")]
    failed_repos = [row for row in inventory if row.get("active3d_relevant") and row.get("clone_status") == "clone_failed"]
    packages = []
    for row in found_repos:
        packages.extend(row.get("packages_found") or [])

    high_confidence = not external_missing and bool(files.get("rrt_star_cpp")) and bool(files.get("global_normalized_gain_cpp"))

    return {
        "stage": "Stage 4A-6.5g",
        "generated_at_utc": utc_now(),
        "context_read": {
            "CURRENT_STATE.md": True,
            "CODEX_LOG.md": True,
            "TODO.md": True,
            "current_stage_confirmed": "Stage 4A-6.5g external active_3d_planning source inspection",
            "hard_boundaries_confirmed": True,
        },
        "external_source_inventory": {
            "rosinstall_dependency_count": len(dependency_json.get("rosinstall_repositories", [])),
            "active3d_repositories": active_repos,
            "found_or_cloned_repositories": found_repos,
            "failed_repositories": failed_repos,
            "packages_found": sorted(set(packages)),
            "confidence": "high" if high_confidence else "low",
        },
        "answers": {
            "rrtstar_tree_generator": {
                "files": [files.get("rrt_star_h", ""), files.get("rrt_star_cpp", ""), files.get("rrt_cpp", "")],
                "classes_functions": ["trajectory_generator::RRTStar", "RRT::selectSegment", "RRTStar::expandSegment", "RRTStar::rewireToBestParent", "RRTStar::rewireRoot"],
                "node_tree_representation": "TrajectorySegment is both tree node and edge/segment: it stores trajectory points plus gain/cost/value, parent, children, and optional info.",
                "interpretation": "RRTStar samples points through RRT, creates a new segment, computes local gain at the new point/segment, tries candidate parents, evaluates cost/value for feasible connections, selects the parent with maximum value, and can rewire existing subtrees.",
                "evidence": evidence_refs(ev["rrtstar_structure"] + ev["rrtstar_expansion_rewire"], 14),
                "confidence": "high" if files.get("rrt_star_cpp") else "missing",
            },
            "trajectory_segment": {
                "file": files.get("trajectory_segment_h", ""),
                "fields": ["trajectory", "gain", "cost", "value", "tg_visited", "parent", "children", "info"],
                "local_accumulated_data": "The struct stores one gain/cost/value per segment. Values can represent accumulated/tree utility depending on the configured ValueComputer; GlobalNormalizedGain writes a subtree-best accumulated gain/cost ratio into value.",
                "interpretation": "TrajectorySegment is the tree node object and contains the edge trajectory from its parent to this node; children encode branches.",
                "evidence": evidence_refs(ev["trajectory_segment_fields"] + ev["trajectory_segment_tree_methods"], 12),
                "confidence": "high" if files.get("trajectory_segment_h") else "missing",
            },
            "segment_time_cost": {
                "file": files.get("segment_time_cpp", ""),
                "formula": "cost = (trajectory.back().time_from_start_ns - trajectory.front().time_from_start_ns) * 1e-9; if accumulate=true and a parent exists, add parent->cost.",
                "distance_yaw_time_handling": "RRT::connectPoses assigns time_from_start from sampling_rate and v_max along a straight-line segment. SegmentTime reads those timestamps; it does not directly include yaw-rate, yaw-acceleration, collision, or acceleration terms.",
                "local_vs_accumulated": "Default parameter is accumulate=false, so SC-Explorer baseline config uses edge-local segment time unless overridden.",
                "evidence": evidence_refs(ev["segment_time"] + ev["rrt_sampling_connect"], 12),
                "confidence": "high" if files.get("segment_time_cpp") else "missing",
            },
            "global_normalized_gain": {
                "file": files.get("global_normalized_gain_cpp", ""),
                "formula": "For a segment, sum ancestor gain/cost, then recursively compute max over current subtree of accumulated_gain / accumulated_cost.",
                "normalization": "Normalized by accumulated cost. No discount, exponent, lambda, or separate normalization parameter appears in this class.",
                "local_vs_accumulated": "Accumulated over the root-to-descendant path, including ancestors and descendants; the stored segment value is the best ratio available in its subtree.",
                "path_tree_behavior": "This is branch/path utility, not one-step gain/cost only.",
                "evidence": evidence_refs(ev["global_normalized_gain"], 14),
                "confidence": "high" if files.get("global_normalized_gain_cpp") else "missing",
            },
            "rrtstar_evaluator_adapter": {
                "file": files.get("rrt_star_cpp", ""),
                "role": "Delegates gain/cost/value/update/visualization to the configured following evaluator, then wraps selectNextBest with RRTStar::rewireRoot.",
                "accumulation_behavior": "It does not itself accumulate gain/cost; accumulation comes from the following evaluator/value computer, specifically GlobalNormalizedGain in this config.",
                "tree_behavior": "On selection, it lets the next selector choose an immediate child, then calls RRTStar::rewireRoot so non-selected root children may survive under the selected new root.",
                "evidence": evidence_refs(ev["rrtstar_evaluator_adapter"] + ev["rrtstar_expansion_rewire"], 12),
                "confidence": "high" if files.get("rrt_star_cpp") else "missing",
            },
            "subsequent_best": {
                "file": files.get("subsequent_best_cpp", ""),
                "best_node_branch_logic": "For each immediate child of the root/current segment, evaluateSingle recursively finds the highest value anywhere in that child's subtree. selectNextBest returns the child index whose subtree contains the highest-value segment, randomizing ties.",
                "replanning_selection_interpretation": "OnlinePlanner then moves current_segment_ to that returned child and publishes that child trajectory. This executes the first/subsequent segment of the best subtree/path, not the far best leaf directly.",
                "path_cost_locality_implication": "This can avoid immediate-neighbor-only selection when a child has lower local value but leads to a high-value descendant.",
                "evidence": evidence_refs(ev["subsequent_best"] + ev["online_planner_selection"], 16),
                "confidence": "high" if files.get("subsequent_best_cpp") else "missing",
            },
            "continuous_yaw": {
                "file": files.get("continuous_yaw_cpp", ""),
                "yaw_handling": "YawPlanningEvaluator samples multiple yaw orientations; ContinuousYawPlanningEvaluator combines adjacent FOV sections, picks the max gain section, sets all trajectory points to the selected yaw, then recomputes cost and value.",
                "yaw_cost": "Yaw can affect raycasting/gain because each orientation is evaluated by the following evaluator. SegmentTime does not directly price yaw motion.",
                "relation_to_gain": "The selected segment gain is the sum of visible section gains for the best yaw window.",
                "evidence": evidence_refs(ev["continuous_yaw"] + ev["yaw_planning_base"], 16),
                "confidence": "high" if files.get("continuous_yaw_cpp") else "missing",
            },
            "sc_gain_integration": {
                "ssc_gain_enters_planner_as": "SSCExplorationEvaluator computes traj_in->gain from visible voxels and voxel type weights. That local segment gain is consumed by GlobalNormalizedGain through TrajectorySegment::gain.",
                "prediction_map_use": "The map checks measured ESDF first for observed voxels, then SSC predicted occupancy/free/unknown through getVoxelSSCState where configured.",
                "measured_prediction_separation": "Measured ESDF and SSC map are separate servers/layers. In sc_explorer.yaml, use_voxblox_information_planning=true and use_ssc_information_planning=false for ray blocking/information map state, while SSCExplorationEvaluator still queries getVoxelSSCState for gain classification of unobserved measured voxels.",
                "collision_ray_blocking_relation": "Collision/traversability can fall back to SSC because use_ssc_planning=true. Ray/information blocking from getVoxelState uses measured voxblox only because use_ssc_information_planning=false in sc_explorer.yaml.",
                "evidence": evidence_refs(ev["local_sc_gain"] + ev["local_sc_map"] + ev["local_config"] + ev["simulated_sensor_flow"], 18),
                "confidence": "high",
            },
            "difference_from_current_simulator_expert": {
                "current_simplification": "Current simulator expert uses one-step reachable_frontier candidates with local gain and A* path_cost; it selects top-1 after a local score.",
                "missing_external_pieces": ["real RRT/RRT* expansion tree", "TrajectorySegment parent/children tree", "GlobalNormalizedGain accumulated subtree best value", "SubsequentBest first-child-of-best-subtree selection", "RRTStar rewiring/root preservation", "continuous yaw section planning"],
                "likely_cause": "Yes. Collapsing the external planner to one-step gain/path_cost removes the subtree-best normalized branch utility and subsequent-child selection, so path-cost/locality dominance can be an artifact of that collapse.",
                "what_to_reproduce_next": "Reproduce GlobalNormalizedGain and SubsequentBest offline over saved candidate sets/tree-like candidate expansions before any rollout.",
                "confidence": "high" if high_confidence else "medium",
            },
        },
        "recommended_next_faithful_step": {
            "choice": "A. offline minimal tree-utility prototype over saved candidates",
            "why": "The external source clearly shows accumulated branch/tree utility plus SubsequentBest immediate-child selection. The smallest faithful next step is to reproduce those formulas offline on saved candidates without Isaac, rollout, map_predict, or planner implementation.",
            "no_rollout": True,
            "no_rl": True,
        },
        "safety": {
            "isaac_startup": "no",
            "rollout": "no",
            "map_predict_rerun": "no",
            "sscnet_training": "no",
            "rl_ppo_bc_il": "no",
            "checkpoint_modified": "no",
            "observed_state_modified": "no",
            "prediction_writeback": "no",
            "target_lr_target_hr_ground_truth_scoring": "no",
            "local_source_modified_by_script": "no",
            "external_source_built": "no",
            "external_source_modified": "no",
        },
        "output_dir": output_dir.as_posix(),
        "missing_or_ambiguous": [
            "No runtime execution was performed, so dynamic ROS parameter overrides were not observed.",
            "The exact sampled RRT tree for the paper experiments is not reproduced; this report is source evidence only.",
        ]
        if not external_missing
        else ["External active_3d_planning source was not available; formulas could not be verified."],
    }


def write_formula_markdown(path: Path, summary: dict[str, object]) -> None:
    answers = summary["answers"]
    sections = [
        ("SegmentTime Cost", answers["segment_time_cost"]),
        ("GlobalNormalizedGain Utility", answers["global_normalized_gain"]),
        ("RRTStarEvaluatorAdapter", answers["rrtstar_evaluator_adapter"]),
        ("SubsequentBest Next Selector", answers["subsequent_best"]),
        ("ContinuousYawPlanningEvaluator", answers["continuous_yaw"]),
        ("SC-Specific Gain Integration", answers["sc_gain_integration"]),
    ]
    lines = ["# Stage 4A-6.5g External Utility Formula Evidence", ""]
    for title, data in sections:
        lines.extend([f"## {title}", ""])
        for key, value in data.items():
            if key == "evidence":
                continue
            lines.append(f"- {key}: {value}")
        lines.append("- evidence:")
        for ref in data.get("evidence", [])[:18]:
            lines.append(f"  - {ref}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tree_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    answers = summary["answers"]
    inv = summary["external_source_inventory"]
    safety = summary["safety"]
    lines = [
        "# Stage 4A-6.5g External active_3d_planning Source Inspection",
        "",
        "## Context Read",
        "- Read CURRENT_STATE.md / CODEX_LOG.md / TODO.md before inspection.",
        f"- Current stage confirmed: {summary['context_read']['current_stage_confirmed']}.",
        "- Hard boundaries confirmed: no Isaac, rollout, map_predict rerun, training, RL/IL, or planner implementation.",
        "",
        "## External Source Inventory",
        f"- .rosinstall dependencies: {inv['rosinstall_dependency_count']}.",
        f"- fetched/found repos: {[row['repo_name'] for row in inv['found_or_cloned_repositories']]}.",
        f"- failed repos: {[row['repo_name'] for row in inv['failed_repositories']]}.",
        f"- packages found: {inv['packages_found']}.",
        f"- confidence: {inv['confidence']}.",
        "",
        "## RRTStar / Tree Generator",
        f"- files: {answers['rrtstar_tree_generator']['files']}",
        f"- classes/functions: {answers['rrtstar_tree_generator']['classes_functions']}",
        f"- tree/node/segment representation: {answers['rrtstar_tree_generator']['node_tree_representation']}",
        f"- interpretation: {answers['rrtstar_tree_generator']['interpretation']}",
        "- evidence lines:",
    ]
    for ref in answers["rrtstar_tree_generator"]["evidence"]:
        lines.append(f"  - {ref}")
    lines.extend(
        [
            "",
            "## TrajectorySegment Structure",
            f"- file: {answers['trajectory_segment']['file']}",
            f"- fields: {answers['trajectory_segment']['fields']}",
            f"- local/accumulated data: {answers['trajectory_segment']['local_accumulated_data']}",
            f"- interpretation: {answers['trajectory_segment']['interpretation']}",
            "- evidence lines:",
        ]
    )
    for ref in answers["trajectory_segment"]["evidence"]:
        lines.append(f"  - {ref}")
    lines.extend(
        [
            "",
            "## SegmentTime Cost",
            f"- file: {answers['segment_time_cost']['file']}",
            f"- formula: {answers['segment_time_cost']['formula']}",
            f"- distance/yaw/time handling: {answers['segment_time_cost']['distance_yaw_time_handling']}",
            f"- local vs accumulated: {answers['segment_time_cost']['local_vs_accumulated']}",
            "- evidence lines:",
        ]
    )
    for ref in answers["segment_time_cost"]["evidence"]:
        lines.append(f"  - {ref}")
    lines.extend(
        [
            "",
            "## GlobalNormalizedGain Utility",
            f"- file: {answers['global_normalized_gain']['file']}",
            f"- formula: {answers['global_normalized_gain']['formula']}",
            f"- normalization: {answers['global_normalized_gain']['normalization']}",
            f"- local vs accumulated: {answers['global_normalized_gain']['local_vs_accumulated']}",
            f"- path/tree behavior: {answers['global_normalized_gain']['path_tree_behavior']}",
            "- evidence lines:",
        ]
    )
    for ref in answers["global_normalized_gain"]["evidence"]:
        lines.append(f"  - {ref}")
    lines.extend(
        [
            "",
            "## RRTStarEvaluatorAdapter",
            f"- role: {answers['rrtstar_evaluator_adapter']['role']}",
            f"- accumulation behavior: {answers['rrtstar_evaluator_adapter']['accumulation_behavior']}",
            f"- tree evaluation behavior: {answers['rrtstar_evaluator_adapter']['tree_behavior']}",
            "- evidence lines:",
        ]
    )
    for ref in answers["rrtstar_evaluator_adapter"]["evidence"]:
        lines.append(f"  - {ref}")
    lines.extend(
        [
            "",
            "## SubsequentBest Next Selector",
            f"- selection logic: {answers['subsequent_best']['best_node_branch_logic']}",
            f"- best node vs best branch vs subsequent segment: {answers['subsequent_best']['replanning_selection_interpretation']}",
            f"- locality interpretation: {answers['subsequent_best']['path_cost_locality_implication']}",
            "- evidence lines:",
        ]
    )
    for ref in answers["subsequent_best"]["evidence"]:
        lines.append(f"  - {ref}")
    lines.extend(
        [
            "",
            "## ContinuousYawPlanningEvaluator",
            f"- yaw handling: {answers['continuous_yaw']['yaw_handling']}",
            f"- relation to gain/raycast/cost: {answers['continuous_yaw']['yaw_cost']}",
            f"- selected gain: {answers['continuous_yaw']['relation_to_gain']}",
            "- evidence lines:",
        ]
    )
    for ref in answers["continuous_yaw"]["evidence"]:
        lines.append(f"  - {ref}")
    lines.extend(
        [
            "",
            "## SC-Specific Gain Integration",
            f"- how SSC gain enters planner: {answers['sc_gain_integration']['ssc_gain_enters_planner_as']}",
            f"- SC prediction map use: {answers['sc_gain_integration']['prediction_map_use']}",
            f"- measured/prediction separation: {answers['sc_gain_integration']['measured_prediction_separation']}",
            f"- collision/ray-blocking configuration relation: {answers['sc_gain_integration']['collision_ray_blocking_relation']}",
            "- evidence lines:",
        ]
    )
    for ref in answers["sc_gain_integration"]["evidence"]:
        lines.append(f"  - {ref}")
    lines.extend(
        [
            "",
            "## Difference From Current Simulator Expert",
            f"- current simplification: {answers['difference_from_current_simulator_expert']['current_simplification']}",
            f"- missing external pieces: {answers['difference_from_current_simulator_expert']['missing_external_pieces']}",
            f"- likely cause of path-cost/locality issue: {answers['difference_from_current_simulator_expert']['likely_cause']}",
            f"- what needs to be reproduced next: {answers['difference_from_current_simulator_expert']['what_to_reproduce_next']}",
            "",
            "## Recommended Next Faithful Step",
            f"- next small task: {summary['recommended_next_faithful_step']['choice']}",
            f"- why: {summary['recommended_next_faithful_step']['why']}",
            "",
            "## Safety / Boundary Check",
        ]
    )
    for key, value in safety.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_missing_report(path: Path, summary: dict[str, object], external_missing: bool) -> None:
    lines = ["# Missing or Ambiguous External Items", ""]
    if external_missing:
        lines.append("- External active_3d_planning source was not available. Formula evidence is missing.")
    for item in summary.get("missing_or_ambiguous", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "- This is source evidence only; no ROS runtime parameter dump was produced.",
            "- This does not claim a full reproduction of the paper planner.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommended_next(path: Path, summary: dict[str, object]) -> None:
    rec = summary["recommended_next_faithful_step"]
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {rec['choice']}",
        f"- why: {rec['why']}",
        "- scope: offline only; use saved candidates/tree-like candidate sets.",
        "- include: GlobalNormalizedGain accumulated path ratio and SubsequentBest child-of-best-subtree selection.",
        "- do not include: Isaac startup, rollout, map_predict rerun, RL/IL, planner implementation, or source modification.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", required=True, type=Path)
    parser.add_argument("--external_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--clone_missing", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    workspace_root = repo_root.parent.resolve()
    external_root = args.external_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repos = parse_rosinstall(repo_root)
    package_deps = parse_package_xmls(repo_root)
    config_refs = collect_config_references(repo_root)

    inventory, clone_status, active_source_roots = inspect_or_clone_repositories(
        repos, repo_root, external_root, args.clone_missing
    )
    external_missing = len(active_source_roots) == 0

    dependency_json = {
        "generated_at_utc": utc_now(),
        "repo_root": repo_root.as_posix(),
        "rosinstall_path": (repo_root / ".rosinstall").as_posix(),
        "rosinstall_repositories": [asdict(repo) for repo in repos],
        "package_xml_dependencies": [asdict(dep) for dep in package_deps],
        "config_references": config_refs,
        "active3d_package_dependencies": [
            asdict(dep)
            for dep in package_deps
            if "active_3d_planning" in dep.dependency or dep.dependency in {"voxblox", "voxblox_ros"}
        ],
    }

    (output_dir / "external_dependency_urls.json").write_text(
        json.dumps(dependency_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "external_clone_status.json").write_text(
        json.dumps(clone_status, indent=2, sort_keys=True), encoding="utf-8"
    )

    write_csv(
        output_dir / "external_source_inventory.csv",
        inventory,
        [
            "repo_name",
            "url",
            "clone_url_used",
            "active3d_relevant",
            "local_path",
            "clone_status",
            "commit_hash",
            "packages_found",
        ],
    )

    external_hits = collect_hits(active_source_roots, workspace_root, "external_active_3d_planning")
    local_hits = collect_local_sc_hits(repo_root, workspace_root)
    all_hits = external_hits + local_hits
    write_csv(
        output_dir / "external_source_hits.csv",
        [asdict(hit) for hit in all_hits],
        ["source_scope", "repo_name", "file_path", "line_number", "keyword", "line_text"],
    )

    symbol_index = build_symbol_index(active_source_roots, workspace_root)
    (output_dir / "planner_symbol_index.json").write_text(
        json.dumps(symbol_index, indent=2, sort_keys=True), encoding="utf-8"
    )

    formula_evidence = collect_named_evidence(active_source_roots, repo_root, workspace_root)
    summary = build_summary(
        inventory,
        dependency_json,
        symbol_index,
        formula_evidence,
        output_dir,
        external_missing,
    )

    formula_json = {
        "generated_at_utc": utc_now(),
        "files": formula_evidence["files"],
        "line_evidence": formula_evidence["line_evidence"],
        "formula_answers": {
            key: summary["answers"][key]
            for key in [
                "segment_time_cost",
                "global_normalized_gain",
                "rrtstar_evaluator_adapter",
                "subsequent_best",
                "continuous_yaw",
                "sc_gain_integration",
            ]
        },
    }
    (output_dir / "external_utility_formula_evidence.json").write_text(
        json.dumps(formula_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_formula_markdown(output_dir / "external_utility_formula_evidence.md", summary)

    (output_dir / "external_tree_utility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tree_summary_markdown(output_dir / "external_tree_utility_summary.md", summary)
    write_missing_report(output_dir / "missing_or_ambiguous_external_items.md", summary, external_missing)
    write_recommended_next(output_dir / "recommended_next_faithful_step.md", summary)

    if external_missing:
        (output_dir / "external_source_missing_report.md").write_text(
            "# External Source Missing Report\n\n"
            "- No active_3d_planning source root was found or cloned.\n"
            "- See external_clone_status.json for clone errors.\n",
            encoding="utf-8",
        )

    # A small manifest helps verify no hidden build/run outputs were created.
    manifest = {
        "generated_at_utc": utc_now(),
        "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "active_source_roots": [path.as_posix() for path in active_source_roots],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "external_source_built": False,
        "external_source_modified": False,
        "local_source_modified_by_script": False,
        "isaac_startup": False,
        "rollout": False,
        "map_predict_rerun": False,
        "training_or_rl": False,
    }
    (output_dir / "inspection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps({"ok": True, "output_dir": output_dir.as_posix(), "external_missing": external_missing}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
