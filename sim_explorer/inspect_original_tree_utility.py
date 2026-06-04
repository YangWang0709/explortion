#!/usr/bin/env python3
"""Inspect the original SC-Explorer tree/planner utility evidence.

This is a static source inspection helper. It searches the SC-Explorer repo,
records keyword hits, extracts planner/evaluator evidence, and writes a compact
human-readable audit. It intentionally does not import or execute ROS, Isaac,
map_predict, SSCNet, or planner code.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


KEYWORDS = [
    "RRT",
    "rrt",
    "Tree",
    "tree",
    "Node",
    "node",
    "utility",
    "gain",
    "information gain",
    "cost",
    "score",
    "branch",
    "parent",
    "child",
    "path",
    "bestNode",
    "best_node",
    "next_best_view",
    "NBV",
    "frontier",
    "map_predict",
    "scene completion",
    "completion",
    "hierarchical",
    "SemanticMap",
    "SCLayer",
    "occupied gain",
    "confidence gain",
    "hybrid gain",
    "raycast",
    "ray casting",
    "planner",
    "planning",
    "execute",
    "replan",
    "yaw",
    "pose",
    "viewpoint",
    "viewpoint gain",
    "RRTStar",
    "RRTStarEvaluatorAdapter",
    "GlobalNormalizedGain",
    "SegmentTime",
    "SubsequentBest",
    "ContinuousYawPlanningEvaluator",
    "TrajectorySegment",
    "IterativeRayCaster",
    "SSCExplorationEvaluator",
    "SSCVoxelEvaluator",
    "SSCVoxbloxOccupancyMap",
    "SSCServer",
    "SSCGrid",
]

SOURCE_SUFFIXES = {
    ".cpp",
    ".hpp",
    ".h",
    ".cc",
    ".py",
    ".yaml",
    ".yml",
    ".launch",
    ".xml",
    ".md",
    ".txt",
    ".rosinstall",
}

SPECIAL_NAMES = {"CMakeLists.txt", "package.xml", "README", ".rosinstall"}

OFFICIAL_IMPORTANT_FILES = [
    "README.md",
    ".rosinstall",
    "ssc_planning/CMakeLists.txt",
    "ssc_planning/package.xml",
    "ssc_planning/launch/run.launch",
    "ssc_planning/cfg/planners/baseline.yaml",
    "ssc_planning/cfg/planners/sc_explorer.yaml",
    "ssc_planning/cfg/planners/exploration.yaml",
    "ssc_planning/src/planner/exploration_planner_node.cpp",
    "ssc_planning/include/ssc_planning/trajectory_evaluator/ssc_exploration_evaluator.h",
    "ssc_planning/src/trajectory_evaluator/ssc_exploration_evaluator.cpp",
    "ssc_planning/include/ssc_planning/trajectory_evaluator/ssc_voxel_evaluator.h",
    "ssc_planning/src/trajectory_evaluator/ssc_voxel_evaluator.cpp",
    "ssc_planning/include/ssc_planning/map/ssc_voxblox_map.h",
    "ssc_planning/src/map/ssc_voxblox_map.cpp",
    "ssc_mapping/include/ssc_mapping/ros/ssc_server.h",
    "ssc_mapping/src/ros/ssc_server.cpp",
    "ssc_mapping/include/ssc_mapping/core/ssc_map.h",
    "ssc_mapping/src/core/ssc_map.cpp",
    "ssc_network/src/ssc_network_node.py",
    "ssc_msgs/msg/SSCGrid.msg",
]

EVIDENCE_PATTERNS = [
    ("planner_structure", "external dependency", r"mav_active_3d_planning|active_3d_planning_core|active_3d_planning_ros|active_3d_planning_voxblox"),
    ("planner_structure", "planner executable", r"exploration_planner_node|RosPlanner|planningLoop|cs_add_executable"),
    ("planner_structure", "launch", r"planner_config_file|planner_node|rosparam.*planners|run\.launch"),
    ("tree", "rrtstar", r'RRTStar|RRTStarEvaluatorAdapter|rewire_|n_neighbors|max_extension_range|max_rewire_range'),
    ("tree", "tree internals external", r"TrajectorySegment|ModuleFactoryRegistry"),
    ("gain", "ssc evaluator", r"SSCExplorationEvaluator|computeGainFromVisibleVoxels|getVoxelType|traj_in->gain"),
    ("gain", "gain weights", r"predicted_occ_weight|predicted_free_weight|unobserved_weight|weight_by_confidence|c_gains_"),
    ("gain", "raycasting", r"IterativeRayCaster|visible_voxels|sensor_model|ray_length|focal_length"),
    ("gain", "prediction map", r"getSSCServer|getSSCMapPtr|getSSCLayer|getVoxelSSCState|probability_log|SSCGrid|sscCallback|base_fusion_->fuse"),
    ("gain", "measured/prediction separation", r"getESDFServer|isObserved|use_voxblox_information_planning|use_ssc_information_planning|use_voxblox_planning|use_ssc_planning"),
    ("cost", "cost module", r"SegmentTime|v_max|yaw_rate_max|yaw_accel_max|system_constraints|cost_computer"),
    ("cost", "collision/traversability", r"isTraversable|getDistanceAtPosition|collision_radius|recheck_collisions|crop_segments"),
    ("utility", "value module", r"GlobalNormalizedGain|value_computer|next_selector|SubsequentBest"),
    ("utility", "replan", r"replan_pos_threshold|replan_yaw_threshold|back_tracker|RotateReverse"),
    ("network_connection", "network publish", r"SSCGrid|ssc_pub|publish|scene_completion|infer|Softmax"),
]


@dataclass
class LineHit:
    file_path: str
    line_number: int
    keyword: str
    line_text: str
    git_state: str


@dataclass
class Evidence:
    category: str
    label: str
    file_path: str
    line_number: int
    function_or_class: str
    evidence_text: str
    git_state: str
    source_or_inference: str = "source"


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_source_file(path: Path) -> bool:
    if path.name in SPECIAL_NAMES or path.name.startswith("README"):
        return True
    return path.suffix in SOURCE_SUFFIXES


def run_git(repo_root: Path, args: list[str]) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return completed.stdout.splitlines()


def git_file_states(repo_root: Path) -> tuple[set[str], dict[str, str]]:
    tracked = set(run_git(repo_root, ["ls-files"]))
    states: dict[str, str] = {path: "tracked" for path in tracked}
    for line in run_git(repo_root, ["status", "--short"]):
        if not line:
            continue
        state = line[:2]
        raw_path = line[3:].strip()
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        if state == "??":
            states[raw_path] = "untracked-local"
        elif raw_path:
            states[raw_path] = "modified-tracked"
    return tracked, states


def iter_files(repo_root: Path) -> Iterable[Path]:
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if is_source_file(path):
            yield path


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def collect_hits(repo_root: Path, git_states: dict[str, str]) -> tuple[list[LineHit], dict[str, Counter]]:
    hits: list[LineHit] = []
    per_file: dict[str, Counter] = defaultdict(Counter)
    lowered_keywords = [(keyword, keyword.lower()) for keyword in KEYWORDS]
    for path in iter_files(repo_root):
        rel = relpath(path, repo_root)
        state = git_states.get(rel, "not-in-git")
        for line_number, line in enumerate(read_lines(path), 1):
            lower = line.lower()
            for keyword, keyword_lower in lowered_keywords:
                if keyword_lower in lower:
                    hit = LineHit(
                        file_path=rel,
                        line_number=line_number,
                        keyword=keyword,
                        line_text=line.strip()[:600],
                        git_state=state,
                    )
                    hits.append(hit)
                    per_file[rel][keyword] += 1
    return hits, per_file


def context_symbol(lines: list[str], index: int) -> str:
    class_re = re.compile(r"\b(class|struct)\s+([A-Za-z_][A-Za-z0-9_:]*)")
    py_re = re.compile(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
    cpp_func_re = re.compile(
        r"^\s*(?:[A-Za-z_][\w:<>,&*\s~]*\s+)?([A-Za-z_][\w:~]*)\s*\([^;]*\)\s*(?:const\s*)?(?:override\s*)?\{?\s*$"
    )
    for j in range(index, max(-1, index - 80), -1):
        text = lines[j].strip()
        for regex in (class_re, py_re):
            match = regex.search(text)
            if match:
                return match.group(2)
        if "{" in text or ")" in text:
            match = cpp_func_re.search(text)
            if match:
                return match.group(1)
    return ""


def collect_evidence(repo_root: Path, git_states: dict[str, str]) -> list[Evidence]:
    evidence: list[Evidence] = []
    compiled = [
        (category, label, re.compile(pattern, re.IGNORECASE))
        for category, label, pattern in EVIDENCE_PATTERNS
    ]
    for path in iter_files(repo_root):
        rel = relpath(path, repo_root)
        lines = read_lines(path)
        state = git_states.get(rel, "not-in-git")
        for i, line in enumerate(lines):
            for category, label, regex in compiled:
                if regex.search(line):
                    evidence.append(
                        Evidence(
                            category=category,
                            label=label,
                            file_path=rel,
                            line_number=i + 1,
                            function_or_class=context_symbol(lines, i),
                            evidence_text=line.strip()[:600],
                            git_state=state,
                        )
                    )
    return evidence


def score_candidate(rel: str, counter: Counter, git_state: str, evidence: list[Evidence]) -> tuple[int, list[str]]:
    text = " ".join(counter.keys()).lower()
    reasons: list[str] = []
    score = 0

    weight_terms = {
        "RRTStar": 30,
        "RRTStarEvaluatorAdapter": 30,
        "GlobalNormalizedGain": 24,
        "SegmentTime": 18,
        "SubsequentBest": 18,
        "ContinuousYawPlanningEvaluator": 15,
        "SSCExplorationEvaluator": 24,
        "SSCVoxelEvaluator": 16,
        "SSCVoxbloxOccupancyMap": 18,
        "RosPlanner": 22,
        "planner": 8,
        "planning": 6,
        "gain": 5,
        "cost": 5,
        "raycast": 5,
        "replan": 5,
        "SSCServer": 10,
        "SSCGrid": 8,
    }
    for term, weight in weight_terms.items():
        count = counter.get(term, 0)
        if count:
            score += weight * min(count, 4)
            reasons.append(f"{term} hits={count}")

    labels = [item.label for item in evidence if item.file_path == rel]
    if labels:
        label_counts = Counter(labels)
        score += 8 * len(label_counts)
        reasons.extend([f"{label} evidence={count}" for label, count in label_counts.items()])

    if rel in OFFICIAL_IMPORTANT_FILES:
        score += 25
        reasons.append("known official planner/evaluator/config path")
    if git_state == "untracked-local":
        score -= 60
        reasons.append("untracked local file; not treated as original evidence")
    if "ssc_network/il/" in rel or rel.endswith("sc_explorer_paper_expert.py"):
        score -= 80
        reasons.append("workspace-generated expert/IL file; excluded from original planner conclusions")
    if any(token in text for token in ("rrt", "planner", "gain", "cost")):
        score += 5
    return score, reasons


def select_candidates(
    per_file: dict[str, Counter], git_states: dict[str, str], evidence: list[Evidence]
) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    excluded_local: list[dict] = []
    for rel, counter in per_file.items():
        state = git_states.get(rel, "not-in-git")
        score, reasons = score_candidate(rel, counter, state, evidence)
        if score <= 0:
            continue
        item = {
            "file_path": rel,
            "score": score,
            "git_state": state,
            "reasons": reasons[:8],
        }
        if state == "untracked-local" or "workspace-generated" in " ".join(reasons):
            excluded_local.append(item)
        else:
            candidates.append(item)
    candidates.sort(key=lambda item: (-item["score"], item["file_path"]))
    excluded_local.sort(key=lambda item: (-item["score"], item["file_path"]))
    return candidates[:30], excluded_local[:20]


def evidence_dict(item: Evidence) -> dict:
    return {
        "category": item.category,
        "label": item.label,
        "file_path": item.file_path,
        "line_number": item.line_number,
        "function_or_class": item.function_or_class,
        "evidence_text": item.evidence_text,
        "git_state": item.git_state,
        "source_or_inference": item.source_or_inference,
    }


def by_file_line(evidence: list[Evidence], labels: set[str] | None = None) -> list[Evidence]:
    selected = [item for item in evidence if labels is None or item.label in labels]
    selected.sort(key=lambda item: (item.file_path, item.line_number, item.label))
    return selected


def original_source_evidence(evidence: list[Evidence]) -> list[Evidence]:
    return [item for item in evidence if item.git_state != "untracked-local"]


def fmt_evidence(item: Evidence) -> str:
    context = f" ({item.function_or_class})" if item.function_or_class else ""
    state = "" if item.git_state == "tracked" else f" [{item.git_state}]"
    return f"`{item.file_path}:{item.line_number}`{context}{state}: {item.evidence_text}"


def first_evidence(
    evidence: list[Evidence],
    label: str,
    limit: int = 6,
    contains: str | None = None,
) -> list[str]:
    selected = original_source_evidence(by_file_line(evidence, {label}))
    if contains is not None:
        selected = [item for item in selected if contains in item.file_path]
    return [fmt_evidence(item) for item in selected[:limit]]


def write_hits_csv(path: Path, hits: list[LineHit]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_path", "line_number", "keyword", "line_text", "git_state"],
        )
        writer.writeheader()
        for hit in hits:
            writer.writerow(
                {
                    "file_path": hit.file_path,
                    "line_number": hit.line_number,
                    "keyword": hit.keyword,
                    "line_text": hit.line_text,
                    "git_state": hit.git_state,
                }
            )


def write_candidates(path: Path, candidates: list[dict], excluded_local: list[dict]) -> None:
    lines = [
        "# Planner File Candidates",
        "",
        "Tracked/official-source candidates are ranked first. Untracked local files are listed separately so previous workspace additions do not get mistaken for original SC-Explorer planner code.",
        "",
        "## Tracked Candidates",
        "",
    ]
    if candidates:
        for item in candidates:
            lines.append(f"- {item['file_path']} (score={item['score']}, git_state={item['git_state']})")
            lines.append(f"  reason: {'; '.join(item['reasons'])}")
    else:
        lines.append("- No tracked planner candidate files found.")
    lines.extend(["", "## Excluded Local / Non-Original Hits", ""])
    if excluded_local:
        for item in excluded_local:
            lines.append(f"- {item['file_path']} (score={item['score']}, git_state={item['git_state']})")
            lines.append(f"  reason: {'; '.join(item['reasons'])}")
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_utility_evidence(path_json: Path, path_md: Path, evidence: list[Evidence]) -> None:
    utility_labels = {
        "rrtstar",
        "cost module",
        "value module",
        "replan",
        "gain weights",
        "ssc evaluator",
        "prediction map",
        "raycasting",
        "measured/prediction separation",
        "collision/traversability",
        "tree internals external",
    }
    selected = original_source_evidence(by_file_line(evidence, utility_labels))
    path_json.write_text(
        json.dumps([evidence_dict(item) for item in selected], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Utility Formula Evidence",
        "",
        "## Source Evidence",
        "",
        "The local `ssc_exploration` repo configures an external RRT* planner stack and provides SC-specific map/evaluator plugins. The exact implementations of `RRTStar`, `RRTStarEvaluatorAdapter`, `SegmentTime`, `GlobalNormalizedGain`, and `SubsequentBest` are not present in this repo.",
        "",
        "Untracked local workspace files are intentionally excluded from this evidence file; they remain visible in `source_file_hits.csv` and `planner_file_candidates.txt` under excluded local hits.",
        "",
    ]
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in selected:
        grouped[item.category].append(item)
    for category in sorted(grouped):
        lines.append(f"## {category.replace('_', ' ').title()}")
        for item in grouped[category][:40]:
            lines.append(f"- {fmt_evidence(item)}")
        if len(grouped[category]) > 40:
            lines.append(f"- ... {len(grouped[category]) - 40} more evidence lines omitted here; see JSON.")
        lines.append("")
    lines.extend(
        [
            "## Inference Boundary",
            "",
            "- Source evidence confirms `type: \"RRTStar\"`, `RRTStarEvaluatorAdapter`, `SegmentTime`, `GlobalNormalizedGain`, and `SubsequentBest` are selected by config.",
            "- Source evidence does not include the underlying node/tree data structures or exact value-computer formula because those live in the external `mav_active_3d_planning` / `active_3d_planning_*` packages.",
            "- Any statement about accumulated branch utility is therefore an inference from module names/config, not a proved implementation detail from this repo.",
        ]
    )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(candidates: list[dict], evidence: list[Evidence], hit_count: int) -> dict:
    summary = {
        "stage": "Stage 4A-6.5f Original SC-Explorer RRT/Tree Utility Inspection",
        "context_read": {
            "CURRENT_STATE": "read before running inspection",
            "CODEX_LOG": "read before running inspection",
            "TODO": "read before running inspection",
            "current_stage": "Stage 4A-6.5f",
            "recent_completed_stage": "Stage 4A-6.5e",
            "recent_key_conclusion": "Full-64 runtime candidates contain farther alternatives, but saved top-N and selected actions stay local/path-cost dominated.",
            "next_small_task": "original SC-Explorer RRT/tree utility source-code inspection",
        },
        "hard_boundaries": {
            "isaac_startup": False,
            "rollout": False,
            "expert_step": False,
            "map_predict_rerun": False,
            "sscnet_inference": False,
            "training_rl_bc_il": False,
            "checkpoint_modified": False,
            "observed_state_modified": False,
            "prediction_writeback": False,
            "target_or_ground_truth_scoring": False,
            "original_source_modified": False,
        },
        "search_summary": {
            "keyword_hit_count": hit_count,
            "candidate_count": len(candidates),
            "candidate_files": [item["file_path"] for item in candidates],
        },
        "original_planner_files": {
            "has_rrt_tree_planner": "configured yes; implementation external to local ssc_exploration repo",
            "language": "C++ ROS planner integration plus YAML config; network bridge is Python",
            "key_files": [
                "ssc_planning/cfg/planners/baseline.yaml",
                "ssc_planning/cfg/planners/sc_explorer.yaml",
                "ssc_planning/src/planner/exploration_planner_node.cpp",
                "ssc_planning/src/trajectory_evaluator/ssc_exploration_evaluator.cpp",
                "ssc_planning/src/map/ssc_voxblox_map.cpp",
                "ssc_mapping/src/ros/ssc_server.cpp",
                "ssc_network/src/ssc_network_node.py",
            ],
            "external_missing_packages": [
                "mav_active_3d_planning",
                "active_3d_planning_core",
                "active_3d_planning_ros",
                "active_3d_planning_voxblox",
            ],
            "confidence": "high for SC-Explorer integration/config; low for exact external tree internals because source is absent",
        },
        "node_tree_structure": {
            "local_repo_node_struct_found": False,
            "local_fields_found": ["TrajectorySegment.gain", "TrajectorySegment.info", "visible_voxels", "trajectory.back().position_W"],
            "missing_external_fields": ["parent", "children", "accumulated gain", "accumulated cost", "branch utility", "best path"],
            "local_vs_accumulated": "local SC evaluator writes local trajectory gain; accumulated RRT* path/tree fields are not defined in this repo",
        },
        "gain": {
            "primary_function": "SSCExplorationEvaluator::computeGainFromVisibleVoxels",
            "variants": {
                "exploration_gain": "measured unobserved / unknown weight through evaluator and map state",
                "scene_completion_gain": "predicted occupied/free weights via getVoxelSSCState",
                "hybrid_gain": "not a separate explicit hybrid variable in original C++; behavior is weighted sum of predicted occupied/free/unobserved voxel classes",
                "occupied_gain": "predicted occupied weight",
                "confidence_gain": "optional weight_by_confidence scales by 2*abs(confidence-0.5)",
            },
            "raycasting": "IterativeRayCaster configured; visible_voxels consumed by evaluator",
            "prediction_ray_blocking": "config disables SSC information planning for ray blocking in sc_explorer.yaml; measured voxblox information planning stays enabled",
            "prediction_writeback": "no write into ESDF/TSDF measured map found; SSCServer maintains/fuses a separate SSC layer",
        },
        "cost": {
            "function_found_locally": False,
            "configured_module": "SegmentTime",
            "inputs_visible_in_config": ["v_max", "a_max", "yaw_rate_max", "yaw_accel_max"],
            "collision": "RRTStar generator has recheck_collisions/crop_segments; map isTraversable uses measured ESDF then optional SSC fallback",
            "edge_vs_accumulated": "not proven from local source",
        },
        "utility": {
            "exact_formula_found_locally": False,
            "configured_value_computer": "GlobalNormalizedGain",
            "configured_next_selector": "SubsequentBest",
            "local_vs_accumulated": "not proven from local source; likely external RRTStarEvaluatorAdapter/value computer handles tree/path values",
            "best_action_selection": "not proven; selection logic lives in external active_3d_planning modules",
            "replan": "replan_pos_threshold and replan_yaw_threshold configured; logic external",
            "path_cost_dominance_mitigation": "original config likely has path/tree-level normalized value modules, but exact mitigation cannot be confirmed without external source",
        },
        "difference_from_current_simulator_expert": {
            "consistent_parts": [
                "uses visible-voxel gain idea",
                "keeps scene completion as information/planning layer rather than target-label scoring",
                "SC predictions do not block rays in the sc_explorer.yaml configuration",
            ],
            "missing_current_parts": [
                "external RRT*/tree trajectory generator",
                "RRTStarEvaluatorAdapter",
                "SegmentTime cost computer",
                "GlobalNormalizedGain value computer",
                "SubsequentBest next selector",
                "ContinuousYawPlanningEvaluator",
                "tree/path/branch-level selection and replanning semantics",
            ],
            "current_simplification": "one-step candidate gain divided by A* path cost",
            "path_cost_issue": "could plausibly come from collapsing external tree/path utility into local one-step gain/path_cost, but the exact source mechanism is not available in this repo",
        },
        "recommended_next_faithful_step": {
            "choice": "E. source evidence insufficient, inspect more files manually",
            "why": "the local SC-Explorer repo configures RRT* and utility modules but does not include their implementation; inspect/fetch mav_active_3d_planning before implementing any tree utility prototype",
        },
    }
    # Keep a few source line samples close to the structured summary.
    summary["evidence_samples"] = {
        "rrtstar": first_evidence(evidence, "rrtstar", 8),
        "value_module": first_evidence(evidence, "value module", 8),
        "gain": first_evidence(evidence, "ssc evaluator", 8),
        "prediction_map": first_evidence(evidence, "prediction map", 8),
        "cost": first_evidence(evidence, "cost module", 8),
        "replan": first_evidence(evidence, "replan", 8),
    }
    return summary


def write_summary_md(path: Path, summary: dict, evidence: list[Evidence]) -> None:
    candidates = summary["original_planner_files"]["key_files"]
    lines = [
        "# Stage 4A-6.5f Original SC-Explorer RRT/Tree Utility Inspection",
        "",
        "## Context Read",
        "",
        "- 已读取 CURRENT_STATE.md / CODEX_LOG.md / TODO.md.",
        "- 当前阶段确认: Stage 4A-6.5f, original SC-Explorer RRT/tree utility source-code inspection.",
        "- 最近完成阶段: Stage 4A-6.5e; 关键结论是 full-64 runtime candidates 有更远候选，但 saved top-N / selected action 仍明显局部化，并且 selected candidate 基本被 min path_cost / inverse path_cost 主导.",
        "- hard boundaries 确认: no Isaac, no rollout, no map_predict rerun, no training/RL/IL, no checkpoint or observed_state edits, no original planner source edits.",
        "",
        "## Original Planner Files",
        "",
        "- In-repo evidence says the planner is launched as a C++ ROS node, but the concrete RRT*/tree implementation is external to this repo.",
        "- Key tracked files:",
    ]
    lines.extend([f"  - `{item}`" for item in candidates])
    lines.extend(
        [
            "- Relevant classes/functions/modules:",
            "  - `exploration_planner_node.cpp`: creates `active_3d_planning::ros::RosPlanner` and calls `planningLoop()`.",
            "  - `baseline.yaml`: selects `trajectory_generator.type: \"RRTStar\"`, `trajectory_evaluator.type: \"RRTStarEvaluatorAdapter\"`, `cost_computer.type: \"SegmentTime\"`, `value_computer.type: \"GlobalNormalizedGain\"`, and `next_selector.type: \"SubsequentBest\"`.",
            "  - `sc_explorer.yaml`: selects `SSCVoxbloxOccupancyMap` and `SSCExplorationEvaluator` with `IterativeRayCaster`.",
            "  - `ssc_network_node.py` publishes `SSCGrid`; `SSCServer::sscCallback` receives and fuses that into an SSC layer.",
            "- Launch/executable evidence:",
        ]
    )
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "planner executable", 10, contains="ssc_planning")])
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "launch", 8)])
    lines.extend(
        [
            "- Confidence level: high that original config uses an RRT*/tree planner stack; low for exact node/tree internals because `mav_active_3d_planning` / `active_3d_planning_*` source is not present in `ssc_exploration`.",
            "",
            "## Original Node / Tree Structure",
            "",
            "- Local node/tree struct definition: not found in tracked `ssc_exploration` source.",
            "- Local fields visible through `TrajectorySegment`: `gain`, `info`, `visible_voxels`, and trajectory pose access such as `trajectory.back().position_W`.",
            "- Parent/child/path representation: not found locally; likely in external `RRTStar` / active_3d_planning data structures.",
            "- Local vs accumulated gain/cost: `SSCExplorationEvaluator::computeGainFromVisibleVoxels` writes a local `traj_in->gain`; no accumulated gain/cost fields were found locally.",
            "- Subtree / branch utility: not found locally.",
            "- Evidence:",
        ]
    )
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "tree internals external", 10)])
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "ssc evaluator", 10)])
    lines.extend(
        [
            "",
            "## Original Gain Computation",
            "",
            "- Gain function: `SSCExplorationEvaluator::computeGainFromVisibleVoxels`.",
            "- It consumes ray-cast `visible_voxels`; `IterativeRayCaster` is selected in config.",
            "- Exploration / SC behavior: observed measured ESDF voxels contribute zero; unobserved voxels are typed through the SSC map as predicted occupied, predicted free, or unknown, then weighted by `predicted_occ_weight`, `predicted_free_weight`, and `unobserved_weight`.",
            "- `sc_explorer.yaml` sets predicted occupied/free weights to `1.0` and unobserved weight to `0.0`, so its main gain is scene-completion-driven for predicted occupied/free voxels.",
            "- Occupied gain exists as the predicted occupied class weight. Confidence gain exists only as optional `weight_by_confidence`, scaling by confidence distance from 0.5.",
            "- A separate explicit `gain_hybrid` variable was not found in original C++; the hybrid-like behavior is the weighted sum over voxel classes.",
            "- Prediction map use: `SSCServer` subscribes to `SSCGrid` and fuses predictions into an SSC layer; evaluator reads the SSC layer through `getVoxelSSCState`.",
            "- Prediction ray blocking: for `sc_explorer.yaml`, measured voxblox information planning is enabled and SSC information planning is disabled; prediction is used for gain and collision fallback, not ray blocking.",
            "- Measured map vs prediction map separation: measured ESDF/TSDF server and SSC server are separate members in `SSCVoxbloxOccupancyMap`.",
            "- Evidence:",
        ]
    )
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "gain weights", 12, contains="ssc_planning")])
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "raycasting", 12, contains="ssc_planning")])
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "prediction map", 12)])
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "measured/prediction separation", 12, contains="ssc_planning")])
    lines.extend(
        [
            "",
            "## Original Cost Computation",
            "",
            "- Cost implementation: not found in local `ssc_exploration` source.",
            "- Configured cost module: `SegmentTime`.",
            "- Configured system limits include `v_max`, `a_max`, `yaw_rate_max`, and `yaw_accel_max`, so the cost is likely time-based and yaw-aware, but exact formula is external and not proven here.",
            "- Collision/traversability: `RRTStar` config enables collision rechecks/cropping; `SSCVoxbloxOccupancyMap::isTraversable` checks measured ESDF first and then optionally falls back to SSC predicted occupancy.",
            "- Edge vs accumulated path cost: not determined from local source.",
            "- Evidence:",
        ]
    )
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "cost module", 12, contains="ssc_planning")])
    lines.extend([f"  - {item}" for item in first_evidence(evidence, "collision/traversability", 12, contains="ssc_planning")])
    lines.extend(
        [
            "",
            "## Original Utility Formula",
            "",
            "- Exact formula: not found in local `ssc_exploration` source.",
            "- Source-proven configuration: `RRTStarEvaluatorAdapter` wraps `ContinuousYawPlanningEvaluator`, which wraps `SSCExplorationEvaluator`; cost is `SegmentTime`; value is `GlobalNormalizedGain`; next action selector is `SubsequentBest`.",
            "- Local vs accumulated: local SC gain is per trajectory segment; the value computer / RRT* adapter likely handles path/tree normalization externally, but this is inference.",
            "- Node vs branch/tree: tree generator is configured as RRT*, but node/branch utility internals are external.",
            "- Best node / branch / first-path-node selection: not determined. `SubsequentBest` is configured, but its source is absent.",
            "- Replanning: thresholds are configured; RosPlanner loop logic is external.",
            "- Low path-cost dominance mitigation: not directly proven. The original stack may mitigate local low-cost dominance through RRT*/path-level/global normalized value and next selection, but the exact mechanism requires external active_3d_planning source.",
            "- Evidence:",
        ]
    )
    for label in ["rrtstar", "value module", "replan"]:
        lines.extend([f"  - {item}" for item in first_evidence(evidence, label, 14)])
    lines.extend(
        [
            "",
            "## Difference From Current Simulator Expert",
            "",
            "- Current simplification: one-step reachable-frontier candidates, per-candidate visible-voxel gain, A* path cost, and final score close to `gain / path_cost`.",
            "- Consistent with original: uses visible-voxel gain; prediction is read as information; prediction is not ray-blocking in the SC-Explorer config.",
            "- Missing original pieces: external RRT*/tree generator, RRTStarEvaluatorAdapter, SegmentTime, GlobalNormalizedGain, SubsequentBest, ContinuousYawPlanningEvaluator, tree/path/branch selection, and original replanning semantics.",
            "- Our current `gain/path_cost` is a local one-step simplification of a configured RRT*/trajectory value stack, not a faithful reproduction of the original tree utility.",
            "- The current path-cost/locality issue could plausibly be caused by collapsing path/tree evaluation into one local candidate score. Source evidence is insufficient to claim the exact original mitigation.",
            "",
            "## Recommended Next Faithful Step",
            "",
            "- Choice: E. source evidence insufficient, inspect more files manually.",
            "- Why: the repo clearly configures RRT*/tree utility modules, but the actual implementations live in `mav_active_3d_planning` / `active_3d_planning_*`, which are referenced by `.rosinstall` and `package.xml` but not present under this local source tree. The next faithful move is to inspect/fetch those external dependency sources and only then decide whether an offline minimal tree-utility prototype over saved candidates is justified.",
            "",
            "## Safety / Boundary Check",
            "",
            "- Isaac startup: no",
            "- rollout: no",
            "- map_predict rerun: no",
            "- training/RL: no",
            "- checkpoint modified: no",
            "- observed_state modified: no",
            "- source modified: no original `ssc_exploration` source modified by this inspection",
            "- target/ground-truth scoring: no",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_missing(path: Path) -> None:
    lines = [
        "# Missing Or Ambiguous Items",
        "",
        "## Missing Source",
        "",
        "- `mav_active_3d_planning` is referenced in `.rosinstall`, but its source is not present under the inspected `ssc_exploration` tree.",
        "- `active_3d_planning_core`, `active_3d_planning_ros`, and `active_3d_planning_voxblox` are package dependencies, but their implementations are not present under the inspected repo.",
        "- Exact definitions for `RRTStar`, `RRTStarEvaluatorAdapter`, `SegmentTime`, `GlobalNormalizedGain`, `SubsequentBest`, `ContinuousYawPlanningEvaluator`, `TrajectorySegment`, and tree node data structures are therefore missing from this audit scope.",
        "",
        "## Ambiguous Answers",
        "",
        "- Whether utility is accumulated branch/path/tree utility or local segment utility is not provable from local source alone.",
        "- Whether final execution takes the best node, best branch, or first node on the best path is not provable from local source alone.",
        "- The exact cost formula and whether it is edge cost or accumulated path cost is not provable from local source alone.",
        "- The exact low-path-cost-dominance mitigation mechanism is not provable from local source alone.",
        "",
        "## Clear Source Evidence Despite Missing Internals",
        "",
        "- `baseline.yaml` configures RRT*, the RRT* evaluator adapter, SegmentTime, GlobalNormalizedGain, and SubsequentBest.",
        "- `sc_explorer.yaml` configures the SSC map and SSC gain evaluator.",
        "- `SSCExplorationEvaluator` computes local gain over ray-cast visible voxels and reads predicted SSC voxels separately from measured ESDF observed voxels.",
        "- `sc_explorer.yaml` disables SSC information planning for ray blocking while leaving SSC planning enabled for collision fallback.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_next(path: Path) -> None:
    lines = [
        "# Recommended Next Faithful Step",
        "",
        "Choice: E. source evidence insufficient, inspect more files manually.",
        "",
        "Why:",
        "",
        "- The local SC-Explorer repo is enough to prove that the original configuration uses an external RRT*/tree planner stack.",
        "- It is not enough to prove the exact node/tree fields, accumulated utility formula, or best-path action selection.",
        "- The next small faithful task should inspect or fetch the external `mav_active_3d_planning` source named in `.rosinstall`, focusing on `RRTStar`, `RRTStarEvaluatorAdapter`, `SegmentTime`, `GlobalNormalizedGain`, `SubsequentBest`, `ContinuousYawPlanningEvaluator`, and `TrajectorySegment`.",
        "- Do not implement a new planner, do not run rollout, and do not push any counterfactual formula into execution until those source-level details are pinned down.",
        "",
        "Conditional follow-up after that inspection:",
        "",
        "- If the external source confirms accumulated branch/path utility, then the next implementation task can become option A: offline minimal tree-utility prototype over saved candidates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(repo_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, git_states = git_file_states(repo_root)
    hits, per_file = collect_hits(repo_root, git_states)
    evidence = collect_evidence(repo_root, git_states)
    candidates, excluded_local = select_candidates(per_file, git_states, evidence)

    write_hits_csv(output_dir / "source_file_hits.csv", hits)
    write_candidates(output_dir / "planner_file_candidates.txt", candidates, excluded_local)
    write_utility_evidence(
        output_dir / "utility_formula_evidence.json",
        output_dir / "utility_formula_evidence.md",
        evidence,
    )
    summary = build_summary(candidates, evidence, len(hits))
    (output_dir / "original_tree_utility_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_md(output_dir / "original_tree_utility_summary.md", summary, evidence)
    write_missing(output_dir / "missing_or_ambiguous_items.md")
    write_next(output_dir / "recommended_next_faithful_step.md")

    return {
        "output_dir": str(output_dir),
        "keyword_hit_count": len(hits),
        "candidate_count": len(candidates),
        "candidate_files": [item["file_path"] for item in candidates[:10]],
        "excluded_local_hit_files": [item["file_path"] for item in excluded_local[:10]],
        "files_written": [
            "source_file_hits.csv",
            "planner_file_candidates.txt",
            "utility_formula_evidence.json",
            "utility_formula_evidence.md",
            "original_tree_utility_summary.json",
            "original_tree_utility_summary.md",
            "missing_or_ambiguous_items.md",
            "recommended_next_faithful_step.md",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", required=True, help="Path to the SC-Explorer repo root")
    parser.add_argument("--output_dir", required=True, help="Directory for inspection outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root does not exist: {repo_root}")
    summary = write_outputs(repo_root, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
