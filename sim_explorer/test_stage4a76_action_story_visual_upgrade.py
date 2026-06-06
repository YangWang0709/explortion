from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a76_stage4a72_manual_topdown_review_packet"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_nonblank(path: Path) -> bool:
    if not path.is_file():
        return False
    with Image.open(path) as image:
        if image.width < 500 or image.height < 350:
            return False
        extrema = image.convert("RGB").getextrema()
        return any(lo != hi for lo, hi in extrema)


def main():
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    report_path = OUT / "stage4a76_action_story_visual_upgrade_report.json"
    report = load_json(report_path) if report_path.is_file() else {}
    checks["report_exists"] = report_path.is_file()
    checks["report_completed"] = report.get("completed") is True
    checks["report_not_blocked"] = report.get("blocked") is False
    checks["main_html_exists"] = (OUT / "stage4a72_action_story_review_index.html").is_file()
    checks["sample_index_csv_exists"] = (OUT / "stage4a72_action_story_sample_index.csv").is_file()
    checks["sample_index_json_exists"] = (OUT / "stage4a72_action_story_sample_index.json").is_file()

    main_html = (OUT / "stage4a72_action_story_review_index.html").read_text(encoding="utf-8") if checks["main_html_exists"] else ""
    checks["main_html_mentions_action_story"] = "Action Story Review" in main_html
    checks["main_html_links_original_index"] = "stage4a72_topdown_review_index.html" in main_html
    checks["main_html_safety_scope"] = all(
        token in main_html
        for token in [
            "No Isaac startup",
            "no BC training",
            "no checkpoint",
            "no label promotion",
            "lambda48 is shadow only",
        ]
    )
    original_html = (OUT / "stage4a72_topdown_review_index.html").read_text(encoding="utf-8")
    checks["original_index_has_banner"] = "stage4a72_action_story_review_index.html" in original_html

    csv_rows = []
    if checks["sample_index_csv_exists"]:
        with (OUT / "stage4a72_action_story_sample_index.csv").open(newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
    json_rows = load_json(OUT / "stage4a72_action_story_sample_index.json") if checks["sample_index_json_exists"] else []
    checks["sample_index_csv_30_rows"] = len(csv_rows) == 30
    checks["sample_index_json_30_rows"] = len(json_rows) == 30

    story_ok = 0
    floorplan_ok = 0
    camera_ok = 0
    review_field_ok = 0
    card_link_ok = 0
    has_next_true = 0
    has_next_false = 0
    for row in csv_rows:
        story = OUT / row["action_storyboard"]
        floorplan = OUT / row["action_floorplan"]
        camera = OUT / row["camera_transition"]
        card = OUT / row["sample_review_card"]
        if image_nonblank(story):
            story_ok += 1
        if image_nonblank(floorplan):
            floorplan_ok += 1
        if image_nonblank(camera):
            camera_ok += 1
        if card.is_file() and "action_review_storyboard.png" in card.read_text(encoding="utf-8"):
            card_link_ok += 1
        review = load_json(card.parent / "review_row.json")
        required = [
            "visual_action_story_upgrade",
            "visual_action_arrow_floorplan",
            "visual_camera_transition",
            "visual_action_review_storyboard",
            "has_next_camera_after",
            "action_distance_m",
        ]
        if all(key in review for key in required) and review.get("visual_action_story_upgrade") is True:
            review_field_ok += 1
        if row.get("has_next_camera_after") == "True":
            has_next_true += 1
        elif row.get("has_next_camera_after") == "False":
            has_next_false += 1

    checks["storyboards_nonblank_30"] = story_ok == 30
    checks["floorplans_nonblank_30"] = floorplan_ok == 30
    checks["camera_transitions_nonblank_30"] = camera_ok == 30
    checks["review_rows_have_visual_fields_30"] = review_field_ok == 30
    checks["sample_cards_link_visual_story_30"] = card_link_ok == 30
    checks["next_camera_counts_expected"] = has_next_true == 20 and has_next_false == 10

    checks["report_counts_expected"] = (
        report.get("total_actions") == 30
        and report.get("action_storyboards") == 30
        and report.get("action_floorplans") == 30
        and report.get("camera_transitions") == 30
        and report.get("has_next_camera_after_count") == 20
        and report.get("missing_next_camera_after_count") == 10
    )
    checks["no_training_runtime_promotion"] = all(
        report.get(key) is False
        for key in [
            "stage4a72_promoted",
            "expert_action_index_primary_created",
            "lambda48_primary_use",
            "bc_training",
            "checkpoint",
            "isaac_startup",
            "capture",
            "map_predict",
            "rollout",
            "rl_gdpo_ppo",
        ]
    )
    checks["primary_label_source_preserved"] = "uncertainty_bonus_composite_beta8" in str(report.get("label_source", ""))
    checks["lambda48_shadow_only"] = "shadow" in str(report.get("lambda48_role", "")).lower()

    for key, ok in checks.items():
        if not ok:
            blockers.append(key)

    result = {
        "all_passed": not blockers,
        "blockers": blockers,
        "checks": checks,
        "counts": {
            "csv_rows": len(csv_rows),
            "json_rows": len(json_rows),
            "storyboards_nonblank": story_ok,
            "floorplans_nonblank": floorplan_ok,
            "camera_transitions_nonblank": camera_ok,
            "review_rows_have_visual_fields": review_field_ok,
            "sample_cards_link_visual_story": card_link_ok,
            "has_next_camera_after_true": has_next_true,
            "has_next_camera_after_false": has_next_false,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
