from __future__ import annotations

import ast
import csv
import hashlib
import html
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path("/home/ubuntu22/sc_explorer_ws")
OUT = ROOT / "outputs/isaac_stage4a76_stage4a72_manual_topdown_review_packet"
RUNTIME = ROOT / "outputs/isaac_stage4a72_bounded_short_rollout_runtime"
TRANSITION_CSV = RUNTIME / "transition_decisions.csv"
LOG_DIR = ROOT / "logs"

CELL_SIZE = 0.1
ORIGIN_X = -10.5
ORIGIN_Y = -0.5


def parse_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return list(ast.literal_eval(str(value)))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def rel(path: Path) -> str:
    return path.relative_to(OUT).as_posix()


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if text_width(draw, trial, font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (246, 247, 248))
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def load_font(size: int = 16):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


@dataclass
class ActionMeta:
    start_id: int
    step_id: int
    sample_id: str
    current: tuple[float, float, float]
    current_yaw: float
    action_target: tuple[float, float, float]
    action_yaw: float
    next_capture: tuple[float, float, float] | None
    next_yaw: float | None
    has_next_camera_after: bool
    action_distance_m: float
    primary_vs_lambda48_distance: float | None
    primary_vs_measured_distance: float | None
    quality_warning: str
    quality_blocker: str
    auto_quality_verdict: str
    selected_confidence: float | None
    selected_entropy: float | None
    selected_margin: float | None
    uncertainty_composite: float | None
    action_floorplan: Path
    camera_transition: Path
    action_storyboard: Path
    original_card: Path


def observed_topdown(path: Path) -> np.ndarray:
    obs = np.load(path)
    top = np.full(obs.shape[:2], -1, dtype=np.int8)
    free = (obs == 0).any(axis=2)
    occ = (obs == 1).any(axis=2)
    top[free] = 0
    top[occ] = 1
    return top


def candidate_worlds(candidate_csv: Path):
    df = pd.read_csv(candidate_csv)
    pts = []
    score = []
    for _, row in df.iterrows():
        try:
            world = parse_list(row["world"])
            pts.append((float(world[0]), float(world[1])))
            score.append(safe_float(row.get("uncertainty_composite"), 0.0))
        except Exception:
            continue
    return np.array(pts, dtype=float), np.array(score, dtype=float), df


def draw_yaw(ax, xy, yaw, color, label):
    if yaw is None:
        return
    length = 0.42
    ax.arrow(
        xy[0],
        xy[1],
        length * math.cos(float(yaw)),
        length * math.sin(float(yaw)),
        width=0.025,
        head_width=0.13,
        head_length=0.16,
        color=color,
        alpha=0.95,
        length_includes_head=True,
        label=label,
        zorder=7,
    )


def plot_action_floorplan(
    row,
    review: dict,
    pose: dict,
    next_pose: dict | None,
    start_rows: pd.DataFrame,
    out_path: Path,
):
    current = tuple(float(x) for x in pose["position"])
    target = tuple(float(x) for x in parse_list(row["action_world_xyz"]))
    next_position = tuple(float(x) for x in next_pose["position"]) if next_pose else None

    top = observed_topdown(Path(row["observed_state_reference"]))
    extent = [
        ORIGIN_X,
        ORIGIN_X + top.shape[0] * CELL_SIZE,
        ORIGIN_Y,
        ORIGIN_Y + top.shape[1] * CELL_SIZE,
    ]
    pts, score, _ = candidate_worlds(Path(row["candidate_features"]))

    fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=150)
    cmap = colors.ListedColormap(["#24313b", "#9edbe1", "#dd5a5a"])
    norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=cmap, norm=norm, alpha=0.9)

    if len(pts):
        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=score if len(score) else "#f7d06f",
            cmap="viridis",
            s=22,
            alpha=0.55,
            edgecolors="none",
            label="candidate points",
            zorder=4,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.02)
        cb.set_label("uncertainty_composite", fontsize=8)

    history_x = []
    history_y = []
    for _, hist_row in start_rows.iterrows():
        if int(hist_row["step_id"]) > int(row["step_id"]):
            continue
        hist_pose = read_json(Path(hist_row["pose"]))
        history_x.append(float(hist_pose["position"][0]))
        history_y.append(float(hist_pose["position"][1]))
    if history_x:
        ax.plot(
            history_x,
            history_y,
            color="#111827",
            linewidth=1.8,
            linestyle="--",
            alpha=0.65,
            label="captured path so far",
            zorder=5,
        )

    ax.annotate(
        "",
        xy=(target[0], target[1]),
        xytext=(current[0], current[1]),
        arrowprops=dict(arrowstyle="-|>", color="#e11d48", lw=3.4, shrinkA=2, shrinkB=2),
        zorder=9,
    )
    ax.scatter([current[0]], [current[1]], s=170, c="#2563eb", marker="o", edgecolors="white", linewidths=1.5, label="current camera", zorder=10)
    ax.scatter([target[0]], [target[1]], s=220, c="#e11d48", marker="*", edgecolors="white", linewidths=1.0, label="primary action target", zorder=11)
    if next_position:
        ax.scatter(
            [next_position[0]],
            [next_position[1]],
            s=130,
            c="#16a34a",
            marker="D",
            edgecolors="white",
            linewidths=1.0,
            label="next captured camera",
            zorder=10,
        )

    lx = safe_float(review.get("lambda48_shadow_target_x"))
    ly = safe_float(review.get("lambda48_shadow_target_y"))
    if lx is not None and ly is not None:
        ax.scatter([lx], [ly], s=95, c="#f59e0b", marker="s", edgecolors="#7c2d12", linewidths=1.0, label="lambda48 shadow", zorder=8)
    mx = safe_float(review.get("measured_shadow_target_x"))
    my = safe_float(review.get("measured_shadow_target_y"))
    if mx is not None and my is not None:
        ax.scatter([mx], [my], s=110, c="#8b5cf6", marker="^", edgecolors="#3b0764", linewidths=1.0, label="measured shadow", zorder=8)

    draw_yaw(ax, (current[0], current[1]), safe_float(pose.get("yaw_rad", pose.get("yaw"))), "#1d4ed8", "current yaw")
    draw_yaw(ax, (target[0], target[1]), safe_float(row.get("action_yaw")), "#be123c", "target yaw")

    distance = math.dist(current[:2], target[:2])
    ax.text(
        0.015,
        0.985,
        f"start {int(row['start_variant_id']):03d} step {int(row['step_id']):03d}\n"
        f"current ({current[0]:.2f}, {current[1]:.2f}) -> target ({target[0]:.2f}, {target[1]:.2f})\n"
        f"distance {distance:.2f} m | primary source: uncertainty_bonus_composite_beta8",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#9ca3af", alpha=0.92),
        zorder=20,
    )

    focus = [(current[0], current[1]), (target[0], target[1])]
    if next_position:
        focus.append((next_position[0], next_position[1]))
    for x_key, y_key in [
        ("lambda48_shadow_target_x", "lambda48_shadow_target_y"),
        ("measured_shadow_target_x", "measured_shadow_target_y"),
    ]:
        sx = safe_float(review.get(x_key))
        sy = safe_float(review.get(y_key))
        if sx is not None and sy is not None:
            focus.append((sx, sy))
    xs = [p[0] for p in focus]
    ys = [p[1] for p in focus]
    pad = max(1.2, distance + 0.55)
    ax.set_xlim(max(extent[0], min(xs) - pad), min(extent[1], max(xs) + pad))
    ax.set_ylim(max(extent[2], min(ys) - pad), min(extent[3], max(ys) + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="white", alpha=0.5, linewidth=0.6)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("Action movement on observed topdown map", fontsize=13, pad=10)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys(), loc="lower right", fontsize=8, framealpha=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def image_diff(before: Image.Image, after: Image.Image, width: int, height: int) -> Image.Image:
    b = fit_image(before, width, height)
    a = fit_image(after, width, height)
    diff = ImageChops.difference(b, a)
    arr = np.asarray(diff).astype(np.float32).mean(axis=2)
    scale = float(arr.max()) or 1.0
    heat = (cm.magma(arr / scale)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(heat)


def draw_panel_title(draw, x, y, width, title, subtitle, font_title, font_small):
    draw.rectangle([x, y, x + width, y + 46], fill=(17, 24, 39))
    draw.text((x + 12, y + 8), title, fill=(255, 255, 255), font=font_title)
    draw.text((x + 12, y + 28), subtitle, fill=(209, 213, 219), font=font_small)


def make_camera_transition(row, pose: dict, next_row, next_pose: dict | None, out_path: Path):
    before = Image.open(Path(row["rgb"])).convert("RGB")
    after = Image.open(Path(next_row["rgb"])).convert("RGB") if next_row is not None else None
    panel_w = 420
    panel_h = 260
    header_h = 58
    bottom_h = 104
    gap = 16
    width = panel_w * 3 + gap * 4
    height = header_h + panel_h + bottom_h
    canvas = Image.new("RGB", (width, height), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(18)
    font_small = load_font(12)
    font_body = load_font(14)

    start_id = int(row["start_variant_id"])
    step_id = int(row["step_id"])
    target = parse_list(row["action_world_xyz"])
    current = pose["position"]
    draw.rectangle([0, 0, width, header_h], fill=(248, 250, 252))
    draw.text(
        (18, 12),
        f"Camera transition: start {start_id:03d} step {step_id:03d}",
        fill=(15, 23, 42),
        font=font_title,
    )
    draw.text(
        (18, 36),
        "offline review view | no Isaac/runtime/training/checkpoint | primary source uncertainty_bonus_composite_beta8",
        fill=(71, 85, 105),
        font=font_small,
    )

    x1 = gap
    y_panel = header_h
    draw_panel_title(draw, x1, y_panel, panel_w, "BEFORE", f"current camera ({current[0]:.2f}, {current[1]:.2f})", font_body, font_small)
    canvas.paste(fit_image(before, panel_w, panel_h - 46), (x1, y_panel + 46))

    x2 = x1 + panel_w + gap
    if after is not None and next_pose is not None:
        np_pos = next_pose["position"]
        draw_panel_title(draw, x2, y_panel, panel_w, "AFTER", f"next capture ({np_pos[0]:.2f}, {np_pos[1]:.2f})", font_body, font_small)
        canvas.paste(fit_image(after, panel_w, panel_h - 46), (x2, y_panel + 46))
        x3 = x2 + panel_w + gap
        draw_panel_title(draw, x3, y_panel, panel_w, "RGB DELTA", "absolute visual change", font_body, font_small)
        canvas.paste(image_diff(before, after, panel_w, panel_h - 46), (x3, y_panel + 46))
    else:
        draw_panel_title(draw, x2, y_panel, panel_w, "AFTER", "not captured in bounded 7.2 sample", font_body, font_small)
        draw.rectangle([x2, y_panel + 46, x2 + panel_w, y_panel + panel_h], fill=(226, 232, 240))
        for line_i, line in enumerate(wrap_text(draw, "No next RGB frame exists for this final sampled step. Use the floorplan target/yaw to review the action intent.", font_body, panel_w - 40)):
            draw.text((x2 + 20, y_panel + 82 + line_i * 22), line, fill=(51, 65, 85), font=font_body)
        x3 = x2 + panel_w + gap
        draw_panel_title(draw, x3, y_panel, panel_w, "ACTION TARGET", f"target ({target[0]:.2f}, {target[1]:.2f})", font_body, font_small)
        draw.rectangle([x3, y_panel + 46, x3 + panel_w, y_panel + panel_h], fill=(255, 247, 237))
        summary = f"Target position: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}); target yaw {safe_float(row.get('action_yaw'), 0.0):.2f} rad. No after-camera frame was added."
        for line_i, line in enumerate(wrap_text(draw, summary, font_body, panel_w - 40)):
            draw.text((x3 + 20, y_panel + 82 + line_i * 22), line, fill=(124, 45, 18), font=font_body)

    bottom_y = header_h + panel_h + 14
    distance = math.dist((current[0], current[1]), (target[0], target[1]))
    lines = [
        f"movement: ({current[0]:.2f}, {current[1]:.2f}) -> ({target[0]:.2f}, {target[1]:.2f})  distance={distance:.2f}m",
        f"action_yaw={safe_float(row.get('action_yaw'), 0.0):.3f} rad | source row: Stage 4A-7.2 transition_decisions.csv start={start_id}, step={step_id}",
        "lambda48 remains shadow/baseline only; this view does not promote labels.",
    ]
    for i, line in enumerate(lines):
        draw.text((18, bottom_y + i * 24), line, fill=(15, 23, 42), font=font_body)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_storyboard(action_floorplan: Path, camera_transition: Path, meta_text: list[str], out_path: Path):
    font_title = load_font(20)
    font_body = load_font(14)
    fp = Image.open(action_floorplan).convert("RGB")
    cam_img = Image.open(camera_transition).convert("RGB")
    fp = fit_image(fp, 760, 760)
    cam_img.thumbnail((900, 420), Image.Resampling.LANCZOS)
    width = 1710
    height = 860
    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), meta_text[0], fill=(15, 23, 42), font=font_title)
    draw.text((24, 48), meta_text[1], fill=(71, 85, 105), font=font_body)
    canvas.paste(fp, (24, 82))
    canvas.paste(cam_img, (790, 96))
    y = 96 + cam_img.height + 22
    for line in meta_text[2:]:
        for wrapped in wrap_text(draw, line, font_body, 860):
            draw.text((800, y), wrapped, fill=(30, 41, 59), font=font_body)
            y += 24
    draw.rectangle([24, 812, width - 24, 844], fill=(15, 23, 42))
    draw.text(
        (36, 820),
        "Review intent: judge the primary action geometry and camera change only. No training, checkpoint, rollout, or label promotion happened.",
        fill=(255, 255, 255),
        font=font_body,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def update_sample_card(card: Path):
    html_text = card.read_text(encoding="utf-8")
    section = """
<section id="action-story-upgrade" style="margin:16px 0;padding:12px;background:#eef6ff;border:1px solid #9cc7f5">
<h2>Action Story View</h2>
<p>Use this first: it shows current camera -> primary action target on the observed topdown map, plus before/after camera frames when the next capture exists.</p>
<figure><img src="action_review_storyboard.png" loading="lazy" style="max-width:1000px"><figcaption>action_review_storyboard.png</figcaption></figure>
<figure><img src="action_arrow_floorplan.png" loading="lazy"><figcaption>action_arrow_floorplan.png</figcaption></figure>
<figure><img src="camera_transition.png" loading="lazy" style="max-width:1000px"><figcaption>camera_transition.png</figcaption></figure>
</section>
"""
    start = html_text.find('<section id="action-story-upgrade"')
    if start != -1:
        end = html_text.find("</section>", start)
        if end != -1:
            html_text = html_text[:start] + section + html_text[end + len("</section>") :]
    else:
        marker = "<section><figure>"
        html_text = html_text.replace(marker, section + marker, 1)
    card.write_text(html_text, encoding="utf-8")


def make_main_html(actions: list[ActionMeta], out_path: Path):
    grouped: dict[int, list[ActionMeta]] = {}
    for item in actions:
        grouped.setdefault(item.start_id, []).append(item)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Stage 4A-7.2 Action Story Review</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#f8fafc;color:#0f172a}",
        "header{position:sticky;top:0;background:#0f172a;color:white;padding:14px 22px;z-index:10}",
        "main{max-width:1320px;margin:0 auto;padding:20px}",
        ".notice{background:#fff7ed;border:1px solid #fdba74;padding:12px;margin:14px 0}",
        ".start{margin:22px 0;padding-top:6px;border-top:3px solid #cbd5e1}",
        ".action{background:white;border:1px solid #cbd5e1;border-radius:8px;margin:16px 0;padding:14px;box-shadow:0 1px 2px #0001}",
        ".story{width:100%;max-width:1240px;border:1px solid #cbd5e1;background:white}",
        ".meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:10px 0}",
        ".meta div{background:#f1f5f9;padding:8px;border-radius:6px;font-size:13px}",
        ".links a{margin-right:12px}",
        ".warn{border-left:6px solid #f59e0b}.pass{border-left:6px solid #16a34a}.block{border-left:6px solid #dc2626}",
        "code{background:#e2e8f0;padding:1px 4px;border-radius:4px}",
        "</style></head><body>",
        "<header><strong>Stage 4A-7.2 Action Story Review</strong> | visual upgrade for manual action review</header>",
        "<main>",
        "<div class='notice'><strong>Safety:</strong> offline visualization only. No Isaac startup, no capture, no map_predict, no rollout, no BC training, no checkpoint, no label promotion, no RL/GDPO/PPO. Primary action source remains <code>stage4a613_uncertainty_bonus_executed_primary</code> / <code>uncertainty_bonus_composite_beta8</code>; lambda48 is shadow only.</div>",
        "<p>Use each storyboard to judge whether the action makes sense: left side is current camera to primary target on observed topdown; right side is camera before/after plus RGB delta when a next frame exists.</p>",
        "<p><a href='stage4a72_topdown_review_index.html'>Original 7.6 index</a> | <a href='stage4a72_manual_review_template.csv'>review CSV template</a> | <a href='stage4a72_action_story_sample_index.csv'>action story sample index CSV</a></p>",
    ]
    for start_id in sorted(grouped):
        parts.append(f"<section class='start'><h2>Start {start_id:03d}</h2>")
        parts.append(f"<p><img src='samples/start_{start_id:03d}/start_path_floorplan.png' style='max-width:460px;border:1px solid #cbd5e1' loading='lazy'> <img src='samples/start_{start_id:03d}/start_action_sequence_floorplan.png' style='max-width:460px;border:1px solid #cbd5e1' loading='lazy'></p>")
        for item in sorted(grouped[start_id], key=lambda x: x.step_id):
            cls = "block" if item.quality_blocker else ("warn" if item.quality_warning else "pass")
            next_text = "yes" if item.has_next_camera_after else "no - final sampled step"
            parts.append(f"<article class='action {cls}' id='{html.escape(item.sample_id)}'>")
            parts.append(f"<h3>{html.escape(item.sample_id)}</h3>")
            parts.append("<div class='meta'>")
            parts.append(f"<div><strong>movement</strong><br>({item.current[0]:.2f},{item.current[1]:.2f}) -> ({item.action_target[0]:.2f},{item.action_target[1]:.2f})</div>")
            parts.append(f"<div><strong>distance</strong><br>{item.action_distance_m:.2f} m</div>")
            parts.append(f"<div><strong>after camera</strong><br>{next_text}</div>")
            parts.append(f"<div><strong>verdict</strong><br>{html.escape(item.auto_quality_verdict)}</div>")
            parts.append(f"<div><strong>confidence / entropy / margin</strong><br>{item.selected_confidence} / {item.selected_entropy} / {item.selected_margin}</div>")
            parts.append(f"<div><strong>uncertainty composite</strong><br>{item.uncertainty_composite}</div>")
            parts.append(f"<div><strong>vs lambda48 shadow</strong><br>{item.primary_vs_lambda48_distance}</div>")
            parts.append(f"<div><strong>vs measured shadow</strong><br>{item.primary_vs_measured_distance}</div>")
            parts.append("</div>")
            parts.append(f"<img class='story' src='{rel(item.action_storyboard)}' loading='lazy'>")
            parts.append("<p class='links'>")
            parts.append(f"<a href='{rel(item.action_floorplan)}'>action floorplan</a>")
            parts.append(f"<a href='{rel(item.camera_transition)}'>camera transition</a>")
            parts.append(f"<a href='{rel(item.original_card)}'>original card/form</a>")
            parts.append("</p></article>")
        parts.append("</section>")
    parts.extend(["</main></body></html>"])
    out_path.write_text("\n".join(parts), encoding="utf-8")


def insert_main_banner():
    path = OUT / "stage4a72_topdown_review_index.html"
    text = path.read_text(encoding="utf-8")
    banner = """
<div id="action-story-upgrade-banner" style="margin:12px;padding:12px;background:#eef6ff;border:1px solid #93c5fd">
<strong>New visual action story view:</strong>
<a href="stage4a72_action_story_review_index.html">open the intuitive action-by-action review index</a>.
It shows current camera -> primary target on topdown plus camera before/after transition.
</div>
"""
    start = text.find('<div id="action-story-upgrade-banner"')
    if start != -1:
        end = text.find("</div>", start)
        if end != -1:
            text = text[:start] + banner + text[end + len("</div>") :]
    else:
        text = text.replace("<body>", "<body>" + banner, 1)
    path.write_text(text, encoding="utf-8")


def update_instructions():
    path = OUT / "stage4a72_manual_review_instructions.md"
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    section = """# Stage 4A-7.2 Manual Review Instructions - Visual Action Story Upgrade

Start with `stage4a72_action_story_review_index.html`.

For every action, review:

1. The left topdown panel: current camera position, primary action target, yaw direction, next captured camera position when available, and lambda48/measured shadow markers.
2. The right camera panel: before RGB, after RGB, and RGB delta when the next frame exists.
3. The original review fields in `sample_review_card.html` or `stage4a72_manual_review_template.csv`.

This upgrade is visualization only. It did not run Isaac, capture new frames, execute actions, train BC, create checkpoints, promote labels, or run RL/GDPO/PPO. Lambda48 remains a shadow/baseline label only.

---

"""
    if "Visual Action Story Upgrade" in old:
        tail = old.split("---", 1)[-1].lstrip() if "---" in old else old
        path.write_text(section + tail, encoding="utf-8")
    else:
        path.write_text(section + old, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def refresh_manifest():
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": rel(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(OUT / "artifact_manifest.json", {"artifact_count": len(rows), "artifacts": rows})
    md = ["# Artifact Manifest", "", f"artifact_count: {len(rows)}", ""]
    for item in rows:
        md.append(f"- `{item['path']}` ({item['size_bytes']} bytes)")
    (OUT / "artifact_manifest.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def git_status_text() -> str:
    return subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True)


def main():
    LOG_DIR.mkdir(exist_ok=True)
    (OUT / "git_status_before_visual_upgrade.txt").write_text(git_status_text(), encoding="utf-8")

    df = pd.read_csv(TRANSITION_CSV).sort_values(["start_variant_id", "step_id"])
    actions: list[ActionMeta] = []
    story_rows = []

    for _, row in df.iterrows():
        start_id = int(row["start_variant_id"])
        step_id = int(row["step_id"])
        sample_id = f"stage4a72_start{start_id:03d}_step{step_id:03d}"
        sample_dir = OUT / "samples" / f"start_{start_id:03d}" / f"step_{step_id:03d}"
        pose = read_json(Path(row["pose"]))

        next_subset = df[(df["start_variant_id"] == start_id) & (df["step_id"] == step_id + 1)]
        next_row = next_subset.iloc[0] if len(next_subset) else None
        next_pose = read_json(Path(next_row["pose"])) if next_row is not None else None
        review = read_json(sample_dir / "review_row.json")

        action_floorplan = sample_dir / "action_arrow_floorplan.png"
        camera_transition = sample_dir / "camera_transition.png"
        action_storyboard = sample_dir / "action_review_storyboard.png"

        start_rows = df[df["start_variant_id"] == start_id]
        plot_action_floorplan(row, review, pose, next_pose, start_rows, action_floorplan)
        make_camera_transition(row, pose, next_row, next_pose, camera_transition)

        current = tuple(float(x) for x in pose["position"])
        target = tuple(float(x) for x in parse_list(row["action_world_xyz"]))
        next_capture = tuple(float(x) for x in next_pose["position"]) if next_pose else None
        distance = math.dist(current[:2], target[:2])
        meta_lines = [
            f"{sample_id}: action story",
            "Primary action source: stage4a613_uncertainty_bonus_executed_primary / uncertainty_bonus_composite_beta8",
            f"Current camera: ({current[0]:.2f}, {current[1]:.2f}, {current[2]:.2f}); primary target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}); distance {distance:.2f} m.",
            f"Next camera frame: {'available' if next_capture else 'not available in this bounded sample'}" + (f" at ({next_capture[0]:.2f}, {next_capture[1]:.2f}, {next_capture[2]:.2f})." if next_capture else "."),
            f"Auto quality: {review.get('auto_quality_verdict', '')}; warning={review.get('quality_warning', '') or 'none'}; blocker={review.get('quality_blocker', '') or 'none'}.",
            f"Shadow comparisons: lambda48 distance={review.get('primary_vs_lambda48_distance')}; measured distance={review.get('primary_vs_measured_distance')}. Lambda48 is not primary.",
        ]
        make_storyboard(action_floorplan, camera_transition, meta_lines, action_storyboard)

        review.update(
            {
                "visual_action_story_upgrade": True,
                "visual_action_arrow_floorplan": rel(action_floorplan),
                "visual_camera_transition": rel(camera_transition),
                "visual_action_review_storyboard": rel(action_storyboard),
                "action_story_current_x": current[0],
                "action_story_current_y": current[1],
                "action_story_target_x": target[0],
                "action_story_target_y": target[1],
                "action_story_next_capture_x": next_capture[0] if next_capture else None,
                "action_story_next_capture_y": next_capture[1] if next_capture else None,
                "has_next_camera_after": bool(next_capture),
                "action_distance_m": distance,
                "visual_upgrade_note": "offline visualization only; no label promotion/training/runtime/checkpoint/RL",
            }
        )
        write_json(sample_dir / "review_row.json", review)
        update_sample_card(sample_dir / "sample_review_card.html")

        meta = ActionMeta(
            start_id=start_id,
            step_id=step_id,
            sample_id=sample_id,
            current=current,
            current_yaw=safe_float(pose.get("yaw_rad", pose.get("yaw")), 0.0),
            action_target=target,
            action_yaw=safe_float(row.get("action_yaw"), 0.0),
            next_capture=next_capture,
            next_yaw=safe_float(next_pose.get("yaw_rad", next_pose.get("yaw")), None) if next_pose else None,
            has_next_camera_after=bool(next_capture),
            action_distance_m=distance,
            primary_vs_lambda48_distance=safe_float(review.get("primary_vs_lambda48_distance")),
            primary_vs_measured_distance=safe_float(review.get("primary_vs_measured_distance")),
            quality_warning=str(review.get("quality_warning") or ""),
            quality_blocker=str(review.get("quality_blocker") or ""),
            auto_quality_verdict=str(review.get("auto_quality_verdict") or ""),
            selected_confidence=safe_float(review.get("selected_confidence")),
            selected_entropy=safe_float(review.get("selected_entropy")),
            selected_margin=safe_float(review.get("selected_margin")),
            uncertainty_composite=safe_float(review.get("uncertainty_composite")),
            action_floorplan=action_floorplan,
            camera_transition=camera_transition,
            action_storyboard=action_storyboard,
            original_card=sample_dir / "sample_review_card.html",
        )
        actions.append(meta)
        story_rows.append(
            {
                "sample_id": sample_id,
                "start_id": start_id,
                "step_id": step_id,
                "current_x": current[0],
                "current_y": current[1],
                "target_x": target[0],
                "target_y": target[1],
                "action_distance_m": distance,
                "has_next_camera_after": bool(next_capture),
                "action_storyboard": rel(action_storyboard),
                "action_floorplan": rel(action_floorplan),
                "camera_transition": rel(camera_transition),
                "sample_review_card": rel(sample_dir / "sample_review_card.html"),
            }
        )

    make_main_html(actions, OUT / "stage4a72_action_story_review_index.html")
    insert_main_banner()
    update_instructions()

    write_json(OUT / "stage4a72_action_story_sample_index.json", story_rows)
    with (OUT / "stage4a72_action_story_sample_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(story_rows[0].keys()))
        writer.writeheader()
        writer.writerows(story_rows)

    report = {
        "completed": True,
        "blocked": False,
        "main_html": str(OUT / "stage4a72_action_story_review_index.html"),
        "total_actions": len(actions),
        "action_storyboards": len([x for x in actions if x.action_storyboard.is_file()]),
        "action_floorplans": len([x for x in actions if x.action_floorplan.is_file()]),
        "camera_transitions": len([x for x in actions if x.camera_transition.is_file()]),
        "has_next_camera_after_count": sum(1 for x in actions if x.has_next_camera_after),
        "missing_next_camera_after_count": sum(1 for x in actions if not x.has_next_camera_after),
        "stage4a72_promoted": False,
        "expert_action_index_primary_created": False,
        "lambda48_primary_use": False,
        "bc_training": False,
        "checkpoint": False,
        "isaac_startup": False,
        "capture": False,
        "map_predict": False,
        "rollout": False,
        "rl_gdpo_ppo": False,
        "label_source": "stage4a613_uncertainty_bonus_executed_primary / uncertainty_bonus_composite_beta8",
        "lambda48_role": "shadow/baseline only",
    }
    write_json(OUT / "stage4a76_action_story_visual_upgrade_report.json", report)
    md = [
        "# Stage 4A-7.6 Action Story Visual Upgrade Report",
        "",
        f"- completed: {report['completed']}",
        f"- blocked: {report['blocked']}",
        f"- main_html: `{report['main_html']}`",
        f"- total_actions: {report['total_actions']}",
        f"- action_storyboards: {report['action_storyboards']}",
        f"- action_floorplans: {report['action_floorplans']}",
        f"- camera_transitions: {report['camera_transitions']}",
        f"- has_next_camera_after_count: {report['has_next_camera_after_count']}",
        f"- missing_next_camera_after_count: {report['missing_next_camera_after_count']}",
        "- safety: no Isaac/runtime/capture/map_predict/rollout/training/checkpoint/label promotion/RL/GDPO/PPO",
        "- primary label source preserved: stage4a613_uncertainty_bonus_executed_primary / uncertainty_bonus_composite_beta8",
        "- lambda48 role: shadow/baseline only",
        "",
    ]
    (OUT / "stage4a76_action_story_visual_upgrade_report.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "git_status_after_visual_upgrade.txt").write_text(git_status_text(), encoding="utf-8")
    refresh_manifest()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
