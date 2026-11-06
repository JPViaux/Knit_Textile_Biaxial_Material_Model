"""Approved Figure 9: all candidates along the retrospective greedy search.

Edit the settings below and run this file in VSCode. Saved report scores are
read without fitting or recalculation. Negative scores use baseline triangles
labelled '<0'; their magnitudes are not plotted. The original plotter remains
independent.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# Release-local directory; example: move the whole GitHub_package folder to another computer.
RELEASE_DIRECTORY = Path(__file__).resolve().parent
# Audited campaign input; example: a replacement campaign's full_workbook_payload.json.
REPORT_PATH = RELEASE_DIRECTORY.parent / "archived_results/adaptive_ratio_study_sv_exp_polycon_mo_v1/full_workbook_payload.json"
# Manuscript figure destination; example: Path(r"C:\Users\JP\Downloads\Figures").
OUTPUT_DIRECTORY = RELEASE_DIRECTORY.parent / "generated/figures/greedy"
# Common filename without extension; example: "greedy_r2_candidates".
OUTPUT_BASENAME = "greedy_r2_candidates"
# Save a PNG when running from VSCode; example: False to skip PNG export.
SAVE_PNG = False
# Save an SVG when running from VSCode; example: False to skip SVG export.
SAVE_SVG = True
# Save a PDF independently of PNG/SVG; example: False to skip PDF export.
SAVE_PDF = False
# Display the Matplotlib window; example: False for automated verification.
SHOW_PLOT = True
# Preserve the reference width; canvas height increased 25%; example: (6.3060417222, 3.5) inches.
FIGURE_SIZE = (454.035004 / 72.0, 3.5)
# Draft PNG resolution; example: 150 dots per inch.
DPI = 150
# Non-selected point opacity and area; examples: 0.35 and 15 square points.
CANDIDATE_ALPHA = 0.35
CANDIDATE_SIZE = 15.0
# Selected point area and curve width; examples: 24 square points and 1.0 point.
SELECTED_SIZE = 24.0
LINE_WIDTH = 1.0
# Maximum horizontal displacement from each batch; example: 0.23 batch units.
POINT_SPREAD = 0.0
# Match the reference figure's STIX font; example: "STIXGeneral".
FONT_FAMILY = "STIXGeneral"
# Math glyphs matching the original figure; example: "stix" for the R-squared label.
MATH_FONTSET = "stix"
# Panel titles match Figure 8; examples: 12.0 points and "#333333".
TITLE_FONT_SIZE = 12.0
TITLE_FONT_COLOR = "#333333"
# Title weight and padding match Figure 8; examples: "normal" and 4.0 points.
TITLE_FONT_WEIGHT = "normal"
TITLE_PAD = 4.0
# Shared y-axis label matches Figure 8; examples: 9.0 points and "black".
AXIS_LABEL_FONT_SIZE = 9.0
AXIS_LABEL_FONT_COLOR = "black"
# Tick, ratio, and negative-label sizes in points; examples: 7.0, 6.5, and 6.0.
TICK_FONT_SIZE = 7.0
RATIO_FONT_SIZE = 6.5
NEGATIVE_FONT_SIZE = 6.0
# Baseline marker and label positions for negative scores; examples: 0.018 and 0.01.
NEGATIVE_MARKER_Y = 0.018
NEGATIVE_LABEL_Y = 0.01
# Put '<0' beside the triangles in batch units; example: 0.32 to the right.
NEGATIVE_LABEL_X_OFFSET = 0.32
# Width of each cumulative vertical bar in batch units; example: 0.58.
RATIO_BAR_WIDTH = 0.58
# Panel heights increased 25% from the manual PDF; examples: 1.084125 and 0.487875 inches.
SCORE_PANEL_HEIGHT_INCHES = 1.084125
RATIO_PANEL_HEIGHT_INCHES = 0.487875
# Transparent ratio axes overlap the score axes by this height; example: 0.2126 inches (0 for no overlap).
RATIO_OVERLAP_INCHES = 0.2126
# Space between complete panel rows, including title clearance; example: 0.33 inches.
PANEL_ROW_GAP_INCHES = 0.33
# Clearance from the canvas top to the upper score axes; example: 0.20 inches.
TOP_MARGIN_INCHES = 0.20
# Gap above each bar for its newly added ratio label, in ratio counts; example: 0.20.
RATIO_LABEL_PAD = 0.20
# Ratio-axis headroom for labels above the nine-segment bar; example: 12.5.
RATIO_AXIS_TOP = 12.5
# Panel order, display titles, and colors; example: ("Rib", "Rib", "#FFD140").
KNIT_PANELS = (
    ("Interlock", "Interlock", "#3BD5AE"),
    ("Purl", "Purl", "#62DAFC"),
    ("Rib", "Rib", "#FFD140"),
    ("SingleJersey", "Single Jersey", "#FB471F"),
)
# Stored ratio tokens supported by this nine-path study; example: "0.75:1".
RATIOS = ("0:1", "0.25:1", "0.5:1", "0.75:1", "1:0", "1:0.25", "1:0.5", "1:0.75", "1:1")
# Exact square-fill colors extracted from greedy_r2_manual.pdf; example: "0:1": "#C65E37".
MANUAL_RATIO_COLORS = {
    "0:1": "#C65E37",
    "0.25:1": "#005EAC",
    "0.5:1": "#C65EAC",
    "0.75:1": "#00B9AC",
    "1:0": "#A4D270",
    "1:0.25": "#6F66BC",
    "1:0.5": "#A4D2B5",
    "1:0.75": "#2A5E37",
    "1:1": "#F5D78A",
}

CAPTION = (
    "Dots show all tested next-ratio candidates; solid dots and the connecting curve show the selected maxima. "
    "From B2 onward, candidates extend the preceding selected greedy path. Their spread describes "
    "sensitivity to the next ratio choice, not failure probability across all possible calibration sets. "
    "Scores are pooled PK1 R² across all nine evaluation ratios. Negative scores are represented "
    "by baseline triangles labelled '<0', without displaying their magnitudes. Cumulative "
    "vertical bars overlapping the lower edge of each score panel show the included calibration ratios, with one colored "
    "segment per ratio in selection order from bottom to top. Only the newly added ratio is "
    "labelled above each bar; Bk contains k calibration ratios. Labels and colors use Course:wale "
    "ratios, with Interlock's stored labels reversed for display only. Bar heights count ratios "
    "and do not represent R² values."
)


def physical_ratio(knit: str, stored_ratio: str) -> str:
    """Convert Interlock's wale:course tokens to common Course:wale labels."""
    first, second = stored_ratio.split(":")
    return f"{second}:{first}" if knit == "Interlock" else stored_ratio


def load_report(path: Path = REPORT_PATH) -> tuple[dict, dict]:
    """Validate complete candidate groups and their corresponding selected fits."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("study_id") != "adaptive_ratio_greedy_sv_exp_polycon_mo_v1":
        raise ValueError("Expected the SV_exp_polycon_mo v1 greedy campaign")
    if payload.get("audit", {}).get("status") != "passed":
        raise ValueError("Report audit must have status 'passed'")
    sheets = payload.get("sheets", {})
    candidates = sheets.get("Candidates", [])
    selections = sheets.get("Selected Path", [])
    if len(candidates) != 180 or len(selections) != 36:
        raise ValueError("Expected 180 candidate rows and 36 selected-path rows")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    knits = {p[0] for p in KNIT_PANELS}
    batches = {f"B{i:02d}" for i in range(1, 10)}
    seen_ids: set[str] = set()
    for row in candidates:
        key = (row["knit"], row["batch"])
        if key[0] not in knits or key[1] not in batches:
            raise ValueError(f"Unexpected knit/batch: {key}")
        if row["candidate_id"] in seen_ids:
            raise ValueError(f"Duplicate candidate: {row['candidate_id']}")
        seen_ids.add(row["candidate_id"])
        value = float(row["all_ratio_pooled_R2"])
        if not math.isfinite(value) or value > 1.0:
            raise ValueError(f"Invalid R² for {row['candidate_id']}: {value}")
        if row["added_ratio"] not in RATIOS:
            raise ValueError(f"Unexpected ratio: {row['added_ratio']}")
        grouped[key].append(row)
    for row in selections:
        key = (row["knit"], row["batch"])
        if key in selected or key not in grouped:
            raise ValueError(f"Duplicate or unexpected selection: {key}")
        if row["model"] != "SV_exp_polycon_mo":
            raise ValueError(f"Unexpected selected model at {key}")
        selected[key] = row
    expected_keys = {(k, b) for k in knits for b in batches}
    if set(grouped) != expected_keys or set(selected) != expected_keys:
        raise ValueError("Missing knit/batch groups")
    for knit in knits:
        previous: set[str] = set()
        for step in range(1, 10):
            key = (knit, f"B{step:02d}")
            rows, choice = grouped[key], selected[key]
            if len(rows) != 10 - step:
                raise ValueError(f"Expected {10 - step} candidates at {key}")
            additions = {r["added_ratio"] for r in rows}
            if additions != set(RATIOS) - previous:
                raise ValueError(f"Unexpected remaining ratios at {key}")
            for row in rows:
                training = {ratio.strip() for ratio in row["training_ratios"].split(",")}
                if training != previous | {row["added_ratio"]}:
                    raise ValueError(f"Candidate does not extend the greedy path: {row['candidate_id']}")
            matches = [r for r in rows if r["candidate_id"] == choice["candidate_id"]]
            flags = [r for r in rows if r["selected_at_step"] is True]
            if len(matches) != 1 or len(flags) != 1 or flags[0] != matches[0]:
                raise ValueError(f"Inconsistent selected candidate at {key}")
            match = matches[0]
            for field in ("fit_code", "added_ratio", "all_ratio_pooled_R2"):
                if match[field] != choice[field]:
                    raise ValueError(f"Selection/candidate mismatch in {field} at {key}")
            maximum = max(float(r["all_ratio_pooled_R2"]) for r in rows)
            if not math.isclose(float(choice["all_ratio_pooled_R2"]), maximum, abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Selection is not a batch maximum at {key}")
            if set(choice["training_ratios"]) != previous | {choice["added_ratio"]}:
                raise ValueError(f"Selected training set is inconsistent at {key}")
            previous.add(choice["added_ratio"])
    return dict(grouped), selected


def build_figure(grouped: dict, selected: dict) -> tuple[Any, dict[str, int]]:
    """Draw scores and cumulative bars labelled only with the newly added ratio."""
    if not 0 < CANDIDATE_ALPHA <= 1 or not 0 <= POINT_SPREAD < 0.5:
        raise ValueError("Opacity must be in (0, 1] and horizontal spread in [0, 0.5)")
    if not 0 < RATIO_BAR_WIDTH <= 1 or min(SCORE_PANEL_HEIGHT_INCHES, RATIO_PANEL_HEIGHT_INCHES) <= 0:
        raise ValueError("Bar width must be in (0, 1] and panel heights positive")
    if not 0 <= RATIO_OVERLAP_INCHES < min(SCORE_PANEL_HEIGHT_INCHES, RATIO_PANEL_HEIGHT_INCHES):
        raise ValueError("Overlap must be nonnegative and smaller than both panel heights")
    if PANEL_ROW_GAP_INCHES < 0 or TOP_MARGIN_INCHES < 0:
        raise ValueError("Row spacing and top margin must be nonnegative")
    if RATIO_LABEL_PAD < 0 or RATIO_AXIS_TOP <= 9 + RATIO_LABEL_PAD:
        raise ValueError("Ratio labels need nonnegative padding and headroom above B9")
    if set(MANUAL_RATIO_COLORS) != set(RATIOS):
        raise ValueError("The manual palette must cover exactly the nine study ratios")
    plt.rcParams.update({"font.family": FONT_FAMILY, "mathtext.fontset": MATH_FONTSET,
                         "font.size": TICK_FONT_SIZE,
                         # Preserve the approved font in SVG paths and embedded PDF TrueType fonts.
                         "svg.fonttype": "path", "pdf.fonttype": 42})
    figure = plt.figure(figsize=FIGURE_SIZE)
    # Keep the approved panel widths while matching the manual PDF's physical vertical geometry.
    horizontal_grid = figure.add_gridspec(1, 2, left=0.075, right=0.99, wspace=0.09)
    figure_height = FIGURE_SIZE[1]
    combined_height = SCORE_PANEL_HEIGHT_INCHES + RATIO_PANEL_HEIGHT_INCHES - RATIO_OVERLAP_INCHES
    row_pitch = combined_height + PANEL_ROW_GAP_INCHES
    bottom_clearance = figure_height - TOP_MARGIN_INCHES - row_pitch - combined_height
    if bottom_clearance <= 0:
        plt.close(figure)
        raise ValueError("FIGURE_SIZE height is too small for the panel heights, overlap, and margins")
    counts = {"main_candidates": 0, "negative_candidates": 0, "selected": 0}
    main_axes = []
    for index, (knit, title, color) in enumerate(KNIT_PANELS):
        column = horizontal_grid[0, index % 2].get_position(figure)
        score_top = figure_height - TOP_MARGIN_INCHES - (index // 2) * row_pitch
        score_bottom = score_top - SCORE_PANEL_HEIGHT_INCHES
        ratio_bottom = score_bottom + RATIO_OVERLAP_INCHES - RATIO_PANEL_HEIGHT_INCHES
        axis = figure.add_axes((column.x0, score_bottom / figure_height, column.width,
                                SCORE_PANEL_HEIGHT_INCHES / figure_height),
                               sharey=main_axes[0] if main_axes else None, zorder=2)
        main_axes.append(axis)
        ratio_axis = figure.add_axes((column.x0, ratio_bottom / figure_height, column.width,
                                      RATIO_PANEL_HEIGHT_INCHES / figure_height), sharex=axis, zorder=3)
        # Hide only the ratio-axis background so the underlying scatter stays visible.
        ratio_axis.patch.set_visible(False)
        axis.set_title(title, fontsize=TITLE_FONT_SIZE, color=TITLE_FONT_COLOR,
                       fontweight=TITLE_FONT_WEIGHT, pad=TITLE_PAD)
        axis.set(xlim=(0.45, 9.55), ylim=(0, 1), yticks=(0, 0.5, 1))
        axis.tick_params(axis="x", bottom=False, labelbottom=False)
        axis.tick_params(axis="y", labelleft=index % 2 == 0, width=0.5, length=2)
        axis.grid(axis="y", color="0.9", linewidth=0.4)
        for spine in ("top", "right", "bottom"):
            axis.spines[spine].set_visible(False)
        axis.spines["left"].set_linewidth(0.5)
        ratio_axis.set(ylim=(0, RATIO_AXIS_TOP), yticks=[])
        for spine in ratio_axis.spines.values():
            spine.set_visible(False)
        # Batch labels appear only under the second row; bar-top ratio labels remain.
        ratio_axis.tick_params(axis="x", bottom=False, labelbottom=index >= 2,
                               pad=3, labelsize=TICK_FONT_SIZE)
        choices, training_sequence = [], []
        for step in range(1, 10):
            key = (knit, f"B{step:02d}")
            rows = sorted(grouped[key], key=lambda r: RATIOS.index(r["added_ratio"]))
            offsets = np.linspace(-POINT_SPREAD, POINT_SPREAD, len(rows)) if len(rows) > 1 else [0.0]
            choice = selected[key]
            negatives = [r for r in rows if float(r["all_ratio_pooled_R2"]) < 0]
            # Give collapsed negatives the full batch width so multiple triangles remain visible.
            negative_offsets = np.linspace(-POINT_SPREAD, POINT_SPREAD, len(negatives)) if len(negatives) > 1 else [0.0]
            negative_positions = {r["candidate_id"]: step + offset for r, offset in zip(negatives, negative_offsets)}
            for row, offset in zip(rows, offsets):
                value = float(row["all_ratio_pooled_R2"])
                is_selected = row["candidate_id"] == choice["candidate_id"]
                if value < 0:
                    dots = axis.scatter(negative_positions[row["candidate_id"]], NEGATIVE_MARKER_Y, marker="v",
                                        s=CANDIDATE_SIZE, color=color, alpha=CANDIDATE_ALPHA,
                                        edgecolors="none", zorder=3)
                    counts["negative_candidates"] += 1
                else:
                    # The selected candidate stays centered so the curve follows B1–B9.
                    dots = axis.scatter(step if is_selected else step + offset, value,
                                        s=SELECTED_SIZE if is_selected else CANDIDATE_SIZE,
                                        color=color, alpha=1.0 if is_selected else CANDIDATE_ALPHA,
                                        edgecolors="0.25" if is_selected else "none",
                                        linewidths=0.35, zorder=4 if is_selected else 3)
                    counts["main_candidates"] += 1
                dots.set_gid(f"candidate:{row['candidate_id']}")
            if negatives:
                axis.text(step + NEGATIVE_LABEL_X_OFFSET, NEGATIVE_LABEL_Y, "<0", ha="left", va="bottom",
                          fontsize=NEGATIVE_FONT_SIZE)
            choices.append(float(choice["all_ratio_pooled_R2"]))
            ratio = physical_ratio(knit, choice["added_ratio"])
            training_sequence.append(ratio)
            # Retain all previous segments, adding the new ratio at the top of this bar.
            for stack_index, included_ratio in enumerate(training_sequence):
                segment = ratio_axis.bar(step, 1, bottom=stack_index, width=RATIO_BAR_WIDTH,
                                         color=MANUAL_RATIO_COLORS[included_ratio], edgecolor="white",
                                         linewidth=0.25)[0]
                segment.set_gid(f"ratio:{knit}:B{step:02d}:{included_ratio}")
            ratio_label = ratio_axis.text(step, step + RATIO_LABEL_PAD, ratio, ha="center", va="bottom",
                                         fontsize=RATIO_FONT_SIZE)
            ratio_label.set_gid(f"ratio_label:{knit}:B{step:02d}:{ratio}")
            counts["selected"] += 1
        axis.plot(range(1, 10), choices, color=color, linewidth=LINE_WIDTH, zorder=2)
        ratio_axis.set_xticks(range(1, 10), [f"B{step}" for step in range(1, 10)])
    figure.text(0.015, 0.55, "All-nine pooled $R^2$", rotation=90, va="center",
                fontsize=AXIS_LABEL_FONT_SIZE, color=AXIS_LABEL_FONT_COLOR)
    if counts != {"main_candidates": 171, "negative_candidates": 9, "selected": 36}:
        raise ValueError(f"Unexpected rendering counts for the audited v1 campaign: {counts}")
    return figure, counts


def save_figure(figure: Any) -> list[Path]:
    """Write only the enabled formats; skip directory creation if saving is disabled."""
    enabled_formats = [extension for extension, enabled in
                       (("png", SAVE_PNG), ("svg", SAVE_SVG), ("pdf", SAVE_PDF)) if enabled]
    if not enabled_formats:
        return []
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for extension in enabled_formats:
        path = OUTPUT_DIRECTORY / f"{OUTPUT_BASENAME}.{extension}"
        figure.savefig(path, **({"dpi": DPI} if extension == "png" else {}))
        saved_paths.append(path)
    return saved_paths


def main() -> None:
    """Save the selected formats, optionally display, and print the suggested caption."""
    grouped, selected = load_report()
    figure, counts = build_figure(grouped, selected)
    saved_paths = save_figure(figure)
    for path in saved_paths:
        print(f"Saved: {path}")
    if not saved_paths:
        print("Saving disabled for PNG, SVG, and PDF; no files written.")
    print(f"Rendered: {counts}\nSuggested caption: {CAPTION}")
    if SHOW_PLOT:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()

