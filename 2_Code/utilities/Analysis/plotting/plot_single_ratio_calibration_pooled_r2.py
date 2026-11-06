"""Plot all-nine-ratio pooled R-squared after independent ratio calibrations.

Edit the configuration values below in VS Code, then run this file directly.
The audited greedy-study payload is opened read-only; this script never refits a
model or modifies the fit registry.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT,
    FIGURE_ROOT as RELEASE_FIGURE_ROOT)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from release_paths import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


CURRENT_FILE = Path(__file__).resolve()
SUPPORT_ROOT = next(parent for parent in CURRENT_FILE.parents if parent.name == "Analysis").parent
CODE_ROOT = SUPPORT_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

import uml_plot_style as uml_style


uml_style.apply()

# Audited output from the latest completed greedy study. Example: a replacement v2 payload.
GREEDY_REPORT_PAYLOAD = (
    RELEASE_ARCHIVE_ROOT
    / "adaptive_ratio_study_sv_exp_polycon_mo_v1"
    / "full_workbook_payload.json"
)
# Material model represented by the report. Example: "SV_exp_polycon_mo".
MODEL_ID = "SV_exp_polycon_mo"
# B01 contains the nine independent one-ratio calibrations. Example: "B01".
SINGLE_RATIO_BATCH = "B01"
# Editable vector output. Example: RELEASE_FIGURE_ROOT / "figure_08/plot.svg".
OUTPUT_PATH = (
    RELEASE_FIGURE_ROOT
    / "figure_08"
    / "single_ratio_calibration_all_nine_pooled_r2.svg"
)
# Save a PDF beside the SVG. Example: False saves only the SVG.
SAVE_PDF = True
# Open the interactive Matplotlib window after saving. Example: False for batch runs.
SHOW_PLOT = True
# Label nonnegative bars with pooled R-squared values. Example: False for no labels.
SHOW_R2_VALUE_LABELS = True
# Add the greedy-study fit code above each bar. Example: True for an audit figure.
SHOW_FIT_CODE_LABELS = False

# Full manuscript text-width figure size in inches. Example: (6.5, 4.2).
FIGURE_SIZE_INCHES = (6.5, 3)
# Width of each calibration-ratio bar. Example: 0.78 leaves a narrow gap.
BAR_WIDTH = 0.78
# Requested displayed pooled-R-squared range. Example: (0.0, 1.0).
Y_LIMITS = (0.0, 1.0)
# Shared pooled-R-squared ticks. Example: (0.0, 0.25, 0.5, 0.75, 1.0).
Y_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Rotation of lower-row calibration-ratio labels. Example: 45.0 degrees.
RATIO_LABEL_ROTATION = 45.0
# Editable plot typography in points. Example: 10.0 for compact panel titles.
PANEL_TITLE_FONT_SIZE = 12.0
PANEL_TITLE_FONT_COLOR = "#333333"
AXIS_LABEL_FONT_SIZE = 9.0
TICK_FONT_SIZE = 7.0
VALUE_LABEL_FONT_SIZE = 8.0
FIT_CODE_FONT_SIZE = 8
# Plot line widths in points. Example: 0.35 for a compact manuscript figure.
AXIS_LINE_WIDTH = 0.15
BAR_BORDER_WIDTH = 0.0
GRID_LINE_WIDTH = 0.0

# Common physical Course:wale calibration-ratio order used in every panel.
RATIO_ORDER = (
    "0:1",
    "0.25:1",
    "0.5:1",
    "0.75:1",
    "1:0",
    "1:0.25",
    "1:0.5",
    "1:0.75",
    "1:1",
)
# Interlock stores ratios as wale:course; this is a display-only axis conversion.
INTERLOCK_STUDY_RATIO_BY_PHYSICAL_RATIO = {
    "0:1": "1:0",
    "0.25:1": "1:0.25",
    "0.5:1": "1:0.5",
    "0.75:1": "1:0.75",
    "1:0": "0:1",
    "1:0.25": "0.25:1",
    "1:0.5": "0.5:1",
    "1:0.75": "0.75:1",
    "1:1": "1:1",
}
# Panel order, display names, and the knit colors used by the adaptive-ratio plotter.
KNIT_PANELS = (
    ("Interlock", "Interlock", "#3BD5AE"),
    ("Rib", "Rib", "#FFD140"),
    ("Purl", "Purl", "#62DAFC"),
    ("SingleJersey", "Single Jersey", "#FB471F"),
)


@dataclass(frozen=True)
class SingleRatioSelection:
    """Validated B01 pooled R-squared values and their greedy-study fit codes."""

    pooled_r2: dict[str, dict[str, float]]
    fit_codes: dict[str, dict[str, int]]


def read_audited_candidates(path: Path = GREEDY_REPORT_PAYLOAD) -> list[dict[str, Any]]:
    """Read candidate rows from the complete, audited greedy-study payload."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Greedy-study report was not found: {path}. Run the polycon finalizer first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != "adaptive_ratio_greedy_sv_exp_polycon_mo_v1":
        raise ValueError(f"Unexpected greedy study_id in {path}: {payload.get('study_id')!r}")
    audit = payload.get("audit")
    if not isinstance(audit, dict) or audit.get("status") != "passed":
        raise ValueError(f"Greedy-study payload is not audited as passed: {path}")
    sheets = payload.get("sheets")
    candidates = sheets.get("Candidates") if isinstance(sheets, dict) else None
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Greedy-study payload has no Candidates rows: {path}")
    return candidates


def require_columns(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require the fields needed to identify fits and plot all-nine performance."""
    required = {
        "batch",
        "knit",
        "added_ratio",
        "training_ratios",
        "fit_code",
        "all_ratio_pooled_R2",
    }
    if not rows:
        raise ValueError("The Candidates sheet contains no rows")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Candidates rows are missing required columns: {sorted(missing)}")


def select_single_ratio_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> SingleRatioSelection:
    """Select exactly nine independent B01 calibrations for each knit type."""
    require_columns(rows)
    selected_rows = [row for row in rows if str(row["batch"]) == SINGLE_RATIO_BATCH]
    expected_knits = {knit for knit, _display_name, _color in KNIT_PANELS}
    expected_count = len(expected_knits) * len(RATIO_ORDER)
    if len(selected_rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {SINGLE_RATIO_BATCH} candidates; "
            f"found {len(selected_rows)}"
        )

    pooled_r2 = {knit: {} for knit in expected_knits}
    fit_codes = {knit: {} for knit in expected_knits}
    used_fit_codes: set[int] = set()
    for row in selected_rows:
        knit = str(row["knit"])
        ratio = str(row["added_ratio"])
        if knit not in expected_knits:
            raise ValueError(f"Unexpected knit in {SINGLE_RATIO_BATCH}: {knit!r}")
        if ratio not in RATIO_ORDER:
            raise ValueError(f"Unexpected calibration ratio for {knit}: {ratio!r}")
        if ratio in pooled_r2[knit]:
            raise ValueError(f"Duplicate {SINGLE_RATIO_BATCH} candidate for {knit}, {ratio}")
        if str(row["training_ratios"]) != ratio:
            raise ValueError(
                f"Fit {row['fit_code']} is not an independent {ratio} calibration: "
                f"training_ratios={row['training_ratios']!r}"
            )

        fit_code = int(row["fit_code"])
        value = float(row["all_ratio_pooled_R2"])
        if fit_code in used_fit_codes:
            raise ValueError(f"Duplicate fit code in {SINGLE_RATIO_BATCH}: {fit_code}")
        if not math.isfinite(value) or value > Y_LIMITS[1] + 1e-12:
            raise ValueError(f"Invalid all-nine pooled R2 for fit {fit_code}: {value}")
        used_fit_codes.add(fit_code)
        pooled_r2[knit][ratio] = value
        fit_codes[knit][ratio] = fit_code

    for knit in expected_knits:
        missing_ratios = set(RATIO_ORDER).difference(pooled_r2[knit])
        if missing_ratios:
            raise ValueError(f"Missing {SINGLE_RATIO_BATCH} ratios for {knit}: {sorted(missing_ratios)}")
    return SingleRatioSelection(pooled_r2=pooled_r2, fit_codes=fit_codes)


def study_ratio_for_physical_ratio(knit: str, physical_ratio: str) -> str:
    """Map one physical Course:wale label to the stored greedy-study ratio."""
    if knit == "Interlock":
        return INTERLOCK_STUDY_RATIO_BY_PHYSICAL_RATIO[physical_ratio]
    return physical_ratio


def create_figure(selection: SingleRatioSelection) -> Figure:
    """Create the four-panel, nine-bar pooled-R-squared figure."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=FIGURE_SIZE_INCHES,
        sharex=True,
        sharey=True,
    )
    x_positions = list(range(len(RATIO_ORDER)))

    for panel_index, (axis, (knit, display_name, color)) in enumerate(
        zip(axes.flat, KNIT_PANELS, strict=True)
    ):
        stored_ratios = [
            study_ratio_for_physical_ratio(knit, physical_ratio)
            for physical_ratio in RATIO_ORDER
        ]
        raw_values = [selection.pooled_r2[knit][ratio] for ratio in stored_ratios]
        displayed_values = [max(Y_LIMITS[0], value) for value in raw_values]
        bars = axis.bar(
            x_positions,
            displayed_values,
            width=BAR_WIDTH,
            color=color,
            edgecolor="black",
            linewidth=BAR_BORDER_WIDTH,
        )

        for x_position, bar, raw_value, stored_ratio in zip(
            x_positions, bars, raw_values, stored_ratios, strict=True
        ):
            if raw_value < Y_LIMITS[0]:
                # A baseline marker communicates a negative raw value without expanding
                # the requested 0--1 display range, for example pooled R2 = -0.34.
                axis.scatter(
                    [x_position],
                    [0.018],
                    marker="v",
                    s=12,
                    color=color,
                    edgecolor="black",
                    linewidth=BAR_BORDER_WIDTH,
                    zorder=3,
                )
                axis.text(
                    x_position,
                    0.045,
                    "<0",
                    ha="center",
                    va="bottom",
                    fontsize=VALUE_LABEL_FONT_SIZE,
                )
            elif SHOW_R2_VALUE_LABELS:
                axis.text(
                    x_position,
                    min(raw_value + 0.018, 0.985),
                    f"{raw_value:.2f}",
                    ha="center",
                    va="top" if raw_value > 0.95 else "bottom",
                    fontsize=VALUE_LABEL_FONT_SIZE,
                )
            if SHOW_FIT_CODE_LABELS:
                axis.text(
                    x_position,
                    0.11,
                    f"fit {selection.fit_codes[knit][stored_ratio]}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=FIT_CODE_FONT_SIZE,
                )

        axis.set_title(
            display_name,
            fontsize=PANEL_TITLE_FONT_SIZE,
            fontweight="normal",
            color=PANEL_TITLE_FONT_COLOR,
            pad=4,
        )
        axis.set_xticks(x_positions, RATIO_ORDER)
        axis.set_ylim(*Y_LIMITS)
        axis.set_yticks(Y_TICKS)
        axis.tick_params(
            axis="both",
            labelsize=TICK_FONT_SIZE,
            width=AXIS_LINE_WIDTH,
            length=2.5,
        )
        if panel_index < 2:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            for label in axis.get_xticklabels():
                label.set_rotation(RATIO_LABEL_ROTATION)
                label.set_horizontalalignment("right")
                label.set_rotation_mode("anchor")

        axis.set_axisbelow(True)
        uml_style.style_axes(axis)
        # Retain only horizontal pooled-R-squared guides after shared styling.
        axis.grid(True, axis="y", color="#878A8F", linewidth=GRID_LINE_WIDTH, alpha=0.65)
        axis.grid(False, axis="x")
        for spine in axis.spines.values():
            spine.set_linewidth(AXIS_LINE_WIDTH)

    figure.supylabel(r"All-nine-ratio pooled $R^2$", fontsize=AXIS_LABEL_FONT_SIZE, x=0.015)
    #figure.supxlabel("Course:wale calibration ratio", fontsize=AXIS_LABEL_FONT_SIZE, y=0.035)
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.16,
        top=0.94,
        hspace=0.28,
        wspace=0.12,
    )
    return figure


def print_fit_inventory(selection: SingleRatioSelection) -> None:
    """Print the stored study ratio, fit code, and raw pooled R-squared value."""
    for knit, _display_name, _color in KNIT_PANELS:
        print(f"{knit}:")
        for physical_ratio in RATIO_ORDER:
            stored_ratio = study_ratio_for_physical_ratio(knit, physical_ratio)
            print(
                f"  Course:wale {physical_ratio}: fit {selection.fit_codes[knit][stored_ratio]}, "
                f"all-nine pooled R2={selection.pooled_r2[knit][stored_ratio]:.12g}"
            )


def main() -> None:
    """Read, validate, plot, save, and optionally display the figure."""
    selection = select_single_ratio_candidates(read_audited_candidates())
    figure = create_figure(selection)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="svg", transparent=True)
    print_fit_inventory(selection)
    print(f"Saved figure: {OUTPUT_PATH}")
    if SAVE_PDF:
        pdf_output_path = OUTPUT_PATH.with_suffix(".pdf")
        figure.savefig(pdf_output_path, format="pdf", transparent=True)
        print(f"Saved figure: {pdf_output_path}")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
