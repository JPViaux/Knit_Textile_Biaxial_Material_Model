"""Create the manuscript pooled-R-squared comparison figure from its source table.

Edit the configuration values below in VS Code when the manuscript table or
output location changes, then run this file directly.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (INPUT_ROOT as RELEASE_INPUT_ROOT,
    FIGURE_ROOT as RELEASE_FIGURE_ROOT)

from collections.abc import Mapping, Sequence
from pathlib import Path
import math
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from openpyxl import load_workbook


CURRENT_FILE = Path(__file__).resolve()
SUPPORT_ROOT = next(parent for parent in CURRENT_FILE.parents if parent.name == "Analysis").parent
CODE_ROOT = SUPPORT_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

import uml_plot_style as uml_style


uml_style.apply()

# Workbook containing the manuscript table. Example: PROJECT_ROOT / "6_Manuscript/Tables/file.xlsx".
WORKBOOK_PATH = RELEASE_INPUT_ROOT / "Biaxial_paper_tables_exp_polycon.xlsx"
# Worksheet containing one row per knit and calibration--fitting ratio-count pair. Example: "Data".
WORKSHEET_NAME = "Data"
# Final editable vector figure path. Example: RELEASE_FIGURE_ROOT / "figure_06/figure.svg".
OUTPUT_PATH = RELEASE_FIGURE_ROOT / "figure_06" / "pooled_r2_by_calibration_fitting_pair.svg"
# Also save a PDF beside the SVG with the same base filename. Example: False saves only the SVG.
SAVE_PDF = True
# Open the interactive plot window after saving. Example: False for batch or validation runs.
SHOW_PLOT = True
# Display the shared x-axis title. Example: False hides the title completely.
SHOW_X_AXIS_TITLE = False
# Shared x-axis title text when SHOW_X_AXIS_TITLE is True. Example: "Knit type".
X_AXIS_TITLE = "Knit type"
# Display the individual subplot titles. Example: False hides every pair title.
SHOW_SUBPLOT_TITLES = True
# One title per subplot, in PAIR_DEFINITIONS order. Example: ("3–3", "3–6", "3–9", "9–9").
SUBPLOT_TITLES = ("C3 – E3", "C3 – E6", "C3 – E9", "C9 – E9")

# Left-to-right panel order and the color assigned to each knit type. Example: ("Interlock", "#0072B2").
KNIT_PANELS = (
    ("Interlock", "#3BD5AE"),
    ("Purl", "#62DAFC"),
    ("Rib", "#FFD140"),
    ("Single Jersey", "#FB471F"),
)
# X-axis order and visible label for each calibration-ratio--fitted-ratio count pair. Example: ((3, 6), "3–6").
PAIR_DEFINITIONS = (
    ((3, 3), "C3 – E3"),
    ((3, 6), "C3 – E6"),
    ((3, 9), "C3 – E9"),
    ((9, 9), "C9 – E9"),
)

# Compact, full-width manuscript figure dimensions in inches. Example: (7.2, 2.2).
FIGURE_SIZE_INCHES = (7.2, 1.5)
# Bar width in x-axis category units. Example: 0.68.
BAR_WIDTH = 0.68
# Show each pooled R-squared value above its bar. Example: False hides all value labels.
SHOW_BAR_VALUE_LABELS = True
# Number of decimal places in each value label. Example: 3 displays 0.977.
BAR_VALUE_DECIMAL_PLACES = 2
# Value-label text size in points. Example: 6.5.
BAR_VALUE_LABEL_FONT_SIZE = 9
# Value-label text color. Example: "#333333".
BAR_VALUE_LABEL_COLOR = "#333333"
# Gap between each bar and its label in points. Example: 2.0.
BAR_VALUE_LABEL_PADDING_POINTS = 2.0
# Extra y-axis space reserved above 1.0 for the labels. Example: 0.08.
BAR_VALUE_LABEL_HEADROOM = 0.1
# Shared pooled-R-squared limits. Example: (0.0, 1.0) preserves the bar-chart baseline.
Y_LIMITS = (0.0, 1.0)
# Shared R-squared tick positions. Example: (0.0, 0.25, 0.5, 0.75, 1.0).
Y_TICKS = (0.0, 0.25, 0.50, 0.75, 1.0)
# Text sizes in points. Example: panel title 8.0 pt and tick labels 7.0 pt.
PANEL_TITLE_FONT_SIZE = 13.0
AXIS_LABEL_FONT_SIZE = 11.0
TICK_FONT_SIZE = 9.0
# One title font size per subplot, in left-to-right order. Example: (8.0, 8.0, 8.0, 8.0).
SUBPLOT_TITLE_FONT_SIZES = (PANEL_TITLE_FONT_SIZE,) * len(PAIR_DEFINITIONS)
# One title font color per subplot. Example: ("#00274C", "#00274C", "#00274C", "#00274C"); None uses the style default.
SUBPLOT_TITLE_FONT_COLORS = ("#333333",) * len(PAIR_DEFINITIONS)
# One knit-label font size per subplot, in left-to-right order. Example: (7.0, 7.0, 7.0, 7.0).
SUBPLOT_X_LABEL_FONT_SIZES = (TICK_FONT_SIZE,) * len(PAIR_DEFINITIONS)
# One knit-label font color per subplot. Example: ("#000000", "#000000", "#000000", "#000000"); None uses the style default.
SUBPLOT_X_LABEL_FONT_COLORS = (None,) * len(PAIR_DEFINITIONS)
# Axis and grid stroke widths in points. Example: 0.7 and 0.45.
AXIS_LINE_WIDTH = 0.15
GRID_LINE_WIDTH = 0.0

REQUIRED_COLUMNS = {"Knit", "# Calibration Ratios", "# Fitted Ratios", "Pooled R2"}


def read_data_rows(
    workbook_path: Path = WORKBOOK_PATH,
    worksheet_name: str = WORKSHEET_NAME,
) -> list[dict[str, Any]]:
    """Read nonblank `Data` rows without altering the source workbook."""
    workbook_path = Path(workbook_path)
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Workbook was not found: {workbook_path}")

    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    if worksheet_name not in workbook.sheetnames:
        raise ValueError(
            f"Worksheet {worksheet_name!r} was not found. Available sheets: {workbook.sheetnames}"
        )

    worksheet = workbook[worksheet_name]
    values = worksheet.iter_rows(values_only=True)
    headers = next(values, None)
    if headers is None:
        raise ValueError(f"Worksheet {worksheet_name!r} is empty")

    header_names = [str(header).strip() if header is not None else "" for header in headers]
    if any(not header for header in header_names):
        raise ValueError(f"Worksheet {worksheet_name!r} contains a blank header")
    missing_columns = REQUIRED_COLUMNS - set(header_names)
    if missing_columns:
        raise ValueError(f"Worksheet {worksheet_name!r} is missing columns: {sorted(missing_columns)}")

    rows: list[dict[str, Any]] = []
    for row_values in values:
        if not any(value not in (None, "") for value in row_values):
            continue
        padded_values = tuple(row_values) + (None,) * (len(header_names) - len(row_values))
        rows.append(dict(zip(header_names, padded_values[: len(header_names)], strict=True)))
    return rows


def select_pooled_r2_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    """Return one validated pooled R-squared value per requested knit and pair."""
    expected_pairs = [pair for pair, _label in PAIR_DEFINITIONS]
    values_by_knit: dict[str, list[float]] = {}

    for knit_name, _color in KNIT_PANELS:
        knit_rows = [row for row in rows if str(row.get("Knit", "")).strip() == knit_name]
        selected_values: list[float] = []
        for calibration_count, fitted_count in expected_pairs:
            matches = [
                row
                for row in knit_rows
                if integer_value(row.get("# Calibration Ratios"), "# Calibration Ratios", row)
                == calibration_count
                and integer_value(row.get("# Fitted Ratios"), "# Fitted Ratios", row) == fitted_count
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one row for knit={knit_name!r}, "
                    f"calibration–fitted pair={calibration_count}–{fitted_count}; found {len(matches)}."
                )
            pooled_r2 = float_value(matches[0].get("Pooled R2"), "Pooled R2", matches[0])
            if not Y_LIMITS[0] <= pooled_r2 <= Y_LIMITS[1]:
                raise ValueError(
                    f"Pooled R2 must be within {Y_LIMITS}, found {pooled_r2} for "
                    f"knit={knit_name!r}, pair={calibration_count}–{fitted_count}."
                )
            selected_values.append(pooled_r2)
        values_by_knit[knit_name] = selected_values

    expected_row_count = len(KNIT_PANELS) * len(PAIR_DEFINITIONS)
    if len(rows) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} nonblank Data rows, found {len(rows)}. "
            "Update the panel or pair configuration if the table was intentionally expanded."
        )
    return values_by_knit


def integer_value(value: Any, column_name: str, row: Mapping[str, Any]) -> int:
    """Return an integer count with a contextual error for malformed table cells."""
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column_name!r} value {value!r} in row {dict(row)!r}") from exc
    if integer != value:
        raise ValueError(f"Noninteger {column_name!r} value {value!r} in row {dict(row)!r}")
    return integer


def float_value(value: Any, column_name: str, row: Mapping[str, Any]) -> float:
    """Return a finite numeric metric with a contextual error for blank table cells."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column_name!r} value {value!r} in row {dict(row)!r}") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"Nonfinite {column_name!r} value {value!r} in row {dict(row)!r}")
    return numeric_value


def create_figure(values_by_knit: Mapping[str, Sequence[float]]) -> Figure:
    """Create four calibration--fitted-pair panels with knit-type bars."""
    validate_subplot_settings()
    knit_names = [knit_name for knit_name, _color in KNIT_PANELS]
    # Use uppercase initials in the compact panels. Examples: "Interlock" -> "I"; "Single Jersey" -> "SJ".
    knit_labels = ["".join(word[0] for word in knit_name.split()).upper() for knit_name in knit_names]
    knit_colors = [color for _knit_name, color in KNIT_PANELS]
    x_positions = list(range(len(knit_names)))
    figure, axes = plt.subplots(
        1,
        len(PAIR_DEFINITIONS),
        figsize=FIGURE_SIZE_INCHES,
        sharey=True,
        layout="constrained",
    )

    for pair_index, (axis, (_pair, _pair_label)) in enumerate(
        zip(axes, PAIR_DEFINITIONS, strict=True)
    ):
        pair_values = [values_by_knit[knit_name][pair_index] for knit_name in knit_names]
        bars = axis.bar(
            x_positions,
            pair_values,
            width=BAR_WIDTH,
            color=knit_colors,
            edgecolor="none",
        )
        if SHOW_BAR_VALUE_LABELS:
            axis.bar_label(
                bars,
                labels=[f"{value:.{BAR_VALUE_DECIMAL_PLACES}f}" for value in pair_values],
                padding=BAR_VALUE_LABEL_PADDING_POINTS,
                fontsize=BAR_VALUE_LABEL_FONT_SIZE,
                color=BAR_VALUE_LABEL_COLOR,
            )
        if SHOW_SUBPLOT_TITLES:
            title_options: dict[str, Any] = {
                "fontsize": SUBPLOT_TITLE_FONT_SIZES[pair_index],
                "fontweight": "normal",
                "pad": 3,
            }
            title_color = SUBPLOT_TITLE_FONT_COLORS[pair_index]
            if title_color is not None:
                title_options["color"] = title_color
            axis.set_title(SUBPLOT_TITLES[pair_index], **title_options)
        axis.set_xticks(x_positions, knit_labels)
        axis.set_ylim(Y_LIMITS[0], Y_LIMITS[1] + BAR_VALUE_LABEL_HEADROOM)
        axis.set_yticks(Y_TICKS)
        axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE, width=AXIS_LINE_WIDTH, length=2.5)
        x_label_color = SUBPLOT_X_LABEL_FONT_COLORS[pair_index]
        for label in axis.get_xticklabels():
            label.set_fontsize(SUBPLOT_X_LABEL_FONT_SIZES[pair_index])
            if x_label_color is not None:
                label.set_color(x_label_color)
        axis.set_axisbelow(True)
        uml_style.style_axes(axis)
        # Apply this after the shared style, which otherwise enables both grid directions.
        axis.grid(True, axis="y", color="#878A8F", linewidth=GRID_LINE_WIDTH)
        axis.grid(False, axis="x")
        for spine in axis.spines.values():
            spine.set_linewidth(AXIS_LINE_WIDTH)

    # A label on the first shared axis reserves margin without repeating it on every panel.
    axes[0].set_ylabel(r"Pooled $R^2$", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=5)
    if SHOW_X_AXIS_TITLE:
        # This position remains inside the constrained figure canvas with compact x labels.
        figure.supxlabel(X_AXIS_TITLE, fontsize=AXIS_LABEL_FONT_SIZE, y=0.01)
    return figure


def validate_subplot_settings() -> None:
    """Ensure every per-subplot title and label setting matches the panel count."""
    expected_length = len(PAIR_DEFINITIONS)
    settings = {
        "SUBPLOT_TITLES": SUBPLOT_TITLES,
        "SUBPLOT_TITLE_FONT_SIZES": SUBPLOT_TITLE_FONT_SIZES,
        "SUBPLOT_TITLE_FONT_COLORS": SUBPLOT_TITLE_FONT_COLORS,
        "SUBPLOT_X_LABEL_FONT_SIZES": SUBPLOT_X_LABEL_FONT_SIZES,
        "SUBPLOT_X_LABEL_FONT_COLORS": SUBPLOT_X_LABEL_FONT_COLORS,
    }
    for setting_name, setting_values in settings.items():
        if len(setting_values) != expected_length:
            raise ValueError(
                f"{setting_name} must contain one value for each PAIR_DEFINITIONS entry."
            )


def main() -> None:
    """Read the manuscript table, save the SVG, and optionally display the figure."""
    values_by_knit = select_pooled_r2_values(read_data_rows())
    figure = create_figure(values_by_knit)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="svg", transparent=True)
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
