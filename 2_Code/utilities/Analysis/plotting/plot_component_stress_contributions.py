"""Plot SV_exp_polycon_mo PK1 or PK2 component percentages for all ratios.

Edit the configuration values below in VS Code, then run this file directly.
The source CSV is opened read-only and is never modified.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT,
    GENERATED_ROOT as RELEASE_GENERATED_ROOT)

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch


CURRENT_FILE = Path(__file__).resolve()
SUPPORT_ROOT = next(parent for parent in CURRENT_FILE.parents if parent.name == "Analysis").parent
CODE_ROOT = SUPPORT_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

import uml_plot_style as uml_style


uml_style.apply()

# Source table. Example: SUPPORT_ROOT / "Outputs/study/results.csv".
SOURCE_CSV_PATH = (
    RELEASE_ARCHIVE_ROOT
    / "model_component_contributions_sv_exp_polycon_mo"
    / "component_stress_contributions.csv"
)
# Fixed model selector. Example: "SV_exp_polycon_mo".
MODEL_ID = "SV_exp_polycon_mo"
# Stress measure selector. Examples: "PK1" or "PK2".
STRESS_MEASURE = "PK2"
# Normalization selector. Examples: "per_ratio" or "all_ratios".
NORMALIZATION_SCOPE = "all_ratios"
# Final editable vector figure. Example: PROJECT_ROOT / "6_Manuscript/Figures/figure.svg".
OUTPUT_PATH = (
    RELEASE_GENERATED_ROOT
    / "figures"
    / "Fig_7_Component_contributions"
    / (
        "component_stress_contributions_"
        f"{STRESS_MEASURE.lower()}_sv_exp_polycon_mo_{NORMALIZATION_SCOPE}.svg"
    )
)
# Also save a PDF beside the SVG. Example: False saves only the SVG.
SAVE_PDF = True
# Open the interactive plot after saving. Example: False for automated validation.
SHOW_PLOT = True

# Full manuscript text-width figure size in inches. Example: (6.5, 4.2).
FIGURE_SIZE_INCHES = (6.5, 3)
# Width of each stacked ratio bar. Example: 0.78 leaves a narrow gap between bars.
BAR_WIDTH = 0.78
# Shared percentage-axis limits. Example: (0.0, 100.0).
Y_LIMITS = (0.0, 100.0)
# Shared percentage tick positions. Example: (0, 25, 50, 75, 100).
Y_TICKS = (0, 25, 50, 75, 100)
# Rotation of lower-row ratio labels in degrees. Example: 45.0.
RATIO_LABEL_ROTATION = 45.0
# Text sizes in points. Example: 10.0 for compact panel titles.
PANEL_TITLE_FONT_SIZE = 12.0
# Knit-panel title color. Example: "#333333" for neutral dark gray.
PANEL_TITLE_FONT_COLOR = "#333333"
# Shared physical ratio label. Example: "Course:wale loading ratio".
X_AXIS_LABEL = None
AXIS_LABEL_FONT_SIZE = 9.0
TICK_FONT_SIZE = 7.0
LEGEND_FONT_SIZE = 7.0
# Hatch applied to nonlinear components; examples: "///", "xxx", or "...".
NONLINEAR_HATCH = "///"
# Solid bar-border width in points; example: 0.7 remains visible in print.
COMPONENT_BORDER_WIDTH = 0.1
AXIS_LINE_WIDTH = 0.2
GRID_LINE_WIDTH = 0
# Numeric tolerance for each independently normalized 100% bar. Example: 1e-8.
PERCENT_SUM_TOLERANCE = 1e-8

# Common physical course:wale order used from left to right in every panel.
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
# Interlock stores ratios as wale:course, so translate the common physical labels
# back to the workbook's axis 1:axis 2 keys. Example: course:wale 0:1 -> 1:0.
INTERLOCK_WORKBOOK_RATIO_BY_PHYSICAL_RATIO = {
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
# Panel order and display names. Example: ("SingleJersey", "Single Jersey").
KNIT_PANELS = (
    ("Interlock", "Interlock"),
    ("Rib", "Rib"),
    ("Purl", "Purl"),
    ("SingleJersey", "Single Jersey"),
)
# Reviewed all-nine fit codes, one complete nine-ratio fit per knit.
EXPECTED_FIT_CODES = {
    "Interlock": 1962,
    "Rib": 1961,
    "Purl": 1963,
    "SingleJersey": 1964,
}

# Use hatch for patterns in nonlinear bar plots instead of discontinuous border lines

@dataclass(frozen=True)
class ComponentDefinition:
    """Workbook selector and visual encoding for one stacked component."""

    linearity: str
    workbook_direction: str
    label: str
    color: str
    hatch: str

    @property
    def key(self) -> tuple[str, str]:
        """Return the exact workbook classification key."""
        return self.linearity, self.workbook_direction


# Fill identifies direction; hatching identifies nonlinear terms.
COMPONENT_DEFINITIONS = (
    ComponentDefinition("Linear", "Course", "Linear - course", "#1257D1", ""),
    ComponentDefinition(
        "Nonlinear", "Course", "Nonlinear - course", "#1257D1", NONLINEAR_HATCH
    ),
    ComponentDefinition("Linear", "Wales", "Linear - wale", "#878A8F", ""),
    ComponentDefinition(
        "Nonlinear", "Wales", "Nonlinear - wale", "#878A8F", NONLINEAR_HATCH
    ),
    ComponentDefinition("Linear", "Coupling", "Linear - coupling", "#D0AF22", ""),
    ComponentDefinition(
        "Nonlinear",
        "Coupling",
        "Nonlinear - coupling",
        "#D0AF22",
        NONLINEAR_HATCH,
    ),
)

REQUIRED_COLUMNS = {
    "model_id",
    "fit_code",
    "knit",
    "ratio",
    "Linear/Nonlinear",
    "Direction",
    "stress_contribution_percent",
    "stress_measure",
    "normalization_scope",
}

ContributionValues = dict[str, dict[str, dict[tuple[str, str], float]]]


@dataclass(frozen=True)
class ContributionSelection:
    """Validated contribution percentages and one selected fit code per knit."""

    values: ContributionValues
    fit_codes: dict[str, int]


def read_contribution_rows(
    source_path: Path = SOURCE_CSV_PATH,
) -> list[dict[str, Any]]:
    """Read nonblank contribution rows without altering the source CSV."""
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Contribution CSV was not found: {source_path}")
    with source_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        header_names = [str(header).strip() for header in (reader.fieldnames or ())]
        if not header_names:
            raise ValueError(f"Contribution CSV is empty: {source_path}")
        if any(not header for header in header_names):
            raise ValueError(f"Contribution CSV contains a blank header: {source_path}")
        missing_columns = REQUIRED_COLUMNS - set(header_names)
        if missing_columns:
            raise ValueError(
                f"Contribution CSV is missing columns: {sorted(missing_columns)}"
            )
        return [
            dict(row)
            for row in reader
            if any(value not in (None, "") for value in row.values())
        ]


def select_component_percentages(
    rows: Sequence[Mapping[str, Any]],
) -> ContributionSelection:
    """Select and validate one stress measure and normalization result."""
    if STRESS_MEASURE not in {"PK1", "PK2"}:
        raise ValueError(
            f"STRESS_MEASURE must be 'PK1' or 'PK2', got {STRESS_MEASURE!r}."
        )
    selected_rows = [
        row
        for row in rows
        if str(row.get("model_id", "")).strip() == MODEL_ID
        and str(row.get("stress_measure", "")).strip() == STRESS_MEASURE
        and str(row.get("normalization_scope", "")).strip()
        == NORMALIZATION_SCOPE
    ]
    expected_row_count = len(KNIT_PANELS) * len(RATIO_ORDER) * len(COMPONENT_DEFINITIONS)
    if len(selected_rows) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} {STRESS_MEASURE} {MODEL_ID} rows "
            f"with normalization_scope={NORMALIZATION_SCOPE!r}, "
            f"found {len(selected_rows)}."
        )

    expected_knits = {knit for knit, _display_name in KNIT_PANELS}
    expected_ratios = set(RATIO_ORDER)
    expected_components = {definition.key for definition in COMPONENT_DEFINITIONS}
    values: ContributionValues = {
        knit: {ratio: {} for ratio in RATIO_ORDER} for knit in expected_knits
    }
    fit_code_sets: dict[str, set[int]] = {knit: set() for knit in expected_knits}

    for row in selected_rows:
        knit = str(row.get("knit", "")).strip()
        ratio = str(row.get("ratio", "")).strip()
        component_key = (
            str(row.get("Linear/Nonlinear", "")).strip(),
            str(row.get("Direction", "")).strip(),
        )
        if knit not in expected_knits:
            raise ValueError(f"Unexpected knit in selected rows: {knit!r}")
        if ratio not in expected_ratios:
            raise ValueError(f"Unexpected ratio for knit {knit!r}: {ratio!r}")
        if component_key not in expected_components:
            raise ValueError(
                f"Unexpected component classification for knit={knit!r}, "
                f"ratio={ratio!r}: {component_key!r}"
            )
        if component_key in values[knit][ratio]:
            raise ValueError(
                f"Duplicate component for knit={knit!r}, ratio={ratio!r}: "
                f"{component_key!r}"
            )

        percentage = finite_float(
            row.get("stress_contribution_percent"),
            "stress_contribution_percent",
            row,
        )
        if percentage < 0.0:
            raise ValueError(
                f"Contribution percentages must be nonnegative; found {percentage} "
                f"for knit={knit!r}, ratio={ratio!r}, component={component_key!r}."
            )
        values[knit][ratio][component_key] = percentage
        fit_code_sets[knit].add(integer_value(row.get("fit_code"), "fit_code", row))

    fit_codes: dict[str, int] = {}
    for knit, _display_name in KNIT_PANELS:
        if fit_code_sets[knit] != {EXPECTED_FIT_CODES[knit]}:
            raise ValueError(
                f"Expected all-nine {MODEL_ID} fit {EXPECTED_FIT_CODES[knit]} "
                f"for {knit}, found {sorted(fit_code_sets[knit])}."
            )
        fit_codes[knit] = next(iter(fit_code_sets[knit]))
        for ratio in RATIO_ORDER:
            components = values[knit][ratio]
            missing_components = expected_components - set(components)
            if missing_components:
                raise ValueError(
                    f"Missing components for knit={knit!r}, ratio={ratio!r}: "
                    f"{sorted(missing_components)}"
                )
            percentage_sum = sum(components.values())
            if not math.isclose(
                percentage_sum,
                100.0,
                rel_tol=0.0,
                abs_tol=PERCENT_SUM_TOLERANCE,
            ):
                raise ValueError(
                    f"Component percentages must sum to 100 for knit={knit!r}, "
                    f"ratio={ratio!r}; found {percentage_sum:.12g}."
                )

    return ContributionSelection(values=values, fit_codes=fit_codes)


def finite_float(value: Any, column_name: str, row: Mapping[str, Any]) -> float:
    """Return a finite numeric cell value with contextual errors."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column_name!r} value {value!r} in row {dict(row)!r}") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"Nonfinite {column_name!r} value {value!r} in row {dict(row)!r}")
    return numeric_value


def integer_value(value: Any, column_name: str, row: Mapping[str, Any]) -> int:
    """Return an integer cell value with contextual errors."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column_name!r} value {value!r} in row {dict(row)!r}") from exc
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"Noninteger {column_name!r} value {value!r} in row {dict(row)!r}")
    return int(numeric_value)


def workbook_ratio_for_physical_ratio(knit: str, physical_ratio: str) -> str:
    """Return the workbook ratio corresponding to a physical course:wale ratio."""
    if knit == "Interlock":
        return INTERLOCK_WORKBOOK_RATIO_BY_PHYSICAL_RATIO[physical_ratio]
    return physical_ratio


def create_figure(selection: ContributionSelection) -> Figure:
    """Create a compact four-panel figure of independently normalized ratio bars."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=FIGURE_SIZE_INCHES,
        sharex=True,
        sharey=True,
    )
    x_positions = list(range(len(RATIO_ORDER)))

    for panel_index, (axis, (knit, display_name)) in enumerate(
        zip(axes.flat, KNIT_PANELS, strict=True)
    ):
        bottoms = [0.0] * len(RATIO_ORDER)
        for definition in COMPONENT_DEFINITIONS:
            component_values = [
                selection.values[knit][
                    workbook_ratio_for_physical_ratio(knit, physical_ratio)
                ][definition.key]
                for physical_ratio in RATIO_ORDER
            ]
            axis.bar(
                x_positions,
                component_values,
                width=BAR_WIDTH,
                bottom=bottoms,
                color=definition.color,
                edgecolor="black",
                linewidth=COMPONENT_BORDER_WIDTH,
                linestyle="solid",
                hatch=definition.hatch,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, component_values, strict=True)]

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
        # Retain only horizontal percentage guides after applying the shared style.
        axis.grid(True, axis="y", color="#878A8F", linewidth=GRID_LINE_WIDTH, alpha=0.65)
        axis.grid(False, axis="x")
        for spine in axis.spines.values():
            spine.set_linewidth(AXIS_LINE_WIDTH)

    legend_handles = [
        Patch(
            facecolor=definition.color,
            edgecolor="none" if definition.linearity == "Linear" else "black",
            linewidth=(
                0.0 if definition.linearity == "Linear" else COMPONENT_BORDER_WIDTH
            ),
            linestyle="solid",
            hatch=definition.hatch,
            label=definition.label,
        )
        for definition in COMPONENT_DEFINITIONS
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.06, 0.96, 0.05),
        mode="expand",
        ncol=6,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.7,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    figure.supylabel(
        f"{STRESS_MEASURE} stress contribution (%)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        x=0.015,
    )
    figure.supxlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE, y=0.095)
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.20,
        top=0.94,
        hspace=0.28,
        wspace=0.12,
    )
    return figure


def main() -> None:
    """Read, validate, plot, save, and optionally display the manuscript figure."""
    selection = select_component_percentages(read_contribution_rows())
    figure = create_figure(selection)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="svg", transparent=True)
    print(
        f"Selected {STRESS_MEASURE} {MODEL_ID} fit codes with "
        f"{NORMALIZATION_SCOPE} normalization: {selection.fit_codes}"
    )
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
