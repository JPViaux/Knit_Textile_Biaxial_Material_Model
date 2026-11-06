"""Postprocess saved component-necessity scores over cost tolerances.

This script never refits a model. It reads the completed 39-candidate score
table for each knit and writes one long-format CSV for filtering or plotting.
Run the file directly from VS Code after editing the user configuration below.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

import csv
from dataclasses import dataclass
import math
import pathlib
from typing import Iterable, Sequence


ANALYSIS_DIR = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis")
SUPPORT_DIR = ANALYSIS_DIR.parent
STUDY_OUTPUT_DIR = (
    RELEASE_ARCHIVE_ROOT
    / "sv_exp_polycon_mo_component_necessity"
    / "all_nine_v1"
)


# ---------------------------------------------------------------------------
# VS CODE USER CONFIGURATION - edit these values, then press Run.
# ---------------------------------------------------------------------------
TOLERANCE_PERCENTAGES = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
# Percentage values use 2.0 to mean 2%; examples: (0.0, 1.0, 2.0, 5.0).

SCORE_COLUMN = "primary_cost"
# Score used to define adequacy; examples: "primary_cost" or "equal_ratio_cost".

INPUT_CSV_PATH = STUDY_OUTPUT_DIR / "candidate_scores.csv"
# Saved score table; example: STUDY_OUTPUT_DIR / "candidate_scores.csv".

OUTPUT_CSV_PATH = (
    STUDY_OUTPUT_DIR
    / "tolerance_sensitivity"
    / "component_necessity_tolerance_sensitivity.csv"
)
# New isolated output; example: STUDY_OUTPUT_DIR / "tolerance_sensitivity" / "results.csv".


SUPPORTED_SCORE_COLUMNS = ("primary_cost", "equal_ratio_cost")
EXPECTED_KNITS = ("Interlock", "Rib", "Purl", "Single Jersey")
EXPECTED_CANDIDATES_PER_KNIT = 39
COST_COMPARISON_EPSILON = 1.0e-15
# Absolute guard for score comparisons; example: 1.0e-15 reproduces the original study rule.

COMPONENT_ORDER = (
    "linear_course",
    "linear_wale",
    "linear_coupling",
    "nonlinear_course",
    "nonlinear_wale",
    "nonlinear_coupling",
)

OUTPUT_FIELDS = (
    "score_column",
    "knit",
    "tolerance_percent",
    "winner_candidate_id",
    "winner_cost",
    "adequate_cost_limit",
    "adequate_candidate_count",
    "adequate_candidate_ids",
    "selected_candidate_id",
    "selected_effective_parameter_count",
    "selected_relative_cost_percent",
    "selected_components",
    "component",
    "selected_contains_component",
    "classification",
    "zero_tolerance_classification",
    "classification_transition_threshold_percent",
    "classification_transition",
    "threshold_candidate_ids",
)


@dataclass(frozen=True)
class CandidateScore:
    """One saved candidate score, for example Rib M31 with seven parameters."""

    knit: str
    candidate_id: str
    mask: int
    components: tuple[str, ...]
    effective_parameter_count: int
    score: float


@dataclass(frozen=True)
class ComponentThreshold:
    """First tolerance where a component leaves its zero-tolerance status."""

    zero_status: str
    threshold_percent: float | None
    transition: str
    candidate_ids: tuple[str, ...]


def normalized_tolerances(values: Iterable[float]) -> tuple[float, ...]:
    """Validate, deduplicate, and sort percentage tolerances."""

    tolerances = tuple(float(value) for value in values)
    if not tolerances:
        raise ValueError("TOLERANCE_PERCENTAGES must contain at least one value")
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("Tolerance percentages must be finite and nonnegative")
    return tuple(sorted(set(tolerances)))


def load_candidate_scores(path: pathlib.Path, score_column: str) -> dict[str, list[CandidateScore]]:
    """Load and validate the completed four-knit candidate-score table."""

    if score_column not in SUPPORTED_SCORE_COLUMNS:
        raise ValueError(
            f"SCORE_COLUMN must be one of {SUPPORTED_SCORE_COLUMNS}, found {score_column!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Saved candidate score table not found: {path}")

    grouped: dict[str, list[CandidateScore]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "knit",
            "candidate_id",
            "mask",
            "components",
            "effective_parameter_count",
            score_column,
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Candidate score table is missing columns: {sorted(missing)}")

        for row in reader:
            components = tuple(item for item in row["components"].split(";") if item)
            unknown = set(components).difference(COMPONENT_ORDER)
            if unknown:
                raise ValueError(f"Unknown components for {row['candidate_id']}: {sorted(unknown)}")
            score = float(row[score_column])
            if not math.isfinite(score) or score <= 0.0:
                raise ValueError(f"Invalid {score_column} for {row['candidate_id']}: {score}")
            candidate = CandidateScore(
                knit=row["knit"],
                candidate_id=row["candidate_id"],
                mask=int(row["mask"]),
                components=components,
                effective_parameter_count=int(row["effective_parameter_count"]),
                score=score,
            )
            grouped.setdefault(candidate.knit, []).append(candidate)

    if set(grouped) != set(EXPECTED_KNITS):
        raise ValueError(f"Expected knits {EXPECTED_KNITS}, found {tuple(grouped)}")
    for knit, rows in grouped.items():
        if len(rows) != EXPECTED_CANDIDATES_PER_KNIT:
            raise ValueError(
                f"{knit}: expected {EXPECTED_CANDIDATES_PER_KNIT} candidates, found {len(rows)}"
            )
        if len({row.mask for row in rows}) != len(rows):
            raise ValueError(f"{knit}: candidate masks are not unique")
    return grouped


def winner(rows: Sequence[CandidateScore]) -> CandidateScore:
    """Return the minimum-cost candidate with mask order as a deterministic tie-break."""

    return min(rows, key=lambda row: (row.score, row.mask))


def relative_cost_percent(row: CandidateScore, winning_cost: float) -> float:
    """Return a candidate's cost penalty relative to the winner in percent."""

    return max(0.0, 100.0 * (row.score / winning_cost - 1.0))


def adequate_rows(
    rows: Sequence[CandidateScore], tolerance_percent: float
) -> tuple[CandidateScore, ...]:
    """Return candidates within the requested relative cost tolerance."""

    winning_cost = winner(rows).score
    limit = winning_cost * (1.0 + tolerance_percent / 100.0)
    return tuple(row for row in rows if row.score <= limit + COST_COMPARISON_EPSILON)


def selected_candidate(rows: Sequence[CandidateScore]) -> CandidateScore:
    """Select lowest complexity, then lowest score, then lowest mask."""

    if not rows:
        raise ValueError("At least one adequate candidate is required")
    return min(
        rows,
        key=lambda row: (row.effective_parameter_count, row.score, row.mask),
    )


def component_classification(
    rows: Sequence[CandidateScore], component: str
) -> str:
    """Classify one component across an adequate candidate set."""

    contains = [component in row.components for row in rows]
    if not contains:
        raise ValueError("At least one adequate candidate is required")
    if all(contains):
        return "Required"
    if any(contains):
        return "Optional/substitutable"
    return "Unsupported"


def component_transition_threshold(
    rows: Sequence[CandidateScore], component: str
) -> ComponentThreshold:
    """Find the first exact relative-cost threshold that changes a component's status."""

    winning_cost = winner(rows).score
    zero_adequate = adequate_rows(rows, 0.0)
    zero_status = component_classification(zero_adequate, component)

    if zero_status == "Optional/substitutable":
        return ComponentThreshold(
            zero_status=zero_status,
            threshold_percent=0.0,
            transition="Optional/substitutable at zero tolerance",
            candidate_ids=tuple(sorted(row.candidate_id for row in zero_adequate)),
        )

    seek_component = zero_status == "Unsupported"
    alternatives = [
        row
        for row in rows
        if (component in row.components) == seek_component
    ]
    if not alternatives:
        return ComponentThreshold(
            zero_status=zero_status,
            threshold_percent=None,
            transition=f"Remains {zero_status} across all saved candidates",
            candidate_ids=(),
        )

    threshold = min(relative_cost_percent(row, winning_cost) for row in alternatives)
    threshold_rows = tuple(
        sorted(
            row.candidate_id
            for row in alternatives
            if math.isclose(
                relative_cost_percent(row, winning_cost),
                threshold,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
    )
    return ComponentThreshold(
        zero_status=zero_status,
        threshold_percent=threshold,
        transition=f"{zero_status} -> Optional/substitutable",
        candidate_ids=threshold_rows,
    )


def build_output_rows(
    grouped: dict[str, list[CandidateScore]],
    tolerances: Sequence[float],
    score_column: str,
) -> list[dict[str, object]]:
    """Build one row per knit, requested tolerance, and physical component."""

    output: list[dict[str, object]] = []
    for knit in EXPECTED_KNITS:
        rows = grouped[knit]
        winning = winner(rows)
        thresholds = {
            component: component_transition_threshold(rows, component)
            for component in COMPONENT_ORDER
        }
        for tolerance in tolerances:
            adequate = adequate_rows(rows, tolerance)
            selected = selected_candidate(adequate)
            cost_limit = winning.score * (1.0 + tolerance / 100.0)
            adequate_ids = ";".join(row.candidate_id for row in sorted(adequate, key=lambda row: row.mask))
            for component in COMPONENT_ORDER:
                threshold = thresholds[component]
                output.append(
                    {
                        "score_column": score_column,
                        "knit": knit,
                        "tolerance_percent": tolerance,
                        "winner_candidate_id": winning.candidate_id,
                        "winner_cost": winning.score,
                        "adequate_cost_limit": cost_limit,
                        "adequate_candidate_count": len(adequate),
                        "adequate_candidate_ids": adequate_ids,
                        "selected_candidate_id": selected.candidate_id,
                        "selected_effective_parameter_count": selected.effective_parameter_count,
                        "selected_relative_cost_percent": relative_cost_percent(selected, winning.score),
                        "selected_components": ";".join(selected.components),
                        "component": component,
                        "selected_contains_component": int(component in selected.components),
                        "classification": component_classification(adequate, component),
                        "zero_tolerance_classification": threshold.zero_status,
                        "classification_transition_threshold_percent": threshold.threshold_percent,
                        "classification_transition": threshold.transition,
                        "threshold_candidate_ids": ";".join(threshold.candidate_ids),
                    }
                )
    return output


def write_output(path: pathlib.Path, rows: Sequence[dict[str, object]]) -> None:
    """Write the deterministic long-format sensitivity CSV."""

    if not rows:
        raise ValueError("Cannot write an empty sensitivity table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate the configured tolerance-sensitivity CSV."""

    tolerances = normalized_tolerances(TOLERANCE_PERCENTAGES)
    grouped = load_candidate_scores(INPUT_CSV_PATH, SCORE_COLUMN)
    rows = build_output_rows(grouped, tolerances, SCORE_COLUMN)
    write_output(OUTPUT_CSV_PATH, rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
