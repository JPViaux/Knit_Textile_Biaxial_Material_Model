"""Calibrate operational nonlinear-coupling bounds for a configured SV_exp case.

Raw ``K_coup`` bounds are not comparable across zeta because the coupling basis
shrinks rapidly as its exponent increases. This study compares raw-bound and
weighted-stress-amplitude ladders on the complete dataset, then evaluates one
broad equal-amplitude rule with leave-one-loading-ratio-out prediction.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

from dataclasses import asdict, dataclass
from datetime import datetime
import copy
from release_paths import json
import pathlib
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


ANALYSIS_DIR = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis")
SUPPORT_CODE_DIR = ANALYSIS_DIR.parent
CODE_ROOT = SUPPORT_CODE_DIR.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(SUPPORT_CODE_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

import uml_plot_style as uml_style  # noqa: E402

uml_style.apply()

from Analysis.studies.solver_diagnostics import separable_solver_historical_benchmark as historical  # noqa: E402
from Analysis.studies.solver_diagnostics import separable_solver_leave_one_ratio_out as jackknife  # noqa: E402
from Analysis.studies.solver_diagnostics import separable_solver_model_selection as model_selection  # noqa: E402
from Python_Wrapper import mech, umat  # noqa: E402
from Python_Wrapper.model_registry import load_model_spec  # noqa: E402
from Python_Wrapper.separable_solver import (  # noqa: E402
    GridSearchResult,
    ProfiledSolution,
    SVExpSeparableSolver,
    deterministic_grid_search,
    logarithmic_grid,
    refine_profiled_solution,
)


FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# USER CONFIGURATION - edit these values before pressing Run in VSCode.
# ---------------------------------------------------------------------------

# Verified historical model-selection summary; another completed timestamp may be used.
SOURCE_MODEL_SELECTION_SUMMARY = (
    RELEASE_ARCHIVE_ROOT
    / "separable_solver_model_selection"
    / "fit_0621"
    / "20260727_162954"
    / "summary.json"
)
# Raw K_coup upper bounds; examples are 10, 100, 1000, and 10000.
RAW_COUPLING_UPPER_LADDER = (10.0, 100.0, 1000.0, 10000.0)
# Equal weighted-RMS stress ceilings; 0.01 means 1% normalized RMS stress.
AMPLITUDE_UPPER_LADDER = (0.002, 0.005, 0.01, 0.03, 0.1)
# Broad equal-amplitude ceiling used for leave-one-ratio-out validation.
CROSS_VALIDATION_AMPLITUDE_CEILING = 0.1
# Coarse complete-data map size; examples are 9, 13, or 17 points per rate.
FULL_DATA_GRID_POINTS_PER_AXIS = 13
# Cross-validation map size; 17 matches the preceding model-selection study.
CROSS_VALIDATION_GRID_POINTS_PER_AXIS = 17
# Refine every fixed-zeta map minimum; False retains only mapped points.
REFINE_EACH_ZETA = True
# Maximum Powell iterations per candidate; examples are 50 or 100.
REFINEMENT_MAX_ITERATIONS = 100
# Log-parameter convergence tolerance; examples are 1e-5 or 1e-6.
REFINEMENT_PARAMETER_TOLERANCE = 1.0e-6
# Objective convergence tolerance; examples are 1e-8 or 1e-10.
REFINEMENT_COST_TOLERANCE = 1.0e-10
# A selected bound should exceed observed winning amplitudes by this factor.
BOUND_MARGIN_FACTOR = 2.0
# Full-data costs within 0.1% of the broad reference are considered converged.
COST_CONVERGENCE_TOLERANCE = 0.001
# Models within 2% of the predictive winner are considered predictively tied.
SIMPLICITY_TOLERANCE = 0.02
# Save JSON, NPZ, and PNG outputs; False only prints the completed results.
SAVE_OUTPUTS = True
# Display plots interactively; False is convenient for a VSCode batch run.
SHOW_PLOTS = False
# Standalone output root; for example fit_0621/<timestamp>.
OUTPUT_ROOT = RELEASE_ARCHIVE_ROOT / "separable_solver_coupling_bounds"


ZETA_VALUES = tuple(range(2, 11))
COUPLING_PARAMETER = "K_coup"


@dataclass(frozen=True)
class CouplingBasisMetrics:
    """Describe the unit-K_coup stress basis for one integer exponent."""

    zeta: int
    weighted_rms: float
    weighted_maximum_absolute: float
    physical_rms: float
    physical_maximum_absolute: float
    physical_axis_rms: tuple[float, ...]


@dataclass(frozen=True)
class BoundStrategyResult:
    """Store one complete-data optimization under a coupling-bound rule."""

    label: str
    strategy_type: str
    strategy_value: float
    coupling_upper_bounds: dict[int, float]
    grid_result: GridSearchResult
    final_by_zeta: tuple[ProfiledSolution, ...]
    best_solution: ProfiledSolution
    direct_mean_cost: float
    maximum_prediction_difference: float
    elapsed_seconds: float


@dataclass(frozen=True)
class FoldBoundResult:
    """Store amplitude-bounded fits and held-out costs for one omitted ratio."""

    omitted_label: str
    omitted_path: pathlib.Path
    training_rows: int
    basis_metrics: tuple[CouplingBasisMetrics, ...]
    coupling_upper_bounds: dict[int, float]
    final_by_zeta: tuple[ProfiledSolution, ...]
    free_solution: ProfiledSolution
    held_out: dict[str, historical.FullParameterEvaluation]
    elapsed_seconds: float


class ZetaCouplingBoundSolver(SVExpSeparableSolver):
    """Apply a different physical K_coup upper bound for every zeta value."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._coupling_index = self.definition.linear_parameters.index(
            COUPLING_PARAMETER
        )
        default_upper = float(self.linear_upper_bounds[self._coupling_index])
        self._coupling_upper_by_zeta = {
            zeta: default_upper for zeta in self.integer_values("zeta")
        }

    def set_coupling_upper_bounds(self, upper_by_zeta: Mapping[int, float]) -> None:
        """Set finite nonnegative bounds, using zero for an inactive basis."""
        expected = set(self.integer_values("zeta"))
        supplied = {int(zeta) for zeta in upper_by_zeta}
        if supplied != expected:
            raise ValueError(
                f"Coupling bounds require zeta values {sorted(expected)}, "
                f"received {sorted(supplied)}"
            )
        validated = {int(zeta): float(value) for zeta, value in upper_by_zeta.items()}
        if any(not np.isfinite(value) or value < 0 for value in validated.values()):
            raise ValueError("Every coupling upper bound must be nonnegative and finite")
        self._coupling_upper_by_zeta = validated

    def linear_bounds_at_shapes(
        self,
        shape_values: Mapping[str, float | int],
    ) -> tuple[FloatArray, FloatArray]:
        """Return local bounds without mutating shared solver state."""
        lower, upper = super().linear_bounds_at_shapes(shape_values)
        upper[self._coupling_index] = self._coupling_upper_by_zeta[
            int(shape_values["zeta"])
        ]
        return lower, upper


class ShapeCouplingBoundSolver(SVExpSeparableSolver):
    """Apply parameter-file or amplitude-equalized bounds at each shape."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._coupling_index = self.definition.linear_parameters.index(
            COUPLING_PARAMETER
        )
        self._coupling_dependencies = self.definition.basis_dependencies()[
            COUPLING_PARAMETER
        ]
        self._bound_mode = "parameter_file"
        self._amplitude_ceiling: float | None = None
        self._basis_rms_by_shape: dict[
            tuple[tuple[str, float | int], ...], float
        ] = {}
        self._coupling_upper_by_shape: dict[
            tuple[tuple[str, float | int], ...], float
        ] = {}

    def configure_coupling_bound(
        self,
        mode: str,
        amplitude_ceiling: float | None = None,
    ) -> None:
        """Select ``parameter_file`` or a positive equalized amplitude ceiling."""
        normalized = str(mode).strip().lower()
        if normalized not in {"parameter_file", "amplitude_equalized"}:
            raise ValueError("Unsupported shape-dependent coupling-bound mode")
        if normalized == "amplitude_equalized":
            ceiling = float(amplitude_ceiling) if amplitude_ceiling is not None else np.nan
            if not np.isfinite(ceiling) or ceiling <= 0:
                raise ValueError("Amplitude-equalized coupling requires a positive ceiling")
            self._amplitude_ceiling = ceiling
        else:
            self._amplitude_ceiling = None
        self._bound_mode = normalized
        self._coupling_upper_by_shape.clear()

    def _shape_key(
        self,
        shape_values: Mapping[str, float | int],
    ) -> tuple[tuple[str, float | int], ...]:
        """Return the exact coupling-basis key, for example b=0.5 and zeta=5."""
        validated = self.validate_shape_values(shape_values, require_all=False)
        return tuple((name, validated[name]) for name in self._coupling_dependencies)

    def basis_rms_at_shapes(
        self,
        shape_values: Mapping[str, float | int],
    ) -> float:
        """Return cached weighted RMS for the coupling basis at one shape."""
        key = self._shape_key(shape_values)
        if key not in self._basis_rms_by_shape:
            column = self.basis_column(COUPLING_PARAMETER, dict(key))
            self._basis_rms_by_shape[key] = float(np.sqrt(np.mean(column * column)))
        return self._basis_rms_by_shape[key]

    def coupling_upper_at_shapes(
        self,
        shape_values: Mapping[str, float | int],
    ) -> float:
        """Return the effective physical K_coup upper bound at one shape."""
        key = self._shape_key(shape_values)
        if key not in self._coupling_upper_by_shape:
            if self._bound_mode == "parameter_file":
                upper = float(self.linear_upper_bounds[self._coupling_index])
            else:
                assert self._amplitude_ceiling is not None
                rms = self.basis_rms_at_shapes(dict(key))
                upper = 0.0 if rms == 0.0 else self._amplitude_ceiling / rms
            self._coupling_upper_by_shape[key] = float(upper)
        return self._coupling_upper_by_shape[key]

    def linear_bounds_at_shapes(
        self,
        shape_values: Mapping[str, float | int],
    ) -> tuple[FloatArray, FloatArray]:
        """Insert the shape-specific coupling ceiling into a local bound copy."""
        lower, upper = super().linear_bounds_at_shapes(shape_values)
        upper[self._coupling_index] = self.coupling_upper_at_shapes(shape_values)
        return lower, upper

    def coupling_bound_cache(
        self,
    ) -> dict[tuple[tuple[str, float | int], ...], dict[str, float]]:
        """Return a copy of evaluated basis RMS and effective upper bounds."""
        return {
            key: {
                "basis_weighted_rms": self._basis_rms_by_shape.get(key, np.nan),
                "coupling_upper_bound": upper,
            }
            for key, upper in self._coupling_upper_by_shape.items()
        }


def amplitude_equalized_bounds(
    basis_rms_by_zeta: Mapping[int, float],
    amplitude_ceiling: float,
) -> dict[int, float]:
    """Convert one weighted-RMS ceiling into zeta-specific physical bounds."""
    ceiling = float(amplitude_ceiling)
    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("Amplitude ceiling must be positive and finite")
    bounds: dict[int, float] = {}
    for zeta, basis_rms in basis_rms_by_zeta.items():
        rms = float(basis_rms)
        if not np.isfinite(rms) or rms < 0:
            raise ValueError(f"Coupling basis RMS must be nonnegative for zeta={zeta}")
        # Pure single-axis curves can make the coupling basis identically zero.
        # Its coefficient is then unidentifiable and has no predictive effect, so
        # use the canonical zero-width bound K_coup=0 instead of dividing by zero.
        bounds[int(zeta)] = 0.0 if rms == 0.0 else ceiling / rms
    return bounds


def recommend_amplitude_ceiling(
    ladder: Sequence[float],
    broad_winning_amplitudes: Sequence[float],
    converged_ceiling: float,
    margin_factor: float = BOUND_MARGIN_FACTOR,
) -> float | None:
    """Choose the smallest converged ladder value above a margin on all winners."""
    candidates = sorted(float(value) for value in ladder)
    amplitudes = np.asarray(tuple(broad_winning_amplitudes), dtype=float)
    if not candidates or amplitudes.size == 0:
        raise ValueError("A nonempty ladder and winning-amplitude list are required")
    if any(value <= 0 or not np.isfinite(value) for value in candidates):
        raise ValueError("Amplitude ladder values must be positive and finite")
    if np.any(amplitudes < 0) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("Winning amplitudes must be nonnegative and finite")
    required = float(margin_factor) * float(np.max(amplitudes))
    eligible = [
        value
        for value in candidates
        if value >= required and value >= float(converged_ceiling)
    ]
    return min(eligible) if eligible else None


def load_and_validate_source(path: pathlib.Path) -> dict[str, Any]:
    """Verify the completed model-selection summary and all immutable inputs."""
    summary = historical.load_json(path)
    if summary.get("schema_version") != 1:
        raise ValueError("Unsupported source model-selection schema")
    if summary.get("study") != "SV_exp fixed-weight leave-one-ratio-out model selection":
        raise ValueError("Source summary is not the expected model-selection study")
    input_paths = {
        name: pathlib.Path(value)
        for name, value in summary["provenance"]["input_paths"].items()
    }
    historical.require_input_files(input_paths)
    expected_hashes = summary["provenance"]["input_sha256"]
    changed = [
        name
        for name, source_path in input_paths.items()
        if historical.sha256_file(source_path) != expected_hashes.get(name)
    ]
    if changed:
        raise ValueError("Source model-selection inputs changed: " + ", ".join(changed))
    return summary


def fixed_reference_maxima(summary: Mapping[str, Any]) -> dict[int, float]:
    """Recover full-data scales, for example ``{0: 0.0051, 1: 0.0091}``."""
    stored = summary["configuration"]["fixed_full_data_reference_maxima"]
    references = {int(axis): float(value) for axis, value in stored.items()}
    if set(references) != set(historical.FIT_AXES):
        raise ValueError("Fixed normalization axes changed")
    return references


def widened_specs(
    saved_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Retain historical linear bounds while widening beta/gamma to 1000."""
    specifications = copy.deepcopy(saved_specs)
    specifications["beta_1"]["maximum"] = model_selection.WIDE_SHAPE_MAXIMUM
    specifications["gamma_1"]["maximum"] = model_selection.WIDE_SHAPE_MAXIMUM
    return specifications


def compute_coupling_basis_metrics(
    solver: ZetaCouplingBoundSolver,
) -> tuple[CouplingBasisMetrics, ...]:
    """Measure weighted and physical unit-K_coup basis magnitudes by zeta."""
    beta_reference = float(np.sqrt(np.prod(solver.parameter_bounds("beta_1"))))
    gamma_reference = float(np.sqrt(np.prod(solver.parameter_bounds("gamma_1"))))
    coupling_index = solver.definition.linear_parameters.index(COUPLING_PARAMETER)
    row_count = solver.problem.target.size // len(solver.fit_axes)
    metrics: list[CouplingBasisMetrics] = []
    for zeta in solver.integer_values("zeta"):
        weighted = solver.build_weighted_design(
            beta_reference,
            gamma_reference,
            zeta,
        )[:, coupling_index]
        physical = weighted * solver.problem.scales
        axis_rms = tuple(
            float(np.sqrt(np.mean(physical[index * row_count : (index + 1) * row_count] ** 2)))
            for index in range(len(solver.fit_axes))
        )
        metrics.append(
            CouplingBasisMetrics(
                zeta=int(zeta),
                weighted_rms=float(np.sqrt(np.mean(weighted**2))),
                weighted_maximum_absolute=float(np.max(np.abs(weighted))),
                physical_rms=float(np.sqrt(np.mean(physical**2))),
                physical_maximum_absolute=float(np.max(np.abs(physical))),
                physical_axis_rms=axis_rms,
            )
        )
    return tuple(metrics)


def invariant_excitation_summary(
    wrapper: Any,
    curves: Sequence[jackknife.RatioCurve],
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Summarize I4/I6 coupling activation for each exact loading-ratio block."""
    theta1 = np.deg2rad(float(parameter_specs["theta1"]["initial_guess"]))
    theta2 = np.deg2rad(float(parameter_specs["theta2"]["initial_guess"]))
    n1 = np.array([np.cos(theta1), np.sin(theta1), 0.0])
    n2 = np.array([np.cos(theta2), np.sin(theta2), 0.0])
    c1 = float(parameter_specs["C1"]["initial_guess"])
    c2 = float(parameter_specs["C2"]["initial_guess"])
    invariants = []
    for deformation_voigt in np.asarray(wrapper.dfgrds, dtype=float):
        deformation = mech.voigt_to_matrix(deformation_voigt)
        right_cauchy_green = deformation.T @ deformation
        invariants.append(
            (
                float(n1 @ right_cauchy_green @ n1),
                float(n2 @ right_cauchy_green @ n2),
            )
        )
    invariant_array = np.asarray(invariants, dtype=float)
    summaries: list[dict[str, Any]] = []
    for curve in curves:
        values = invariant_array[curve.start_row : curve.stop_row]
        delta4 = np.maximum(values[:, 0] - c1, 0.0)
        delta6 = np.maximum(values[:, 1] - c2, 0.0)
        product = delta4 * delta6
        jointly_active = (delta4 > 0) & (delta6 > 0)
        summaries.append(
            {
                "label": curve.label,
                "point_count": int(values.shape[0]),
                "jointly_active_fraction": float(np.mean(jointly_active)),
                "I4_minimum": float(np.min(values[:, 0])),
                "I4_maximum": float(np.max(values[:, 0])),
                "I6_minimum": float(np.min(values[:, 1])),
                "I6_maximum": float(np.max(values[:, 1])),
                "delta_I4_maximum": float(np.max(delta4)),
                "delta_I6_maximum": float(np.max(delta6)),
                "coupling_product_median": float(np.median(product)),
                "coupling_product_90th_percentile": float(np.quantile(product, 0.9)),
                "coupling_product_maximum": float(np.max(product)),
            }
        )
    return tuple(summaries)


def refined_grid_solutions(
    solver: ZetaCouplingBoundSolver,
    grid_result: GridSearchResult,
) -> tuple[ProfiledSolution, ...]:
    """Refine every zeta minimum or retain the mapped candidates."""
    if not REFINE_EACH_ZETA:
        return grid_result.best_by_zeta
    return tuple(
        refine_profiled_solution(
            solver,
            solution,
            maximum_iterations=REFINEMENT_MAX_ITERATIONS,
            parameter_tolerance=REFINEMENT_PARAMETER_TOLERANCE,
            cost_tolerance=REFINEMENT_COST_TOLERANCE,
        )
        for solution in grid_result.best_by_zeta
    )


def fit_bound_strategy(
    solver: ZetaCouplingBoundSolver,
    label: str,
    strategy_type: str,
    strategy_value: float,
    coupling_upper_bounds: Mapping[int, float],
    beta_values: FloatArray,
    gamma_values: FloatArray,
) -> BoundStrategyResult:
    """Fit every zeta under one supplied raw or amplitude-equalized bound map."""
    start = time.perf_counter()
    solver.set_coupling_upper_bounds(coupling_upper_bounds)
    grid_result = deterministic_grid_search(
        solver,
        beta_values,
        gamma_values,
        ZETA_VALUES,
    )
    final_by_zeta = refined_grid_solutions(solver, grid_result)
    best = min(final_by_zeta, key=lambda solution: solution.cost)
    validation = solver.validate_solution(best)
    if validation.absolute_cost_difference > 1.0e-10:
        raise ValueError(f"Direct UMAT cost disagrees for strategy {label}")
    return BoundStrategyResult(
        label=label,
        strategy_type=strategy_type,
        strategy_value=float(strategy_value),
        coupling_upper_bounds={int(zeta): float(value) for zeta, value in coupling_upper_bounds.items()},
        grid_result=grid_result,
        final_by_zeta=final_by_zeta,
        best_solution=best,
        direct_mean_cost=validation.direct_cost / float(solver.problem.target.size),
        maximum_prediction_difference=validation.maximum_weighted_prediction_difference,
        elapsed_seconds=float(time.perf_counter() - start),
    )


def scalar_strategy(result: BoundStrategyResult) -> dict[str, Any]:
    """Convert one complete-data result to JSON scalars while maps go to NPZ."""
    return {
        "label": result.label,
        "strategy_type": result.strategy_type,
        "strategy_value": result.strategy_value,
        "coupling_upper_bounds": {
            str(zeta): value for zeta, value in result.coupling_upper_bounds.items()
        },
        "best_solution": asdict(result.best_solution),
        "direct_mean_cost": result.direct_mean_cost,
        "maximum_prediction_difference": result.maximum_prediction_difference,
        "final_by_zeta": [asdict(solution) for solution in result.final_by_zeta],
        "profiled_evaluations": int(np.prod(result.grid_result.costs.shape)),
        "elapsed_seconds": result.elapsed_seconds,
    }


def evaluate_solutions_on_curve(
    curve: jackknife.RatioCurve,
    solutions: Sequence[ProfiledSolution],
    parameter_path: pathlib.Path,
    model_spec: Any,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    reference_maxima: Mapping[int, float],
) -> dict[str, historical.FullParameterEvaluation]:
    """Evaluate fixed-zeta candidates and the training-selected free candidate."""
    wrapper = model_selection.configured_wrapper(
        curve.path,
        parameter_path,
        model_spec,
        parameter_specs,
        reference_maxima,
    )
    evaluations = {}
    for solution in solutions:
        label = f"fixed_zeta_{solution.zeta}"
        parameters = historical.complete_profiled_parameters(solution, parameter_specs)
        evaluations[label] = historical.evaluate_full_parameter_set(
            wrapper,
            parameters,
            historical.FIT_AXES,
        )
    best = min(solutions, key=lambda solution: solution.cost)
    evaluations["free_zeta"] = evaluations[f"fixed_zeta_{best.zeta}"]
    return evaluations


def fit_amplitude_fold(
    training_path: pathlib.Path,
    omitted_curve: jackknife.RatioCurve,
    parameter_path: pathlib.Path,
    model_spec: Any,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    reference_maxima: Mapping[int, float],
) -> FoldBoundResult:
    """Fit one broad equal-amplitude fold and evaluate its omitted curve."""
    start = time.perf_counter()
    wrapper = model_selection.configured_wrapper(
        training_path,
        parameter_path,
        model_spec,
        parameter_specs,
        reference_maxima,
    )
    solver = ZetaCouplingBoundSolver(wrapper, historical.FIT_AXES)
    basis_metrics = compute_coupling_basis_metrics(solver)
    basis_rms = {item.zeta: item.weighted_rms for item in basis_metrics}
    coupling_bounds = amplitude_equalized_bounds(
        basis_rms,
        CROSS_VALIDATION_AMPLITUDE_CEILING,
    )
    solver.set_coupling_upper_bounds(coupling_bounds)
    beta_values = logarithmic_grid(
        solver.parameter_bounds("beta_1"),
        CROSS_VALIDATION_GRID_POINTS_PER_AXIS,
    )
    gamma_values = logarithmic_grid(
        solver.parameter_bounds("gamma_1"),
        CROSS_VALIDATION_GRID_POINTS_PER_AXIS,
    )
    grid_result = deterministic_grid_search(
        solver,
        beta_values,
        gamma_values,
        ZETA_VALUES,
    )
    final_by_zeta = refined_grid_solutions(solver, grid_result)
    free_solution = min(final_by_zeta, key=lambda solution: solution.cost)
    held_out = evaluate_solutions_on_curve(
        omitted_curve,
        final_by_zeta,
        parameter_path,
        model_spec,
        parameter_specs,
        reference_maxima,
    )
    return FoldBoundResult(
        omitted_label=omitted_curve.label,
        omitted_path=omitted_curve.path,
        training_rows=int(wrapper.expstressPK2.shape[0]),
        basis_metrics=basis_metrics,
        coupling_upper_bounds=coupling_bounds,
        final_by_zeta=final_by_zeta,
        free_solution=free_solution,
        held_out=held_out,
        elapsed_seconds=float(time.perf_counter() - start),
    )


def scalar_evaluation(evaluation: historical.FullParameterEvaluation) -> dict[str, float]:
    """Return cost scalars while prediction and residual arrays remain transient."""
    return {
        "raw_cost": evaluation.raw_cost,
        "mean_cost": evaluation.mean_cost,
        "normalized_residual_rmse": float(np.sqrt(np.mean(evaluation.normalized_residuals**2))),
    }


def scalar_fold(fold: FoldBoundResult) -> dict[str, Any]:
    """Convert one completed amplitude-calibrated fold to JSON-safe values."""
    basis_rms = {item.zeta: item.weighted_rms for item in fold.basis_metrics}
    return {
        "omitted_label": fold.omitted_label,
        "omitted_path": str(fold.omitted_path.resolve()),
        "training_rows": fold.training_rows,
        "basis_metrics": [asdict(item) for item in fold.basis_metrics],
        "coupling_upper_bounds": {
            str(zeta): value for zeta, value in fold.coupling_upper_bounds.items()
        },
        "final_by_zeta": [
            {
                **asdict(solution),
                "effective_weighted_rms_amplitude": (
                    solution.linear_values[COUPLING_PARAMETER] * basis_rms[solution.zeta]
                ),
            }
            for solution in fold.final_by_zeta
        ],
        "free_solution": {
            **asdict(fold.free_solution),
            "effective_weighted_rms_amplitude": (
                fold.free_solution.linear_values[COUPLING_PARAMETER]
                * basis_rms[fold.free_solution.zeta]
            ),
        },
        "held_out": {
            label: scalar_evaluation(evaluation)
            for label, evaluation in fold.held_out.items()
        },
        "elapsed_seconds": fold.elapsed_seconds,
    }


def aggregate_amplitude_scores(
    folds: Sequence[FoldBoundResult],
) -> dict[str, float]:
    """Pool every held-out normalized SSE and divide by all observations once."""
    if not folds:
        raise ValueError("At least one completed fold is required")
    labels = tuple(f"fixed_zeta_{zeta}" for zeta in ZETA_VALUES) + ("free_zeta",)
    observation_count = sum(
        fold.held_out["free_zeta"].normalized_residuals.size for fold in folds
    )
    return {
        label: float(sum(fold.held_out[label].raw_cost for fold in folds))
        / float(observation_count)
        for label in labels
    }


def historical_scores(summary: Mapping[str, Any]) -> dict[str, float]:
    """Read prior raw-bound-10 scores, including the reduced K_coup=0 model."""
    return {
        str(score["model_label"]): float(score["held_out_mean_cost"])
        for score in summary["model_scores"]
    }


def select_converged_amplitude_ceiling(
    strategies: Sequence[BoundStrategyResult],
) -> float:
    """Find the smallest amplitude ladder cost-converged to its broadest result."""
    amplitude_results = sorted(
        (result for result in strategies if result.strategy_type == "amplitude"),
        key=lambda result: result.strategy_value,
    )
    broad_cost = amplitude_results[-1].best_solution.cost
    converged = [
        result.strategy_value
        for result in amplitude_results
        if result.best_solution.cost / broad_cost - 1.0 <= COST_CONVERGENCE_TOLERANCE
    ]
    if not converged:
        raise ValueError("No amplitude ceiling converged to the broad reference")
    return float(min(converged))


def create_invariant_figure(summaries: Sequence[Mapping[str, Any]]) -> plt.Figure:
    """Show coupling activation and invariant-product magnitude by loading ratio."""
    labels = [str(item["label"]) for item in summaries]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].bar(
        positions,
        [100.0 * float(item["jointly_active_fraction"]) for item in summaries],
        color="#0072B2",
    )
    axes[0].set_ylabel("Joint activation (% of points)")
    axes[0].set_title(r"Rows with $I_4>C_1$ and $I_6>C_2$")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].semilogy(
        positions,
        [max(float(item["coupling_product_maximum"]), 1.0e-18) for item in summaries],
        marker="o",
        label="Maximum",
    )
    axes[1].semilogy(
        positions,
        [
            max(float(item["coupling_product_90th_percentile"]), 1.0e-18)
            for item in summaries
        ],
        marker="s",
        label="90th percentile",
    )
    axes[1].set_ylabel(r"$(I_4-C_1)_+(I_6-C_2)_+$")
    axes[1].set_xlabel("Loading ratio")
    axes[1].set_title("Coupling invariant product")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=45, ha="right")
    return figure


def create_basis_figure(
    basis_metrics: Sequence[CouplingBasisMetrics],
    recommended_ceiling: float | None,
) -> plt.Figure:
    """Show basis collapse with zeta and the resulting equal-amplitude bounds."""
    zeta = np.asarray([item.zeta for item in basis_metrics], dtype=int)
    rms = np.asarray([item.weighted_rms for item in basis_metrics], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].semilogy(zeta, rms, marker="o")
    axes[0].set_xlabel(r"$\zeta$")
    axes[0].set_ylabel(r"Weighted RMS basis for $K_{coup}=1$")
    axes[0].set_title("Observable coupling basis magnitude")
    axes[0].grid(True, which="both", alpha=0.3)
    for ceiling in AMPLITUDE_UPPER_LADDER:
        bounds = ceiling / rms
        axes[1].semilogy(zeta, bounds, marker="o", label=f"Amax={ceiling:g}")
    if recommended_ceiling is not None:
        axes[1].set_title(f"Equal-amplitude bounds; recommendation Amax={recommended_ceiling:g}")
    else:
        axes[1].set_title("Equal-amplitude physical bounds")
    axes[1].set_xlabel(r"$\zeta$")
    axes[1].set_ylabel(r"Converted $K_{coup}$ upper bound")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    return figure


def create_full_data_ladder_figure(
    strategies: Sequence[BoundStrategyResult],
) -> plt.Figure:
    """Compare refined zeta profiles across raw and equal-amplitude bounds."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for strategy_type, axis in (("raw", axes[0]), ("amplitude", axes[1])):
        relevant = [item for item in strategies if item.strategy_type == strategy_type]
        reference = min(
            solution.cost
            for item in relevant
            for solution in item.final_by_zeta
        )
        for item in relevant:
            axis.plot(
                ZETA_VALUES,
                [100.0 * (solution.cost / reference - 1.0) for solution in item.final_by_zeta],
                marker="o",
                label=f"{item.strategy_value:g}",
            )
        axis.set_xlabel(r"$\zeta$")
        axis.set_ylabel("Cost above broad-study minimum (%)")
        axis.grid(True, alpha=0.3)
        axis.legend(title="Kc max" if strategy_type == "raw" else "Amplitude max")
    axes[0].set_title("Raw K_coup bound ladder")
    axes[1].set_title("Equal weighted-amplitude ladder")
    return figure


def create_cross_validation_figure(
    source_scores: Mapping[str, float],
    amplitude_scores: Mapping[str, float],
) -> plt.Figure:
    """Compare historical raw-bound and broad equal-amplitude predictive scores."""
    labels = ("no_coupling",) + tuple(f"fixed_zeta_{zeta}" for zeta in ZETA_VALUES) + (
        "free_zeta",
    )
    positions = np.arange(len(labels))
    width = 0.38
    historical_values = [source_scores[label] for label in labels]
    amplitude_values = [
        source_scores["no_coupling"] if label == "no_coupling" else amplitude_scores[label]
        for label in labels
    ]
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        historical_values,
        width,
        label=r"Historical raw $K_{coup}\leq10$",
        color="#999999",
    )
    axis.bar(
        positions + width / 2,
        amplitude_values,
        width,
        label=f"Equal amplitude <= {CROSS_VALIDATION_AMPLITUDE_CEILING:g}",
        color="#0072B2",
    )
    axis.set_xticks(
        positions,
        [label.replace("fixed_zeta_", "zeta=") for label in labels],
        rotation=45,
        ha="right",
    )
    axis.set_ylabel("Aggregate held-out mean cost")
    axis.set_title("Predictive model comparison under alternative coupling bounds")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    return figure


def write_checkpoint(
    path: pathlib.Path,
    strategies: Sequence[BoundStrategyResult],
    folds: Sequence[FoldBoundResult],
) -> None:
    """Preserve completed scalar results after each expensive strategy and fold."""
    payload = {
        "schema_version": 1,
        "complete_data_strategies": [scalar_strategy(item) for item in strategies],
        "completed_folds": [scalar_fold(item) for item in folds],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_outputs(
    output_directory: pathlib.Path,
    source_summary_path: pathlib.Path,
    source_summary: Mapping[str, Any],
    knit_name: str,
    invariant_summaries: Sequence[Mapping[str, Any]],
    full_basis_metrics: Sequence[CouplingBasisMetrics],
    strategies: Sequence[BoundStrategyResult],
    folds: Sequence[FoldBoundResult],
    total_elapsed_seconds: float,
) -> pathlib.Path:
    """Write the bound-calibration summary, arrays, and figures."""
    source_scores = historical_scores(source_summary)
    amplitude_scores = aggregate_amplitude_scores(folds)
    converged_ceiling = select_converged_amplitude_ceiling(strategies)
    broad_winning_amplitudes = []
    broad_strategy = next(
        item
        for item in strategies
        if item.strategy_type == "amplitude"
        and np.isclose(item.strategy_value, max(AMPLITUDE_UPPER_LADDER))
    )
    full_basis_rms = {item.zeta: item.weighted_rms for item in full_basis_metrics}
    broad_winning_amplitudes.append(
        broad_strategy.best_solution.linear_values[COUPLING_PARAMETER]
        * full_basis_rms[broad_strategy.best_solution.zeta]
    )
    for fold in folds:
        fold_basis_rms = {item.zeta: item.weighted_rms for item in fold.basis_metrics}
        broad_winning_amplitudes.append(
            fold.free_solution.linear_values[COUPLING_PARAMETER]
            * fold_basis_rms[fold.free_solution.zeta]
        )
    recommended_ceiling = recommend_amplitude_ceiling(
        AMPLITUDE_UPPER_LADDER,
        broad_winning_amplitudes,
        converged_ceiling,
    )

    combined_scores = {"no_coupling": source_scores["no_coupling"], **amplitude_scores}
    predictive_label = min(combined_scores, key=combined_scores.get)
    predictive_cost = combined_scores[predictive_label]
    complexity = {
        "no_coupling": 7,
        **{f"fixed_zeta_{zeta}": 8 for zeta in ZETA_VALUES},
        "free_zeta": 9,
    }
    eligible = [
        label
        for label, cost in combined_scores.items()
        if cost <= predictive_cost * (1.0 + SIMPLICITY_TOLERANCE)
    ]
    simplicity_label = min(eligible, key=lambda label: (complexity[label], combined_scores[label]))
    selected_bounds = (
        amplitude_equalized_bounds(full_basis_rms, recommended_ceiling)
        if recommended_ceiling is not None
        else None
    )
    summary = {
        "schema_version": 1,
        "study": (
            f"SV_exp {knit_name} nonlinear-coupling operational-bound calibration"
        ),
        "interpretation_limit": (
            "The amplitude rule equalizes observable normalized-stress capacity; it is an "
            "operational fitting bound, not a confidence interval or material-property range."
        ),
        "provenance": {
            "source_model_selection_summary": str(source_summary_path.resolve()),
            "source_model_selection_summary_sha256": historical.sha256_file(source_summary_path),
            "verified_input_sha256": source_summary["provenance"]["input_sha256"],
            "source_files_modified": False,
        },
        "configuration": {
            "raw_coupling_upper_ladder": list(RAW_COUPLING_UPPER_LADDER),
            "amplitude_upper_ladder": list(AMPLITUDE_UPPER_LADDER),
            "cross_validation_amplitude_ceiling": CROSS_VALIDATION_AMPLITUDE_CEILING,
            "full_data_grid_points_per_axis": FULL_DATA_GRID_POINTS_PER_AXIS,
            "cross_validation_grid_points_per_axis": CROSS_VALIDATION_GRID_POINTS_PER_AXIS,
            "bound_margin_factor": BOUND_MARGIN_FACTOR,
            "cost_convergence_tolerance": COST_CONVERGENCE_TOLERANCE,
            "simplicity_tolerance": SIMPLICITY_TOLERANCE,
        },
        "invariant_excitation_by_ratio": list(invariant_summaries),
        "full_data_coupling_basis_metrics": [asdict(item) for item in full_basis_metrics],
        "complete_data_strategies": [scalar_strategy(item) for item in strategies],
        "amplitude_cross_validation_folds": [scalar_fold(item) for item in folds],
        "predictive_scores": {
            "historical_raw_bound_10": source_scores,
            "equal_amplitude_broad": combined_scores,
        },
        "predictive_winner_equal_amplitude": {
            "model_label": predictive_label,
            "held_out_mean_cost": predictive_cost,
        },
        "simplicity_recommendation_equal_amplitude": {
            "model_label": simplicity_label,
            "held_out_mean_cost": combined_scores[simplicity_label],
        },
        "bound_selection": {
            "smallest_full_data_converged_amplitude_ceiling": converged_ceiling,
            "broad_winning_effective_amplitudes": broad_winning_amplitudes,
            "maximum_broad_winning_effective_amplitude": max(broad_winning_amplitudes),
            "recommended_operational_amplitude_ceiling": recommended_ceiling,
            "recommended_full_data_K_coup_upper_by_zeta": (
                {str(zeta): value for zeta, value in selected_bounds.items()}
                if selected_bounds is not None
                else None
            ),
            "requires_ladder_expansion": recommended_ceiling is None,
        },
        "total_elapsed_seconds": float(total_elapsed_seconds),
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    np.savez_compressed(
        output_directory / "coupling_bound_arrays.npz",
        zeta_values=np.asarray(ZETA_VALUES, dtype=int),
        full_basis_weighted_rms=np.asarray([item.weighted_rms for item in full_basis_metrics]),
        raw_strategy_values=np.asarray(
            [item.strategy_value for item in strategies if item.strategy_type == "raw"]
        ),
        raw_refined_costs=np.asarray(
            [
                [solution.mean_cost for solution in item.final_by_zeta]
                for item in strategies
                if item.strategy_type == "raw"
            ]
        ),
        amplitude_strategy_values=np.asarray(
            [item.strategy_value for item in strategies if item.strategy_type == "amplitude"]
        ),
        amplitude_refined_costs=np.asarray(
            [
                [solution.mean_cost for solution in item.final_by_zeta]
                for item in strategies
                if item.strategy_type == "amplitude"
            ]
        ),
        amplitude_cross_validated_scores=np.asarray(
            [amplitude_scores[f"fixed_zeta_{zeta}"] for zeta in ZETA_VALUES]
            + [amplitude_scores["free_zeta"]]
        ),
    )
    figures = {
        "invariant_excitation.png": create_invariant_figure(invariant_summaries),
        "coupling_basis_scaling.png": create_basis_figure(
            full_basis_metrics,
            recommended_ceiling,
        ),
        "complete_data_bound_ladders.png": create_full_data_ladder_figure(strategies),
        "cross_validated_bound_comparison.png": create_cross_validation_figure(
            source_scores,
            amplitude_scores,
        ),
    }
    for name, figure in figures.items():
        figure.savefig(output_directory / name, dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)
    return output_directory


def main() -> None:
    """Run complete-data calibration and broad equal-amplitude cross-validation."""
    source_path = SOURCE_MODEL_SELECTION_SUMMARY.resolve()
    source_summary = load_and_validate_source(source_path)
    input_paths = {
        name: pathlib.Path(value)
        for name, value in source_summary["provenance"]["input_paths"].items()
    }
    artifact = historical.load_json(input_paths["fit_artifact"])
    checkpoint = historical.load_json(input_paths["checkpoint"])
    historical.validate_reference_artifact(artifact)
    signature = historical.verify_historical_batch_signature(artifact, checkpoint)
    if signature != source_summary["provenance"]["reconstructed_batch_signature"]:
        raise ValueError("Historical fit signature differs from the source study")

    compiled_path = input_paths["compiled_experimental_data"]
    parameter_path = input_paths["parameter_file"]
    model_spec = load_model_spec(str(artifact["model_id"]))
    parameter_specs = widened_specs(artifact["parameter_specs"])
    reference_maxima = fixed_reference_maxima(source_summary)
    curves = jackknife.discover_ordered_ratio_curves(compiled_path)
    output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = OUTPUT_ROOT / f"fit_{historical.REFERENCE_FIT_CODE:04d}" / output_timestamp
    if SAVE_OUTPUTS:
        output_directory.mkdir(parents=True, exist_ok=False)

    print(f"Verified source model-selection study: {source_path}", flush=True)
    umat.compile_umat_meson(input_paths["umat_source"])
    wrapper = model_selection.configured_wrapper(
        compiled_path,
        parameter_path,
        model_spec,
        parameter_specs,
        reference_maxima,
    )
    solver = ZetaCouplingBoundSolver(wrapper, historical.FIT_AXES)
    basis_metrics = compute_coupling_basis_metrics(solver)
    basis_rms = {item.zeta: item.weighted_rms for item in basis_metrics}
    invariant_summaries = invariant_excitation_summary(wrapper, curves, parameter_specs)
    print("Full-data weighted unit-coupling basis RMS:", flush=True)
    for item in basis_metrics:
        print(f"  zeta={item.zeta}: {item.weighted_rms:.8g}", flush=True)

    beta_values = logarithmic_grid(
        solver.parameter_bounds("beta_1"),
        FULL_DATA_GRID_POINTS_PER_AXIS,
    )
    gamma_values = logarithmic_grid(
        solver.parameter_bounds("gamma_1"),
        FULL_DATA_GRID_POINTS_PER_AXIS,
    )
    total_start = time.perf_counter()
    strategies: list[BoundStrategyResult] = []
    folds: list[FoldBoundResult] = []
    for raw_upper in RAW_COUPLING_UPPER_LADDER:
        label = f"raw_Kc_max_{raw_upper:g}"
        print(f"\nComplete-data strategy {label}...", flush=True)
        result = fit_bound_strategy(
            solver,
            label,
            "raw",
            raw_upper,
            {zeta: raw_upper for zeta in ZETA_VALUES},
            beta_values,
            gamma_values,
        )
        strategies.append(result)
        if SAVE_OUTPUTS:
            write_checkpoint(output_directory / "checkpoint.json", strategies, folds)
        print(
            f"  best zeta={result.best_solution.zeta}, "
            f"Kc={result.best_solution.linear_values[COUPLING_PARAMETER]:.8g}, "
            f"mean={result.best_solution.mean_cost:.12g}, "
            f"elapsed={result.elapsed_seconds:.2f}s",
            flush=True,
        )
    for amplitude_upper in AMPLITUDE_UPPER_LADDER:
        label = f"amplitude_max_{amplitude_upper:g}"
        print(f"\nComplete-data strategy {label}...", flush=True)
        result = fit_bound_strategy(
            solver,
            label,
            "amplitude",
            amplitude_upper,
            amplitude_equalized_bounds(basis_rms, amplitude_upper),
            beta_values,
            gamma_values,
        )
        strategies.append(result)
        if SAVE_OUTPUTS:
            write_checkpoint(output_directory / "checkpoint.json", strategies, folds)
        effective_amplitude = (
            result.best_solution.linear_values[COUPLING_PARAMETER]
            * basis_rms[result.best_solution.zeta]
        )
        print(
            f"  best zeta={result.best_solution.zeta}, "
            f"Kc={result.best_solution.linear_values[COUPLING_PARAMETER]:.8g}, "
            f"A={effective_amplitude:.8g}, "
            f"mean={result.best_solution.mean_cost:.12g}, "
            f"elapsed={result.elapsed_seconds:.2f}s",
            flush=True,
        )

    print(
        f"\nRunning leave-one-ratio-out with amplitude ceiling "
        f"{CROSS_VALIDATION_AMPLITUDE_CEILING:g}...",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="ssmccf_coupling_bounds_") as temporary:
        temporary_directory = pathlib.Path(temporary)
        for index, curve in enumerate(curves, start=1):
            print(f"[{index}/{len(curves)}] Omitting ratio {curve.label}...", flush=True)
            training_path = temporary_directory / f"without_ratio_{index:02d}.dat"
            jackknife.write_omitted_training_data(curves, curve.label, training_path)
            fold = fit_amplitude_fold(
                training_path,
                curve,
                parameter_path,
                model_spec,
                parameter_specs,
                reference_maxima,
            )
            folds.append(fold)
            if SAVE_OUTPUTS:
                write_checkpoint(output_directory / "checkpoint.json", strategies, folds)
            fold_basis = {item.zeta: item.weighted_rms for item in fold.basis_metrics}
            fitted_amplitude = (
                fold.free_solution.linear_values[COUPLING_PARAMETER]
                * fold_basis[fold.free_solution.zeta]
            )
            print(
                f"  zeta={fold.free_solution.zeta}, "
                f"Kc={fold.free_solution.linear_values[COUPLING_PARAMETER]:.8g}, "
                f"A={fitted_amplitude:.8g}, elapsed={fold.elapsed_seconds:.2f}s",
                flush=True,
            )

    total_elapsed = time.perf_counter() - total_start
    amplitude_scores = aggregate_amplitude_scores(folds)
    source_scores = historical_scores(source_summary)
    combined_scores = {"no_coupling": source_scores["no_coupling"], **amplitude_scores}
    print("\nEqual-amplitude aggregate held-out scores:", flush=True)
    for label, cost in sorted(combined_scores.items(), key=lambda item: item[1]):
        print(f"  {label}: {cost:.12g}", flush=True)
    if SAVE_OUTPUTS:
        saved = save_outputs(
            output_directory,
            source_path,
            source_summary,
            str(artifact["experiment_context"]["knit"]),
            invariant_summaries,
            basis_metrics,
            strategies,
            folds,
            total_elapsed,
        )
        print(f"Saved coupling-bound calibration: {saved.resolve()}")
    print(f"Total elapsed time: {total_elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
