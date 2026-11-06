"""Select the SV_exp coupling structure with fold-consistent cross-validation.

The configured complete dataset supplies fixed normalized-stress reference maxima.
Every leave-one-ratio-out fold then compares free zeta, each fixed zeta from 2
through 10, and a reduced ``K_coup=0`` model on exactly the same scales. Model
selection prioritizes aggregate held-out cost and then favors simpler models
inside a user-configured predictive tolerance.
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
from Python_Wrapper import umat  # noqa: E402
from Python_Wrapper.model_registry import load_model_spec  # noqa: E402
from Python_Wrapper.separable_solver import (  # noqa: E402
    ProfiledSolution,
    SeparableModelDefinition,
    SVExpSeparableSolver,
    apply_parameter_specs_in_memory,
    deterministic_grid_search,
    logarithmic_grid,
    refine_profiled_solution,
)


FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# USER CONFIGURATION - edit these values before pressing Run in VSCode.
# ---------------------------------------------------------------------------

# In-memory beta/gamma ceiling; 1000 contains every prior full/jackknife optimum.
WIDE_SHAPE_MAXIMUM = 1000.0
# Logarithmic beta/gamma samples per fold; examples are 13, 17, or 21.
GRID_POINTS_PER_AXIS = 17
# Refine every fixed-zeta candidate after its coarse map.
REFINE_EACH_ZETA = True
# Maximum Powell iterations per candidate; examples are 50 or 100.
REFINEMENT_MAX_ITERATIONS = 100
# Log-parameter convergence tolerance; examples are 1e-5 or 1e-6.
REFINEMENT_PARAMETER_TOLERANCE = 1.0e-6
# Objective convergence tolerance; examples are 1e-8 or 1e-10.
REFINEMENT_COST_TOLERANCE = 1.0e-10

# Prefer the least complex model within this fraction of the predictive winner.
# For example, 0.02 means that a model within 2% is considered predictively tied.
SIMPLICITY_TOLERANCE = 0.02
# Save standalone JSON, NPZ, and PNG outputs; False only prints results.
SAVE_OUTPUTS = True
# Display figures interactively; False is convenient for a full VSCode run.
SHOW_PLOTS = False
# Standalone output root; for example fit_0621/<timestamp>.
OUTPUT_ROOT = RELEASE_ARCHIVE_ROOT / "separable_solver_model_selection"


FIXED_ZETA_VALUES = tuple(range(2, 11))
MODEL_LABELS = (
    "no_coupling",
    *(f"fixed_zeta_{zeta}" for zeta in FIXED_ZETA_VALUES),
    "free_zeta",
)
MODEL_COMPLEXITY = {
    "no_coupling": 7,
    **{f"fixed_zeta_{zeta}": 8 for zeta in FIXED_ZETA_VALUES},
    "free_zeta": 9,
}
NO_COUPLING_DEFINITION = SeparableModelDefinition(
    model_id="SV_exp",
    linear_parameters=("mu", "mu2", "mu3", "K_1ani1", "K_1ani2"),
    continuous_shape_parameters=("beta_1", "gamma_1"),
    integer_shape_parameters=(),
)


@dataclass(frozen=True)
class VariantFit:
    """Store one model variant's deterministic training fit."""

    model_label: str
    solution: ProfiledSolution
    parameters: dict[str, float | int]
    direct_raw_cost: float
    direct_mean_cost: float
    maximum_prediction_difference: float


@dataclass(frozen=True)
class DatasetVariantFits:
    """Store all coupling variants fitted to one training dataset."""

    variants: dict[str, VariantFit]
    coupling_profiled_evaluations: int
    no_coupling_profiled_evaluations: int
    elapsed_seconds: float
    training_rows: int


@dataclass(frozen=True)
class FoldResult:
    """Store one omitted ratio's fits and fixed-scale held-out evaluations."""

    omitted_label: str
    omitted_path: pathlib.Path
    fits: DatasetVariantFits
    held_out: dict[str, historical.FullParameterEvaluation]


@dataclass(frozen=True)
class ModelScore:
    """Store one model's aggregate cross-validated prediction score."""

    model_label: str
    free_parameter_count: int
    held_out_raw_cost: float
    held_out_mean_cost: float
    relative_to_predictive_winner: float


def widened_specs(
    saved_specs: Mapping[str, Mapping[str, Any]],
    *,
    no_coupling: bool = False,
) -> dict[str, dict[str, Any]]:
    """Copy historical bounds and optionally fix K_coup to exactly zero."""
    specifications = copy.deepcopy(saved_specs)
    specifications["beta_1"]["maximum"] = float(WIDE_SHAPE_MAXIMUM)
    specifications["gamma_1"]["maximum"] = float(WIDE_SHAPE_MAXIMUM)
    if no_coupling:
        specifications["K_coup"].update(
            {
                "initial_guess": 0.0,
                "minimum": 0.0,
                "maximum": 10.0,
                "fixed": True,
            }
        )
        specifications["zeta"].update({"initial_guess": 2, "fixed": True})
    return specifications


def configured_wrapper(
    data_path: pathlib.Path,
    parameter_path: pathlib.Path,
    model_spec: Any,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    reference_maxima: Mapping[int, float],
) -> Any:
    """Load one fold while retaining the complete dataset's objective scales."""
    loaded_parameters = umat.Load_props(parameter_path, model_spec=model_spec)
    material = umat.Material_model(loaded_parameters)
    wrapper = umat.Umat_wrapper(
        material,
        historical.HISTORICAL_PARA_PARAMS["umat_type"],
    )
    apply_parameter_specs_in_memory(wrapper, parameter_specs)
    wrapper.load_inputs(
        data_path,
        historical.HISTORICAL_PARA_PARAMS["exp_type"],
    )
    wrapper.set_cost_function(
        historical.HISTORICAL_PARA_PARAMS["cost_type"],
        historical.HISTORICAL_PARA_PARAMS["normalized_stress_floor_factor"],
        reference_maxima,
    )
    return wrapper


def full_data_reference_maxima(
    compiled_path: pathlib.Path,
    parameter_path: pathlib.Path,
    model_spec: Any,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> dict[int, float]:
    """Calculate fixed PK2 maxima, for example one value each for xx and yy."""
    loaded_parameters = umat.Load_props(parameter_path, model_spec=model_spec)
    material = umat.Material_model(loaded_parameters)
    wrapper = umat.Umat_wrapper(
        material,
        historical.HISTORICAL_PARA_PARAMS["umat_type"],
    )
    apply_parameter_specs_in_memory(wrapper, parameter_specs)
    wrapper.load_inputs(
        compiled_path,
        historical.HISTORICAL_PARA_PARAMS["exp_type"],
    )
    return {
        axis: float(np.max(np.abs(wrapper.expstressPK2[:, axis])))
        for axis in historical.FIT_AXES
    }


def refine_candidates(
    solver: SVExpSeparableSolver,
    initial_solutions: Sequence[ProfiledSolution],
) -> tuple[ProfiledSolution, ...]:
    """Refine every supplied zeta candidate or retain mapped solutions."""
    if not REFINE_EACH_ZETA:
        return tuple(initial_solutions)
    return tuple(
        refine_profiled_solution(
            solver,
            solution,
            maximum_iterations=REFINEMENT_MAX_ITERATIONS,
            parameter_tolerance=REFINEMENT_PARAMETER_TOLERANCE,
            cost_tolerance=REFINEMENT_COST_TOLERANCE,
        )
        for solution in initial_solutions
    )


def variant_from_solution(
    model_label: str,
    solver: SVExpSeparableSolver,
    solution: ProfiledSolution,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> VariantFit:
    """Complete one solution and verify it through the fixed-scale wrapper cost."""
    parameters = historical.complete_profiled_parameters(solution, parameter_specs)
    direct = historical.evaluate_full_parameter_set(
        solver.wrapper,
        parameters,
        historical.FIT_AXES,
    )
    validation = solver.validate_solution(solution)
    if not np.isclose(direct.raw_cost, solution.cost, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"Profiled and direct costs disagree for {model_label}")
    if validation.existing_wrapper_cost_difference > 1.0e-10:
        raise ValueError(f"Wrapper fixed-scale cost disagrees for {model_label}")
    return VariantFit(
        model_label=model_label,
        solution=solution,
        parameters=parameters,
        direct_raw_cost=direct.raw_cost,
        direct_mean_cost=direct.mean_cost,
        maximum_prediction_difference=(
            validation.maximum_weighted_prediction_difference
        ),
    )


def fit_all_variants(
    data_path: pathlib.Path,
    parameter_path: pathlib.Path,
    model_spec: Any,
    saved_specs: Mapping[str, Mapping[str, Any]],
    reference_maxima: Mapping[int, float],
) -> DatasetVariantFits:
    """Fit free/fixed zeta from one map plus one reduced no-coupling map."""
    start_time = time.perf_counter()
    coupling_specs = widened_specs(saved_specs)
    coupling_wrapper = configured_wrapper(
        data_path,
        parameter_path,
        model_spec,
        coupling_specs,
        reference_maxima,
    )
    coupling_solver = SVExpSeparableSolver(coupling_wrapper, historical.FIT_AXES)
    beta_values = logarithmic_grid(
        coupling_solver.parameter_bounds("beta_1"),
        GRID_POINTS_PER_AXIS,
    )
    gamma_values = logarithmic_grid(
        coupling_solver.parameter_bounds("gamma_1"),
        GRID_POINTS_PER_AXIS,
    )
    coupling_grid = deterministic_grid_search(
        coupling_solver,
        beta_values,
        gamma_values,
        FIXED_ZETA_VALUES,
    )
    refined_by_zeta = refine_candidates(coupling_solver, coupling_grid.best_by_zeta)
    variants = {
        f"fixed_zeta_{solution.zeta}": variant_from_solution(
            f"fixed_zeta_{solution.zeta}",
            coupling_solver,
            solution,
            coupling_specs,
        )
        for solution in refined_by_zeta
    }
    free_solution = min(refined_by_zeta, key=lambda solution: solution.cost)
    variants["free_zeta"] = variant_from_solution(
        "free_zeta",
        coupling_solver,
        free_solution,
        coupling_specs,
    )

    no_coupling_specs = widened_specs(saved_specs, no_coupling=True)
    no_coupling_wrapper = configured_wrapper(
        data_path,
        parameter_path,
        model_spec,
        no_coupling_specs,
        reference_maxima,
    )
    no_coupling_solver = SVExpSeparableSolver(
        no_coupling_wrapper,
        historical.FIT_AXES,
        definition=NO_COUPLING_DEFINITION,
    )
    no_coupling_grid = deterministic_grid_search(
        no_coupling_solver,
        beta_values,
        gamma_values,
        (2,),
    )
    no_coupling_solution = refine_candidates(
        no_coupling_solver,
        no_coupling_grid.best_by_zeta,
    )[0]
    variants["no_coupling"] = variant_from_solution(
        "no_coupling",
        no_coupling_solver,
        no_coupling_solution,
        no_coupling_specs,
    )
    elapsed_seconds = time.perf_counter() - start_time
    return DatasetVariantFits(
        variants={label: variants[label] for label in MODEL_LABELS},
        coupling_profiled_evaluations=coupling_solver.profiled_evaluation_count(),
        no_coupling_profiled_evaluations=(
            no_coupling_solver.profiled_evaluation_count()
        ),
        elapsed_seconds=float(elapsed_seconds),
        training_rows=int(coupling_wrapper.expstressPK2.shape[0]),
    )


def evaluate_variants_on_curve(
    curve: jackknife.RatioCurve,
    fits: DatasetVariantFits,
    parameter_path: pathlib.Path,
    model_spec: Any,
    saved_specs: Mapping[str, Mapping[str, Any]],
    reference_maxima: Mapping[int, float],
) -> dict[str, historical.FullParameterEvaluation]:
    """Evaluate all model variants on one omitted curve using fixed scales."""
    wrapper = configured_wrapper(
        curve.path,
        parameter_path,
        model_spec,
        widened_specs(saved_specs),
        reference_maxima,
    )
    return {
        label: historical.evaluate_full_parameter_set(
            wrapper,
            fits.variants[label].parameters,
            historical.FIT_AXES,
        )
        for label in MODEL_LABELS
    }


def aggregate_model_scores(
    folds: Sequence[FoldResult],
) -> tuple[ModelScore, ...]:
    """Aggregate each held-out curve once and normalize by all observations."""
    if not folds:
        raise ValueError("At least one held-out fold is required")
    observation_count = sum(
        next(iter(fold.held_out.values())).normalized_residuals.size for fold in folds
    )
    raw_costs = {
        label: float(sum(fold.held_out[label].raw_cost for fold in folds))
        for label in MODEL_LABELS
    }
    best_cost = min(raw_costs.values())
    return tuple(
        ModelScore(
            model_label=label,
            free_parameter_count=MODEL_COMPLEXITY[label],
            held_out_raw_cost=raw_costs[label],
            held_out_mean_cost=raw_costs[label] / float(observation_count),
            relative_to_predictive_winner=raw_costs[label] / best_cost - 1.0,
        )
        for label in MODEL_LABELS
    )


def select_models(
    scores: Sequence[ModelScore],
    simplicity_tolerance: float = SIMPLICITY_TOLERANCE,
) -> tuple[ModelScore, ModelScore]:
    """Return the predictive winner and simplest model inside the tolerance."""
    if not scores:
        raise ValueError("At least one model score is required")
    predictive_winner = min(scores, key=lambda score: score.held_out_raw_cost)
    threshold = predictive_winner.held_out_raw_cost * (1.0 + simplicity_tolerance)
    eligible = [score for score in scores if score.held_out_raw_cost <= threshold]
    simplicity_choice = min(
        eligible,
        key=lambda score: (score.free_parameter_count, score.held_out_raw_cost),
    )
    return predictive_winner, simplicity_choice


def parameter_stability(
    full_fit: VariantFit,
    folds: Sequence[FoldResult],
    model_label: str,
) -> dict[str, dict[str, float | str | None]]:
    """Summarize fold ranges for one selected model structure."""
    names = tuple(full_fit.parameters)
    summary: dict[str, dict[str, float | str | None]] = {}
    for name in names:
        full_value = float(full_fit.parameters[name])
        values = np.asarray(
            [float(fold.fits.variants[model_label].parameters[name]) for fold in folds]
        )
        changes = np.abs(values - full_value)
        index = int(np.argmax(changes))
        summary[name] = {
            "full_data_value": full_value,
            "fold_minimum": float(np.min(values)),
            "fold_maximum": float(np.max(values)),
            "maximum_absolute_change": float(changes[index]),
            "maximum_relative_change": (
                float(changes[index] / abs(full_value)) if full_value != 0 else None
            ),
            "most_influential_omission": folds[index].omitted_label,
        }
    return summary


def scalar_evaluation(evaluation: historical.FullParameterEvaluation) -> dict[str, float]:
    """Return held-out scalar values while arrays remain in the NPZ output."""
    return {
        "raw_cost": evaluation.raw_cost,
        "mean_cost": evaluation.mean_cost,
        "normalized_residual_rmse": float(
            np.sqrt(np.mean(evaluation.normalized_residuals**2))
        ),
    }


def create_model_score_figure(
    scores: Sequence[ModelScore],
    full_data_fits: DatasetVariantFits,
    predictive_winner: ModelScore,
    simplicity_choice: ModelScore,
) -> plt.Figure:
    """Compare aggregate held-out and complete-data training objectives."""
    labels = [score.model_label.replace("fixed_zeta_", "zeta=") for score in scores]
    positions = np.arange(len(labels))
    held_out = [score.held_out_mean_cost for score in scores]
    training = [
        full_data_fits.variants[score.model_label].direct_mean_cost for score in scores
    ]
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    colors = [
        "tab:green"
        if score.model_label == simplicity_choice.model_label
        else "tab:orange"
        if score.model_label == predictive_winner.model_label
        else "tab:blue"
        for score in scores
    ]
    axes[0].bar(positions, held_out, color=colors)
    axes[0].set_ylabel("Aggregate held-out mean cost")
    axes[0].set_title("Fold-consistent predictive model comparison")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(positions, training, color=colors)
    axes[1].set_ylabel("Complete-data training mean cost")
    axes[1].set_title("Training objective for the same model structures")
    axes[1].grid(True, axis="y", alpha=0.3)
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=45, ha="right")
    return figure


def create_held_out_heatmap(
    folds: Sequence[FoldResult],
) -> plt.Figure:
    """Plot log10 held-out mean cost for each model and omitted ratio."""
    matrix = np.array(
        [
            [fold.held_out[label].mean_cost for fold in folds]
            for label in MODEL_LABELS
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = axis.imshow(np.log10(matrix), aspect="auto", cmap="viridis")
    axis.set_xticks(
        np.arange(len(folds)),
        [fold.omitted_label for fold in folds],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(MODEL_LABELS)),
        [label.replace("fixed_zeta_", "zeta=") for label in MODEL_LABELS],
    )
    axis.set_xlabel("Held-out loading ratio")
    axis.set_ylabel("Model structure")
    axis.set_title("Held-out objective by model and ratio")
    figure.colorbar(image, ax=axis, label="log10(mean normalized cost)")
    return figure


def write_checkpoint(
    path: pathlib.Path,
    full_data_fits: DatasetVariantFits,
    folds: Sequence[FoldResult],
) -> None:
    """Preserve completed optimization scalars after every expensive fold."""
    payload = {
        "schema_version": 1,
        "full_data_fits": asdict(full_data_fits),
        "completed_folds": [
            {
                "omitted_label": fold.omitted_label,
                "omitted_path": str(fold.omitted_path.resolve()),
                "fits": asdict(fold.fits),
                "held_out": {
                    label: scalar_evaluation(evaluation)
                    for label, evaluation in fold.held_out.items()
                },
            }
            for fold in folds
        ],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_final_outputs(
    output_directory: pathlib.Path,
    artifact: Mapping[str, Any],
    input_paths: Mapping[str, pathlib.Path],
    input_hashes: Mapping[str, str],
    signature: str,
    reference_maxima: Mapping[int, float],
    curves: Sequence[jackknife.RatioCurve],
    full_data_fits: DatasetVariantFits,
    folds: Sequence[FoldResult],
    total_seconds: float,
) -> pathlib.Path:
    """Save the complete model-selection summary, arrays, and figures."""
    scores = aggregate_model_scores(folds)
    predictive_winner, simplicity_choice = select_models(scores)
    summary = {
        "schema_version": 1,
        "study": "SV_exp fixed-weight leave-one-ratio-out model selection",
        "interpretation_limit": (
            "Loading ratios are diagnostic folds, not replicate specimens; "
            "scores are predictive comparisons, not population intervals."
        ),
        "provenance": {
            "reference_fit_code": historical.REFERENCE_FIT_CODE,
            "reconstructed_batch_signature": signature,
            "input_paths": {name: str(path.resolve()) for name, path in input_paths.items()},
            "input_sha256": dict(input_hashes),
            "source_files_modified": False,
        },
        "configuration": {
            "fit_axes": list(historical.FIT_AXES),
            "fixed_full_data_reference_maxima": {
                str(axis): value for axis, value in reference_maxima.items()
            },
            "wide_shape_maximum": WIDE_SHAPE_MAXIMUM,
            "grid_points_per_axis": GRID_POINTS_PER_AXIS,
            "fixed_zeta_values": list(FIXED_ZETA_VALUES),
            "simplicity_tolerance": SIMPLICITY_TOLERANCE,
            "model_complexity": MODEL_COMPLEXITY,
        },
        "full_data_fits": asdict(full_data_fits),
        "folds": [
            {
                "omitted_label": fold.omitted_label,
                "omitted_path": str(fold.omitted_path.resolve()),
                "fits": asdict(fold.fits),
                "held_out": {
                    label: scalar_evaluation(evaluation)
                    for label, evaluation in fold.held_out.items()
                },
            }
            for fold in folds
        ],
        "model_scores": [asdict(score) for score in scores],
        "predictive_winner": asdict(predictive_winner),
        "simplicity_recommendation": asdict(simplicity_choice),
        "recommended_parameter_stability": parameter_stability(
            full_data_fits.variants[simplicity_choice.model_label],
            folds,
            simplicity_choice.model_label,
        ),
        "free_zeta_parameter_stability": parameter_stability(
            full_data_fits.variants["free_zeta"],
            folds,
            "free_zeta",
        ),
        "total_elapsed_seconds": float(total_seconds),
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    held_out_matrix = np.array(
        [
            [fold.held_out[label].mean_cost for fold in folds]
            for label in MODEL_LABELS
        ],
        dtype=float,
    )
    np.savez_compressed(
        output_directory / "model_selection_arrays.npz",
        model_labels=np.asarray(MODEL_LABELS),
        ratio_labels=np.asarray([fold.omitted_label for fold in folds]),
        held_out_mean_costs=held_out_matrix,
        aggregate_held_out_mean_costs=np.asarray(
            [score.held_out_mean_cost for score in scores]
        ),
        full_data_training_mean_costs=np.asarray(
            [full_data_fits.variants[label].direct_mean_cost for label in MODEL_LABELS]
        ),
    )
    score_figure = create_model_score_figure(
        scores,
        full_data_fits,
        predictive_winner,
        simplicity_choice,
    )
    heatmap_figure = create_held_out_heatmap(folds)
    score_figure.savefig(output_directory / "model_score_comparison.png", dpi=200)
    heatmap_figure.savefig(output_directory / "held_out_model_heatmap.png", dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(score_figure)
        plt.close(heatmap_figure)
    return output_directory


def main() -> None:
    """Run fixed-scale cross-validation and select the coupling structure."""
    reference_files = historical.reference_paths()
    historical.require_input_files(reference_files)
    artifact = historical.load_json(reference_files["fit_artifact"])
    checkpoint = historical.load_json(reference_files["checkpoint"])
    historical.validate_reference_artifact(artifact)
    signature = historical.verify_historical_batch_signature(artifact, checkpoint)
    metadata = artifact["metadata"]
    parameter_path = pathlib.Path(metadata["Param_File_Path"])
    compiled_path = pathlib.Path(metadata["Exp_Data_Path"])
    model_spec = load_model_spec(str(artifact["model_id"]))
    model_source = pathlib.Path(model_spec.fortran_source)
    curves = jackknife.discover_ordered_ratio_curves(compiled_path)
    coupling_specs = widened_specs(artifact["parameter_specs"])
    input_paths = {
        **reference_files,
        "parameter_file": parameter_path,
        "compiled_experimental_data": compiled_path,
        "umat_source": model_source,
        **{f"ratio_{curve.label}": curve.path for curve in curves},
    }
    historical.require_input_files(input_paths)
    input_hashes = {
        name: historical.sha256_file(path) for name, path in input_paths.items()
    }
    output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = (
        OUTPUT_ROOT / f"fit_{historical.REFERENCE_FIT_CODE:04d}" / output_timestamp
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    print(f"Verified historical signature: {signature}", flush=True)
    umat.compile_umat_meson(model_source)
    reference_maxima = full_data_reference_maxima(
        compiled_path,
        parameter_path,
        model_spec,
        coupling_specs,
    )
    print(f"Fixed full-data PK2 reference maxima: {reference_maxima}", flush=True)
    print(
        f"Comparing no coupling, fixed zeta {min(FIXED_ZETA_VALUES)}-"
        f"{max(FIXED_ZETA_VALUES)}, and free zeta with "
        f"a {100.0 * SIMPLICITY_TOLERANCE:g}% simplicity tolerance.",
        flush=True,
    )
    total_start = time.perf_counter()
    print("\nFitting all variants to the complete dataset...", flush=True)
    full_data_fits = fit_all_variants(
        compiled_path,
        parameter_path,
        model_spec,
        artifact["parameter_specs"],
        reference_maxima,
    )
    print(
        f"Complete-data variants finished in {full_data_fits.elapsed_seconds:.2f}s; "
        f"free zeta={full_data_fits.variants['free_zeta'].parameters['zeta']}, "
        f"free mean cost={full_data_fits.variants['free_zeta'].direct_mean_cost:.12g}.",
        flush=True,
    )

    folds: list[FoldResult] = []
    write_checkpoint(output_directory / "checkpoint.json", full_data_fits, folds)
    with tempfile.TemporaryDirectory(prefix="ssmccf_model_selection_") as temporary:
        temporary_directory = pathlib.Path(temporary)
        for index, curve in enumerate(curves, start=1):
            print(
                f"\n[{index}/{len(curves)}] Omitting ratio {curve.label}...",
                flush=True,
            )
            training_path = temporary_directory / f"without_ratio_{index:02d}.dat"
            jackknife.write_omitted_training_data(
                curves,
                curve.label,
                training_path,
            )
            fits = fit_all_variants(
                training_path,
                parameter_path,
                model_spec,
                artifact["parameter_specs"],
                reference_maxima,
            )
            held_out = evaluate_variants_on_curve(
                curve,
                fits,
                parameter_path,
                model_spec,
                artifact["parameter_specs"],
                reference_maxima,
            )
            fold = FoldResult(
                omitted_label=curve.label,
                omitted_path=curve.path,
                fits=fits,
                held_out=held_out,
            )
            folds.append(fold)
            write_checkpoint(output_directory / "checkpoint.json", full_data_fits, folds)
            interim_scores = aggregate_model_scores(folds)
            interim_winner, interim_simple = select_models(interim_scores)
            print(
                f"Fold finished in {fits.elapsed_seconds:.2f}s; "
                f"free zeta={fits.variants['free_zeta'].parameters['zeta']}; "
                f"current predictive winner={interim_winner.model_label}; "
                f"current simple choice={interim_simple.model_label}.",
                flush=True,
            )

    total_seconds = time.perf_counter() - total_start
    scores = aggregate_model_scores(folds)
    predictive_winner, simplicity_choice = select_models(scores)
    print("\nAggregate held-out model scores", flush=True)
    for score in sorted(scores, key=lambda value: value.held_out_mean_cost):
        print(
            f"{score.model_label}: mean={score.held_out_mean_cost:.12g}, "
            f"relative={100.0 * score.relative_to_predictive_winner:.4f}%, "
            f"parameters={score.free_parameter_count}",
            flush=True,
        )
    print(f"Predictive winner: {predictive_winner.model_label}", flush=True)
    print(
        f"Simplicity recommendation: {simplicity_choice.model_label}",
        flush=True,
    )
    print(f"Total elapsed time: {total_seconds:.2f}s", flush=True)

    if SAVE_OUTPUTS:
        saved = save_final_outputs(
            output_directory,
            artifact,
            input_paths,
            input_hashes,
            signature,
            reference_maxima,
            curves,
            full_data_fits,
            folds,
            total_seconds,
        )
        print(f"Saved model-selection study: {saved}", flush=True)


if __name__ == "__main__":
    main()
