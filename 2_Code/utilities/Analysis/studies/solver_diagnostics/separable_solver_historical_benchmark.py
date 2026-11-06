"""Benchmark the SV_exp separable solver against a configured historical fit.

The script is independent from the fitting batch and simulated-annealing code.
It reads the immutable fit artifact, applies its saved parameter specifications
in memory, validates the historical batch signature, and evaluates both methods
through the same current UMAT objective. It never rewrites the parameter file,
fit artifact, checkpoint, or shared results CSV.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from release_paths import json
import pathlib
import sys
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

from Python_Wrapper import umat  # noqa: E402
from Python_Wrapper.model_registry import load_model_spec  # noqa: E402
from Python_Wrapper.separable_solver import (  # noqa: E402
    GridSearchResult,
    ProfiledSolution,
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

# Historical fit code used for the like-for-like benchmark; for example 621.
REFERENCE_FIT_CODE = 621
# Expected knit stored in the artifact; examples are "Rib" or "Interlock".
REFERENCE_KNIT = "Rib"
# Batch-run timestamp containing the checkpoint; for example 20260724_170524.
REFERENCE_BATCH_RUN = "20260724_170524"
# Historical worker number containing the checkpoint; fit 621 came from job 6.
REFERENCE_JOB_INDEX = 6
# Original batch job name used in its signature; for example rib_sv_exp_raw_all.
REFERENCE_JOB_NAME = "rib_sv_exp_raw_all"
# Stress components fitted historically; (0, 1) means xx and yy.
FIT_AXES = (0, 1)

# Logarithmic beta/gamma samples per zeta; examples are 11, 21, or 41.
GRID_POINTS_PER_AXIS = 21
# Refine every integer-zeta map minimum after the deterministic grid search.
REFINE_EACH_ZETA = True
# Maximum Powell iterations for each zeta; examples are 50 or 100.
REFINEMENT_MAX_ITERATIONS = 100
# Log-parameter tolerance for Powell; examples are 1e-5 or 1e-6.
REFINEMENT_PARAMETER_TOLERANCE = 1.0e-6
# Objective tolerance for Powell; examples are 1e-8 or 1e-10.
REFINEMENT_COST_TOLERANCE = 1.0e-10

# Save standalone JSON, NPZ, and PNG outputs; False only prints the comparison.
SAVE_OUTPUTS = True
# Display plots in an interactive window; False is convenient for a full run.
SHOW_PLOTS = False
# Root for standalone benchmark artifacts; for example fit_0621/<timestamp>.
OUTPUT_ROOT = RELEASE_ARCHIVE_ROOT / "separable_solver_benchmarks"

# These settings reconstruct job 6's checkpoint signature exactly. Changing a
# value such as fit_axes to [0, 1, 3] intentionally invalidates provenance.
HISTORICAL_PARA_PARAMS: dict[str, Any] = {
    "exp_type": ["H", "P"],
    "umat_type": ["F", "Sigma"],
    "n_sims": 30,
    "temp": 0.1,
    "temp_change": 0.6,
    "n_mc_steps": 5000,
    "exc_thr": False,
    "stepsize": 0.05,
    "change_stepsize": 0.85,
    "fit_axes": list(FIT_AXES),
    "cost_type": "normalized",
    "normalized_stress_floor_factor": 1,
}


@dataclass(frozen=True)
class FullParameterEvaluation:
    """Store one direct current-UMAT evaluation of a complete parameter set."""

    raw_cost: float
    mean_cost: float
    predicted_pk2: FloatArray
    normalized_residuals: FloatArray


def reference_paths() -> dict[str, pathlib.Path]:
    """Return the fit artifact and worker checkpoint, for example fit 0621."""
    fit_folder = RELEASE_ARCHIVE_ROOT / "fits" / f"{REFERENCE_FIT_CODE:04d}"
    worker_folder = (
        RELEASE_ARCHIVE_ROOT
        / "fitting_batch_runs"
        / REFERENCE_BATCH_RUN
        / f"job_{REFERENCE_JOB_INDEX:02d}"
    )
    return {
        "fit_artifact": fit_folder / "fit.json",
        "checkpoint": worker_folder / "run_checkpoint.json",
    }


def load_json(path: pathlib.Path) -> dict[str, Any]:
    """Read one JSON object, for example the saved fit artifact."""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def sha256_file(path: pathlib.Path) -> str:
    """Hash one current input so a benchmark run can be reproduced later."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_reference_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject a fit that is not the configured raw SV_exp comparison case."""
    metadata = artifact.get("metadata", {})
    experiment = artifact.get("experiment_context", {})
    required_top_level = {"fit_code", "model_id", "parameters", "parameter_specs"}
    missing = sorted(required_top_level - set(artifact))
    if missing:
        raise ValueError("Reference fit artifact is missing: " + ", ".join(missing))
    if int(artifact["fit_code"]) != REFERENCE_FIT_CODE:
        raise ValueError(f"Expected fit {REFERENCE_FIT_CODE}, found {artifact['fit_code']}")
    if artifact["model_id"] != "SV_exp":
        raise ValueError(f"Expected SV_exp, found {artifact['model_id']!r}")
    if (
        experiment.get("knit") != REFERENCE_KNIT
        or experiment.get("exp_data_type") != "raw"
    ):
        raise ValueError(
            f"Reference fit must be the raw {REFERENCE_KNIT} SV_exp case"
        )
    if experiment.get("exp_type") != HISTORICAL_PARA_PARAMS["exp_type"]:
        raise ValueError("Reference experimental stress/strain types changed")
    if [experiment.get("umat_strain"), experiment.get("umat_stress")] != (
        HISTORICAL_PARA_PARAMS["umat_type"]
    ):
        raise ValueError("Reference UMAT stress/strain types changed")
    if metadata.get("Cost_Type") != HISTORICAL_PARA_PARAMS["cost_type"]:
        raise ValueError("Reference cost type changed")
    if float(metadata.get("Normalized_Stress_Floor_Factor")) != float(
        HISTORICAL_PARA_PARAMS["normalized_stress_floor_factor"]
    ):
        raise ValueError("Reference normalized-stress floor changed")


def verify_historical_batch_signature(
    artifact: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> str:
    """Prove axes and objective settings by reproducing the worker signature."""
    from Analysis.fitting import run_fitting_batch as batch

    metadata = artifact["metadata"]
    knit_type = batch.knit_type_from_name(str(metadata["Knit"]))
    job = batch.default_job(
        REFERENCE_JOB_NAME,
        str(artifact["model_id"]),
        knit_type,
        exp_data_type=str(metadata["Exp_Data_Type"]),
        exp_path=pathlib.Path(metadata["Exp_Data_Path"]),
        props_path=pathlib.Path(metadata["Param_File_Path"]),
    )
    job["n_runs"] = int(metadata["N_Runs"])
    job["para_params"] = dict(HISTORICAL_PARA_PARAMS)
    reconstructed = batch.job_signature(job, REFERENCE_JOB_INDEX)
    recorded = str(checkpoint.get("signature", ""))
    if reconstructed != recorded:
        raise ValueError(
            "Historical batch signature could not be reproduced; the fitted axes "
            "or another reference setting may no longer match"
        )
    return reconstructed


def require_input_files(paths: Mapping[str, pathlib.Path]) -> None:
    """Fail before compilation when any immutable reference input is absent."""
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark inputs:\n" + "\n".join(missing))


def set_parameter_value(wrapper: Any, name: str, value: float | int) -> None:
    """Set one value while preserving metadata, for example zeta=9 as an int."""
    _, data_type, lower, upper, counter, fixed = wrapper.getv(name)
    typed_value = int(round(value)) if data_type is int else float(value)
    wrapper.setv(name, typed_value, data_type, lower, upper, counter, fixed)


def direct_prediction_pk2(wrapper: Any) -> FloatArray:
    """Run the current UMAT and convert its prediction to PK2 when necessary."""
    predicted = np.asarray(wrapper.calc_stress_umat_Abaqus(), dtype=float)
    if wrapper.umat_out_stress_type != "S":
        predicted = np.asarray(
            wrapper.stress_to_PK2(predicted, wrapper.umat_out_stress_type),
            dtype=float,
        )
    return predicted


def normalized_residuals(
    experimental_pk2: FloatArray,
    predicted_pk2: FloatArray,
    axes: Sequence[int],
    floor_factor: float,
    reference_maxima: Mapping[int, float] | None = None,
) -> FloatArray:
    """Return residual/scale columns used by the normalized historical cost."""
    columns: list[FloatArray] = []
    for axis in axes:
        experimental = np.asarray(experimental_pk2[:, axis], dtype=float)
        maximum = float(
            np.max(np.abs(experimental))
            if reference_maxima is None or axis not in reference_maxima
            else reference_maxima[axis]
        )
        scale = np.abs(experimental) + float(floor_factor) * maximum
        columns.append((np.asarray(predicted_pk2[:, axis]) - experimental) / scale)
    return np.column_stack(columns)


def evaluate_full_parameter_set(
    wrapper: Any,
    values: Mapping[str, float | int],
    axes: Sequence[int],
) -> FullParameterEvaluation:
    """Evaluate arbitrary saved values through the established wrapper cost."""
    parameter_names = set(wrapper.get_props_dic())
    if set(values) != parameter_names:
        missing = sorted(parameter_names - set(values))
        extra = sorted(set(values) - parameter_names)
        raise ValueError(f"Complete parameter set required; missing={missing}, extra={extra}")

    snapshot = {name: tuple(record) for name, record in wrapper.get_props_dic().items()}
    try:
        for name, value in values.items():
            set_parameter_value(wrapper, name, value)
        predicted = direct_prediction_pk2(wrapper)
        raw_cost = float(wrapper.cost_calculations(get_plots=False, axis=list(axes)))
    finally:
        for name, record in snapshot.items():
            wrapper.setv(name, *record)

    observation_count = int(predicted.shape[0] * len(tuple(axes)))
    residuals = normalized_residuals(
        np.asarray(wrapper.expstressPK2, dtype=float),
        predicted,
        axes,
        wrapper.normalized_stress_floor_factor,
        getattr(wrapper, "normalized_stress_reference_maxima", None),
    )
    return FullParameterEvaluation(
        raw_cost=raw_cost,
        mean_cost=raw_cost / float(observation_count),
        predicted_pk2=predicted,
        normalized_residuals=residuals,
    )


def complete_profiled_parameters(
    solution: ProfiledSolution,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | int]:
    """Combine profiled values with fixed C/theta values from the artifact."""
    values: dict[str, float | int] = {
        name: specification["initial_guess"]
        for name, specification in parameter_specs.items()
    }
    values.update(solution.linear_values)
    # Include model-specific shapes such as Stapleton beta_2 and gamma_2 while
    # preserving the legacy beta_1/gamma_1/zeta compatibility attributes.
    values.update(solution.complete_shape_values())
    return values


def print_zeta_progress(zeta: int, solution: ProfiledSolution) -> None:
    """Print one mapped result, for example zeta=5 and mean cost=0.0015."""
    print(
        f"Mapped zeta={zeta}: mean cost={solution.mean_cost:.12g}, "
        f"beta_1={solution.beta_1:.8g}, gamma_1={solution.gamma_1:.8g}"
    )


def refine_grid_minima(
    solver: SVExpSeparableSolver,
    grid_result: GridSearchResult,
) -> tuple[ProfiledSolution, ...]:
    """Polish every integer-zeta minimum or retain the grid-only results."""
    if not REFINE_EACH_ZETA:
        return grid_result.best_by_zeta
    refined: list[ProfiledSolution] = []
    for initial in grid_result.best_by_zeta:
        solution = refine_profiled_solution(
            solver,
            initial,
            maximum_iterations=REFINEMENT_MAX_ITERATIONS,
            parameter_tolerance=REFINEMENT_PARAMETER_TOLERANCE,
            cost_tolerance=REFINEMENT_COST_TOLERANCE,
        )
        refined.append(solution)
        print(
            f"Refined zeta={solution.zeta}: mean cost={solution.mean_cost:.12g}, "
            f"beta_1={solution.beta_1:.8g}, gamma_1={solution.gamma_1:.8g}"
        )
    return tuple(refined)


def create_cost_figure(
    grid_result: GridSearchResult,
    final_by_zeta: Sequence[ProfiledSolution],
    reference_mean_cost: float,
    observation_count: int,
) -> plt.Figure:
    """Plot the best cost map and compare its zeta profile with fit 621."""
    best = min(final_by_zeta, key=lambda solution: solution.cost)
    best_index = grid_result.zeta_values.index(best.zeta)
    mean_cost_map = grid_result.costs[best_index] / float(observation_count)
    log_cost = np.log10(np.maximum(mean_cost_map, np.finfo(float).tiny))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    contour = axes[0].contourf(
        grid_result.beta_values,
        grid_result.gamma_values,
        log_cost.T,
        levels=30,
    )
    axes[0].scatter([best.beta_1], [best.gamma_1], color="red", marker="x", s=70)
    axes[0].set(xscale="log", yscale="log", xlabel=r"$\beta_1$", ylabel=r"$\gamma_1$")
    axes[0].set_title(f"Mean profiled cost map, zeta={best.zeta}")
    figure.colorbar(contour, ax=axes[0], label=r"$\log_{10}(\mathrm{mean\ cost})$")

    zeta_values = [solution.zeta for solution in final_by_zeta]
    zeta_costs = [solution.mean_cost for solution in final_by_zeta]
    axes[1].plot(zeta_values, zeta_costs, marker="o", label="Separable solver")
    axes[1].axhline(
        reference_mean_cost,
        color="tab:orange",
        linestyle="--",
        label=f"Fit {REFERENCE_FIT_CODE}",
    )
    axes[1].scatter([best.zeta], [best.mean_cost], color="red", zorder=3)
    axes[1].set(yscale="log", xlabel=r"$\zeta$", ylabel="Mean normalized cost")
    axes[1].set_title("Best beta/gamma solution for every integer zeta")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    return figure


def create_residual_figure(
    experimental_pk2: FloatArray,
    reference: FullParameterEvaluation,
    separable: FullParameterEvaluation,
) -> plt.Figure:
    """Compare predicted stresses and normalized residuals on both fitted axes."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axis_labels = ("xx", "yy")
    for column, (axis, label) in enumerate(zip(FIT_AXES, axis_labels)):
        experimental = experimental_pk2[:, axis]
        reference_prediction = reference.predicted_pk2[:, axis]
        separable_prediction = separable.predicted_pk2[:, axis]
        lower = float(min(experimental.min(), reference_prediction.min(), separable_prediction.min()))
        upper = float(max(experimental.max(), reference_prediction.max(), separable_prediction.max()))
        axes[0, column].plot([lower, upper], [lower, upper], color="black", linewidth=1)
        axes[0, column].scatter(
            experimental,
            reference_prediction,
            s=8,
            alpha=0.45,
            label=f"Fit {REFERENCE_FIT_CODE}",
        )
        axes[0, column].scatter(
            experimental,
            separable_prediction,
            s=8,
            alpha=0.45,
            label="Separable",
        )
        axes[0, column].set_xlabel(f"Experimental PK2 {label}")
        axes[0, column].set_ylabel(f"Predicted PK2 {label}")
        axes[0, column].set_title(f"{label} prediction agreement")
        axes[0, column].legend()

        axes[1, column].hist(
            reference.normalized_residuals[:, column],
            bins=35,
            alpha=0.55,
            label=f"Fit {REFERENCE_FIT_CODE}",
        )
        axes[1, column].hist(
            separable.normalized_residuals[:, column],
            bins=35,
            alpha=0.55,
            label="Separable",
        )
        axes[1, column].set_xlabel(f"Normalized residual {label}")
        axes[1, column].set_ylabel("Observation count")
        axes[1, column].set_title(f"{label} objective residuals")
        axes[1, column].legend()
    return figure


def serializable_evaluation(evaluation: FullParameterEvaluation) -> dict[str, float]:
    """Return only scalar evaluation fields for JSON; arrays go to the NPZ."""
    return {
        "raw_cost": evaluation.raw_cost,
        "mean_cost": evaluation.mean_cost,
        "maximum_absolute_normalized_residual": float(
            np.max(np.abs(evaluation.normalized_residuals))
        ),
        "normalized_residual_rmse": float(
            np.sqrt(np.mean(evaluation.normalized_residuals**2))
        ),
    }


def save_outputs(
    artifact: Mapping[str, Any],
    input_paths: Mapping[str, pathlib.Path],
    input_hashes: Mapping[str, str],
    signature: str,
    solver: SVExpSeparableSolver,
    grid_result: GridSearchResult,
    final_by_zeta: Sequence[ProfiledSolution],
    best_parameters: Mapping[str, float | int],
    reference: FullParameterEvaluation,
    separable: FullParameterEvaluation,
    search_seconds: float,
) -> pathlib.Path:
    """Write standalone benchmark artifacts without modifying fitting outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = OUTPUT_ROOT / f"fit_{REFERENCE_FIT_CODE:04d}" / timestamp
    output_directory.mkdir(parents=True, exist_ok=False)
    best = min(final_by_zeta, key=lambda solution: solution.cost)
    validation = solver.validate_solution(best)
    reported_mean_cost = float(artifact["metadata"]["Min_Cost"])
    improvement = reference.mean_cost - separable.mean_cost
    reference_parameters = artifact["parameters"]
    parameter_comparison = {
        name: {
            "historical_sa": reference_parameters[name],
            "separable": best_parameters[name],
            "difference": float(best_parameters[name]) - float(reference_parameters[name]),
        }
        for name in reference_parameters
    }

    summary = {
        "schema_version": 1,
        "benchmark": f"historical fit {REFERENCE_FIT_CODE} versus SV_exp separable solver",
        "provenance": {
            "reconstructed_batch_signature": signature,
            "signature_matches_checkpoint": True,
            "fit_axes": list(FIT_AXES),
            "input_paths": {name: str(path.resolve()) for name, path in input_paths.items()},
            "input_sha256": dict(input_hashes),
            "historical_source_hash_available": False,
            "historical_source_hash_note": (
                "The 2026 fit artifact records the UMAT source path but not its content hash."
            ),
        },
        "configuration": {
            "historical_para_params": HISTORICAL_PARA_PARAMS,
            "grid_points_per_axis": GRID_POINTS_PER_AXIS,
            "refine_each_zeta": REFINE_EACH_ZETA,
            "refinement_max_iterations": REFINEMENT_MAX_ITERATIONS,
            "zeta_values": list(grid_result.zeta_values),
            "fitted_observation_count": int(solver.problem.target.size),
        },
        "historical_sa": {
            "reported_mean_cost": reported_mean_cost,
            "recalculated": serializable_evaluation(reference),
            "reported_minus_recalculated": reported_mean_cost - reference.mean_cost,
            "parameters": dict(reference_parameters),
            "saved_wall_seconds": None,
        },
        "separable_solver": {
            "grid_best": asdict(grid_result.best_solution),
            "final_best": asdict(best),
            "final_by_zeta": [asdict(solution) for solution in final_by_zeta],
            "direct_evaluation": serializable_evaluation(separable),
            "direct_profile_validation": asdict(validation),
            "parameters": dict(best_parameters),
            "profiled_evaluation_count": solver.profiled_evaluation_count(),
            "basis_cache_sizes": solver.cache_sizes(),
            "search_wall_seconds": float(search_seconds),
        },
        "comparison": {
            "mean_cost_reduction": improvement,
            "relative_mean_cost_reduction": improvement / reference.mean_cost,
            "separable_to_historical_cost_ratio": separable.mean_cost / reference.mean_cost,
            "parameter_values": parameter_comparison,
        },
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    np.savez_compressed(
        output_directory / "benchmark_arrays.npz",
        beta_values=grid_result.beta_values,
        gamma_values=grid_result.gamma_values,
        zeta_values=np.asarray(grid_result.zeta_values, dtype=int),
        costs=grid_result.costs,
        experimental_pk2=np.asarray(solver.wrapper.expstressPK2, dtype=float),
        historical_predicted_pk2=reference.predicted_pk2,
        separable_predicted_pk2=separable.predicted_pk2,
        historical_normalized_residuals=reference.normalized_residuals,
        separable_normalized_residuals=separable.normalized_residuals,
    )
    cost_figure = create_cost_figure(
        grid_result,
        final_by_zeta,
        reference.mean_cost,
        solver.problem.target.size,
    )
    residual_figure = create_residual_figure(
        np.asarray(solver.wrapper.expstressPK2, dtype=float),
        reference,
        separable,
    )
    cost_figure.savefig(output_directory / "cost_comparison.png", dpi=200)
    residual_figure.savefig(output_directory / "prediction_residual_comparison.png", dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(cost_figure)
        plt.close(residual_figure)
    return output_directory


def main() -> None:
    """Reconstruct fit 621, run the separable search, and save the comparison."""
    immutable_paths = reference_paths()
    require_input_files(immutable_paths)
    artifact = load_json(immutable_paths["fit_artifact"])
    checkpoint = load_json(immutable_paths["checkpoint"])
    validate_reference_artifact(artifact)
    signature = verify_historical_batch_signature(artifact, checkpoint)

    metadata = artifact["metadata"]
    parameter_path = pathlib.Path(metadata["Param_File_Path"])
    experimental_path = pathlib.Path(metadata["Exp_Data_Path"])
    model_spec = load_model_spec(str(artifact["model_id"]))
    model_source = pathlib.Path(model_spec.fortran_source)
    runtime_inputs = {
        **immutable_paths,
        "parameter_file": parameter_path,
        "experimental_data": experimental_path,
        "umat_source": model_source,
    }
    require_input_files(runtime_inputs)
    input_hashes = {name: sha256_file(path) for name, path in runtime_inputs.items()}

    print(f"Historical fit: {REFERENCE_FIT_CODE}")
    print(f"Verified checkpoint signature: {signature}")
    print(f"Experimental data: {experimental_path}")
    print("Applying saved fit-621 parameter specifications in memory only.")
    umat.compile_umat_meson(model_source)
    loaded_parameters = umat.Load_props(parameter_path, model_spec=model_spec)
    material = umat.Material_model(loaded_parameters)
    wrapper = umat.Umat_wrapper(material, HISTORICAL_PARA_PARAMS["umat_type"])
    apply_parameter_specs_in_memory(wrapper, artifact["parameter_specs"])
    wrapper.load_inputs(experimental_path, HISTORICAL_PARA_PARAMS["exp_type"])
    wrapper.set_cost_function(
        HISTORICAL_PARA_PARAMS["cost_type"],
        HISTORICAL_PARA_PARAMS["normalized_stress_floor_factor"],
    )
    solver = SVExpSeparableSolver(wrapper, FIT_AXES)

    reference_evaluation = evaluate_full_parameter_set(
        wrapper,
        artifact["parameters"],
        FIT_AXES,
    )
    reported_mean_cost = float(metadata["Min_Cost"])
    if not np.isclose(
        reference_evaluation.mean_cost,
        reported_mean_cost,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Current-UMAT recalculation does not reproduce fit 621's saved cost; "
            "a strict like-for-like comparison is not valid"
        )
    print(
        f"Reproduced historical mean cost: {reference_evaluation.mean_cost:.12g} "
        f"(saved {reported_mean_cost:.12g})"
    )

    beta_values = logarithmic_grid(
        solver.parameter_bounds("beta_1"), GRID_POINTS_PER_AXIS
    )
    gamma_values = logarithmic_grid(
        solver.parameter_bounds("gamma_1"), GRID_POINTS_PER_AXIS
    )
    zeta_values = solver.integer_values("zeta")
    print(
        f"Mapping {len(zeta_values)} zeta values on a "
        f"{len(beta_values)} x {len(gamma_values)} historical-bound grid."
    )
    start_time = time.perf_counter()
    grid_result = deterministic_grid_search(
        solver,
        beta_values,
        gamma_values,
        zeta_values,
        progress_callback=print_zeta_progress,
    )
    final_by_zeta = refine_grid_minima(solver, grid_result)
    search_seconds = time.perf_counter() - start_time
    best = min(final_by_zeta, key=lambda solution: solution.cost)
    best_parameters = complete_profiled_parameters(best, artifact["parameter_specs"])
    separable_evaluation = evaluate_full_parameter_set(
        wrapper,
        best_parameters,
        FIT_AXES,
    )
    profile_validation = solver.validate_solution(best)
    if not np.isclose(separable_evaluation.raw_cost, best.cost, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Profiled and direct separable-solver objectives do not agree")

    relative_reduction = (
        reference_evaluation.mean_cost - separable_evaluation.mean_cost
    ) / reference_evaluation.mean_cost
    print("\nLike-for-like result")
    print(f"Historical fit {REFERENCE_FIT_CODE}: {reference_evaluation.mean_cost:.12g}")
    print(f"Separable solver:       {separable_evaluation.mean_cost:.12g}")
    print(f"Relative cost reduction: {100.0 * relative_reduction:.3f}%")
    print(
        f"Best shape: beta_1={best.beta_1:.10g}, gamma_1={best.gamma_1:.10g}, "
        f"zeta={best.zeta}"
    )
    print(f"Profiled evaluations: {solver.profiled_evaluation_count()}")
    print(f"Search wall time: {search_seconds:.2f} seconds")
    print(
        "Maximum profiled/direct weighted prediction difference: "
        f"{profile_validation.maximum_weighted_prediction_difference:.3e}"
    )

    if SAVE_OUTPUTS:
        output_directory = save_outputs(
            artifact,
            runtime_inputs,
            input_hashes,
            signature,
            solver,
            grid_result,
            final_by_zeta,
            best_parameters,
            reference_evaluation,
            separable_evaluation,
            search_seconds,
        )
        print(f"Saved standalone benchmark: {output_directory}")
    elif SHOW_PLOTS:
        create_cost_figure(
            grid_result,
            final_by_zeta,
            reference_evaluation.mean_cost,
            solver.problem.target.size,
        )
        create_residual_figure(
            np.asarray(wrapper.expstressPK2, dtype=float),
            reference_evaluation,
            separable_evaluation,
        )
        plt.show()


if __name__ == "__main__":
    main()
