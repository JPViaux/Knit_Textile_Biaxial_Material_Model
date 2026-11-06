"""Run a deterministic leave-one-loading-ratio-out SV_exp diagnostic.

Each complete Rib loading curve is omitted once, the widened-bound separable
problem is refitted, and the resulting parameter movement and held-out error
are compared with the full-data fit. The analysis is a sensitivity jackknife,
not a population confidence interval, because only one specimen exists per
ratio. Existing experimental files, ``.dat`` bounds, fitting code, and shared
results remain unchanged.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import copy
from release_paths import json
import pathlib
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import matplotlib.colors as colors
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
from Python_Wrapper import umat  # noqa: E402
from Python_Wrapper.model_registry import load_model_spec  # noqa: E402
from Python_Wrapper.separable_solver import (  # noqa: E402
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

# In-memory shape-rate ceiling; 1000 contains the full-data optimum near beta=199.
WIDE_SHAPE_MAXIMUM = 1000.0
# Coarse log-grid points per refit; examples are 13, 17, or 21.
GRID_POINTS_PER_AXIS = 17
# Refine every integer-zeta minimum after each coarse deterministic map.
REFINE_EACH_ZETA = True
# Maximum Powell iterations per zeta; examples are 50 or 100.
REFINEMENT_MAX_ITERATIONS = 100
# Log-parameter convergence tolerance; examples are 1e-5 or 1e-6.
REFINEMENT_PARAMETER_TOLERANCE = 1.0e-6
# Objective convergence tolerance; examples are 1e-8 or 1e-10.
REFINEMENT_COST_TOLERANCE = 1.0e-10

# Save standalone JSON, NPZ, and PNG outputs; False only prints results.
SAVE_OUTPUTS = True
# Display figures interactively; False is convenient for a complete VSCode run.
SHOW_PLOTS = False
# Standalone result root; for example fit_0621/<timestamp>.
OUTPUT_ROOT = RELEASE_ARCHIVE_ROOT / "separable_solver_leave_one_ratio_out"

# Free parameters reported in influence diagnostics; fixed C/theta are excluded.
REPORTED_PARAMETERS = (
    "mu",
    "mu2",
    "mu3",
    "K_1ani1",
    "beta_1",
    "K_1ani2",
    "gamma_1",
    "K_coup",
    "zeta",
)


@dataclass(frozen=True)
class RatioCurve:
    """Describe one exact loading-ratio block inside the complete data file."""

    label: str
    path: pathlib.Path
    data: FloatArray
    start_row: int
    stop_row: int


@dataclass(frozen=True)
class DatasetFit:
    """Store one deterministic full or leave-one-out fit."""

    solution: ProfiledSolution
    parameters: dict[str, float | int]
    direct_raw_cost: float
    direct_mean_cost: float
    validation_maximum_prediction_difference: float
    profiled_evaluations: int
    basis_cache_sizes: dict[str, int]
    elapsed_seconds: float
    training_rows: int


@dataclass(frozen=True)
class OmissionResult:
    """Store one omitted ratio's fit and two held-out evaluations."""

    omitted_label: str
    omitted_path: pathlib.Path
    fit: DatasetFit
    full_fit_held_out: historical.FullParameterEvaluation
    omission_fit_held_out: historical.FullParameterEvaluation


def ratio_label_from_stem(stem: str) -> str:
    """Convert a filename such as ``0p25 to 1`` to the label ``0.25:1``."""
    parts = stem.lower().split(" to ")
    if len(parts) != 2:
        raise ValueError(f"Ratio filename must contain ' to ': {stem!r}")
    formatted = [part.replace("p", ".") for part in parts]
    return f"{formatted[0]}:{formatted[1]}"


def discover_ordered_ratio_curves(
    compiled_path: pathlib.Path,
) -> tuple[RatioCurve, ...]:
    """Prove that separate ratio files exactly concatenate to the compiled file."""
    compiled = np.atleast_2d(np.loadtxt(compiled_path, dtype=float))
    candidates = sorted(compiled_path.parent.glob("* to *.dat"))
    if not candidates:
        raise FileNotFoundError(f"No individual ratio files found beside {compiled_path}")
    remaining = {
        path: np.atleast_2d(np.loadtxt(path, dtype=float)) for path in candidates
    }
    curves: list[RatioCurve] = []
    start_row = 0
    while remaining:
        matches = [
            (path, data)
            for path, data in remaining.items()
            if start_row + data.shape[0] <= compiled.shape[0]
            and np.array_equal(
                compiled[start_row : start_row + data.shape[0]],
                data,
            )
        ]
        if len(matches) != 1:
            names = [path.name for path, _ in matches]
            raise ValueError(
                f"Expected one exact ratio block at compiled row {start_row}; "
                f"matches={names}"
            )
        path, data = matches[0]
        stop_row = start_row + data.shape[0]
        curves.append(
            RatioCurve(
                label=ratio_label_from_stem(path.stem),
                path=path,
                data=data,
                start_row=start_row,
                stop_row=stop_row,
            )
        )
        del remaining[path]
        start_row = stop_row
    if start_row != compiled.shape[0]:
        raise ValueError(
            f"Individual curves cover {start_row} rows, compiled file has {compiled.shape[0]}"
        )
    return tuple(curves)


def write_omitted_training_data(
    curves: Sequence[RatioCurve],
    omitted_label: str,
    output_path: pathlib.Path,
) -> int:
    """Write all curves except one to a temporary fitting file."""
    retained = [curve.data for curve in curves if curve.label != omitted_label]
    if len(retained) != len(curves) - 1:
        raise ValueError(f"Expected exactly one curve labeled {omitted_label!r}")
    combined = np.vstack(retained)
    np.savetxt(output_path, combined, delimiter="\t", fmt="%.17g")
    return int(combined.shape[0])


def widened_parameter_specs(
    saved_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Copy historical specifications and widen only beta/gamma in memory."""
    specifications = copy.deepcopy(saved_specs)
    specifications["beta_1"]["maximum"] = float(WIDE_SHAPE_MAXIMUM)
    specifications["gamma_1"]["maximum"] = float(WIDE_SHAPE_MAXIMUM)
    return specifications


def configured_wrapper(
    data_path: pathlib.Path,
    parameter_path: pathlib.Path,
    model_spec: Any,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> Any:
    """Load one dataset with the historical objective and in-memory wide bounds."""
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
    )
    return wrapper


def fit_dataset(
    data_path: pathlib.Path,
    parameter_path: pathlib.Path,
    model_spec: Any,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> DatasetFit:
    """Map and refine one full or leave-one-ratio-out fitting problem."""
    wrapper = configured_wrapper(data_path, parameter_path, model_spec, parameter_specs)
    solver = SVExpSeparableSolver(wrapper, historical.FIT_AXES)
    beta_values = logarithmic_grid(
        solver.parameter_bounds("beta_1"),
        GRID_POINTS_PER_AXIS,
    )
    gamma_values = logarithmic_grid(
        solver.parameter_bounds("gamma_1"),
        GRID_POINTS_PER_AXIS,
    )
    start_time = time.perf_counter()
    grid_result = deterministic_grid_search(
        solver,
        beta_values,
        gamma_values,
        solver.integer_values("zeta"),
    )
    if REFINE_EACH_ZETA:
        final_by_zeta = tuple(
            refine_profiled_solution(
                solver,
                initial,
                maximum_iterations=REFINEMENT_MAX_ITERATIONS,
                parameter_tolerance=REFINEMENT_PARAMETER_TOLERANCE,
                cost_tolerance=REFINEMENT_COST_TOLERANCE,
            )
            for initial in grid_result.best_by_zeta
        )
    else:
        final_by_zeta = grid_result.best_by_zeta
    elapsed_seconds = time.perf_counter() - start_time
    best = min(final_by_zeta, key=lambda solution: solution.cost)
    parameters = historical.complete_profiled_parameters(best, parameter_specs)
    direct = historical.evaluate_full_parameter_set(
        wrapper,
        parameters,
        historical.FIT_AXES,
    )
    validation = solver.validate_solution(best)
    if not np.isclose(direct.raw_cost, best.cost, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Profiled and direct costs disagree for a jackknife fit")
    return DatasetFit(
        solution=best,
        parameters=parameters,
        direct_raw_cost=direct.raw_cost,
        direct_mean_cost=direct.mean_cost,
        validation_maximum_prediction_difference=(
            validation.maximum_weighted_prediction_difference
        ),
        profiled_evaluations=solver.profiled_evaluation_count(),
        basis_cache_sizes=solver.cache_sizes(),
        elapsed_seconds=float(elapsed_seconds),
        training_rows=int(wrapper.expstressPK2.shape[0]),
    )


def evaluate_on_curve(
    curve: RatioCurve,
    parameters: Mapping[str, float | int],
    parameter_path: pathlib.Path,
    model_spec: Any,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> historical.FullParameterEvaluation:
    """Evaluate one parameter set on one complete held-out loading curve."""
    wrapper = configured_wrapper(
        curve.path,
        parameter_path,
        model_spec,
        parameter_specs,
    )
    return historical.evaluate_full_parameter_set(
        wrapper,
        parameters,
        historical.FIT_AXES,
    )


def parameter_sensitivity_summary(
    full_parameters: Mapping[str, float | int],
    omissions: Sequence[OmissionResult],
    parameter_names: Sequence[str] = REPORTED_PARAMETERS,
) -> dict[str, dict[str, Any]]:
    """Summarize descriptive parameter spread and the most influential ratio."""
    summary: dict[str, dict[str, Any]] = {}
    for name in parameter_names:
        full_value = float(full_parameters[name])
        values = np.asarray(
            [float(result.fit.parameters[name]) for result in omissions],
            dtype=float,
        )
        absolute_changes = np.abs(values - full_value)
        influential_index = int(np.argmax(absolute_changes))
        summary[name] = {
            "full_data_value": full_value,
            "leave_one_out_minimum": float(np.min(values)),
            "leave_one_out_maximum": float(np.max(values)),
            "leave_one_out_mean": float(np.mean(values)),
            "leave_one_out_standard_deviation": float(np.std(values, ddof=1)),
            "maximum_absolute_change": float(absolute_changes[influential_index]),
            "maximum_relative_change": (
                float(absolute_changes[influential_index] / abs(full_value))
                if full_value != 0
                else None
            ),
            "most_influential_omission": omissions[influential_index].omitted_label,
        }
    return summary


def scalar_evaluation(evaluation: historical.FullParameterEvaluation) -> dict[str, float]:
    """Return held-out scalar metrics while storing predictions separately."""
    return {
        "raw_cost": evaluation.raw_cost,
        "mean_cost": evaluation.mean_cost,
        "normalized_residual_rmse": float(
            np.sqrt(np.mean(evaluation.normalized_residuals**2))
        ),
        "maximum_absolute_normalized_residual": float(
            np.max(np.abs(evaluation.normalized_residuals))
        ),
    }


def parameter_relative_change_matrix(
    full_fit: DatasetFit,
    omissions: Sequence[OmissionResult],
) -> FloatArray:
    """Return signed percent changes, allowing values such as K_coup=0."""
    matrix = np.empty((len(omissions), len(REPORTED_PARAMETERS)), dtype=float)
    for row, result in enumerate(omissions):
        for column, name in enumerate(REPORTED_PARAMETERS):
            full_value = float(full_fit.parameters[name])
            if full_value == 0:
                raise ValueError(f"Relative influence requires nonzero full-fit {name}")
            omitted_value = float(result.fit.parameters[name])
            matrix[row, column] = 100.0 * (omitted_value - full_value) / abs(full_value)
    return matrix


def create_parameter_influence_figure(
    full_fit: DatasetFit,
    omissions: Sequence[OmissionResult],
) -> plt.Figure:
    """Plot signed percent changes from the full-data parameter estimates."""
    matrix = parameter_relative_change_matrix(full_fit, omissions)

    maximum = max(float(np.max(np.abs(matrix))), 0.1)
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        norm=colors.SymLogNorm(
            linthresh=10.0,
            linscale=1.0,
            vmin=-maximum,
            vmax=maximum,
        ),
    )
    axis.set_xticks(
        np.arange(len(REPORTED_PARAMETERS)),
        REPORTED_PARAMETERS,
        rotation=45,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(omissions)),
        [result.omitted_label for result in omissions],
    )
    axis.set_xlabel("Refitted parameter")
    axis.set_ylabel("Omitted loading ratio")
    axis.set_title("Leave-one-ratio-out parameter influence")
    figure.colorbar(image, ax=axis, label="Change from full-data estimate (%)")
    return figure


def create_cost_figure(
    full_fit: DatasetFit,
    omissions: Sequence[OmissionResult],
) -> plt.Figure:
    """Compare training objectives and genuine held-out curve objectives."""
    labels = [result.omitted_label for result in omissions]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    axes[0].bar(
        positions,
        [result.fit.direct_mean_cost for result in omissions],
        color="tab:blue",
    )
    axes[0].axhline(
        full_fit.direct_mean_cost,
        color="black",
        linestyle="--",
        label="Full-data fit",
    )
    axes[0].set_ylabel("Training mean normalized cost")
    axes[0].set_title("Objective after omitting each ratio")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    width = 0.38
    axes[1].bar(
        positions - width / 2,
        [result.full_fit_held_out.mean_cost for result in omissions],
        width,
        label="Full-data parameters",
    )
    axes[1].bar(
        positions + width / 2,
        [result.omission_fit_held_out.mean_cost for result in omissions],
        width,
        label="Parameters fitted without ratio",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Held-out mean normalized cost")
    axes[1].set_title("Prediction on the omitted curve")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=45, ha="right")
    return figure


def create_held_out_prediction_figure(
    omissions: Sequence[OmissionResult],
) -> plt.Figure:
    """Compare held-out residuals for all nine loading ratios."""
    figure, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
    for axis, result in zip(axes.flat, omissions):
        # Both evaluations use the same held-out curve and normalization scales.
        full_residual = result.full_fit_held_out.normalized_residuals.T.ravel()
        omitted_residual = result.omission_fit_held_out.normalized_residuals.T.ravel()
        positions = np.arange(full_residual.size)
        axis.scatter(
            positions,
            full_residual,
            s=7,
            alpha=0.55,
            label="Full-data fit",
        )
        axis.scatter(
            positions,
            omitted_residual,
            s=7,
            alpha=0.55,
            label="Fit without ratio",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        rows_per_axis = result.full_fit_held_out.normalized_residuals.shape[0]
        axis.axvline(rows_per_axis - 0.5, color="gray", linewidth=0.8, linestyle=":")
        axis.set_title(result.omitted_label)
        axis.set_xlabel("Held-out observation (xx, then yy)")
        axis.set_ylabel("Normalized residual")
    axes.flat[0].legend()
    figure.suptitle("Held-out residuals by omitted loading ratio", fontsize=16)
    return figure


def save_outputs(
    artifact: Mapping[str, Any],
    input_paths: Mapping[str, pathlib.Path],
    input_hashes: Mapping[str, str],
    signature: str,
    curves: Sequence[RatioCurve],
    full_fit: DatasetFit,
    omissions: Sequence[OmissionResult],
    total_seconds: float,
    *,
    output_directory: pathlib.Path | None = None,
) -> pathlib.Path:
    """Save a standalone jackknife package under one timestamp."""
    if output_directory is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = (
            OUTPUT_ROOT / f"fit_{historical.REFERENCE_FIT_CODE:04d}" / timestamp
        )
        output_directory.mkdir(parents=True, exist_ok=False)
    else:
        output_directory.mkdir(parents=True, exist_ok=True)
    sensitivity = parameter_sensitivity_summary(full_fit.parameters, omissions)
    omission_payload = []
    for result in omissions:
        parameter_changes = {
            name: float(result.fit.parameters[name]) - float(full_fit.parameters[name])
            for name in REPORTED_PARAMETERS
        }
        omission_payload.append(
            {
                "omitted_label": result.omitted_label,
                "omitted_path": str(result.omitted_path.resolve()),
                "fit": asdict(result.fit),
                "parameter_changes_from_full_fit": parameter_changes,
                "full_fit_on_held_out_curve": scalar_evaluation(
                    result.full_fit_held_out
                ),
                "omission_fit_on_held_out_curve": scalar_evaluation(
                    result.omission_fit_held_out
                ),
                "held_out_cost_ratio_omission_to_full": (
                    result.omission_fit_held_out.mean_cost
                    / result.full_fit_held_out.mean_cost
                ),
            }
        )

    summary = {
        "schema_version": 1,
        "study": "SV_exp deterministic leave-one-loading-ratio-out sensitivity",
        "interpretation_limit": (
            "One specimen exists per ratio, so these are influence diagnostics, "
            "not population confidence intervals."
        ),
        "provenance": {
            "reference_fit_code": historical.REFERENCE_FIT_CODE,
            "reconstructed_batch_signature": signature,
            "input_paths": {name: str(path.resolve()) for name, path in input_paths.items()},
            "input_sha256": dict(input_hashes),
            "compiled_file_exactly_reconstructed_from_ratio_files": True,
            "source_files_modified": False,
        },
        "configuration": {
            "fit_axes": list(historical.FIT_AXES),
            "wide_shape_maximum": WIDE_SHAPE_MAXIMUM,
            "grid_points_per_axis": GRID_POINTS_PER_AXIS,
            "refine_each_zeta": REFINE_EACH_ZETA,
            "zeta_values": list(range(2, 11)),
            "ratio_count": len(curves),
        },
        "ratio_blocks": [
            {
                "label": curve.label,
                "path": str(curve.path.resolve()),
                "rows": int(curve.data.shape[0]),
                "start_row": curve.start_row,
                "stop_row_exclusive": curve.stop_row,
            }
            for curve in curves
        ],
        "full_data_fit": asdict(full_fit),
        "leave_one_out_results": omission_payload,
        "parameter_sensitivity": sensitivity,
        "zeta_frequency": {
            str(key): value
            for key, value in sorted(
                Counter(int(result.fit.parameters["zeta"]) for result in omissions).items()
            )
        },
        "total_elapsed_seconds": float(total_seconds),
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    influence_matrix = parameter_relative_change_matrix(full_fit, omissions)
    np.savez_compressed(
        output_directory / "jackknife_arrays.npz",
        ratio_labels=np.asarray([result.omitted_label for result in omissions]),
        parameter_names=np.asarray(REPORTED_PARAMETERS),
        parameter_relative_changes_percent=influence_matrix,
        training_mean_costs=np.asarray(
            [result.fit.direct_mean_cost for result in omissions]
        ),
        full_fit_held_out_mean_costs=np.asarray(
            [result.full_fit_held_out.mean_cost for result in omissions]
        ),
        omission_fit_held_out_mean_costs=np.asarray(
            [result.omission_fit_held_out.mean_cost for result in omissions]
        ),
    )
    influence_figure = create_parameter_influence_figure(full_fit, omissions)
    cost_figure = create_cost_figure(full_fit, omissions)
    prediction_figure = create_held_out_prediction_figure(omissions)
    influence_figure.savefig(output_directory / "parameter_influence.png", dpi=200)
    cost_figure.savefig(output_directory / "training_and_held_out_costs.png", dpi=200)
    prediction_figure.savefig(output_directory / "held_out_residuals.png", dpi=200)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(influence_figure)
        plt.close(cost_figure)
        plt.close(prediction_figure)
    return output_directory


def dataset_fit_from_payload(payload: Mapping[str, Any]) -> DatasetFit:
    """Reconstruct one completed fit, for example after a plotting-only failure."""
    solution_payload = dict(payload["solution"])
    return DatasetFit(
        solution=ProfiledSolution(**solution_payload),
        parameters=dict(payload["parameters"]),
        direct_raw_cost=float(payload["direct_raw_cost"]),
        direct_mean_cost=float(payload["direct_mean_cost"]),
        validation_maximum_prediction_difference=float(
            payload["validation_maximum_prediction_difference"]
        ),
        profiled_evaluations=int(payload["profiled_evaluations"]),
        basis_cache_sizes={
            str(name): int(value)
            for name, value in payload["basis_cache_sizes"].items()
        },
        elapsed_seconds=float(payload["elapsed_seconds"]),
        training_rows=int(payload["training_rows"]),
    )


def regenerate_saved_run(summary_path: pathlib.Path) -> pathlib.Path:
    """Regenerate figures from completed fits without rerunning optimization."""
    saved = historical.load_json(summary_path)
    input_paths = {
        name: pathlib.Path(path)
        for name, path in saved["provenance"]["input_paths"].items()
    }
    historical.require_input_files(input_paths)
    for name, expected_hash in saved["provenance"]["input_sha256"].items():
        actual_hash = historical.sha256_file(input_paths[name])
        if actual_hash != expected_hash:
            raise ValueError(f"Input changed since the saved run: {input_paths[name]}")

    artifact = historical.load_json(input_paths["fit_artifact"])
    model_spec = load_model_spec(str(artifact["model_id"]))
    parameter_path = input_paths["parameter_file"]
    curves = discover_ordered_ratio_curves(input_paths["compiled_experimental_data"])
    curves_by_label = {curve.label: curve for curve in curves}
    parameter_specs = widened_parameter_specs(artifact["parameter_specs"])
    umat.compile_umat_meson(input_paths["umat_source"])

    full_fit = dataset_fit_from_payload(saved["full_data_fit"])
    omissions: list[OmissionResult] = []
    for item in saved["leave_one_out_results"]:
        label = str(item["omitted_label"])
        curve = curves_by_label[label]
        fit = dataset_fit_from_payload(item["fit"])
        full_held_out = evaluate_on_curve(
            curve,
            full_fit.parameters,
            parameter_path,
            model_spec,
            parameter_specs,
        )
        omission_held_out = evaluate_on_curve(
            curve,
            fit.parameters,
            parameter_path,
            model_spec,
            parameter_specs,
        )
        saved_full_cost = float(item["full_fit_on_held_out_curve"]["mean_cost"])
        saved_omission_cost = float(
            item["omission_fit_on_held_out_curve"]["mean_cost"]
        )
        if not np.isclose(full_held_out.mean_cost, saved_full_cost, rtol=1.0e-12):
            raise ValueError(f"Full-fit held-out cost changed for ratio {label}")
        if not np.isclose(
            omission_held_out.mean_cost,
            saved_omission_cost,
            rtol=1.0e-12,
        ):
            raise ValueError(f"Omission-fit held-out cost changed for ratio {label}")
        omissions.append(
            OmissionResult(
                omitted_label=label,
                omitted_path=curve.path,
                fit=fit,
                full_fit_held_out=full_held_out,
                omission_fit_held_out=omission_held_out,
            )
        )

    return save_outputs(
        artifact,
        input_paths,
        dict(saved["provenance"]["input_sha256"]),
        str(saved["provenance"]["reconstructed_batch_signature"]),
        curves,
        full_fit,
        omissions,
        float(saved["total_elapsed_seconds"]),
        output_directory=summary_path.parent,
    )


def main() -> None:
    """Run the full fit, nine omission refits, and held-out evaluations."""
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
    curves = discover_ordered_ratio_curves(compiled_path)
    if len(curves) != 9:
        raise ValueError(f"Expected nine Rib loading ratios, found {len(curves)}")
    parameter_specs = widened_parameter_specs(artifact["parameter_specs"])
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

    print(f"Verified historical signature: {signature}", flush=True)
    print(
        "Exact compiled ratio order: "
        + ", ".join(curve.label for curve in curves),
        flush=True,
    )
    print(
        f"Using a {GRID_POINTS_PER_AXIS} x {GRID_POINTS_PER_AXIS} map, "
        "zeta 2-10, and per-zeta refinement for every refit.",
        flush=True,
    )
    umat.compile_umat_meson(model_source)
    total_start = time.perf_counter()

    print("\nFitting the complete 550-row dataset...", flush=True)
    full_fit = fit_dataset(
        compiled_path,
        parameter_path,
        model_spec,
        parameter_specs,
    )
    print(
        f"Full fit: mean cost={full_fit.direct_mean_cost:.12g}, "
        f"beta={full_fit.parameters['beta_1']:.8g}, "
        f"gamma={full_fit.parameters['gamma_1']:.8g}, "
        f"zeta={full_fit.parameters['zeta']}, "
        f"time={full_fit.elapsed_seconds:.2f}s",
        flush=True,
    )

    omissions: list[OmissionResult] = []
    with tempfile.TemporaryDirectory(prefix="ssmccf_leave_one_ratio_") as temporary:
        temporary_directory = pathlib.Path(temporary)
        for index, curve in enumerate(curves, start=1):
            print(
                f"\n[{index}/{len(curves)}] Omitting ratio {curve.label} "
                f"({curve.data.shape[0]} rows)...",
                flush=True,
            )
            training_path = temporary_directory / f"without_ratio_{index:02d}.dat"
            training_rows = write_omitted_training_data(
                curves,
                curve.label,
                training_path,
            )
            fit = fit_dataset(
                training_path,
                parameter_path,
                model_spec,
                parameter_specs,
            )
            if fit.training_rows != training_rows:
                raise ValueError("Written and loaded jackknife row counts disagree")
            full_held_out = evaluate_on_curve(
                curve,
                full_fit.parameters,
                parameter_path,
                model_spec,
                parameter_specs,
            )
            omission_held_out = evaluate_on_curve(
                curve,
                fit.parameters,
                parameter_path,
                model_spec,
                parameter_specs,
            )
            result = OmissionResult(
                omitted_label=curve.label,
                omitted_path=curve.path,
                fit=fit,
                full_fit_held_out=full_held_out,
                omission_fit_held_out=omission_held_out,
            )
            omissions.append(result)
            print(
                f"Result: beta={fit.parameters['beta_1']:.8g}, "
                f"gamma={fit.parameters['gamma_1']:.8g}, "
                f"zeta={fit.parameters['zeta']}, "
                f"train cost={fit.direct_mean_cost:.12g}, "
                f"held-out ratio={omission_held_out.mean_cost / full_held_out.mean_cost:.4g}, "
                f"time={fit.elapsed_seconds:.2f}s",
                flush=True,
            )

    total_seconds = time.perf_counter() - total_start
    sensitivity = parameter_sensitivity_summary(full_fit.parameters, omissions)
    print("\nLargest leave-one-ratio-out parameter influences", flush=True)
    for name in REPORTED_PARAMETERS:
        item = sensitivity[name]
        print(
            f"{name}: {item['leave_one_out_minimum']:.8g} to "
            f"{item['leave_one_out_maximum']:.8g}; "
            f"largest omission={item['most_influential_omission']}",
            flush=True,
        )
    print(
        "zeta frequency: "
        + str(Counter(int(result.fit.parameters["zeta"]) for result in omissions)),
        flush=True,
    )
    print(f"Total elapsed time: {total_seconds:.2f}s", flush=True)

    if SAVE_OUTPUTS:
        output_directory = save_outputs(
            artifact,
            input_paths,
            input_hashes,
            signature,
            curves,
            full_fit,
            omissions,
            total_seconds,
        )
        print(f"Saved standalone jackknife study: {output_directory}", flush=True)


if __name__ == "__main__":
    main()
