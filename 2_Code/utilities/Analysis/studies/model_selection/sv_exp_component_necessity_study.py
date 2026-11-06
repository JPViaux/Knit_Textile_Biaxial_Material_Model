"""Select necessary SV_exp stress components for the four complete knit datasets.

Edit the USER CONFIGURATION block and press Run in VSCode. The study is
standalone: it reads immutable fit/data artifacts, never allocates fit codes,
and never changes the SV_exp UMAT, parameter files, or shared fit registry.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (INPUT_ROOT as RELEASE_INPUT_ROOT,
    ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
import argparse
import copy
import csv
import hashlib
from release_paths import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear, minimize


ANALYSIS_DIR = next(
    parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis"
)
SUPPORT_DIR = ANALYSIS_DIR.parent
CODE_ROOT = SUPPORT_DIR.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(SUPPORT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

import uml_plot_style as uml_style  # noqa: E402

uml_style.apply()

from Analysis.diagnostics import constitutive_stability_study as stability  # noqa: E402
from Analysis.diagnostics import model_component_contribution_study as components  # noqa: E402
from Analysis.studies.solver_diagnostics import separable_solver_historical_benchmark as historical  # noqa: E402
from Analysis.studies.solver_diagnostics import separable_solver_leave_one_ratio_out as jackknife  # noqa: E402
from Python_Wrapper import mech, umat  # noqa: E402
from Python_Wrapper.model_registry import load_model_spec  # noqa: E402
from Python_Wrapper.separable_solver import (  # noqa: E402
    SV_EXP_DEFINITION,
    SVExpSeparableSolver,
    apply_parameter_specs_in_memory,
    logarithmic_grid,
)


FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# USER CONFIGURATION - edit these values before pressing Run in VSCode.
# ---------------------------------------------------------------------------

# Complete raw Tripoly references; example: Interlock uses fit 726 and sample S06.
FIT_CODES: dict[str, int] = {
    "Interlock": 726,
    "Rib": 722,
    "Purl": 723,
    "SingleJersey": 725,
}
# Knits to run; example: ("Interlock", "Rib") runs only two resumable cases.
KNITS = ("Interlock", "Rib", "Purl", "SingleJersey")
# Log-grid points for each active beta/gamma; examples are 13, 17, or 21.
GRID_POINTS_PER_AXIS = 17
# Integer nonlinear-coupling exponents; example: tuple(range(2, 16)) gives 2..15.
ZETA_VALUES = tuple(range(2, 16))
# Weighted-RMS coupling-response ceiling used by production SV_exp fits; example: 0.1.
COUPLING_AMPLITUDE_CEILING = 0.1
# Maximum Powell iterations for an active continuous shape; examples are 50 or 100.
REFINEMENT_MAX_ITERATIONS = 100
# Log-shape convergence tolerance; examples are 1e-5 or 1e-6.
REFINEMENT_PARAMETER_TOLERANCE = 1.0e-6
# Profiled-cost convergence tolerance; examples are 1e-8 or 1e-10.
REFINEMENT_COST_TOLERANCE = 1.0e-10
# Predictive adequacy fraction; example: 0.02 means within 2% of the winner.
SIMPLICITY_TOLERANCE = 0.02
# Full stability envelope resolution; example: 121 gives a 121 x 121 grid.
STABILITY_GRID_POINTS = 121
# Propagation angles for stability screening; example: 721 covers 0..pi densely.
STABILITY_ANGLE_POINTS = 721
# Local envelope minima refined per scope; example: 32 matches the validated study.
STABILITY_REFINEMENT_STARTS = 32
# True runs measured/envelope stability and compiled-UMAT critical-state checks.
RUN_STABILITY = True
# True resumes matching checkpoints; False requires an empty output directory.
RESUME_EXISTING = True
# False protects completed outputs; set True only to intentionally rebuild them.
OVERWRITE_EXISTING = False
# False is appropriate for a complete VSCode run; True opens generated figures.
SHOW_PLOTS = False
# Stable standalone output; example: Outputs/sv_exp_component_necessity/b09_v1.
OUTPUT_DIRECTORY = (
    RELEASE_ARCHIVE_ROOT / "sv_exp_component_necessity" / "b09_v1"
)


COMPONENT_ORDER = (
    "linear_course",
    "linear_wale",
    "linear_coupling",
    "nonlinear_course",
    "nonlinear_wale",
    "nonlinear_coupling",
)
AMPLITUDE_PARAMETERS = (
    "mu",
    "mu2",
    "mu3",
    "K_1ani1",
    "K_1ani2",
    "K_coup",
)
SHAPE_PARAMETER_BY_AMPLITUDE = {
    "K_1ani1": "beta_1",
    "K_1ani2": "gamma_1",
    "K_coup": "zeta",
}
FIXED_PARAMETERS = {"C1": 1.0, "C2": 1.0, "theta1": 0.0, "theta2": 90.0}
FIT_AXES = (0, 1)
RATIO_ORDER = components.RATIO_ORDER
STATUS_ORDER = ("Required", "Optional/substitutable", "Unsupported")
STATUS_COLORS = {
    "Required": "#C43B3B",
    "Optional/substitutable": "#D0AF22",
    "Unsupported": "#878A8F",
}


@dataclass(frozen=True)
class CandidateDefinition:
    """One nonempty physical component subset, for example mask M17."""

    mask: int
    candidate_id: str
    components: tuple[str, ...]
    linear_parameters: tuple[str, ...]
    continuous_shapes: tuple[str, ...]
    integer_shapes: tuple[str, ...]
    effective_parameter_count: int


@dataclass(frozen=True)
class CaseInputs:
    """Immutable files and physical-axis metadata for one knit specimen."""

    knit: str
    fit_code: int
    sample_folder: str
    axis_directions: tuple[str, str]
    artifact_path: pathlib.Path
    parameter_path: pathlib.Path
    compiled_path: pathlib.Path
    snapshot_path: pathlib.Path
    umat_source: pathlib.Path
    artifact: dict[str, Any]
    parameter_specs: dict[str, dict[str, Any]]
    curves: tuple[jackknife.RatioCurve, ...]
    input_hashes: dict[str, str]


@dataclass(frozen=True)
class SubsetSolution:
    """Best profiled solution for one component subset and one dataset."""

    candidate_id: str
    mask: int
    components: tuple[str, ...]
    effective_parameter_count: int
    shape_values: dict[str, float | int]
    linear_values: dict[str, float]
    active_bounds: dict[str, int]
    raw_cost: float
    mean_cost: float
    optimality: float
    iterations: int
    profiled_evaluations: int
    elapsed_seconds: float
    design_matrix_rank: int
    design_matrix_singular_values: tuple[float, ...]
    design_matrix_condition_number: float
    coupling_upper_bound: float | None
    maximum_weighted_prediction_difference: float
    direct_cost_difference: float


def utc_now() -> str:
    """Return an ISO timestamp, for example 2026-08-12T18:30:00+00:00."""

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: pathlib.Path) -> str:
    """Hash one immutable input or generated deliverable."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Write one checkpoint atomically so an interrupted run remains resumable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # PID-specific temporary names avoid collisions among isolated fold workers.
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_ready(value: Any) -> Any:
    """Convert paths, NumPy values, and dataclasses into JSON-compatible values."""

    if hasattr(value, "__dataclass_fields__"):
        return json_ready(asdict(value))
    if isinstance(value, pathlib.Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def normalize_direction(value: str) -> str:
    """Normalize metadata labels, for example ``wales`` to physical ``wale``."""

    token = str(value).strip().lower()
    aliases = {"course": "course", "courses": "course", "wale": "wale", "wales": "wale"}
    try:
        return aliases[token]
    except KeyError as exc:
        raise ValueError(f"Unsupported physical direction {value!r}") from exc


def metadata_axis_directions(sample_folder: str) -> tuple[str, str]:
    """Read one sample's constant physical axes from the durable metadata CSV."""

    metadata_path = RELEASE_INPUT_ROOT / "Experiment_Data" / "biaxial_test_metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8-sig") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row.get("canonical_sample_folder") == sample_folder
            and str(row.get("is_conditioning", "")).strip().lower() not in {"true", "1"}
        ]
    if not rows:
        raise ValueError(f"No nonconditioning metadata rows found for {sample_folder}")
    directions = {
        (
            normalize_direction(row["axis1_direction"]),
            normalize_direction(row["axis2_direction"]),
        )
        for row in rows
    }
    if len(directions) != 1:
        raise ValueError(f"Axis directions change within {sample_folder}: {sorted(directions)}")
    axis_directions = next(iter(directions))
    if set(axis_directions) != {"course", "wale"}:
        raise ValueError(f"Expected course/wale axes for {sample_folder}: {axis_directions}")
    return axis_directions


def component_parameter_map(axis_directions: tuple[str, str]) -> dict[str, str]:
    """Map physical component labels to family-1/family-2 SV_exp parameters."""

    first, second = axis_directions
    return {
        f"linear_{first}": "mu",
        f"linear_{second}": "mu2",
        "linear_coupling": "mu3",
        f"nonlinear_{first}": "K_1ani1",
        f"nonlinear_{second}": "K_1ani2",
        "nonlinear_coupling": "K_coup",
    }


def enumerate_component_subsets(
    axis_directions: tuple[str, str],
) -> tuple[CandidateDefinition, ...]:
    """Return all 63 nonempty physical subsets in deterministic mask order."""

    mapping = component_parameter_map(axis_directions)
    candidates: list[CandidateDefinition] = []
    for mask in range(1, 1 << len(COMPONENT_ORDER)):
        selected_components = tuple(
            name for index, name in enumerate(COMPONENT_ORDER) if mask & (1 << index)
        )
        selected_parameters = {mapping[name] for name in selected_components}
        linear_parameters = tuple(
            name for name in SV_EXP_DEFINITION.linear_parameters if name in selected_parameters
        )
        shapes = {
            SHAPE_PARAMETER_BY_AMPLITUDE[name]
            for name in linear_parameters
            if name in SHAPE_PARAMETER_BY_AMPLITUDE
        }
        continuous = tuple(name for name in ("beta_1", "gamma_1") if name in shapes)
        integers = ("zeta",) if "zeta" in shapes else ()
        candidates.append(
            CandidateDefinition(
                mask=mask,
                candidate_id=f"M{mask:02d}",
                components=selected_components,
                linear_parameters=linear_parameters,
                continuous_shapes=continuous,
                integer_shapes=integers,
                effective_parameter_count=len(linear_parameters) + len(shapes),
            )
        )
    if len(candidates) != 63 or len({item.components for item in candidates}) != 63:
        raise RuntimeError("Component enumeration did not produce 63 unique subsets")
    return tuple(candidates)


def raw_parameter_specs(artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Recover broad production bounds and enforce the prescribed fixed geometry."""

    provenance = artifact.get("parameter_provenance", {})
    source = provenance.get("raw_parameter_specs", artifact["parameter_specs"])
    specifications = copy.deepcopy(source)
    specifications["beta_1"]["maximum"] = 1000.0
    specifications["gamma_1"]["maximum"] = 1000.0
    specifications["zeta"].update(
        {"initial_guess": 2, "minimum": min(ZETA_VALUES), "maximum": max(ZETA_VALUES), "fixed": False}
    )
    for name, value in FIXED_PARAMETERS.items():
        specifications[name].update(
            {"initial_guess": value, "minimum": value, "maximum": value, "fixed": True}
        )
    return specifications


def load_case_inputs(knit: str) -> CaseInputs:
    """Load and reconcile one approved B09 fit, snapshot, raw file, and nine curves."""

    fit_code = int(FIT_CODES[knit])
    artifact_path = RELEASE_ARCHIVE_ROOT / "fits" / f"{fit_code:04d}" / "fit.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    context = artifact["experiment_context"]
    if artifact.get("model_id") != "SV_exp" or context.get("knit") != knit:
        raise ValueError(f"Fit {fit_code} is not the requested {knit} SV_exp artifact")
    if artifact.get("result_role") != "training_optimum":
        raise ValueError(f"Fit {fit_code} is not a production training_optimum")
    sample_folder = str(context["sample_folder"])
    axis_directions = metadata_axis_directions(sample_folder)
    parameter_path = pathlib.Path(artifact["input_files"]["material_parameters"]["snapshot_path"])
    snapshot_path = pathlib.Path(artifact["input_files"]["experimental_data"]["snapshot_path"])
    compiled_path = pathlib.Path(context["exp_data_path"])
    umat_source = pathlib.Path(artifact["input_files"]["umat_source"]["snapshot_path"])
    for path in (artifact_path, parameter_path, snapshot_path, compiled_path, umat_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    compiled = np.atleast_2d(np.loadtxt(compiled_path, dtype=float))
    snapshot = np.atleast_2d(np.loadtxt(snapshot_path, dtype=float))
    if not np.array_equal(compiled, snapshot):
        raise ValueError(f"Fit {fit_code} snapshot does not exactly match its raw B09 product")
    curves = jackknife.discover_ordered_ratio_curves(compiled_path)
    if len(curves) != 9 or {curve.label for curve in curves} != set(RATIO_ORDER):
        raise ValueError(f"Expected the canonical nine ratios for {knit}")
    reconstructed = np.vstack([curve.data for curve in curves])
    if not np.array_equal(reconstructed, snapshot):
        raise ValueError(
            f"The nine raw ratio curves do not exactly reconstruct fit {fit_code}'s B09 snapshot"
        )
    hashes = {
        "fit_artifact": sha256_file(artifact_path),
        "material_parameters": sha256_file(parameter_path),
        "compiled_experimental_data": sha256_file(compiled_path),
        "experimental_snapshot": sha256_file(snapshot_path),
        "umat_source": sha256_file(umat_source),
        **{f"ratio_{curve.label}": sha256_file(curve.path) for curve in curves},
    }
    declared_hashes = {
        "material_parameters": artifact["input_files"]["material_parameters"]["sha256"],
        "experimental_snapshot": artifact["input_files"]["experimental_data"]["sha256"],
        "umat_source": artifact["input_files"]["umat_source"]["sha256"],
    }
    for name, declared in declared_hashes.items():
        if hashes[name] != declared:
            raise ValueError(f"Fit {fit_code} frozen {name} hash changed")
    return CaseInputs(
        knit=knit,
        fit_code=fit_code,
        sample_folder=sample_folder,
        axis_directions=axis_directions,
        artifact_path=artifact_path,
        parameter_path=parameter_path,
        compiled_path=compiled_path,
        snapshot_path=snapshot_path,
        umat_source=umat_source,
        artifact=artifact,
        parameter_specs=raw_parameter_specs(artifact),
        curves=curves,
        input_hashes=hashes,
    )


def configured_wrapper(
    data_path: pathlib.Path,
    case: CaseInputs,
    reference_maxima: Mapping[int, float] | None,
) -> Any:
    """Load one dataset with the production normalized PK2 objective."""

    model_spec = load_model_spec("SV_exp")
    loaded = umat.Load_props(case.parameter_path, model_spec=model_spec)
    wrapper = umat.Umat_wrapper(
        umat.Material_model(loaded),
        [case.artifact["experiment_context"]["umat_strain"], case.artifact["experiment_context"]["umat_stress"]],
    )
    apply_parameter_specs_in_memory(wrapper, case.parameter_specs)
    wrapper.load_inputs(data_path, case.artifact["experiment_context"]["exp_type"])
    wrapper.set_cost_function("normalized", 1.0, reference_maxima)
    return wrapper


def reference_maxima(case: CaseInputs) -> dict[int, float]:
    """Compute fixed full-data PK2 maxima for axes 0 and 1."""

    wrapper = configured_wrapper(case.compiled_path, case, None)
    return {
        axis: float(np.max(np.abs(wrapper.expstressPK2[:, axis]))) for axis in FIT_AXES
    }


def shape_grid(solver: SVExpSeparableSolver, name: str) -> FloatArray:
    """Return a 17-point log grid for beta or gamma."""

    return logarithmic_grid(solver.parameter_bounds(name), GRID_POINTS_PER_AXIS)


def candidate_design(
    solver: SVExpSeparableSolver,
    candidate: CandidateDefinition,
    shape_values: Mapping[str, float | int],
) -> FloatArray:
    """Build only the weighted basis columns permitted by one candidate."""

    dependencies = SV_EXP_DEFINITION.basis_dependencies()
    columns = []
    for parameter in candidate.linear_parameters:
        controlling = {name: shape_values[name] for name in dependencies[parameter]}
        columns.append(solver.basis_column(parameter, controlling))
    return np.column_stack(columns)


def coupling_upper_bound(weighted_column: FloatArray) -> float:
    """Apply the production weighted-RMS coupling-response ceiling."""

    rms = float(np.sqrt(np.mean(np.asarray(weighted_column, dtype=float) ** 2)))
    return 0.0 if rms == 0.0 else COUPLING_AMPLITUDE_CEILING / rms


def solve_at_shapes(
    solver: SVExpSeparableSolver,
    candidate: CandidateDefinition,
    shape_values: Mapping[str, float | int],
) -> dict[str, Any]:
    """Profile one candidate's bounded amplitudes at fixed active shapes."""

    design = candidate_design(solver, candidate, shape_values)
    lower = np.asarray([float(solver.wrapper.getv(name)[2]) for name in candidate.linear_parameters])
    upper = np.asarray([float(solver.wrapper.getv(name)[3]) for name in candidate.linear_parameters])
    coupling_upper: float | None = None
    if "K_coup" in candidate.linear_parameters:
        coupling_index = candidate.linear_parameters.index("K_coup")
        coupling_upper = coupling_upper_bound(design[:, coupling_index])
        upper[coupling_index] = coupling_upper
    fixed = np.isclose(upper, lower, rtol=0.0, atol=0.0)
    coefficients = lower.copy()
    variable = np.flatnonzero(~fixed)
    active_mask = np.zeros(len(candidate.linear_parameters), dtype=int)
    optimality = 0.0
    iterations = 0
    if variable.size:
        variable_design = design[:, variable]
        spans = upper[variable] - lower[variable]
        shifted_target = solver.problem.weighted_target - design @ coefficients
        result = lsq_linear(
            variable_design * spans[np.newaxis, :],
            shifted_target,
            bounds=(0.0, 1.0),
            method="bvls",
            tol=1.0e-12,
            max_iter=100,
        )
        if not result.success:
            raise RuntimeError(f"Candidate BVLS solve failed: {result.message}")
        coefficients[variable] = lower[variable] + spans * np.asarray(result.x)
        active_mask[variable] = np.asarray(result.active_mask, dtype=int)
        optimality = float(result.optimality)
        iterations = int(result.nit)
    prediction = design @ coefficients
    residual = prediction - solver.problem.weighted_target
    cost = float(residual @ residual)
    return {
        "shape_values": dict(shape_values),
        "linear_values": {
            name: float(value) for name, value in zip(candidate.linear_parameters, coefficients)
        },
        "active_bounds": {
            name: int(value) for name, value in zip(candidate.linear_parameters, active_mask)
        },
        "raw_cost": cost,
        "mean_cost": cost / float(solver.problem.target.size),
        "optimality": optimality,
        "iterations": iterations,
        "weighted_design": design,
        "weighted_prediction": prediction,
        "coupling_upper_bound": coupling_upper,
    }


def refine_one_zeta(
    solver: SVExpSeparableSolver,
    candidate: CandidateDefinition,
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    """Polish active beta/gamma values in log space at fixed zeta."""

    if not candidate.continuous_shapes:
        retained = dict(initial)
        retained["refinement_evaluations"] = 0
        return retained
    names = candidate.continuous_shapes
    bounds = [tuple(np.log(solver.parameter_bounds(name))) for name in names]
    initial_log = np.log([float(initial["shape_values"][name]) for name in names])
    fixed_shapes = {
        name: value
        for name, value in initial["shape_values"].items()
        if name not in names
    }

    evaluations = 0

    def objective(log_values: FloatArray) -> float:
        nonlocal evaluations
        evaluations += 1
        shapes = dict(fixed_shapes)
        shapes.update(
            {name: float(value) for name, value in zip(names, np.exp(log_values))}
        )
        return float(solve_at_shapes(solver, candidate, shapes)["raw_cost"])

    result = minimize(
        objective,
        initial_log,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": REFINEMENT_MAX_ITERATIONS,
            "xtol": REFINEMENT_PARAMETER_TOLERANCE,
            "ftol": REFINEMENT_COST_TOLERANCE,
        },
    )
    refined_shapes = dict(fixed_shapes)
    refined_shapes.update(
        {name: float(value) for name, value in zip(names, np.exp(result.x))}
    )
    refined = solve_at_shapes(solver, candidate, refined_shapes)
    evaluations += 1
    retained = refined if refined["raw_cost"] < initial["raw_cost"] else dict(initial)
    retained["refinement_evaluations"] = evaluations
    return retained


def complete_parameters(
    case: CaseInputs,
    candidate: CandidateDefinition,
    solution: Mapping[str, Any],
) -> dict[str, float | int]:
    """Create a full 13-parameter SV_exp record with omitted amplitudes at zero."""

    values: dict[str, float | int] = {
        name: specification["initial_guess"]
        for name, specification in case.parameter_specs.items()
    }
    values.update({name: 0.0 for name in AMPLITUDE_PARAMETERS})
    values.update(FIXED_PARAMETERS)
    values.update(solution["linear_values"])
    values.update(solution["shape_values"])
    if "beta_1" not in candidate.continuous_shapes:
        values["beta_1"] = 1.0
    if "gamma_1" not in candidate.continuous_shapes:
        values["gamma_1"] = 1.0
    if "zeta" not in candidate.integer_shapes:
        values["zeta"] = 2
    return values


def fit_candidate(
    solver: SVExpSeparableSolver,
    case: CaseInputs,
    candidate: CandidateDefinition,
) -> tuple[SubsetSolution, dict[str, float | int]]:
    """Map, refine, and directly validate one component subset."""

    start = time.perf_counter()
    grids: dict[str, Sequence[float | int]] = {}
    for name in candidate.continuous_shapes:
        grids[name] = shape_grid(solver, name)
    for name in candidate.integer_shapes:
        grids[name] = ZETA_VALUES
    if not grids:
        shape_combinations = [dict()]
    else:
        names = tuple(grids)
        shape_combinations = [
            dict(zip(names, values)) for values in product(*(grids[name] for name in names))
        ]
    best_by_zeta: dict[int, dict[str, Any]] = {}
    evaluations = 0
    for shapes in shape_combinations:
        candidate_solution = solve_at_shapes(solver, candidate, shapes)
        evaluations += 1
        zeta = int(shapes.get("zeta", 2))
        if zeta not in best_by_zeta or candidate_solution["raw_cost"] < best_by_zeta[zeta]["raw_cost"]:
            best_by_zeta[zeta] = candidate_solution
    refined = []
    for initial in best_by_zeta.values():
        final = refine_one_zeta(solver, candidate, initial)
        evaluations += int(final.pop("refinement_evaluations"))
        refined.append(final)
    best = min(refined, key=lambda item: item["raw_cost"])
    parameters = complete_parameters(case, candidate, best)
    direct = historical.evaluate_full_parameter_set(solver.wrapper, parameters, FIT_AXES)
    direct_weighted = np.concatenate(
        [direct.predicted_pk2[:, axis] for axis in FIT_AXES]
    ) / solver.problem.scales
    maximum_difference = float(np.max(np.abs(direct_weighted - best["weighted_prediction"])))
    direct_cost_difference = abs(float(direct.raw_cost) - float(best["raw_cost"]))
    if maximum_difference >= 1.0e-10 or direct_cost_difference >= 1.0e-10:
        raise ValueError(
            f"Direct UMAT validation failed for {case.knit} {candidate.candidate_id}: "
            f"prediction={maximum_difference:.3e}, cost={direct_cost_difference:.3e}"
        )
    singular_values = np.linalg.svd(best["weighted_design"], compute_uv=False)
    rank = int(np.linalg.matrix_rank(best["weighted_design"]))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == len(candidate.linear_parameters)
        and singular_values.size
        and singular_values[-1] > 0
        else math.inf
    )
    solution = SubsetSolution(
        candidate_id=candidate.candidate_id,
        mask=candidate.mask,
        components=candidate.components,
        effective_parameter_count=candidate.effective_parameter_count,
        shape_values={name: value for name, value in best["shape_values"].items()},
        linear_values={name: float(value) for name, value in best["linear_values"].items()},
        active_bounds={name: int(value) for name, value in best["active_bounds"].items()},
        raw_cost=float(direct.raw_cost),
        mean_cost=float(direct.mean_cost),
        optimality=float(best["optimality"]),
        iterations=int(best["iterations"]),
        profiled_evaluations=int(evaluations),
        elapsed_seconds=float(time.perf_counter() - start),
        design_matrix_rank=rank,
        design_matrix_singular_values=tuple(float(value) for value in singular_values),
        design_matrix_condition_number=condition,
        coupling_upper_bound=(
            None if best["coupling_upper_bound"] is None else float(best["coupling_upper_bound"])
        ),
        maximum_weighted_prediction_difference=maximum_difference,
        direct_cost_difference=direct_cost_difference,
    )
    return solution, parameters


def solution_from_payload(payload: Mapping[str, Any]) -> SubsetSolution:
    """Reconstruct one saved candidate solution from a checkpoint."""

    converted = dict(payload)
    converted["components"] = tuple(converted["components"])
    converted["design_matrix_singular_values"] = tuple(
        float(value) for value in converted["design_matrix_singular_values"]
    )
    return SubsetSolution(**converted)


def manifest_payload(cases: Sequence[CaseInputs]) -> dict[str, Any]:
    """Return the immutable configuration used to validate checkpoint reuse."""

    return {
        "schema_version": 1,
        "study_id": "sv_exp_component_necessity_b09_v1",
        "fit_codes": FIT_CODES,
        "knits": list(KNITS),
        "components": list(COMPONENT_ORDER),
        "candidate_count": 63,
        "fit_axes": list(FIT_AXES),
        "grid_points_per_axis": GRID_POINTS_PER_AXIS,
        "zeta_values": list(ZETA_VALUES),
        "coupling_amplitude_ceiling": COUPLING_AMPLITUDE_CEILING,
        "simplicity_tolerance": SIMPLICITY_TOLERANCE,
        "fixed_parameters": FIXED_PARAMETERS,
        "stability_grid_points": STABILITY_GRID_POINTS,
        "stability_angle_points": STABILITY_ANGLE_POINTS,
        "stability_refinement_starts": STABILITY_REFINEMENT_STARTS,
        "inputs": {case.knit: case.input_hashes for case in cases},
    }


def manifest_hash(payload: Mapping[str, Any]) -> str:
    """Hash the canonical manifest so incompatible checkpoints cannot be mixed."""

    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_or_initialize_checkpoint(
    path: pathlib.Path,
    knit: str,
    expected_manifest_hash: str,
) -> dict[str, Any]:
    """Load a matching resumable checkpoint or create an empty state."""

    if path.is_file():
        if not RESUME_EXISTING:
            raise FileExistsError(path)
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("manifest_hash") != expected_manifest_hash or state.get("knit") != knit:
            raise ValueError(f"Checkpoint configuration mismatch: {path}")
        return state
    return {
        "schema_version": 1,
        "manifest_hash": expected_manifest_hash,
        "knit": knit,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "full_fits": {},
        "folds": {},
    }


def evaluate_held_out(
    wrapper: Any,
    parameters: Mapping[str, float | int],
) -> dict[str, Any]:
    """Evaluate one fold fit on its omitted curve using the fixed full-data scales."""

    evaluation = historical.evaluate_full_parameter_set(wrapper, parameters, FIT_AXES)
    return {
        "raw_cost": float(evaluation.raw_cost),
        "mean_cost": float(evaluation.mean_cost),
        "observation_count": int(evaluation.normalized_residuals.size),
        "normalized_residual_rmse": float(np.sqrt(np.mean(evaluation.normalized_residuals**2))),
        "maximum_absolute_normalized_residual": float(np.max(np.abs(evaluation.normalized_residuals))),
    }


def write_omitted_data(
    case: CaseInputs,
    omitted_label: str,
    path: pathlib.Path,
) -> None:
    """Write eight retained curves to one deterministic checkpoint input."""

    path.parent.mkdir(parents=True, exist_ok=True)
    retained = [curve.data for curve in case.curves if curve.label != omitted_label]
    if len(retained) != 8:
        raise ValueError(f"Expected one omitted ratio {omitted_label}")
    np.savetxt(path, np.vstack(retained), delimiter="\t", fmt="%.17g")


def run_knit(
    case: CaseInputs,
    candidates: Sequence[CandidateDefinition],
    fixed_maxima: Mapping[int, float],
    checkpoint_path: pathlib.Path,
    study_manifest_hash: str,
) -> dict[str, Any]:
    """Fit all candidates to full data and all nine ratio-omission folds."""

    state = load_or_initialize_checkpoint(
        checkpoint_path, case.knit, study_manifest_hash
    )
    full_wrapper = configured_wrapper(case.compiled_path, case, fixed_maxima)
    full_solver = SVExpSeparableSolver(full_wrapper, FIT_AXES)
    for index, candidate in enumerate(candidates, start=1):
        if candidate.candidate_id in state["full_fits"]:
            continue
        print(
            f"{case.knit} full [{index:02d}/63] {candidate.candidate_id} "
            f"({', '.join(candidate.components)})",
            flush=True,
        )
        solution, parameters = fit_candidate(full_solver, case, candidate)
        state["full_fits"][candidate.candidate_id] = {
            "solution": asdict(solution),
            "parameters": parameters,
        }
        state["updated_at"] = utc_now()
        atomic_json(checkpoint_path, state)

    training_directory = checkpoint_path.parent / "training_data"
    for fold_index, curve in enumerate(case.curves, start=1):
        fold = state["folds"].setdefault(
            curve.label,
            {
                "omitted_label": curve.label,
                "omitted_path": str(curve.path.resolve()),
                "fits": {},
            },
        )
        training_path = training_directory / f"without_{curve.label.replace(':', '_')}.dat"
        write_omitted_data(case, curve.label, training_path)
        fold_wrapper = configured_wrapper(training_path, case, fixed_maxima)
        fold_solver = SVExpSeparableSolver(fold_wrapper, FIT_AXES)
        held_wrapper = configured_wrapper(curve.path, case, fixed_maxima)
        for candidate_index, candidate in enumerate(candidates, start=1):
            if candidate.candidate_id in fold["fits"]:
                continue
            print(
                f"{case.knit} fold {fold_index}/9 {curve.label} "
                f"[{candidate_index:02d}/63] {candidate.candidate_id}",
                flush=True,
            )
            solution, parameters = fit_candidate(fold_solver, case, candidate)
            fold["fits"][candidate.candidate_id] = {
                "solution": asdict(solution),
                "parameters": parameters,
                "held_out": evaluate_held_out(held_wrapper, parameters),
            }
            state["updated_at"] = utc_now()
            atomic_json(checkpoint_path, state)
    state["completed_at"] = utc_now()
    atomic_json(checkpoint_path, state)
    return state


def fold_checkpoint_path(base_checkpoint: pathlib.Path, ratio: str) -> pathlib.Path:
    """Return one independent fold checkpoint, for example fold_checkpoints/1_0.json."""

    return base_checkpoint.parent / "fold_checkpoints" / f"{ratio.replace(':', '_')}.json"


def run_fold_worker(
    case: CaseInputs,
    candidates: Sequence[CandidateDefinition],
    fixed_maxima: Mapping[int, float],
    base_checkpoint: pathlib.Path,
    study_manifest_hash: str,
    ratio: str,
) -> dict[str, Any]:
    """Fit one ratio omission in an isolated atomic checkpoint."""

    base = load_or_initialize_checkpoint(base_checkpoint, case.knit, study_manifest_hash)
    if len(base["full_fits"]) != 63:
        raise ValueError(f"Full-data checkpoint is incomplete for {case.knit}")
    curve = next(item for item in case.curves if item.label == ratio)
    checkpoint_path = fold_checkpoint_path(base_checkpoint, ratio)
    if checkpoint_path.is_file():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            state.get("manifest_hash") != study_manifest_hash
            or state.get("knit") != case.knit
            or state.get("omitted_label") != ratio
        ):
            raise ValueError(f"Fold checkpoint configuration mismatch: {checkpoint_path}")
    else:
        existing = base.get("folds", {}).get(ratio, {})
        state = {
            "schema_version": 1,
            "manifest_hash": study_manifest_hash,
            "knit": case.knit,
            "omitted_label": ratio,
            "omitted_path": str(curve.path.resolve()),
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "fits": copy.deepcopy(existing.get("fits", {})),
        }
    training_path = base_checkpoint.parent / "training_data" / f"without_{ratio.replace(':', '_')}.dat"
    write_omitted_data(case, ratio, training_path)
    fold_wrapper = configured_wrapper(training_path, case, fixed_maxima)
    fold_solver = SVExpSeparableSolver(fold_wrapper, FIT_AXES)
    held_wrapper = configured_wrapper(curve.path, case, fixed_maxima)
    for candidate_index, candidate in enumerate(candidates, start=1):
        if candidate.candidate_id in state["fits"]:
            continue
        print(
            f"{case.knit} isolated fold {ratio} "
            f"[{candidate_index:02d}/63] {candidate.candidate_id}",
            flush=True,
        )
        solution, parameters = fit_candidate(fold_solver, case, candidate)
        state["fits"][candidate.candidate_id] = {
            "solution": asdict(solution),
            "parameters": parameters,
            "held_out": evaluate_held_out(held_wrapper, parameters),
        }
        state["updated_at"] = utc_now()
        atomic_json(checkpoint_path, state)
    state["completed_at"] = utc_now()
    atomic_json(checkpoint_path, state)
    return state


def combine_complete_checkpoint(
    base_checkpoint: pathlib.Path,
    knit: str,
    study_manifest_hash: str,
) -> dict[str, Any]:
    """Merge the base state and independent fold checkpoints for reporting."""

    state = copy.deepcopy(
        load_or_initialize_checkpoint(base_checkpoint, knit, study_manifest_hash)
    )
    if len(state["full_fits"]) != 63:
        raise ValueError(f"Incomplete full-data checkpoint for {knit}")
    for ratio in RATIO_ORDER:
        separate_path = fold_checkpoint_path(base_checkpoint, ratio)
        if separate_path.is_file():
            separate = json.loads(separate_path.read_text(encoding="utf-8"))
            if (
                separate.get("manifest_hash") != study_manifest_hash
                or separate.get("knit") != knit
                or separate.get("omitted_label") != ratio
            ):
                raise ValueError(f"Fold checkpoint configuration mismatch: {separate_path}")
            fold = state["folds"].setdefault(
                ratio,
                {
                    "omitted_label": ratio,
                    "omitted_path": separate["omitted_path"],
                    "fits": {},
                },
            )
            fold["fits"].update(separate["fits"])
        if ratio not in state["folds"] or len(state["folds"][ratio]["fits"]) != 63:
            count = len(state.get("folds", {}).get(ratio, {}).get("fits", {}))
            raise ValueError(f"Incomplete fold checkpoint for {knit} {ratio}: {count}/63")
    state["completed_at"] = state.get("completed_at", utc_now())
    return state


def aggregate_candidate_scores(
    knit: str,
    candidates: Sequence[CandidateDefinition],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate observation-weighted and equal-ratio held-out costs."""

    rows: list[dict[str, Any]] = []
    folds = list(state["folds"].values())
    if len(folds) != 9:
        raise ValueError(f"Expected nine completed folds for {knit}")
    for candidate in candidates:
        evaluations = [fold["fits"][candidate.candidate_id]["held_out"] for fold in folds]
        raw_cost = float(sum(item["raw_cost"] for item in evaluations))
        observations = int(sum(item["observation_count"] for item in evaluations))
        rows.append(
            {
                "knit": knit,
                "candidate_id": candidate.candidate_id,
                "mask": candidate.mask,
                "components": "; ".join(candidate.components),
                "effective_parameter_count": candidate.effective_parameter_count,
                "held_out_raw_cost": raw_cost,
                "held_out_mean_cost": raw_cost / observations,
                "equal_ratio_mean_cost": float(np.mean([item["mean_cost"] for item in evaluations])),
                "held_out_observation_count": observations,
                "full_data_training_mean_cost": float(
                    state["full_fits"][candidate.candidate_id]["solution"]["mean_cost"]
                ),
            }
        )
    primary_best = min(row["held_out_mean_cost"] for row in rows)
    equal_best = min(row["equal_ratio_mean_cost"] for row in rows)
    for row in rows:
        row["relative_to_primary_winner"] = row["held_out_mean_cost"] / primary_best - 1.0
        row["relative_to_equal_ratio_winner"] = row["equal_ratio_mean_cost"] / equal_best - 1.0
        row["primary_adequate"] = row["relative_to_primary_winner"] <= SIMPLICITY_TOLERANCE
        row["equal_ratio_adequate"] = row["relative_to_equal_ratio_winner"] <= SIMPLICITY_TOLERANCE
    return rows


def select_recommended_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose the simplest primary-adequate model with deterministic tie-breaking."""

    adequate = [row for row in rows if bool(row["primary_adequate"])]
    if not adequate:
        raise ValueError("No primary-adequate model exists")
    selected = min(
        adequate,
        key=lambda row: (
            int(row["effective_parameter_count"]),
            float(row["held_out_mean_cost"]),
            int(row["mask"]),
        ),
    )
    winner = min(rows, key=lambda row: (float(row["held_out_mean_cost"]), int(row["mask"])))
    return {"predictive_winner": dict(winner), "recommended": dict(selected)}


def classify_components(
    knit: str,
    candidates: Sequence[CandidateDefinition],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Classify required, substitutable, unsupported, and weighting-sensitive terms."""

    candidate_by_id = {item.candidate_id: item for item in candidates}
    primary_adequate = [row for row in rows if bool(row["primary_adequate"])]
    equal_adequate = [row for row in rows if bool(row["equal_ratio_adequate"])]
    best_primary = min(float(row["held_out_mean_cost"]) for row in rows)

    def status(component: str, adequate: Sequence[Mapping[str, Any]]) -> str:
        included = [component in candidate_by_id[str(row["candidate_id"])].components for row in adequate]
        if all(included):
            return "Required"
        if any(included):
            return "Optional/substitutable"
        return "Unsupported"

    output = []
    for component in COMPONENT_ORDER:
        without = [
            row
            for row in rows
            if component not in candidate_by_id[str(row["candidate_id"])].components
        ]
        with_component = [
            row
            for row in rows
            if component in candidate_by_id[str(row["candidate_id"])].components
        ]
        primary_status = status(component, primary_adequate)
        equal_status = status(component, equal_adequate)
        best_without = min(float(row["held_out_mean_cost"]) for row in without)
        best_with = min(float(row["held_out_mean_cost"]) for row in with_component)
        output.append(
            {
                "knit": knit,
                "component": component,
                "primary_status": primary_status,
                "equal_ratio_status": equal_status,
                "weighting_sensitive": primary_status != equal_status,
                "best_without_mean_cost": best_without,
                "best_with_mean_cost": best_with,
                "best_without_penalty_fraction": best_without / best_primary - 1.0,
                "best_with_penalty_fraction": best_with / best_primary - 1.0,
                "required_threshold_fraction": SIMPLICITY_TOLERANCE,
            }
        )
    return output


def fold_score_rows(
    knit: str,
    candidates: Sequence[CandidateDefinition],
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten every candidate/fold training and held-out result."""

    candidate_by_id = {item.candidate_id: item for item in candidates}
    rows = []
    for ratio in RATIO_ORDER:
        fold = state["folds"][ratio]
        for candidate_id, payload in fold["fits"].items():
            candidate = candidate_by_id[candidate_id]
            solution = payload["solution"]
            rows.append(
                {
                    "knit": knit,
                    "ratio": ratio,
                    "candidate_id": candidate_id,
                    "mask": candidate.mask,
                    "effective_parameter_count": candidate.effective_parameter_count,
                    "training_mean_cost": solution["mean_cost"],
                    "held_out_raw_cost": payload["held_out"]["raw_cost"],
                    "held_out_mean_cost": payload["held_out"]["mean_cost"],
                    "held_out_observation_count": payload["held_out"]["observation_count"],
                    "normalized_residual_rmse": payload["held_out"]["normalized_residual_rmse"],
                    "design_matrix_rank": solution["design_matrix_rank"],
                    "design_matrix_columns": len(solution["linear_values"]),
                    "condition_number": solution["design_matrix_condition_number"],
                    "active_bounds": json.dumps(solution["active_bounds"], sort_keys=True),
                    "coupling_upper_bound": solution["coupling_upper_bound"],
                    "maximum_weighted_prediction_difference": solution["maximum_weighted_prediction_difference"],
                    "direct_cost_difference": solution["direct_cost_difference"],
                }
            )
    return rows


def parameter_rows(
    case: CaseInputs,
    candidate_id: str,
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Report full-data and fold parameter stability for the selected structure."""

    full_payload = state["full_fits"][candidate_id]
    full = full_payload["parameters"]
    fold_payloads = [state["folds"][ratio]["fits"][candidate_id] for ratio in RATIO_ORDER]
    folds = [payload["parameters"] for payload in fold_payloads]
    active_linear = set(full_payload["solution"]["linear_values"])
    active_shapes = set(full_payload["solution"]["shape_values"])

    def status(name: str, payload: Mapping[str, Any]) -> str:
        solution = payload["solution"]
        if name in AMPLITUDE_PARAMETERS and name not in solution["linear_values"]:
            return "inactive_fixed_zero"
        if name in SHAPE_PARAMETER_BY_AMPLITUDE.values() and name not in solution["shape_values"]:
            return "inactive_removed"
        if name in solution["active_bounds"]:
            if name == "K_coup" and np.isclose(
                float(solution["coupling_upper_bound"]),
                float(case.parameter_specs[name]["minimum"]),
                rtol=0.0,
                atol=1.0e-14,
            ):
                return "fixed_equal_bounds"
            return {-1: "lower", 0: "interior", 1: "upper"}[
                int(solution["active_bounds"][name])
            ]
        minimum = float(case.parameter_specs[name]["minimum"])
        maximum = float(case.parameter_specs[name]["maximum"])
        value = float(payload["parameters"][name])
        if np.isclose(value, minimum, rtol=0.0, atol=1.0e-10):
            return "lower"
        if np.isclose(value, maximum, rtol=0.0, atol=1.0e-10):
            return "upper"
        return "interior"

    rows = []
    for name in ("mu", "mu2", "mu3", "K_1ani1", "beta_1", "K_1ani2", "gamma_1", "K_coup", "zeta"):
        values = np.asarray([float(payload[name]) for payload in folds], dtype=float)
        full_value = float(full[name])
        changes = np.abs(values - full_value)
        influential = int(np.argmax(changes))
        fold_statuses = [status(name, payload) for payload in fold_payloads]
        specification = case.parameter_specs[name]
        declared_maximum = float(specification["maximum"])
        full_effective_upper = (
            full_payload["solution"]["coupling_upper_bound"]
            if name == "K_coup" and name in active_linear
            else declared_maximum
        )
        fold_effective_uppers = [
            (
                payload["solution"]["coupling_upper_bound"]
                if name == "K_coup" and name in payload["solution"]["linear_values"]
                else declared_maximum
            )
            for payload in fold_payloads
        ]
        rows.append(
            {
                "knit": case.knit,
                "candidate_id": candidate_id,
                "parameter": name,
                "active_in_selected_structure": name in active_linear or name in active_shapes,
                "declared_minimum": float(specification["minimum"]),
                "declared_maximum": declared_maximum,
                "full_data_effective_upper_bound": full_effective_upper,
                "fold_effective_upper_minimum": float(min(fold_effective_uppers)),
                "fold_effective_upper_maximum": float(max(fold_effective_uppers)),
                "full_data_bound_status": status(name, full_payload),
                "fold_lower_bound_count": fold_statuses.count("lower"),
                "fold_upper_bound_count": fold_statuses.count("upper"),
                "fold_interior_count": fold_statuses.count("interior"),
                "fold_fixed_equal_bound_count": fold_statuses.count("fixed_equal_bounds"),
                "fold_inactive_count": sum(item.startswith("inactive") for item in fold_statuses),
                "full_data_value": full_value,
                "fold_minimum": float(np.min(values)),
                "fold_maximum": float(np.max(values)),
                "fold_mean": float(np.mean(values)),
                "fold_standard_deviation": float(np.std(values, ddof=1)),
                "maximum_absolute_change": float(changes[influential]),
                "maximum_relative_change": (
                    float(changes[influential] / abs(full_value)) if full_value != 0 else None
                ),
                "most_influential_omission": RATIO_ORDER[influential],
            }
        )
    return rows


def profiled_umat_validation_rows(
    knit: str,
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Flatten direct compiled-UMAT agreement for every retained fit solution."""

    rows: list[dict[str, Any]] = []
    for candidate_id, payload in state["full_fits"].items():
        solution = payload["solution"]
        rows.append(
            {
                "validation_scope": "profiled_prediction_vs_compiled_umat",
                "knit": knit,
                "dataset_scope": "full_data",
                "ratio": "ALL",
                "candidate_id": candidate_id,
                "maximum_weighted_prediction_difference": solution["maximum_weighted_prediction_difference"],
                "direct_cost_difference": solution["direct_cost_difference"],
                "tolerance": 1.0e-10,
                "passed": (
                    float(solution["maximum_weighted_prediction_difference"]) < 1.0e-10
                    and float(solution["direct_cost_difference"]) < 1.0e-10
                ),
            }
        )
    for ratio in RATIO_ORDER:
        for candidate_id, payload in state["folds"][ratio]["fits"].items():
            solution = payload["solution"]
            rows.append(
                {
                    "validation_scope": "profiled_prediction_vs_compiled_umat",
                    "knit": knit,
                    "dataset_scope": "leave_one_ratio_out_training",
                    "ratio": ratio,
                    "candidate_id": candidate_id,
                    "maximum_weighted_prediction_difference": solution["maximum_weighted_prediction_difference"],
                    "direct_cost_difference": solution["direct_cost_difference"],
                    "tolerance": 1.0e-10,
                    "passed": (
                        float(solution["maximum_weighted_prediction_difference"]) < 1.0e-10
                        and float(solution["direct_cost_difference"]) < 1.0e-10
                    ),
                }
            )
    return rows


def pk2_to_pk1_history(dfgrds: FloatArray, pk2: FloatArray) -> FloatArray:
    """Convert a PK2 history to first Piola stress row by row."""

    return np.asarray(
        [
            mech.matrix_to_voigt(
                mech.pk2_to_pk1(mech.voigt_to_matrix(stress), mech.voigt_to_matrix(F))
            )
            for F, stress in zip(dfgrds, pk2)
        ],
        dtype=float,
    )


def r2_parts(observed: FloatArray, predicted: FloatArray) -> tuple[float, float, float]:
    """Return R2, SSE, and SST without clipping negative performance."""

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    sse = float(np.sum((observed - predicted) ** 2))
    sst = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = math.nan if sst == 0 else 1.0 - sse / sst
    return r2, sse, sst


def prediction_and_metric_rows(
    case: CaseInputs,
    selected_id: str,
    state: Mapping[str, Any],
    fixed_maxima: Mapping[int, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate selected/full structures for calibration and held-out PK1 reporting."""

    full_id = "M63"
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    umat_validation_rows: list[dict[str, Any]] = []
    direction_axis = {direction: index for index, direction in enumerate(case.axis_directions)}
    modes = {
        "selected_full_data": state["full_fits"][selected_id]["parameters"],
        "full_model_full_data": state["full_fits"][full_id]["parameters"],
    }
    for ratio in RATIO_ORDER:
        curve = next(item for item in case.curves if item.label == ratio)
        wrapper = configured_wrapper(curve.path, case, fixed_maxima)
        evaluations = {
            label: historical.evaluate_full_parameter_set(wrapper, parameters, FIT_AXES)
            for label, parameters in modes.items()
        }
        fold_parameters = state["folds"][ratio]["fits"][selected_id]["parameters"]
        evaluations["selected_held_out"] = historical.evaluate_full_parameter_set(
            wrapper, fold_parameters, FIT_AXES
        )
        full_fold_parameters = state["folds"][ratio]["fits"][full_id]["parameters"]
        evaluations["full_model_held_out"] = historical.evaluate_full_parameter_set(
            wrapper, full_fold_parameters, FIT_AXES
        )
        predicted_pk1 = {
            label: pk2_to_pk1_history(np.asarray(wrapper.dfgrds), evaluation.predicted_pk2)
            for label, evaluation in evaluations.items()
        }
        experimental_pk1 = np.asarray(wrapper.expstresses, dtype=float)
        experimental_pk2 = np.asarray(wrapper.expstressPK2, dtype=float)
        for direction in ("course", "wale"):
            axis = direction_axis[direction]
            for label, values in predicted_pk1.items():
                r2, sse, sst = r2_parts(experimental_pk1[:, axis], values[:, axis])
                metric_rows.append(
                    {
                        "knit": case.knit,
                        "ratio": ratio,
                        "direction": direction,
                        "evaluation": label,
                        "evidence_scope": (
                            "calibration"
                            if label.endswith("full_data")
                            else "within_specimen_loading_path_transfer"
                        ),
                        "R2": r2,
                        "SSE": sse,
                        "SST": sst,
                        "observation_count": int(experimental_pk1.shape[0]),
                    }
                )
            for row_index in range(experimental_pk1.shape[0]):
                prediction_rows.append(
                    {
                        "knit": case.knit,
                        "ratio": ratio,
                        "row_index": row_index,
                        "direction": direction,
                        "axis": axis,
                        "stretch_axis1": float(mech.voigt_to_matrix(wrapper.dfgrds[row_index])[0, 0]),
                        "stretch_axis2": float(mech.voigt_to_matrix(wrapper.dfgrds[row_index])[1, 1]),
                        "experimental_pk1": float(experimental_pk1[row_index, axis]),
                        "selected_full_data_pk1": float(predicted_pk1["selected_full_data"][row_index, axis]),
                        "full_model_full_data_pk1": float(predicted_pk1["full_model_full_data"][row_index, axis]),
                        "selected_held_out_pk1": float(predicted_pk1["selected_held_out"][row_index, axis]),
                        "full_model_held_out_pk1": float(predicted_pk1["full_model_held_out"][row_index, axis]),
                        "experimental_pk2": float(experimental_pk2[row_index, axis]),
                        "selected_full_data_pk2": float(evaluations["selected_full_data"].predicted_pk2[row_index, axis]),
                        "full_model_full_data_pk2": float(evaluations["full_model_full_data"].predicted_pk2[row_index, axis]),
                        "selected_held_out_pk2": float(evaluations["selected_held_out"].predicted_pk2[row_index, axis]),
                        "full_model_held_out_pk2": float(evaluations["full_model_held_out"].predicted_pk2[row_index, axis]),
                    }
                )
    for evaluation in (
        "selected_full_data",
        "full_model_full_data",
        "selected_held_out",
        "full_model_held_out",
    ):
        selected = [row for row in metric_rows if row["evaluation"] == evaluation]
        sse = float(sum(row["SSE"] for row in selected))
        sst = float(sum(row["SST"] for row in selected))
        metric_rows.append(
            {
                "knit": case.knit,
                "ratio": "ALL",
                "direction": "pooled",
                "evaluation": evaluation,
                "evidence_scope": (
                    "calibration"
                    if evaluation.endswith("full_data")
                    else "within_specimen_loading_path_transfer"
                ),
                "R2": math.nan if sst == 0 else 1.0 - sse / sst,
                "SSE": sse,
                "SST": sst,
                "observation_count": int(sum(row["observation_count"] for row in selected)),
            }
        )
    return prediction_rows, metric_rows


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one flat machine-readable table with stable headers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(name for name in row if name not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def create_component_matrix_figure(rows: Sequence[Mapping[str, Any]]) -> plt.Figure:
    """Show required/optional/unsupported status for six components and four knits."""

    knit_order = list(KNITS)
    matrix = np.zeros((len(knit_order), len(COMPONENT_ORDER)), dtype=int)
    status_value = {name: index for index, name in enumerate(STATUS_ORDER)}
    sensitive = np.zeros_like(matrix, dtype=bool)
    lookup = {(row["knit"], row["component"]): row for row in rows}
    for i, knit in enumerate(knit_order):
        for j, component in enumerate(COMPONENT_ORDER):
            item = lookup[(knit, component)]
            matrix[i, j] = status_value[item["primary_status"]]
            sensitive[i, j] = bool(item["weighting_sensitive"])
    cmap = mcolors.ListedColormap([STATUS_COLORS[name] for name in STATUS_ORDER])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    figure, axis = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(np.arange(len(COMPONENT_ORDER)), [name.replace("_", " ").title() for name in COMPONENT_ORDER], rotation=35, ha="right")
    axis.set_yticks(np.arange(len(knit_order)), knit_order)
    axis.set_title("SV_exp component necessity from leave-one-ratio-out prediction")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = STATUS_ORDER[matrix[i, j]].split("/")[0]
            if sensitive[i, j]:
                label += "*"
            axis.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if matrix[i, j] != 1 else "black")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLORS[name], label=name) for name in STATUS_ORDER]
    handles.append(plt.Line2D([], [], color="none", label="* equal-ratio sensitivity"))
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False)
    return figure


def create_penalty_figure(rows: Sequence[Mapping[str, Any]]) -> plt.Figure:
    """Plot the best predictive penalty after excluding each component."""

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey=True)
    labels = [name.replace("_", "\n") for name in COMPONENT_ORDER]
    for axis, knit in zip(axes.flat, KNITS):
        knit_rows = [row for row in rows if row["knit"] == knit]
        lookup = {row["component"]: row for row in knit_rows}
        penalties = [100.0 * float(lookup[name]["best_without_penalty_fraction"]) for name in COMPONENT_ORDER]
        colors = [STATUS_COLORS[lookup[name]["primary_status"]] for name in COMPONENT_ORDER]
        axis.bar(np.arange(len(labels)), penalties, color=colors)
        axis.axhline(100.0 * SIMPLICITY_TOLERANCE, color="black", linestyle="--", linewidth=1.0, label="2% threshold")
        axis.set_xticks(np.arange(len(labels)), labels, fontsize=8)
        axis.set_title(knit)
        axis.set_ylabel("Best-without penalty (%)")
        axis.grid(True, axis="y", alpha=0.25)
    axes.flat[0].legend(frameon=False)
    figure.suptitle("Predictive penalty for excluding each SV_exp component")
    return figure


def create_curve_figure(knit: str, rows: Sequence[Mapping[str, Any]]) -> plt.Figure:
    """Compare experimental, full-model, and selected-reduced PK1 curves."""

    figure, axes = plt.subplots(3, 3, figsize=(13.5, 12), constrained_layout=False)
    for axis, ratio in zip(axes.flat, RATIO_ORDER):
        ratio_rows = [row for row in rows if row["knit"] == knit and row["ratio"] == ratio]
        for direction, color in (("course", "#1257D1"), ("wale", "#878A8F")):
            direction_rows = sorted(
                (row for row in ratio_rows if row["direction"] == direction),
                key=lambda row: int(row["row_index"]),
            )
            x = np.arange(len(direction_rows))
            axis.scatter(x, [row["experimental_pk1"] for row in direction_rows], s=9, color=color, alpha=0.65, label=f"{direction} experiment")
            axis.plot(x, [row["full_model_held_out_pk1"] for row in direction_rows], color=color, linewidth=1.2, linestyle=":", label=f"{direction} full held-out")
            axis.plot(x, [row["selected_held_out_pk1"] for row in direction_rows], color=color, linewidth=1.5, label=f"{direction} selected held-out")
        axis.set_title(ratio)
        axis.set_xlabel("Loading-path observation")
        axis.set_ylabel("First Piola stress")
        axis.grid(True, axis="y", alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    # Reserve explicit outer margins so PDF/SVG export cannot overlap the legend,
    # titles, or y-axis labels (for example, the nine-panel Single Jersey figure).
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.925, hspace=0.34, wspace=0.25)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        f"{knit}: full versus selected SV_exp model (within-specimen held-out)",
        y=0.975,
    )
    return figure


def save_figure_pair(figure: plt.Figure, stem: pathlib.Path) -> None:
    """Save matching editable SVG and print-ready PDF exports."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(figure)


def create_stability_figure(result: Mapping[str, Any]) -> plt.Figure:
    """Create the selected model's three-panel stability envelope heatmap."""

    case: stability.FitCase = result["case"]
    surface = result["surface_data"]
    lambda_1 = np.asarray(surface["lambda_1"], dtype=float)
    lambda_2 = np.asarray(surface["lambda_2"], dtype=float)
    measured = stability.measured_states(case)
    summary = {
        row["scope"]: row
        for row in result["summary_rows"]
        if row["domain"] == "incompressible_envelope"
    }
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    for axis, scope in zip(axes, stability.SCOPE_ORDER):
        values = np.asarray(surface["values"][scope], dtype=float)
        image = axis.pcolormesh(lambda_1, lambda_2, values.T, shading="auto", cmap="coolwarm", norm=stability._surface_norm(values))
        axis.scatter([item["lambda_1"] for item in measured], [item["lambda_2"] for item in measured], s=5, color="black", alpha=0.22, linewidths=0)
        critical = summary[scope]
        axis.scatter([critical["lambda_1"]], [critical["lambda_2"]], marker="*", s=130, color="#FFD700", edgecolor="black", linewidth=0.8)
        axis.axvline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        axis.set_xlabel(r"$\lambda_1$ (axis 1)")
        axis.set_ylabel(r"$\lambda_2$ (axis 2)")
        axis.set_title(scope.replace("_", " ").title())
        figure.colorbar(image, ax=axis, shrink=0.86, label="Minimum rank-one curvature")
    figure.suptitle(f"SV_exp selected reduced model stability - {case.knit}")
    return figure


def run_selected_stability(
    cases: Sequence[CaseInputs],
    selections: Mapping[str, str],
    states: Mapping[str, Mapping[str, Any]],
    figures_directory: pathlib.Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the validated operational stability screen for four selected fits."""

    fit_cases = []
    for case in cases:
        config = components.FitConfiguration(
            model_id="SV_exp",
            knit=case.knit,
            fit_code=case.fit_code,
            sample_folder=case.sample_folder,
            fit_series="prescribed_b09",
        )
        datasets = components.load_ratio_datasets(config)
        components.verify_b09_training_data(config, case.artifact, datasets)
        parameters = {
            name: float(value)
            for name, value in states[case.knit]["full_fits"][selections[case.knit]]["parameters"].items()
        }
        fit_cases.append(
            stability.FitCase(
                model_id="SV_exp",
                knit=case.knit,
                fit_code=case.fit_code,
                parameters=parameters,
                payload=case.artifact,
                datasets=datasets,
            )
        )
    angles = stability.propagation_angles(STABILITY_ANGLE_POINTS)
    results = []
    for case in fit_cases:
        print(f"Stability screen: {case.knit}", flush=True)
        result = stability.analyze_case(
            case,
            angles,
            points=STABILITY_GRID_POINTS,
            refinement_starts=STABILITY_REFINEMENT_STARTS,
        )
        for collection_name in ("summary_rows", "measured_rows", "envelope_rows"):
            for row in result[collection_name]:
                row["selected_candidate_id"] = selections[case.knit]
                row["fit_code_role"] = "reference_input_artifact_only"
        results.append(result)
        save_figure_pair(
            create_stability_figure(result),
            figures_directory / f"stability_{case.knit.lower()}",
        )
    umat_rows = stability.run_isolated_umat_validations(results)
    for row in umat_rows:
        row["selected_candidate_id"] = selections[str(row["knit"])]
        row["fit_code_role"] = "reference_input_artifact_only"
    summary_rows = [row for result in results for row in result["summary_rows"]]
    measured_rows = [row for result in results for row in result["measured_rows"]]
    envelope_rows = [row for result in results for row in result["envelope_rows"]]
    return summary_rows, measured_rows, envelope_rows, umat_rows


def readme_text(summary: Mapping[str, Any]) -> str:
    """Create a concise methods, results, and interpretation README."""

    lines = [
        "# SV_exp component-necessity study",
        "",
        "This standalone study exhaustively refits all 63 nonempty subsets of the six additive SV_exp stress components for the four complete raw Tripoly specimens.",
        "",
        "## Scope",
        "",
        "The nine loading ratios from one specimen per knit are used in leave-one-ratio-out folds. These are within-specimen loading-path transfer diagnostics, not external validation or population confidence intervals.",
        "",
        "## Methods",
        "",
        "The immutable reference artifacts are Interlock S06 / fit 726, Rib S03 / fit 722, Purl S02 / fit 723, and Single Jersey S02 / fit 725. Their hashes and exact nine-ratio B09 reconstructions are recorded in `study_definition.json`.",
        "",
        f"All 63 nonempty physical component subsets were fitted with C1=C2=1, theta1=0 degrees, and theta2=90 degrees. Active amplitudes use bounded BVLS; active beta/gamma shapes use {GRID_POINTS_PER_AXIS}-point logarithmic grids followed by Powell refinement; active nonlinear coupling searches every integer zeta from {min(ZETA_VALUES)} through {max(ZETA_VALUES)}. Inactive amplitudes are zero and their unused shapes are removed from effective complexity.",
        "",
        "Each of the nine ratios is omitted once. Training and held-out objectives reuse the two full-B09 PK2 reference maxima on axes 0 and 1. Every full/fold candidate is checked against direct evaluation of the compiled frozen SV_exp UMAT.",
        "",
        f"Selected full-data fits receive the validated measured-path and {STABILITY_GRID_POINTS} x {STABILITY_GRID_POINTS} knit-envelope stability screen using {STABILITY_ANGLE_POINTS} propagation directions and {STABILITY_REFINEMENT_STARTS} local refinements, including activation-side compiled-UMAT checks.",
        "",
        "## Selection rule",
        "",
        f"The primary score pools normalized PK2 residuals over every held-out observation. Models within {100 * SIMPLICITY_TOLERANCE:g}% of the predictive winner are adequate; the recommended model is the adequate candidate with the fewest effective parameters.",
        "",
        "Component labels mean: Required = present in every adequate model; Optional/substitutable = adequate models exist with and without it; Unsupported = absent from every adequate model. An asterisk in the matrix indicates that equal-ratio weighting changes the classification.",
        "",
        "## Results",
        "",
        "| Knit | Candidate | Components | Parameters | Held-out PK2 cost selected / full | Held-out pooled PK1 R2 selected / full |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for knit in KNITS:
        item = summary["knits"][knit]
        lines.append(
            f"| {knit} | {item['recommended']['candidate_id']} | {item['recommended']['components']} | {item['recommended']['effective_parameter_count']} | {item['recommended']['held_out_mean_cost']:.6g} / {item['full_model_held_out_mean_pk2_cost']:.6g} | {item['pooled_pk1_r2']['selected_held_out']:.6g} / {item['pooled_pk1_r2']['full_model_held_out']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Post-fit component magnitudes alone were not used as evidence of necessity. Each conclusion comes from constrained refitting and held-out loading-path prediction. PK2 fitting cost and first-Piola R2 are reported separately and are not compared as interchangeable metrics.",
            "",
            "The stability screen is an applicability diagnostic, not a model-selection gate. A finite positive screen is not proof of ellipticity, and removing components does not repair SV_exp's structural lack of arbitrary 3D stabilization.",
            "",
            "## Files",
            "",
            "- `study_summary.json`: compact selections and provenance.",
            "- `candidate_scores.csv`, `fold_scores.csv`, and `component_necessity.csv`: exhaustive predictive results.",
            "- `parameters.csv`, `conditioning.csv`, `curve_metrics.csv`, and `predictions.csv`: fitted values and diagnostics.",
            "- `stability_summary.csv`, `stability_measured.csv`, `stability_envelope.csv`, and `umat_validation.csv`: selected-model stability evidence.",
            "- `sv_exp_component_necessity.xlsx`: validated publication workbook.",
            "- `figures/`: matching SVG and PDF figures.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse optional process-level controls used for safe per-knit parallel runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knit-worker",
        choices=KNITS,
        help="Run only one knit checkpoint, for example --knit-worker Rib.",
    )
    parser.add_argument(
        "--fold-worker",
        choices=RATIO_ORDER,
        help="With --knit-worker, run only one omitted ratio, for example --fold-worker 1:0.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Require complete checkpoints and only rebuild reports/deliverables.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the exhaustive four-knit study and write publication deliverables."""

    args = parse_args()
    if args.knit_worker and args.finalize_only:
        raise ValueError("--knit-worker and --finalize-only are mutually exclusive")
    if args.fold_worker and not args.knit_worker:
        raise ValueError("--fold-worker requires --knit-worker")
    if OUTPUT_DIRECTORY.exists() and not RESUME_EXISTING and not OVERWRITE_EXISTING:
        raise FileExistsError(OUTPUT_DIRECTORY)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figures_directory = OUTPUT_DIRECTORY / "figures"
    checkpoints_directory = OUTPUT_DIRECTORY / "checkpoints"
    cases = [load_case_inputs(knit) for knit in KNITS]
    if len({case.input_hashes["umat_source"] for case in cases}) != 1:
        raise ValueError("The four B09 artifacts do not share one frozen SV_exp UMAT")
    manifest = manifest_payload(cases)
    study_manifest_hash = manifest_hash(manifest)
    manifest["manifest_hash"] = study_manifest_hash
    definition_path = OUTPUT_DIRECTORY / "study_definition.json"
    if not args.knit_worker:
        atomic_json(definition_path, manifest)
    elif definition_path.is_file():
        existing = json.loads(definition_path.read_text(encoding="utf-8"))
        if existing.get("manifest_hash") != study_manifest_hash:
            raise ValueError("Existing study definition does not match the worker")
    print(f"Compiling frozen SV_exp UMAT: {cases[0].umat_source}", flush=True)
    umat.compile_umat_meson(cases[0].umat_source)

    states: dict[str, dict[str, Any]] = {}
    maxima_by_knit: dict[str, dict[int, float]] = {}
    candidates_by_knit: dict[str, tuple[CandidateDefinition, ...]] = {}
    total_start = time.perf_counter()
    active_cases = (
        [next(case for case in cases if case.knit == args.knit_worker)]
        if args.knit_worker
        else cases
    )
    for case in active_cases:
        candidates = enumerate_component_subsets(case.axis_directions)
        candidates_by_knit[case.knit] = candidates
        maxima = reference_maxima(case)
        maxima_by_knit[case.knit] = maxima
        checkpoint_path = checkpoints_directory / case.knit / "checkpoint.json"
        if args.finalize_only:
            states[case.knit] = combine_complete_checkpoint(
                checkpoint_path, case.knit, study_manifest_hash
            )
        elif args.fold_worker:
            states[case.knit] = run_fold_worker(
                case,
                candidates,
                maxima,
                checkpoint_path,
                study_manifest_hash,
                args.fold_worker,
            )
        else:
            states[case.knit] = run_knit(
                case,
                candidates,
                maxima,
                checkpoint_path,
                study_manifest_hash,
            )
    if args.knit_worker:
        print(f"Completed knit worker: {args.knit_worker}", flush=True)
        return
    # Record the derived full-data scales without changing the checkpoint hash.
    manifest["normalization_reference_maxima_pk2"] = {
        knit: {str(axis): value for axis, value in maxima.items()}
        for knit, maxima in maxima_by_knit.items()
    }
    manifest["case_metadata"] = {
        case.knit: {
            "reference_fit_code": case.fit_code,
            "sample_folder": case.sample_folder,
            "axis_directions": list(case.axis_directions),
            "physical_component_parameter_map": component_parameter_map(case.axis_directions),
            "fit_artifact": str(case.artifact_path.resolve()),
            "compiled_b09_data": str(case.compiled_path.resolve()),
        }
        for case in cases
    }
    manifest["manifest_hash_scope"] = (
        "Configuration and immutable input hashes defined before the derived "
        "normalization scales and case metadata were appended."
    )
    atomic_json(definition_path, manifest)

    candidate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    necessity_rows: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    conditioning_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    umat_validation_rows: list[dict[str, Any]] = []
    selections: dict[str, str] = {}
    knit_summaries: dict[str, Any] = {}
    for case in cases:
        candidates = candidates_by_knit[case.knit]
        state = states[case.knit]
        scores = aggregate_candidate_scores(case.knit, candidates, state)
        candidate_rows.extend(scores)
        fold_rows.extend(fold_score_rows(case.knit, candidates, state))
        necessity_rows.extend(classify_components(case.knit, candidates, scores))
        selection = select_recommended_model(scores)
        selected_id = str(selection["recommended"]["candidate_id"])
        selections[case.knit] = selected_id
        parameters.extend(parameter_rows(case, selected_id, state))
        umat_validation_rows.extend(profiled_umat_validation_rows(case.knit, state))
        selected_solution = state["full_fits"][selected_id]["solution"]
        conditioning_rows.append(
            {
                "knit": case.knit,
                "candidate_id": selected_id,
                "design_matrix_rank": selected_solution["design_matrix_rank"],
                "design_matrix_columns": len(selected_solution["linear_values"]),
                "condition_number": selected_solution["design_matrix_condition_number"],
                "optimality": selected_solution["optimality"],
                "active_bounds": json.dumps(selected_solution["active_bounds"], sort_keys=True),
                "maximum_weighted_prediction_difference": selected_solution["maximum_weighted_prediction_difference"],
                "direct_cost_difference": selected_solution["direct_cost_difference"],
            }
        )
        knit_predictions, knit_metrics = prediction_and_metric_rows(
            case, selected_id, state, maxima_by_knit[case.knit]
        )
        prediction_rows.extend(knit_predictions)
        metric_rows.extend(knit_metrics)
        pooled_by_evaluation = {
            row["evaluation"]: row["R2"]
            for row in knit_metrics
            if row["ratio"] == "ALL"
        }
        full_score = next(row for row in scores if row["candidate_id"] == "M63")
        knit_summaries[case.knit] = {
            **selection,
            "axis_directions": list(case.axis_directions),
            "sample_folder": case.sample_folder,
            "reference_fit_code": case.fit_code,
            "held_out_pooled_pk1_r2": pooled_by_evaluation["selected_held_out"],
            "pooled_pk1_r2": pooled_by_evaluation,
            "full_model_held_out_mean_pk2_cost": full_score["held_out_mean_cost"],
            "full_model_calibration_mean_pk2_cost": full_score["full_data_training_mean_cost"],
        }

    stability_summary_rows: list[dict[str, Any]] = []
    stability_measured_rows: list[dict[str, Any]] = []
    stability_envelope_rows: list[dict[str, Any]] = []
    if RUN_STABILITY:
        (
            stability_summary_rows,
            stability_measured_rows,
            stability_envelope_rows,
            stability_umat_rows,
        ) = run_selected_stability(cases, selections, states, figures_directory)
        umat_validation_rows.extend(
            {"validation_scope": "stability_critical_state", **row}
            for row in stability_umat_rows
        )

    expected_counts = {
        "candidate_scores": len(KNITS) * 63,
        "fold_scores": len(KNITS) * len(RATIO_ORDER) * 63,
        "component_necessity": len(KNITS) * len(COMPONENT_ORDER),
        "parameters": len(KNITS) * 9,
        "conditioning": len(KNITS),
        "curve_metrics": len(KNITS) * (len(RATIO_ORDER) * 2 * 4 + 4),
        "profiled_umat_validation": len(KNITS) * 63 * (1 + len(RATIO_ORDER)),
    }
    observed_counts = {
        "candidate_scores": len(candidate_rows),
        "fold_scores": len(fold_rows),
        "component_necessity": len(necessity_rows),
        "parameters": len(parameters),
        "conditioning": len(conditioning_rows),
        "curve_metrics": len(metric_rows),
        "profiled_umat_validation": sum(
            row.get("validation_scope") == "profiled_prediction_vs_compiled_umat"
            for row in umat_validation_rows
        ),
    }
    if observed_counts != expected_counts:
        raise ValueError(f"Report row-count mismatch: {observed_counts} != {expected_counts}")
    if not all(
        bool(row["passed"])
        for row in umat_validation_rows
        if row.get("validation_scope") == "profiled_prediction_vs_compiled_umat"
    ):
        raise ValueError("At least one profiled solution failed compiled-UMAT validation")

    write_csv(OUTPUT_DIRECTORY / "candidate_scores.csv", candidate_rows)
    write_csv(OUTPUT_DIRECTORY / "fold_scores.csv", fold_rows)
    write_csv(OUTPUT_DIRECTORY / "component_necessity.csv", necessity_rows)
    write_csv(OUTPUT_DIRECTORY / "parameters.csv", parameters)
    write_csv(OUTPUT_DIRECTORY / "conditioning.csv", conditioning_rows)
    write_csv(OUTPUT_DIRECTORY / "predictions.csv", prediction_rows)
    write_csv(OUTPUT_DIRECTORY / "curve_metrics.csv", metric_rows)
    if RUN_STABILITY:
        write_csv(OUTPUT_DIRECTORY / "stability_summary.csv", stability_summary_rows)
        write_csv(OUTPUT_DIRECTORY / "stability_measured.csv", stability_measured_rows)
        write_csv(OUTPUT_DIRECTORY / "stability_envelope.csv", stability_envelope_rows)
        write_csv(OUTPUT_DIRECTORY / "umat_validation.csv", umat_validation_rows)

    save_figure_pair(
        create_component_matrix_figure(necessity_rows),
        figures_directory / "component_necessity_matrix",
    )
    save_figure_pair(
        create_penalty_figure(necessity_rows),
        figures_directory / "best_without_component_penalties",
    )
    for knit in KNITS:
        save_figure_pair(
            create_curve_figure(knit, prediction_rows),
            figures_directory / f"full_vs_reduced_{knit.lower()}",
        )

    expected_figure_stems = 2 + len(KNITS) + (len(KNITS) if RUN_STABILITY else 0)
    pdf_count = len(list(figures_directory.glob("*.pdf")))
    svg_count = len(list(figures_directory.glob("*.svg")))
    if pdf_count != expected_figure_stems or svg_count != expected_figure_stems:
        raise ValueError(
            f"Figure-pair mismatch: expected {expected_figure_stems}, "
            f"found {pdf_count} PDF and {svg_count} SVG"
        )

    summary = {
        "schema_version": 1,
        "study_id": manifest["study_id"],
        "completed_at": utc_now(),
        "interpretation_limit": (
            "One complete specimen per knit; ratio folds are within-specimen "
            "loading-path diagnostics, not external validation or population intervals."
        ),
        "manifest_hash": study_manifest_hash,
        "configuration": manifest,
        "knits": knit_summaries,
        "component_necessity": necessity_rows,
        "stability_is_selection_gate": False,
        "profiled_umat_validation_count": sum(
            row.get("validation_scope") == "profiled_prediction_vs_compiled_umat"
            for row in umat_validation_rows
        ),
        "profiled_umat_all_passed": all(
            bool(row["passed"])
            for row in umat_validation_rows
            if row.get("validation_scope") == "profiled_prediction_vs_compiled_umat"
        ),
        "row_counts": {
            **observed_counts,
            "predictions": len(prediction_rows),
            "stability_summary": len(stability_summary_rows),
            "stability_measured": len(stability_measured_rows),
            "stability_envelope": len(stability_envelope_rows),
            "umat_validation_total": len(umat_validation_rows),
        },
        "figure_counts": {"pdf": pdf_count, "svg": svg_count},
        "total_elapsed_seconds": float(time.perf_counter() - total_start),
        "shared_fit_registry_modified": False,
        "source_umat_modified": False,
    }
    atomic_json(OUTPUT_DIRECTORY / "study_summary.json", summary)
    (OUTPUT_DIRECTORY / "README.md").write_text(readme_text(summary), encoding="utf-8")
    print(f"Study calculations complete: {OUTPUT_DIRECTORY}", flush=True)


if __name__ == "__main__":
    main()
