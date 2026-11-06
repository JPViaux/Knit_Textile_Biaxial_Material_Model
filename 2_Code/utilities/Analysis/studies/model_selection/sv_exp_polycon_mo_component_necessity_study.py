"""Run the approved SV_exp_polycon_mo component-necessity study.

The default VSCode run resumes the full four-knit campaign. Set ``RUN_MODE``
to ``"pilot"`` only to reproduce the historical low-resolution Interlock gate.
Both modes write isolated checkpoints and never register a production fit.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (CODE_ROOT as RELEASE_CODE_ROOT,
    ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
from release_paths import json
import os
import pathlib
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ANALYSIS_DIR = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis")
SUPPORT_DIR = ANALYSIS_DIR.parent
sys.path.insert(0, str(SUPPORT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from Analysis.fitting import separable_solver_fit as fit_runner  # noqa: E402
from Analysis.studies.model_selection import sv_exp_component_necessity_study as sv_study  # noqa: E402
from Analysis.studies.solver_diagnostics import separable_solver_coupling_bound_calibration as coupling  # noqa: E402
from Analysis.studies.solver_diagnostics import separable_solver_historical_benchmark as historical  # noqa: E402
from Analysis.studies.solver_diagnostics import separable_solver_leave_one_ratio_out as ratios  # noqa: E402
from Python_Wrapper import umat  # noqa: E402
from Python_Wrapper.model_registry import load_model_spec  # noqa: E402
from Python_Wrapper.separable_solver import (  # noqa: E402
    AffineDerivedParameter,
    RotatedConeConstraint,
    SeparableMaterialSolver,
    SeparableModelDefinition,
    deterministic_shape_grid_search,
    linear_grid,
    logarithmic_grid,
    refine_profiled_shapes,
)


# ---------------------------------------------------------------------------
# VSCode USER CONFIGURATION - press Run to execute or resume the full study.
# ---------------------------------------------------------------------------
RUN_MODE = "full"  # Examples: "pilot" or the approved "full" campaign.
RESUME_CHECKPOINTS = True  # Example: True resumes completed component masks.
FORCE_RECOMPUTE = False  # Example: True ignores existing checkpoints after code changes.

PILOT_FIT_CODE = 1962  # Interlock S06 all-nine provenance fit.
PILOT_GRID_POINTS = 5  # Five log points for beta_1 and gamma_1.
PILOT_B_GRID_POINTS = 3  # Three linear b points with derived a=1-b.
PILOT_ZETA_VALUES = (2, 6, 10, 15)  # Coarse exponent domain for the pilot only.
FULL_ZETA_VALUES = tuple(range(3, 16))  # Standing rule: production zeta must be >2.
ADEQUATE_RELATIVE_TOLERANCE = 0.02  # Candidates within 2% of the score winner.

OUTPUT_ROOT = (
    RELEASE_ARCHIVE_ROOT
    / "sv_exp_polycon_mo_component_necessity"
    / "all_nine_v1"
)
PILOT_OUTPUT = OUTPUT_ROOT / "pilot_interlock"
CHECKPOINT_DIR = PILOT_OUTPUT / "checkpoints"
FULL_CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

FULL_FIT_CODES = {
    "Interlock": 1962,
    "Rib": 1961,
    "Purl": 1963,
    "Single Jersey": 1964,
}
FULL_GRID_POINTS = 17  # Production beta_1/gamma_1 logarithmic grid count.
FULL_B_GRID_POINTS = 9  # Production b grid count; a is always 1-b.
FULL_REFINEMENT_ITERATIONS = 100  # Powell maximum per retained integer zeta.
FULL_PARAMETER_TOLERANCE = 1.0e-6  # Example: convergence in transformed shapes.
FULL_COST_TOLERANCE = 1.0e-10  # Example: Powell objective convergence tolerance.

FIT_AXES = (0, 1)
COMPONENT_ORDER = (
    "linear_course",
    "linear_wale",
    "linear_coupling",
    "nonlinear_course",
    "nonlinear_wale",
    "nonlinear_coupling",
)
FULL_LINEAR_ORDER = ("mu", "mu2", "mu3", "K_1ani1", "K_1ani2", "K_coup")
SHAPE_BY_AMPLITUDE = {
    "K_1ani1": ("beta_1",),
    "K_1ani2": ("gamma_1",),
    "K_coup": ("b", "zeta"),
}


@dataclass(frozen=True)
class Candidate:
    """One structurally admissible component mask."""

    mask: int
    candidate_id: str
    components: tuple[str, ...]
    linear_parameters: tuple[str, ...]
    continuous_shapes: tuple[str, ...]
    integer_shapes: tuple[str, ...]
    effective_parameter_count: int


def utc_now() -> str:
    """Return one stable ISO timestamp, for example 2026-08-26T12:00:00Z."""

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: pathlib.Path) -> str:
    """Return the SHA-256 digest of one immutable input."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    """Convert NumPy, tuple, and Path values to JSON-compatible objects."""

    if isinstance(value, pathlib.Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def atomic_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one JSON checkpoint, for example M39.json."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def physical_parameter_map(axis_directions: tuple[str, str]) -> dict[str, str]:
    """Map course/wale labels to the compiled family-1/family-2 parameters."""

    return sv_study.component_parameter_map(axis_directions)


def enumerate_admissible_subsets(
    axis_directions: tuple[str, str],
) -> tuple[Candidate, ...]:
    """Enumerate 39 masks; mu3 is allowed only with both directional linear terms."""

    mapping = physical_parameter_map(axis_directions)
    candidates: list[Candidate] = []
    for mask in range(1, 1 << len(COMPONENT_ORDER)):
        components = tuple(
            name for index, name in enumerate(COMPONENT_ORDER) if mask & (1 << index)
        )
        if "linear_coupling" in components and not {
            "linear_course",
            "linear_wale",
        }.issubset(components):
            continue
        active = {mapping[name] for name in components}
        linear = tuple(name for name in FULL_LINEAR_ORDER if name in active)
        shapes = {shape for name in linear for shape in SHAPE_BY_AMPLITUDE.get(name, ())}
        continuous = tuple(name for name in ("beta_1", "gamma_1", "b") if name in shapes)
        integers = ("zeta",) if "zeta" in shapes else ()
        candidates.append(
            Candidate(
                mask=mask,
                candidate_id=f"M{mask:02d}",
                components=components,
                linear_parameters=linear,
                continuous_shapes=continuous,
                integer_shapes=integers,
                effective_parameter_count=len(linear) + len(shapes),
            )
        )
    if len(candidates) != 39 or len({item.mask for item in candidates}) != 39:
        raise RuntimeError("Admissible enumeration must produce 39 unique masks")
    return tuple(candidates)


def reduced_definition(candidate: Candidate) -> SeparableModelDefinition:
    """Remove inactive amplitudes and all shape variables owned by them."""

    dependencies = tuple(
        (name, SHAPE_BY_AMPLITUDE[name])
        for name in candidate.linear_parameters
        if name in SHAPE_BY_AMPLITUDE
    )
    derived = (
        (AffineDerivedParameter(target="a", offset=1.0, coefficients=(("b", -1.0),)),)
        if "K_coup" in candidate.linear_parameters
        else ()
    )
    cone = (
        (RotatedConeConstraint(first="mu", second="mu2", cross="mu3"),)
        if "mu3" in candidate.linear_parameters
        else ()
    )
    scales = tuple(
        (name, "log" if name in {"beta_1", "gamma_1"} else "linear")
        for name in candidate.continuous_shapes
    )
    return SeparableModelDefinition(
        model_id="SV_exp_polycon_mo",
        linear_parameters=candidate.linear_parameters,
        continuous_shape_parameters=candidate.continuous_shapes,
        integer_shape_parameters=candidate.integer_shapes,
        basis_shape_parameters=dependencies,
        derived_parameters=derived,
        rotated_cone_constraints=cone,
        continuous_shape_scales=scales,
    )


def adequate_rows(rows: Sequence[Mapping[str, Any]], score_key: str) -> list[Mapping[str, Any]]:
    """Return candidates no more than 2% above the declared score winner."""

    winner = min(float(row[score_key]) for row in rows)
    limit = winner * (1.0 + ADEQUATE_RELATIVE_TOLERANCE)
    return [row for row in rows if float(row[score_key]) <= limit + 1.0e-15]


def select_reduced(rows: Sequence[Mapping[str, Any]], score_key: str) -> Mapping[str, Any]:
    """Select minimum complexity, then score, then deterministic mask order."""

    return min(
        adequate_rows(rows, score_key),
        key=lambda row: (
            int(row["effective_parameter_count"]),
            float(row[score_key]),
            int(row["mask"]),
        ),
    )


def classify(rows: Sequence[Mapping[str, Any]], score_key: str) -> dict[str, str]:
    """Classify components within one 2%-adequate calibration set."""

    adequate = adequate_rows(rows, score_key)
    classifications: dict[str, str] = {}
    for component in COMPONENT_ORDER:
        contains = [component in row["components"] for row in adequate]
        classifications[component] = (
            "Required" if all(contains) else "Unsupported" if not any(contains) else "Optional/substitutable"
        )
    return classifications


def load_pilot_inputs() -> dict[str, Any]:
    """Validate fit 1962, its frozen all-nine snapshot, and raw ratio reconstruction."""

    artifact_path = RELEASE_ARCHIVE_ROOT / "fits" / str(PILOT_FIT_CODE) / "fit.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("model_id") != "SV_exp_polycon_mo":
        raise ValueError(f"Fit {PILOT_FIT_CODE} is not SV_exp_polycon_mo")
    context = artifact["experiment_context"]
    if context.get("knit") != "Interlock" or artifact.get("result_role") != "training_optimum":
        raise ValueError(f"Fit {PILOT_FIT_CODE} is not the Interlock training optimum")
    compiled_path = pathlib.Path(context["exp_data_path"])
    snapshot_path = pathlib.Path(artifact["input_files"]["experimental_data"]["snapshot_path"])
    parameter_path = pathlib.Path(artifact["input_files"]["material_parameters"]["snapshot_path"])
    umat_snapshot = pathlib.Path(artifact["input_files"]["umat_source"]["snapshot_path"])
    compiled = np.atleast_2d(np.loadtxt(compiled_path, dtype=float))
    snapshot = np.atleast_2d(np.loadtxt(snapshot_path, dtype=float))
    if not np.array_equal(compiled, snapshot):
        raise ValueError("Raw Interlock all-nine data do not match fit 1962's snapshot")
    curves = ratios.discover_ordered_ratio_curves(compiled_path)
    if len(curves) != 9 or not np.array_equal(np.vstack([curve.data for curve in curves]), snapshot):
        raise ValueError("The nine ratio files do not exactly reconstruct fit 1962's snapshot")
    axes = sv_study.metadata_axis_directions(str(context["sample_folder"]))
    if axes != ("wale", "course"):
        raise ValueError(f"Interlock axis reversal changed: expected wale/course, found {axes}")
    hashes = {
        "fit_artifact": sha256_file(artifact_path),
        "compiled_experimental_data": sha256_file(compiled_path),
        "experimental_snapshot": sha256_file(snapshot_path),
        "material_parameters": sha256_file(parameter_path),
        "umat_snapshot": sha256_file(umat_snapshot),
        **{f"ratio_{curve.label}": sha256_file(curve.path) for curve in curves},
    }
    for key, artifact_key in (
        ("experimental_snapshot", "experimental_data"),
        ("material_parameters", "material_parameters"),
        ("umat_snapshot", "umat_source"),
    ):
        if hashes[key] != artifact["input_files"][artifact_key]["sha256"]:
            raise ValueError(f"Fit 1962 frozen {key} hash changed")
    return {
        "artifact": artifact,
        "artifact_path": artifact_path,
        "compiled_path": compiled_path,
        "snapshot_path": snapshot_path,
        "parameter_path": parameter_path,
        "umat_source": RELEASE_CODE_ROOT / '1_Material_model/SV_exp_polycon_mo.f',
        "curves": curves,
        "axis_directions": axes,
        "hashes": hashes,
    }


def load_full_inputs(knit: str) -> dict[str, Any]:
    """Validate one reviewed all-nine fit snapshot and its nine raw curves."""

    fit_code = int(FULL_FIT_CODES[knit])
    artifact_path = RELEASE_ARCHIVE_ROOT / "fits" / str(fit_code) / "fit.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    context = artifact["experiment_context"]
    stored_knit = str(context.get("knit", "")).replace(" ", "").lower()
    requested_knit = str(knit).replace(" ", "").lower()
    if artifact.get("model_id") != "SV_exp_polycon_mo" or stored_knit != requested_knit:
        raise ValueError(f"Fit {fit_code} is not the requested {knit} SV_exp_polycon_mo fit")
    if artifact.get("result_role") != "training_optimum":
        raise ValueError(f"Fit {fit_code} is not a training_optimum")
    compiled_path = pathlib.Path(context["exp_data_path"])
    snapshot_path = pathlib.Path(artifact["input_files"]["experimental_data"]["snapshot_path"])
    parameter_path = pathlib.Path(artifact["input_files"]["material_parameters"]["snapshot_path"])
    umat_snapshot = pathlib.Path(artifact["input_files"]["umat_source"]["snapshot_path"])
    compiled = np.atleast_2d(np.loadtxt(compiled_path, dtype=float))
    snapshot = np.atleast_2d(np.loadtxt(snapshot_path, dtype=float))
    if not np.array_equal(compiled, snapshot):
        raise ValueError(f"{knit}: raw all-nine data differ from fit {fit_code}'s snapshot")
    curves = ratios.discover_ordered_ratio_curves(compiled_path)
    reconstructed = np.vstack([curve.data for curve in curves])
    if len(curves) != 9 or not np.array_equal(reconstructed, snapshot):
        raise ValueError(f"{knit}: nine raw curves do not reconstruct fit {fit_code}")
    axis_directions = sv_study.metadata_axis_directions(str(context["sample_folder"]))
    hashes = {
        "fit_artifact": sha256_file(artifact_path),
        "compiled_experimental_data": sha256_file(compiled_path),
        "experimental_snapshot": sha256_file(snapshot_path),
        "material_parameters": sha256_file(parameter_path),
        "umat_snapshot": sha256_file(umat_snapshot),
        **{f"ratio_{curve.label}": sha256_file(curve.path) for curve in curves},
    }
    declared = {
        "experimental_snapshot": artifact["input_files"]["experimental_data"]["sha256"],
        "material_parameters": artifact["input_files"]["material_parameters"]["sha256"],
        "umat_snapshot": artifact["input_files"]["umat_source"]["sha256"],
    }
    for name, expected in declared.items():
        if hashes[name] != expected:
            raise ValueError(f"Fit {fit_code} frozen {name} hash changed")
    return {
        "knit": knit,
        "fit_code": fit_code,
        "artifact": artifact,
        "artifact_path": artifact_path,
        "compiled_path": compiled_path,
        "snapshot_path": snapshot_path,
        "parameter_path": parameter_path,
        "umat_snapshot": umat_snapshot,
        "curves": curves,
        "axis_directions": axis_directions,
        "hashes": hashes,
    }


def configured_wrapper(inputs: Mapping[str, Any]) -> tuple[Any, dict[int, float]]:
    """Compile the production source and load one fixed-scale all-nine objective."""

    model_spec = load_model_spec("SV_exp_polycon_mo")
    umat.compile_umat_meson(pathlib.Path(model_spec.fortran_source))
    loaded = umat.Load_props(inputs["parameter_path"], model_spec=model_spec)
    wrapper = umat.Umat_wrapper(umat.Material_model(loaded), ["F", "Sigma"])
    specs = inputs["artifact"]["parameter_specs"]
    fit_runner.apply_parameter_specs_in_memory(wrapper, specs)
    wrapper.load_inputs(inputs["compiled_path"], ["H", "P"])
    references = {
        axis: float(np.max(np.abs(wrapper.expstressPK2[:, axis]))) for axis in FIT_AXES
    }
    wrapper.set_cost_function("normalized", 1.0, references)
    return wrapper, references


def configured_full_wrapper(
    inputs: Mapping[str, Any],
    *,
    compile_source: bool,
) -> tuple[Any, dict[int, float]]:
    """Load a full-study wrapper, compiling the registered source only once."""

    model_spec = load_model_spec("SV_exp_polycon_mo")
    if compile_source:
        umat.compile_umat_meson(pathlib.Path(model_spec.fortran_source))
    loaded = umat.Load_props(inputs["parameter_path"], model_spec=model_spec)
    wrapper = umat.Umat_wrapper(umat.Material_model(loaded), ["F", "Sigma"])
    fit_runner.apply_parameter_specs_in_memory(wrapper, inputs["artifact"]["parameter_specs"])
    wrapper.load_inputs(inputs["compiled_path"], ["H", "P"])
    references = {
        axis: float(np.max(np.abs(wrapper.expstressPK2[:, axis]))) for axis in FIT_AXES
    }
    wrapper.set_cost_function("normalized", 1.0, references)
    return wrapper, references


def candidate_grids(solver: SeparableMaterialSolver) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    """Construct the approved 5/5/3 continuous and four-value zeta pilot grid."""

    continuous: dict[str, np.ndarray] = {}
    for name in solver.definition.continuous_shape_parameters:
        points = PILOT_B_GRID_POINTS if name == "b" else PILOT_GRID_POINTS
        bounds = solver.parameter_bounds(name)
        continuous[name] = linear_grid(bounds, points) if name == "b" else logarithmic_grid(bounds, points)
    integers = {
        name: PILOT_ZETA_VALUES for name in solver.definition.integer_shape_parameters
    }
    return continuous, integers


def complete_parameters(
    solution: Any,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | int]:
    """Create a complete UMAT vector with every inactive amplitude fixed to zero."""

    values = {name: spec["initial_guess"] for name, spec in parameter_specs.items()}
    for name in FULL_LINEAR_ORDER:
        values[name] = 0.0
    values.update(solution.linear_values)
    values.update(solution.complete_shape_values())
    if "b" not in solution.complete_shape_values():
        values["a"] = 0.5
        values["b"] = 0.5
    return values


def curve_scores(residuals: np.ndarray, curves: Sequence[Any]) -> dict[str, float]:
    """Return mean squared normalized PK2 residual for each ratio block."""

    return {
        curve.label: float(np.mean(residuals[curve.start_row : curve.stop_row, :] ** 2))
        for curve in curves
    }


def solve_candidate(
    wrapper: Any,
    candidate: Candidate,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Map one candidate grid, validate its direct cost, and return checkpoint data."""

    # A reduced solver knows only its retained columns.  Zero every amplitude
    # first so an omitted family cannot leak its saved full-fit value into the
    # compiled UMAT baseline used to construct those columns.
    for name in FULL_LINEAR_ORDER:
        historical.set_parameter_value(wrapper, name, 0.0)
    definition = reduced_definition(candidate)
    solver: SeparableMaterialSolver
    if "K_coup" in candidate.linear_parameters:
        bound_solver = coupling.ShapeCouplingBoundSolver(wrapper, FIT_AXES, definition=definition)
        bound_solver.configure_coupling_bound("amplitude_equalized", 0.1)
        solver = bound_solver
    else:
        solver = SeparableMaterialSolver(wrapper, FIT_AXES, definition=definition)
    continuous, integers = candidate_grids(solver)
    start = time.perf_counter()
    grid = deterministic_shape_grid_search(
        solver,
        continuous,
        integers,
        reject_infeasible_cone_points=True,
    )
    elapsed = time.perf_counter() - start
    best = grid.best_solution
    parameters = complete_parameters(best, inputs["artifact"]["parameter_specs"])
    direct = historical.evaluate_full_parameter_set(wrapper, parameters, FIT_AXES)
    if not np.isclose(direct.raw_cost, best.cost, rtol=1.0e-9, atol=1.0e-11):
        raise ValueError(
            f"{candidate.candidate_id}: profiled/direct costs disagree: "
            f"{best.cost:.17g} versus {direct.raw_cost:.17g}; parameters={parameters}"
        )
    per_ratio = curve_scores(direct.normalized_residuals, inputs["curves"])
    singular = np.linalg.svd(
        solver.build_weighted_design_from_shapes(
            {name: best.complete_shape_values()[name] for name in definition.shape_parameters}
        ),
        compute_uv=False,
    )
    positive = singular[singular > np.finfo(float).eps * max(singular[0], 1.0)]
    return {
        "schema_version": 1,
        "study_stage": "low_resolution_pilot",
        "completed_at": utc_now(),
        **asdict(candidate),
        "parameters": parameters,
        "primary_cost": float(best.mean_cost),
        "equal_ratio_cost": float(np.mean(list(per_ratio.values()))),
        "per_ratio_costs": per_ratio,
        "raw_cost": float(best.cost),
        "observation_count": int(wrapper.expstressPK2.shape[0] * len(FIT_AXES)),
        "profiled_evaluations": int(solver.profiled_evaluation_count()),
        "elapsed_seconds": float(elapsed),
        "linear_solver_backends": solver.linear_backend_counts(),
        "minimum_constraint_margins": solver.minimum_constraint_margins(),
        "rejected_cone_evaluations": len(grid.rejected_evaluations),
        "active_bounds": best.active_bounds,
        "effective_linear_upper_bounds": best.effective_linear_upper_bounds,
        "design_rank": int(np.linalg.matrix_rank(
            solver.build_weighted_design_from_shapes(
                {name: best.complete_shape_values()[name] for name in definition.shape_parameters}
            )
        )),
        "design_condition_number": float(singular[0] / positive[-1]) if positive.size else float("inf"),
        "direct_cost_difference": float(abs(direct.raw_cost - best.cost)),
    }


def full_candidate_grids(
    solver: SeparableMaterialSolver,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    """Build the approved 17/17/9 grids and strict zeta=3..15 domain."""

    continuous: dict[str, np.ndarray] = {}
    for name in solver.definition.continuous_shape_parameters:
        points = FULL_B_GRID_POINTS if name == "b" else FULL_GRID_POINTS
        bounds = solver.parameter_bounds(name)
        continuous[name] = (
            linear_grid(bounds, points)
            if name == "b"
            else logarithmic_grid(bounds, points)
        )
    integers = {
        name: FULL_ZETA_VALUES for name in solver.definition.integer_shape_parameters
    }
    return continuous, integers


def solve_full_candidate(
    wrapper: Any,
    candidate: Candidate,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Grid-map and Powell-refine one all-nine production-resolution mask."""

    for name in FULL_LINEAR_ORDER:
        historical.set_parameter_value(wrapper, name, 0.0)
    definition = reduced_definition(candidate)
    if "K_coup" in candidate.linear_parameters:
        bounded = coupling.ShapeCouplingBoundSolver(wrapper, FIT_AXES, definition=definition)
        bounded.configure_coupling_bound("amplitude_equalized", 0.1)
        solver: SeparableMaterialSolver = bounded
    else:
        solver = SeparableMaterialSolver(wrapper, FIT_AXES, definition=definition)
    continuous, integers = full_candidate_grids(solver)
    started = time.perf_counter()
    grid = deterministic_shape_grid_search(
        solver,
        continuous,
        integers,
        reject_infeasible_cone_points=True,
    )
    refined = tuple(
        refine_profiled_shapes(
            solver,
            initial,
            maximum_iterations=FULL_REFINEMENT_ITERATIONS,
            parameter_tolerance=FULL_PARAMETER_TOLERANCE,
            cost_tolerance=FULL_COST_TOLERANCE,
        )
        for initial in grid.best_by_integer_combination
    )
    best = min(refined, key=lambda solution: (solution.cost, solution.zeta))
    elapsed = time.perf_counter() - started
    parameters = complete_parameters(best, inputs["artifact"]["parameter_specs"])
    if int(parameters["zeta"]) <= 2 and "K_coup" in candidate.linear_parameters:
        raise ValueError(f"{candidate.candidate_id}: production zeta must be >2")
    direct = historical.evaluate_full_parameter_set(wrapper, parameters, FIT_AXES)
    if not np.isclose(direct.raw_cost, best.cost, rtol=1.0e-9, atol=1.0e-11):
        raise ValueError(
            f"{inputs['knit']} {candidate.candidate_id}: profiled/direct costs disagree: "
            f"{best.cost:.17g} versus {direct.raw_cost:.17g}"
        )
    per_ratio = curve_scores(direct.normalized_residuals, inputs["curves"])
    selected_shapes = {
        name: best.complete_shape_values()[name] for name in definition.shape_parameters
    }
    design = solver.build_weighted_design_from_shapes(selected_shapes)
    singular = np.linalg.svd(design, compute_uv=False)
    positive = singular[singular > np.finfo(float).eps * max(singular[0], 1.0)]
    return {
        "schema_version": 1,
        "study_stage": "full_resolution_all_nine",
        "knit": inputs["knit"],
        "fit_code_reference": inputs["fit_code"],
        "completed_at": utc_now(),
        **asdict(candidate),
        "parameters": parameters,
        "primary_cost": float(best.mean_cost),
        "equal_ratio_cost": float(np.mean(list(per_ratio.values()))),
        "per_ratio_costs": per_ratio,
        "raw_cost": float(best.cost),
        "observation_count": int(wrapper.expstressPK2.shape[0] * len(FIT_AXES)),
        "profiled_evaluations": int(solver.profiled_evaluation_count()),
        "elapsed_seconds": float(elapsed),
        "linear_solver_backends": solver.linear_backend_counts(),
        "minimum_constraint_margins": solver.minimum_constraint_margins(),
        "rejected_cone_evaluations": len(grid.rejected_evaluations),
        "active_bounds": best.active_bounds,
        "effective_linear_upper_bounds": best.effective_linear_upper_bounds,
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_singular_values": singular.tolist(),
        "design_condition_number": float(singular[0] / positive[-1]) if positive.size else float("inf"),
        "direct_cost_difference": float(abs(direct.raw_cost - best.cost)),
        "predicted_pk2_axes": np.asarray(direct.predicted_pk2[:, FIT_AXES], dtype=float),
        "experimental_pk2_axes": np.asarray(wrapper.expstressPK2[:, FIT_AXES], dtype=float),
    }


def summarize_knit(
    inputs: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    references: Mapping[int, float],
) -> dict[str, Any]:
    """Apply both 2%-adequate rules and calibration-only classifications."""

    primary_winner = min(rows, key=lambda row: (float(row["primary_cost"]), int(row["mask"])))
    equal_winner = min(rows, key=lambda row: (float(row["equal_ratio_cost"]), int(row["mask"])))
    selected = select_reduced(rows, "primary_cost")
    full = next(row for row in rows if int(row["mask"]) == 63)
    primary_status = classify(rows, "primary_cost")
    equal_status = classify(rows, "equal_ratio_cost")
    components = []
    for component in COMPONENT_ORDER:
        best_without = min(float(row["primary_cost"]) for row in rows if component not in row["components"])
        components.append({
            "knit": inputs["knit"],
            "component": component,
            "primary_classification": primary_status[component],
            "equal_ratio_classification": equal_status[component],
            "weighting_sensitive": primary_status[component] != equal_status[component],
            "best_without_cost": best_without,
            "best_without_penalty_percent": 100.0 * (best_without / float(primary_winner["primary_cost"]) - 1.0),
        })
    return {
        "knit": inputs["knit"],
        "fit_code_reference": inputs["fit_code"],
        "axis_directions": inputs["axis_directions"],
        "physical_component_parameter_map": physical_parameter_map(inputs["axis_directions"]),
        "normalization_reference_maxima": references,
        "primary_winner": primary_winner,
        "equal_ratio_winner": equal_winner,
        "selected_lowest_complexity_adequate": selected,
        "full_model": full,
        "primary_adequate_candidate_ids": [row["candidate_id"] for row in adequate_rows(rows, "primary_cost")],
        "equal_ratio_adequate_candidate_ids": [row["candidate_id"] for row in adequate_rows(rows, "equal_ratio_cost")],
        "component_classifications": components,
    }


def run_full_calibrations() -> pathlib.Path:
    """Run or resume all 156 production-resolution knit/mask calibrations."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FULL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    all_hashes: dict[str, Any] = {}
    compiled = False
    for knit_index, knit in enumerate(FULL_FIT_CODES, start=1):
        inputs = load_full_inputs(knit)
        all_hashes[knit] = inputs["hashes"]
        candidates = enumerate_admissible_subsets(inputs["axis_directions"])
        wrapper, references = configured_full_wrapper(inputs, compile_source=not compiled)
        compiled = True
        knit_token = knit.lower().replace(" ", "_")
        checkpoint_dir = FULL_CHECKPOINT_DIR / knit_token
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            checkpoint = checkpoint_dir / f"{candidate.candidate_id}.json"
            if RESUME_CHECKPOINTS and not FORCE_RECOMPUTE and checkpoint.is_file():
                row = json.loads(checkpoint.read_text(encoding="utf-8"))
                if row.get("study_stage") != "full_resolution_all_nine":
                    raise ValueError(f"Wrong checkpoint stage: {checkpoint}")
                if "K_coup" in row["linear_parameters"] and int(row["parameters"]["zeta"]) <= 2:
                    raise ValueError(f"Forbidden zeta checkpoint: {checkpoint}")
            else:
                print(
                    f"[{knit_index}/4 {knit}] [{candidate_index:02d}/39] "
                    f"{candidate.candidate_id} {candidate.components}",
                    flush=True,
                )
                row = solve_full_candidate(wrapper, candidate, inputs)
                atomic_json(checkpoint, row)
            rows.append(row)
        summary = summarize_knit(inputs, rows, references)
        atomic_json(OUTPUT_ROOT / f"summary_{knit_token}.json", summary)
        summaries.append(summary)
    study_definition = {
        "schema_version": 1,
        "study": "SV_exp_polycon_mo all-nine component necessity",
        "stage": "full_resolution_all_nine",
        "classification_scope": "all-nine calibration support only",
        "fit_codes": FULL_FIT_CODES,
        "candidate_count_per_knit": 39,
        "total_candidate_count": 156,
        "grid": {
            "beta_1_points": FULL_GRID_POINTS,
            "gamma_1_points": FULL_GRID_POINTS,
            "b_points": FULL_B_GRID_POINTS,
            "zeta_values": FULL_ZETA_VALUES,
            "powell_refinement": True,
            "refinement_max_iterations": FULL_REFINEMENT_ITERATIONS,
        },
        "coupling": {
            "linear_constraint": "4*mu*mu2-mu3^2>=0",
            "nonlinear_form": "K_coup*<a*(I4-C1)+b*(I6-C2)>_+^zeta",
            "amplitude_equalized_ceiling": 0.1,
            "a": "1-b",
        },
        "adequate_relative_tolerance": ADEQUATE_RELATIVE_TOLERANCE,
        "input_hashes": all_hashes,
    }
    atomic_json(OUTPUT_ROOT / "study_definition.json", study_definition)
    final = {
        "schema_version": 1,
        "study": study_definition["study"],
        "stage": study_definition["stage"],
        "completed_at": utc_now(),
        "classification_scope": study_definition["classification_scope"],
        "knits": summaries,
    }
    atomic_json(OUTPUT_ROOT / "calibration_summary.json", final)
    return OUTPUT_ROOT / "calibration_summary.json"


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Write one deterministic UTF-8 CSV table."""

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def run_pilot() -> pathlib.Path:
    """Run or resume all 39 Interlock masks and produce provisional diagnostics."""

    if RUN_MODE != "pilot":
        raise RuntimeError("Full resolution is gated pending explicit approval; set RUN_MODE='pilot'")
    PILOT_OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_pilot_inputs()
    candidates = enumerate_admissible_subsets(inputs["axis_directions"])
    definition_payload = {
        "schema_version": 1,
        "stage": "low_resolution_pilot",
        "classification_scope": "all-nine calibration support; provisional",
        "fit_code": PILOT_FIT_CODE,
        "axis_directions": inputs["axis_directions"],
        "component_parameter_map": physical_parameter_map(inputs["axis_directions"]),
        "candidate_count": len(candidates),
        "grid": {
            "beta_1_points": PILOT_GRID_POINTS,
            "gamma_1_points": PILOT_GRID_POINTS,
            "b_points": PILOT_B_GRID_POINTS,
            "zeta_values": PILOT_ZETA_VALUES,
            "powell_refinement": False,
        },
        "adequate_relative_tolerance": ADEQUATE_RELATIVE_TOLERANCE,
        "input_hashes": inputs["hashes"],
    }
    atomic_json(PILOT_OUTPUT / "study_definition.json", definition_payload)
    wrapper, references = configured_wrapper(inputs)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        checkpoint = CHECKPOINT_DIR / f"{candidate.candidate_id}.json"
        if RESUME_CHECKPOINTS and not FORCE_RECOMPUTE and checkpoint.is_file():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            print(f"[{index:02d}/39] {candidate.candidate_id} {candidate.components}", flush=True)
            row = solve_candidate(wrapper, candidate, inputs)
            atomic_json(checkpoint, row)
        rows.append(row)
    primary_adequate = adequate_rows(rows, "primary_cost")
    equal_adequate = adequate_rows(rows, "equal_ratio_cost")
    primary_status = classify(rows, "primary_cost")
    equal_status = classify(rows, "equal_ratio_cost")
    selected = select_reduced(rows, "primary_cost")
    primary_winner = min(rows, key=lambda row: (float(row["primary_cost"]), int(row["mask"])))
    full = next(row for row in rows if int(row["mask"]) == 63)
    component_rows = []
    for component in COMPONENT_ORDER:
        without = min(float(row["primary_cost"]) for row in rows if component not in row["components"])
        component_rows.append({
            "component": component,
            "primary_classification": primary_status[component],
            "equal_ratio_classification": equal_status[component],
            "weighting_sensitive": primary_status[component] != equal_status[component],
            "best_without_cost": without,
            "best_without_penalty_percent": 100.0 * (without / float(primary_winner["primary_cost"]) - 1.0),
        })
    candidate_fields = (
        "candidate_id", "mask", "components", "linear_parameters",
        "effective_parameter_count", "primary_cost", "equal_ratio_cost", "raw_cost",
        "profiled_evaluations", "elapsed_seconds", "design_rank",
        "design_condition_number", "rejected_cone_evaluations", "direct_cost_difference",
    )
    csv_rows = [
        {**row, "components": ";".join(row["components"]), "linear_parameters": ";".join(row["linear_parameters"])}
        for row in sorted(rows, key=lambda item: int(item["mask"]))
    ]
    write_csv(PILOT_OUTPUT / "candidate_scores.csv", csv_rows, candidate_fields)
    write_csv(
        PILOT_OUTPUT / "component_necessity.csv",
        component_rows,
        ("component", "primary_classification", "equal_ratio_classification", "weighting_sensitive", "best_without_cost", "best_without_penalty_percent"),
    )
    total_seconds = float(sum(float(row["elapsed_seconds"]) for row in rows))
    pilot_profiles = sum(int(row["profiled_evaluations"]) for row in rows)
    production_profiles_per_knit = sum(
        (17 if "beta_1" in candidate.continuous_shapes else 1)
        * (17 if "gamma_1" in candidate.continuous_shapes else 1)
        * (9 if "b" in candidate.continuous_shapes else 1)
        * (14 if "zeta" in candidate.integer_shapes else 1)
        for candidate in candidates
    )
    four_knit_grid_scale = 4.0 * production_profiles_per_knit / max(pilot_profiles, 1)
    summary = {
        "schema_version": 1,
        "study": "SV_exp_polycon_mo all-nine component necessity",
        "stage": "low_resolution_pilot",
        "classification_scope": "provisional all-nine calibration support",
        "completed_at": utc_now(),
        "input_validation": "passed",
        "candidate_count": len(rows),
        "normalization_reference_maxima": references,
        "primary_winner": primary_winner,
        "selected_lowest_complexity_adequate": selected,
        "full_model": full,
        "adequate_candidate_ids": [row["candidate_id"] for row in primary_adequate],
        "equal_ratio_adequate_candidate_ids": [row["candidate_id"] for row in equal_adequate],
        "component_classifications": component_rows,
        "timing": {
            "candidate_search_seconds": total_seconds,
            "profiled_evaluations": pilot_profiles,
            "seconds_per_profile": total_seconds / max(pilot_profiles, 1),
            "production_grid_profiles_per_knit": production_profiles_per_knit,
            "four_knit_grid_only_scale_factor": four_knit_grid_scale,
            "estimated_four_knit_full_grid_seconds": total_seconds * four_knit_grid_scale,
            "note": "Grid-only estimate excludes Powell refinement, stability, workbook, figures, and report rendering.",
        },
        "compiled_umat_validation": {
            "scope": [primary_winner["candidate_id"], full["candidate_id"]],
            "status": "passed",
            "maximum_raw_cost_difference": max(float(primary_winner["direct_cost_difference"]), float(full["direct_cost_difference"])),
        },
    }
    atomic_json(PILOT_OUTPUT / "pilot_summary.json", summary)
    return PILOT_OUTPUT / "pilot_summary.json"


def main() -> None:
    """Execute the configured isolated study stage and print its summary path."""

    if RUN_MODE == "pilot":
        path = run_pilot()
        print(f"Pilot complete: {path}", flush=True)
    elif RUN_MODE == "full":
        path = run_full_calibrations()
        print(f"Full calibrations complete: {path}", flush=True)
    else:
        raise ValueError("RUN_MODE must be 'pilot' or 'full'")


if __name__ == "__main__":
    main()
