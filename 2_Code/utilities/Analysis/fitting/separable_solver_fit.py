"""Fit one explicitly configured separable material case and export artifacts.

Select ``CASE_FILE`` and press Run in VSCode. The case JSON points directly to
the material-parameter ``.dat``, experimental ``.dat``, and UMAT source. No
previous fit code or result artifact is used as an optimization input.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (INPUT_ROOT as RELEASE_INPUT_ROOT,
    ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT,
    GENERATED_ROOT as RELEASE_GENERATED_ROOT)

import argparse
import copy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import itertools
from release_paths import json
import math
import os
import pathlib
import re
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

# Keep NumPy/SciPy and the optional Windows SCIP wheel on one OpenMP runtime.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np


ANALYSIS_DIR = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis")
SUPPORT_CODE_DIR = ANALYSIS_DIR.parent
CODE_ROOT = SUPPORT_CODE_DIR.parent
RELEASE_PROJECT_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(SUPPORT_CODE_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from Analysis.studies.solver_diagnostics import separable_solver_coupling_bound_calibration as calibration  # noqa: E402
from Analysis.studies.solver_diagnostics import separable_solver_historical_benchmark as historical  # noqa: E402
from Analysis.core.fit_results import (  # noqa: E402
    DEFAULT_RESULTS_CSV,
    DEFAULT_SEPARABLE_RESULTS_CSV,
    append_separable_fit_results,
    artifact_directory,
    export_fit_artifacts,
    initialize_separable_results_table,
    reserve_fit_codes,
    update_fit_registration,
)
from Python_Wrapper import umat  # noqa: E402
from Python_Wrapper.model_registry import ModelSpec, load_model_spec  # noqa: E402
from Python_Wrapper.separable_solver import (  # noqa: E402
    ProfiledSolution,
    SeparableMaterialSolver,
    SeparableModelDefinition,
    apply_parameter_specs_in_memory,
    deterministic_grid_search,
    deterministic_shape_grid_search,
    linear_grid,
    load_separable_definition,
    logarithmic_grid,
    refine_profiled_shapes,
    refine_profiled_solution,
)
from Python_Wrapper.separable_solver_scip import (  # noqa: E402
    CertifiedDiscreteSeparableSolver,
    CertifiedDiscreteSolution,
)


# ---------------------------------------------------------------------------
# USER CONFIGURATION - select one direct-input case, then press Run in VSCode.
# ---------------------------------------------------------------------------

CASE_DIRECTORY = RELEASE_INPUT_ROOT / "Separable_Solver" / "Cases"
# Example: sv_exp_polycon_mo_rib_s03_raw_all.json.
CASE_FILE = CASE_DIRECTORY / "three_ratio_raw_c063_sv_exp_polycon_mo_interlock_s06.json"

# Shared output locations; tests may temporarily redirect these paths.
LEGACY_RESULTS_CSV = RELEASE_GENERATED_ROOT / "results/calibration_refitting/results_final.csv"
SEPARABLE_RESULTS_CSV = RELEASE_GENERATED_ROOT / "results/calibration_refitting/separable_solver_results.csv"
RUN_SUMMARY_ROOT = RELEASE_GENERATED_ROOT / "results/calibration_refitting/summaries"

SUPPORTED_SCHEMA_VERSION = 1
OPTIMIZER_NAME = "separable_deterministic"
FIXED_ARCHITECTURE = {"C1": 1.0, "C2": 1.0, "theta1": 0.0, "theta2": 90.0}
# Production SV_exp_polycon cases use nine b points unless a manifest overrides it.
DEFAULT_POLYCON_B_GRID_POINTS = 9

@dataclass(frozen=True)
class OptimizationConfiguration:
    """Numerical controls, for example a 17 by 17 shape grid."""

    grid_points_per_axis: int
    refine_each_zeta: bool
    refinement_max_iterations: int
    parameter_tolerance: float
    cost_tolerance: float
    # Optional per-shape counts; for example b=9 while beta_1=gamma_1=17.
    grid_points_by_parameter: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CertifiedStudyConfiguration:
    """Controls for a certified discrete study, for example Rib Stapleton."""

    amplitude_ladder: tuple[float, ...]
    amplitude_extensions: tuple[float, ...]
    relative_cost_tolerance: float
    realized_margin_factor: float
    relative_gap: float
    total_time_limit_seconds: float
    inactive_linear_parameters: tuple[str, ...]
    certification_backend: str
    # Production cases solve one reviewed ceiling, for example 0.2. Diagnostic
    # studies omit this and continue to use their adaptive amplitude ladder.
    production_amplitude_ceiling: float | None


@dataclass(frozen=True)
class CaseConfiguration:
    """One complete, direct-input material-fitting problem."""

    path: pathlib.Path
    payload: dict[str, Any]
    case_id: str
    run_label: str
    workflow: str
    model_id: str
    separable_definition: str
    umat_source: pathlib.Path
    parameter_file: pathlib.Path
    experimental_data: pathlib.Path
    yarn_type: str
    knit: str
    knit_token: str
    sample_folder: str
    data_type: str
    experimental_types: tuple[str, str]
    umat_types: tuple[str, str]
    cost_type: str
    normalized_stress_floor_factor: float
    fit_axes: tuple[int, ...]
    bounds_source: str
    parameter_overrides: dict[str, dict[str, Any]]
    coupling_bound_mode: str
    amplitude_ceiling: float | None
    zeta_selection: str | tuple[int, ...]
    predictive_structure: str | None
    selection_basis: str
    optimization: OptimizationConfiguration
    certified_study: CertifiedStudyConfiguration | None
    result_roles: tuple[str, ...]
    snapshot_experimental_data: bool


@dataclass(frozen=True)
class ExportCandidate:
    """One independently validated solution that will receive a fit code."""

    roles: tuple[str, ...]
    model_labels: tuple[str, ...]
    solution: ProfiledSolution
    parameters: dict[str, float | int]
    parameter_specs: dict[str, dict[str, Any]]
    direct_raw_cost: float
    direct_mean_cost: float
    maximum_prediction_difference: float
    fitted_observation_count: int
    profiled_evaluations: int
    elapsed_seconds: float
    # Keys are integer planes for legacy models and real xi values for the
    # continuous-exponent model (for example, 7 or 7.3125).
    coupling_upper_bounds: dict[int | float, float]
    basis_weighted_rms: dict[int | float, float]
    validation: dict[str, Any]

    @property
    def result_role(self) -> str:
        """Return roles such as ``training_optimum+predictive_model``."""
        return "+".join(self.roles)

    @property
    def model_label(self) -> str:
        """Return structures such as ``free_zeta+fixed_zeta_3``."""
        return "+".join(self.model_labels)


def _mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return one required JSON object, for example the ``objective`` block."""
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Case configuration requires an object named {key!r}")
    return dict(value)


def _string(parent: Mapping[str, Any], key: str) -> str:
    """Return one nonempty string, for example model_id ``SV_exp``."""
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Case configuration requires a nonempty string {key!r}")
    return value.strip()


def _resolve_project_path(value: Any, label: str) -> pathlib.Path:
    """Resolve absolute or project-relative input paths and require a file."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path string")
    path = pathlib.Path(value.strip())
    resolved = path if path.is_absolute() else RELEASE_PROJECT_ROOT / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} was not found: {resolved}")
    return resolved


def _measure_pair(parent: Mapping[str, Any], key: str) -> tuple[str, str]:
    """Return a two-item measure pair, for example ``['H', 'P']``."""
    value = parent.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{key} must contain exactly two strings")
    pair = tuple(str(item).strip() for item in value)
    if not all(pair):
        raise ValueError(f"{key} cannot contain an empty measure")
    return pair[0], pair[1]


def load_case_configuration(path: pathlib.Path = CASE_FILE) -> CaseConfiguration:
    """Load one direct-input case JSON and validate every declared setting."""
    path = pathlib.Path(path).resolve()
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Case configuration must contain one JSON object: {path}")
    if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Case schema_version must equal {SUPPORTED_SCHEMA_VERSION}"
        )

    case_id = _string(payload, "case_id")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", case_id) is None:
        raise ValueError("case_id may contain only letters, numbers, dots, dashes, and underscores")
    run_label = str(payload.get("run_label", case_id)).strip()
    if not run_label:
        raise ValueError("run_label cannot be empty")

    model = _mapping(payload, "model")
    files = _mapping(payload, "files")
    experiment = _mapping(payload, "experiment")
    umat_settings = _mapping(payload, "umat")
    objective = _mapping(payload, "objective")
    parameter_policy = _mapping(payload, "parameter_policy")
    outputs = _mapping(payload, "outputs")

    model_id = _string(model, "model_id")
    definition = _string(model, "separable_definition")
    # The registered pair prevents, for example, using tanh data with an exp basis label.
    definition_spec = load_separable_definition(definition, model_id)
    workflow = str(payload.get("workflow", "production")).strip().lower()
    if workflow not in {"production", "checkpoint", "study_only"}:
        raise ValueError("workflow must be production, checkpoint, or study_only")
    if workflow == "study_only" and definition_spec.backend != "scip_sos1":
        raise ValueError("study_only is currently reserved for certified discrete cases")
    umat_source = _resolve_project_path(model.get("umat_source"), "UMAT source")
    parameter_file = _resolve_project_path(
        files.get("material_parameters"), "Material-parameter file"
    )
    experimental_data = _resolve_project_path(
        files.get("experimental_data"), "Experimental-data file"
    )

    model_spec = load_model_spec(model_id)
    if umat_source != pathlib.Path(model_spec.fortran_source).resolve():
        raise ValueError(
            "The manifest UMAT source disagrees with the registered model source: "
            f"{umat_source} != {model_spec.fortran_source}"
        )

    data_type = _string(experiment, "data_type").lower()
    if data_type not in {"raw"}:
        raise ValueError("experiment.data_type must be raw")
    experimental_types = _measure_pair(experiment, "experimental_types")
    umat_types = _measure_pair(umat_settings, "types")
    cost_type = _string(objective, "cost_type").lower()
    if cost_type not in {"legacy", "normalized"}:
        raise ValueError("objective.cost_type must be legacy or normalized")
    floor_factor = float(objective.get("normalized_stress_floor_factor", 1.0))
    if not math.isfinite(floor_factor) or floor_factor < 0:
        raise ValueError("normalized_stress_floor_factor must be finite and nonnegative")
    raw_axes = objective.get("fit_axes")
    if not isinstance(raw_axes, list) or not raw_axes:
        raise ValueError("objective.fit_axes must be a nonempty integer list")
    fit_axes = tuple(int(axis) for axis in raw_axes)
    if len(set(fit_axes)) != len(fit_axes) or any(axis < 0 or axis > 5 for axis in fit_axes):
        raise ValueError("objective.fit_axes must contain unique Voigt indices 0 through 5")

    bounds_source = _string(parameter_policy, "bounds_source")
    if bounds_source != "parameter_file":
        raise ValueError("parameter_policy.bounds_source must be 'parameter_file'")
    raw_overrides = parameter_policy.get("parameter_overrides", {})
    if not isinstance(raw_overrides, dict):
        raise ValueError("parameter_overrides must be an object keyed by parameter")
    parameter_overrides = {
        str(name): dict(values)
        for name, values in raw_overrides.items()
        if isinstance(values, dict)
    }
    if len(parameter_overrides) != len(raw_overrides):
        raise ValueError("Every parameter override must be a JSON object")

    certified_study: CertifiedStudyConfiguration | None = None
    if definition_spec.backend == "grid_powell":
        coupling = _mapping(payload, "coupling")
        optimization = _mapping(payload, "optimization")
        coupling_mode = _string(coupling, "bound_mode")
        if coupling_mode not in {"amplitude_equalized", "parameter_file"}:
            raise ValueError(
                "coupling.bound_mode must be amplitude_equalized or parameter_file"
            )
        raw_amplitude = coupling.get("amplitude_ceiling")
        amplitude_ceiling = None if raw_amplitude is None else float(raw_amplitude)
        if coupling_mode == "amplitude_equalized" and (
            amplitude_ceiling is None
            or not math.isfinite(amplitude_ceiling)
            or amplitude_ceiling <= 0
        ):
            raise ValueError("Amplitude-equalized coupling requires a positive ceiling")
        exponent_name = definition_spec.coupling_exponent_parameter
        exponent_values_key = f"{exponent_name}_values"
        raw_zeta = coupling.get(exponent_values_key, "parameter_file")
        if raw_zeta == "parameter_file":
            zeta_selection: str | tuple[int, ...] = "parameter_file"
        elif isinstance(raw_zeta, list) and raw_zeta:
            zeta_values = tuple(int(value) for value in raw_zeta)
            if len(set(zeta_values)) != len(zeta_values):
                raise ValueError(
                    f"coupling.{exponent_values_key} cannot contain duplicates"
                )
            zeta_selection = zeta_values
        else:
            raise ValueError(
                f"coupling.{exponent_values_key} must be 'parameter_file' or a list"
            )
        raw_predictive_structure = coupling.get("predictive_structure")
        predictive_structure = (
            None
            if raw_predictive_structure is None
            else str(raw_predictive_structure).strip()
        )
        if predictive_structure == "":
            raise ValueError("coupling.predictive_structure cannot be empty")
        selection_basis = str(coupling.get("selection_basis", "")).strip()

        raw_grid_points = optimization.get("grid_points_by_parameter", {})
        if not isinstance(raw_grid_points, dict):
            raise ValueError("optimization.grid_points_by_parameter must be an object")
        grid_points_by_parameter = {
            str(name): int(points) for name, points in raw_grid_points.items()
        }
        unknown_grid_shapes = sorted(
            set(grid_points_by_parameter)
            - set(definition_spec.continuous_shape_parameters)
        )
        if unknown_grid_shapes:
            raise ValueError(
                "Grid-point overrides reference unknown continuous shapes: "
                + ", ".join(unknown_grid_shapes)
            )
        opt = OptimizationConfiguration(
            grid_points_per_axis=int(optimization.get("grid_points_per_axis", 17)),
            refine_each_zeta=bool(optimization.get("refine_each_zeta", True)),
            refinement_max_iterations=int(
                optimization.get("refinement_max_iterations", 100)
            ),
            parameter_tolerance=float(
                optimization.get("parameter_tolerance", 1.0e-6)
            ),
            cost_tolerance=float(optimization.get("cost_tolerance", 1.0e-10)),
            grid_points_by_parameter=grid_points_by_parameter,
        )
        if (
            opt.grid_points_per_axis < 2
            or any(points < 2 for points in opt.grid_points_by_parameter.values())
            or opt.refinement_max_iterations <= 0
        ):
            raise ValueError(
                "Grid points must be at least 2 and refinement iterations positive"
            )
        if opt.parameter_tolerance <= 0 or opt.cost_tolerance <= 0:
            raise ValueError("Refinement tolerances must be positive")
    else:
        study = _mapping(payload, "certified_study")
        raw_ladder = study.get("amplitude_ladder")
        raw_extensions = study.get("amplitude_extensions")
        if not isinstance(raw_ladder, list) or len(raw_ladder) < 2:
            raise ValueError("certified_study.amplitude_ladder requires at least two values")
        if not isinstance(raw_extensions, list):
            raise ValueError("certified_study.amplitude_extensions must be a list")
        raw_inactive = study.get("inactive_linear_parameters", [])
        if not isinstance(raw_inactive, list) or any(
            not isinstance(name, str) or not name.strip() for name in raw_inactive
        ):
            raise ValueError(
                "certified_study.inactive_linear_parameters must be a string list"
            )
        ladder = tuple(float(value) for value in raw_ladder)
        extensions = tuple(float(value) for value in raw_extensions)
        combined = ladder + extensions
        if any(not math.isfinite(value) or value <= 0 for value in combined):
            raise ValueError("Certified amplitude ceilings must be finite and positive")
        if any(second <= first for first, second in zip(combined, combined[1:])):
            raise ValueError("Certified amplitude ceilings must increase strictly")
        raw_production_ceiling = study.get("production_amplitude_ceiling")
        production_ceiling = (
            None
            if raw_production_ceiling is None
            else float(raw_production_ceiling)
        )
        if production_ceiling is not None and (
            not math.isfinite(production_ceiling) or production_ceiling <= 0
        ):
            raise ValueError(
                "certified_study.production_amplitude_ceiling must be positive"
            )
        if workflow == "production" and production_ceiling is None:
            raise ValueError(
                "Certified production cases require production_amplitude_ceiling"
            )
        if production_ceiling is not None and not any(
            np.isclose(production_ceiling, candidate, rtol=0.0, atol=1.0e-14)
            for candidate in combined
        ):
            raise ValueError(
                "production_amplitude_ceiling must be one of the recorded "
                "amplitude ladder or extension values"
            )
        certified_study = CertifiedStudyConfiguration(
            amplitude_ladder=ladder,
            amplitude_extensions=extensions,
            relative_cost_tolerance=float(
                study.get("relative_cost_tolerance", 0.001)
            ),
            realized_margin_factor=float(study.get("realized_margin_factor", 2.0)),
            relative_gap=float(study.get("relative_gap", 1.0e-6)),
            total_time_limit_seconds=float(
                study.get("total_time_limit_seconds", 14_400.0)
            ),
            inactive_linear_parameters=tuple(
                name.strip() for name in raw_inactive
            ),
            certification_backend=str(
                study.get("certification_backend", "scip_sos1")
            ).strip().lower(),
            production_amplitude_ceiling=production_ceiling,
        )
        if (
            certified_study.relative_cost_tolerance < 0
            or certified_study.realized_margin_factor <= 1
            or certified_study.relative_gap <= 0
            or certified_study.total_time_limit_seconds <= 0
        ):
            raise ValueError("Certified study tolerances and limits are invalid")
        inactive = certified_study.inactive_linear_parameters
        if len(set(inactive)) != len(inactive):
            raise ValueError("Certified inactive linear parameters cannot repeat")
        dependencies = definition_spec.basis_dependencies()
        invalid_inactive = [
            name for name in inactive if name not in dependencies or not dependencies[name]
        ]
        if invalid_inactive:
            raise ValueError(
                "Certified inactive parameters must be nonlinear basis families: "
                + ", ".join(invalid_inactive)
            )
        if certified_study.certification_backend not in {
            "scip_sos1",
            "exhaustive_enumeration",
        }:
            raise ValueError(
                "certified_study.certification_backend must be scip_sos1 "
                "or exhaustive_enumeration"
            )
        remaining_families = [
            name
            for name, controlling in dependencies.items()
            if controlling and name not in inactive
        ]
        if (
            certified_study.certification_backend == "exhaustive_enumeration"
            and len(remaining_families) > 3
        ):
            raise ValueError(
                "Exhaustive certification is restricted to at most three active families"
            )
        coupling_mode = "common_amplitude"
        amplitude_ceiling = None
        zeta_selection = "parameter_file"
        predictive_structure = None
        selection_basis = str(study.get("selection_basis", "")).strip()
        # Retain a concrete compatibility object even though SCIP does not use it.
        opt = OptimizationConfiguration(17, False, 100, 1.0e-6, 1.0e-10)

    raw_roles = outputs.get("result_roles", [])
    if not isinstance(raw_roles, list) or (workflow == "production" and not raw_roles):
        raise ValueError("Production outputs.result_roles must be a nonempty list")
    result_roles = tuple(str(role) for role in raw_roles)
    allowed_roles = {"training_optimum", "predictive_model"}
    if len(set(result_roles)) != len(result_roles) or not set(result_roles) <= allowed_roles:
        raise ValueError("Result roles may contain training_optimum and predictive_model once")
    if (
        definition_spec.backend == "grid_powell"
        and "predictive_model" in result_roles
        and predictive_structure is None
    ):
        raise ValueError(
            "coupling.predictive_structure is required when predictive_model is requested"
        )
    if (
        workflow == "production"
        and definition_spec.backend == "scip_sos1"
        and result_roles != ("training_optimum",)
    ):
        raise ValueError(
            "Certified production cases currently export only training_optimum; "
            "a predictive_model requires a separate held-out structure study"
        )

    return CaseConfiguration(
        path=path,
        payload=dict(payload),
        case_id=case_id,
        run_label=run_label,
        workflow=workflow,
        model_id=model_id,
        separable_definition=definition,
        umat_source=umat_source,
        parameter_file=parameter_file,
        experimental_data=experimental_data,
        yarn_type=_string(experiment, "yarn_type"),
        knit=_string(experiment, "knit"),
        knit_token=_string(experiment, "knit_token"),
        sample_folder=_string(experiment, "sample_folder"),
        data_type=data_type,
        experimental_types=experimental_types,
        umat_types=umat_types,
        cost_type=cost_type,
        normalized_stress_floor_factor=floor_factor,
        fit_axes=fit_axes,
        bounds_source=bounds_source,
        parameter_overrides=parameter_overrides,
        coupling_bound_mode=coupling_mode,
        amplitude_ceiling=amplitude_ceiling,
        zeta_selection=zeta_selection,
        predictive_structure=predictive_structure,
        selection_basis=selection_basis,
        optimization=opt,
        certified_study=certified_study,
        result_roles=result_roles,
        snapshot_experimental_data=bool(
            outputs.get("snapshot_experimental_data", True)
        ),
    )


def parameter_specs_from_file(
    config: CaseConfiguration,
    model_spec: ModelSpec,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load raw `.dat` specifications and apply only explicit JSON overrides."""
    loaded = umat.Load_props(config.parameter_file, model_spec=model_spec)
    records = loaded.get_start_parameters()
    raw_specs = {
        name: {
            "type": "int" if record[1] is int else "float",
            "initial_guess": record[0],
            "minimum": record[2],
            "maximum": record[3],
            "fixed": bool(record[5]),
        }
        for name, record in records.items()
    }
    effective = copy.deepcopy(raw_specs)
    definition = load_separable_definition(
        config.separable_definition,
        config.model_id,
    )
    unknown = sorted(set(config.parameter_overrides) - set(effective))
    if unknown:
        raise ValueError("Overrides reference unknown parameters: " + ", ".join(unknown))
    allowed_fields = {"value", "initial_guess", "minimum", "maximum", "fixed"}
    for name, overrides in config.parameter_overrides.items():
        invalid = sorted(set(overrides) - allowed_fields)
        if invalid:
            raise ValueError(f"Unsupported override fields for {name}: {invalid}")
        specification = effective[name]
        if "value" in overrides and "initial_guess" in overrides:
            raise ValueError(f"Use value or initial_guess for {name}, not both")
        if "value" in overrides:
            specification["initial_guess"] = overrides["value"]
        for field in ("initial_guess", "minimum", "maximum", "fixed"):
            if field in overrides:
                specification[field] = overrides[field]

    for name, specification in effective.items():
        data_type = int if specification["type"] == "int" else float
        for field in ("initial_guess", "minimum", "maximum"):
            value = specification[field]
            numeric = int(value) if data_type is int else float(value)
            if not math.isfinite(float(numeric)):
                raise ValueError(f"{name}.{field} must be finite")
            specification[field] = numeric
        if not isinstance(specification["fixed"], bool):
            raise ValueError(f"{name}.fixed must be true or false")
        if specification["maximum"] <= specification["minimum"]:
            raise ValueError(f"{name} requires minimum < maximum")
        if not specification["minimum"] <= specification["initial_guess"] <= specification["maximum"]:
            raise ValueError(f"{name} initial value lies outside its bounds")

    for name, expected in FIXED_ARCHITECTURE.items():
        if name not in effective or not effective[name]["fixed"]:
            raise ValueError(f"{name} must remain fixed by the project policy")
        if not np.isclose(float(effective[name]["initial_guess"]), expected):
            raise ValueError(f"{name} must equal {expected:g}")
    # Certified reduced models declare inactive basis families explicitly. Their
    # material-file flags must remain fixed so the archived input records the
    # model that was actually solved, for example K_2ani1 = K_2ani2 = 0.
    inactive_linear = set(
        ()
        if config.certified_study is None
        else config.certified_study.inactive_linear_parameters
    )
    dependencies = definition.basis_dependencies()
    inactive_shapes = {
        shape
        for name in inactive_linear
        for shape in dependencies.get(name, ())
    }
    if config.certified_study is not None:
        free_inactive = sorted(
            name for name in inactive_linear if not effective[name]["fixed"]
        )
        if free_inactive:
            raise ValueError(
                "Certified inactive linear parameters must be fixed in the material "
                "file: " + ", ".join(free_inactive)
            )
        nonzero_inactive = sorted(
            name
            for name in inactive_linear
            if not np.isclose(float(effective[name]["initial_guess"]), 0.0)
        )
        if nonzero_inactive:
            raise ValueError(
                "Certified inactive linear parameters must equal zero: "
                + ", ".join(nonzero_inactive)
            )
        free_inactive_shapes = sorted(
            name for name in inactive_shapes if not effective[name]["fixed"]
        )
        if free_inactive_shapes:
            raise ValueError(
                "Shapes belonging only to certified inactive families must be fixed "
                "in the material file: " + ", ".join(free_inactive_shapes)
            )

    fixed_linear = [
        name
        for name in definition.linear_parameters
        if effective[name]["fixed"] and name not in inactive_linear
    ]
    if fixed_linear:
        raise ValueError(
            "The current separable solver requires free linear parameters: "
            + ", ".join(fixed_linear)
        )
    shape_parameters = definition.shape_parameters
    fixed_shapes = [
        name
        for name in shape_parameters
        if effective[name]["fixed"] and name not in inactive_shapes
    ]
    if fixed_shapes:
        raise ValueError(
            "Fixed shape flags are not supported by the free training search: "
            + ", ".join(fixed_shapes)
        )
    for rule in definition.derived_parameters:
        if not effective[rule.target]["fixed"]:
            raise ValueError(
                f"Derived parameter {rule.target} must remain fixed as an independent variable"
            )
        source_names = tuple(name for name, _ in rule.coefficients)
        fixed_sources = [name for name in source_names if effective[name]["fixed"]]
        if fixed_sources:
            raise ValueError(
                f"Derived parameter {rule.target} requires free source shapes: "
                + ", ".join(fixed_sources)
            )
    wrong_integer_types = [
        name
        for name in definition.integer_shape_parameters
        if effective[name]["type"] != "int"
    ]
    if wrong_integer_types:
        raise ValueError(
            "Integer shapes must be declared as int: "
            + ", ".join(wrong_integer_types)
        )
    wrong_continuous_types = [
        name
        for name in definition.continuous_shape_parameters
        if effective[name]["type"] != "float"
    ]
    if wrong_continuous_types:
        raise ValueError(
            "Continuous shapes must be declared as float: "
            + ", ".join(wrong_continuous_types)
        )
    if config.coupling_bound_mode == "amplitude_equalized" and not np.isclose(
        float(effective["K_coup"]["minimum"]), 0.0
    ):
        raise ValueError("Amplitude-equalized coupling requires K_coup minimum 0")
    if config.coupling_bound_mode == "common_amplitude":
        dependencies = definition.basis_dependencies()
        nonlinear = [name for name in definition.linear_parameters if dependencies[name]]
        nonzero_lowers = [
            name for name in nonlinear if not np.isclose(effective[name]["minimum"], 0.0)
        ]
        if nonzero_lowers:
            raise ValueError(
                "Common-amplitude coefficients require zero lower bounds: "
                + ", ".join(nonzero_lowers)
            )
    return raw_specs, effective


def selected_zeta_values(
    config: CaseConfiguration,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[int, ...]:
    """Resolve zeta from the parameter file or an explicit manifest list."""
    zeta_spec = parameter_specs["zeta"]
    available = tuple(range(int(zeta_spec["minimum"]), int(zeta_spec["maximum"]) + 1))
    selected = available if config.zeta_selection == "parameter_file" else config.zeta_selection
    assert isinstance(selected, tuple)
    if not selected or not set(selected) <= set(available):
        raise ValueError(
            f"Selected zeta values {selected} are not contained in parameter-file range {available}"
        )
    allowed_structures = {
        "no_coupling",
        "free_zeta",
        *(f"fixed_zeta_{zeta}" for zeta in selected),
    }
    if (
        config.predictive_structure is not None
        and config.predictive_structure not in allowed_structures
    ):
        raise ValueError(
            f"predictive_structure {config.predictive_structure!r} is incompatible with zeta {selected}"
        )
    return selected


def selected_coupling_exponent_values(
    config: CaseConfiguration,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[int, ...]:
    """Resolve the model's coupling exponent, for example zeta or xi."""
    definition = load_separable_definition(
        config.separable_definition,
        config.model_id,
    )
    name = definition.coupling_exponent_parameter
    if name == "zeta":
        return selected_zeta_values(config, parameter_specs)
    specification = parameter_specs[name]
    available = tuple(
        range(int(specification["minimum"]), int(specification["maximum"]) + 1)
    )
    selected = (
        available
        if config.zeta_selection == "parameter_file"
        else config.zeta_selection
    )
    assert isinstance(selected, tuple)
    if not selected or not set(selected) <= set(available):
        raise ValueError(
            f"Selected {name} values {selected} are not contained in "
            f"parameter-file range {available}"
        )
    return selected


def selected_shape_domains(
    config: CaseConfiguration,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[int, ...] | tuple[float, ...]]:
    """Return model-generic shape domains for preflight and study dispatch."""
    definition = load_separable_definition(
        config.separable_definition,
        config.model_id,
    )
    domains: dict[str, tuple[int, ...] | tuple[float, ...]] = {}
    for name in definition.continuous_shape_parameters:
        specification = parameter_specs[name]
        if specification["fixed"]:
            domains[name] = (float(specification["initial_guess"]),)
        else:
            domains[name] = (
                float(specification["minimum"]),
                float(specification["maximum"]),
            )
    for name in definition.integer_shape_parameters:
        specification = parameter_specs[name]
        if specification["fixed"]:
            domains[name] = (int(specification["initial_guess"]),)
        else:
            domains[name] = tuple(
                range(
                    int(specification["minimum"]),
                    int(specification["maximum"]) + 1,
                )
            )
    if (
        definition.backend == "grid_powell"
        and definition.coupling_exponent_parameter
        in definition.integer_shape_parameters
    ):
        exponent_name = definition.coupling_exponent_parameter
        domains[exponent_name] = selected_coupling_exponent_values(
            config,
            parameter_specs,
        )
    return domains


def case_preflight(
    config: CaseConfiguration,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    *,
    require_production: bool = False,
) -> dict[str, Any]:
    """Return a model-generic deterministic preflight description."""
    if require_production and config.workflow != "production":
        raise ValueError(
            f"Case {config.case_id} is {config.workflow}, not a production fit"
        )
    definition = load_separable_definition(
        config.separable_definition,
        config.model_id,
    )
    domains = selected_shape_domains(config, parameter_specs)
    descriptions = []
    for name, domain in domains.items():
        if len(domain) == 1:
            descriptions.append(f"{name}={domain[0]:g} (fixed)")
        elif name in definition.integer_shape_parameters:
            values = tuple(int(value) for value in domain)
            descriptions.append(f"{name}={values[0]}..{values[-1]}")
        else:
            lower, upper = domain
            descriptions.append(f"{name}={lower:g}..{upper:g}")
    inactive_linear = (
        ()
        if config.certified_study is None
        else config.certified_study.inactive_linear_parameters
    )
    dependencies = definition.basis_dependencies()
    inactive_shapes = tuple(
        dependencies[name][0] for name in inactive_linear
    )
    return {
        "backend": definition.backend,
        "workflow": config.workflow,
        "shape_domains": domains,
        "shape_summary": ", ".join(descriptions),
        "inactive_linear_parameters": inactive_linear,
        "inactive_shape_parameters": inactive_shapes,
        "certification_backend": (
            definition.backend
            if config.certified_study is None
            else config.certified_study.certification_backend
        ),
    }


def build_configured_wrapper(
    config: CaseConfiguration,
    model_spec: ModelSpec,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    reference_maxima: Mapping[int, float] | None = None,
) -> tuple[Any, dict[int, float]]:
    """Load direct files and configure the selected objective and measures."""
    loaded = umat.Load_props(config.parameter_file, model_spec=model_spec)
    material = umat.Material_model(loaded)
    wrapper = umat.Umat_wrapper(material, list(config.umat_types))
    apply_parameter_specs_in_memory(wrapper, parameter_specs)
    wrapper.load_inputs(config.experimental_data, list(config.experimental_types))
    if reference_maxima is None:
        reference_maxima = {
            axis: float(np.max(np.abs(wrapper.expstressPK2[:, axis])))
            for axis in config.fit_axes
        }
    references = {int(axis): float(value) for axis, value in reference_maxima.items()}
    wrapper.set_cost_function(
        config.cost_type,
        config.normalized_stress_floor_factor,
        references if config.cost_type == "normalized" else None,
    )
    return wrapper, references


def refine_candidates(
    solver: SeparableMaterialSolver,
    candidates: Sequence[ProfiledSolution],
    settings: OptimizationConfiguration,
) -> tuple[ProfiledSolution, ...]:
    """Polish each mapped zeta minimum using manifest-controlled tolerances."""
    if not settings.refine_each_zeta:
        return tuple(candidates)
    return tuple(
        refine_profiled_shapes(
            solver,
            candidate,
            maximum_iterations=settings.refinement_max_iterations,
            parameter_tolerance=settings.parameter_tolerance,
            cost_tolerance=settings.cost_tolerance,
        )
        for candidate in candidates
    )


def print_zeta_progress(zeta: int, solution: ProfiledSolution) -> None:
    """Print one deterministic map minimum, for example zeta=3."""
    b_text = (
        f", b={float(solution.complete_shape_values()['b']):.8g}"
        if "b" in solution.complete_shape_values()
        else ""
    )
    print(
        f"Mapped zeta={zeta}: mean cost={solution.mean_cost:.12g}, "
        f"beta_1={solution.beta_1:.8g}, gamma_1={solution.gamma_1:.8g}{b_text}",
        flush=True,
    )


def specs_for_solution(
    base_specs: Mapping[str, Mapping[str, Any]],
    solution: ProfiledSolution,
    *,
    coupling_upper: float | None,
    fixed_zeta: bool,
    coupling_exponent_name: str = "zeta",
) -> dict[str, dict[str, Any]]:
    """Record the effective domain used by one exported result."""
    specifications = copy.deepcopy(base_specs)
    complete_shapes = solution.complete_shape_values()
    for name, value in solution.linear_values.items():
        if name in specifications:
            specifications[name]["initial_guess"] = value
    for name, value in complete_shapes.items():
        if name in specifications:
            specifications[name]["initial_guess"] = value
    exponent_value = complete_shapes[coupling_exponent_name]
    exponent_type = specifications[coupling_exponent_name]["type"]
    specifications[coupling_exponent_name]["initial_guess"] = (
        int(exponent_value) if exponent_type == "int" else float(exponent_value)
    )
    specifications[coupling_exponent_name]["fixed"] = bool(fixed_zeta)
    for name in solution.derived_shape_values:
        specifications[name]["fixed"] = True
    if coupling_upper is not None:
        specifications["K_coup"]["maximum"] = float(coupling_upper)
        if float(coupling_upper) == 0.0:
            # A zero coupling basis makes K_coup structurally unidentifiable.
            specifications["K_coup"].update({"initial_guess": 0.0, "fixed": True})
    return specifications


def solve_pl_polycon_export_candidates(
    config: CaseConfiguration,
    model_spec: ModelSpec,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    definition: SeparableModelDefinition,
) -> tuple[tuple[ExportCandidate, ...], dict[str, Any]]:
    """Exhaust the integer power laws and refine the best continuous b values."""
    if config.result_roles != ("training_optimum",):
        raise ValueError(
            "SV_pl_polycon production currently exports only training_optimum"
        )
    xi_values = selected_coupling_exponent_values(config, parameter_specs)
    if min(xi_values) <= 2:
        raise ValueError("SV_pl_polycon requires xi > 2")
    wrapper, reference_maxima = build_configured_wrapper(
        config,
        model_spec,
        parameter_specs,
    )
    solver = calibration.ShapeCouplingBoundSolver(
        wrapper,
        config.fit_axes,
        definition=definition,
    )
    solver.configure_coupling_bound(
        config.coupling_bound_mode,
        config.amplitude_ceiling,
    )
    b_grid = continuous_grids_for_solver(solver, config.optimization)["b"]
    beta_values = solver.integer_values("beta")
    gamma_values = solver.integer_values("gamma")
    retained_count = 20  # Example: refine the 20 best distinct grid profiles.
    retained: list[ProfiledSolution] = []
    rejected_count = 0
    rejected_examples: list[dict[str, float | int | str]] = []

    start = time.perf_counter()
    for xi in xi_values:
        xi_best: ProfiledSolution | None = None
        for beta, gamma, b in itertools.product(beta_values, gamma_values, b_grid):
            shapes = {
                "b": float(b),
                "beta": int(beta),
                "gamma": int(gamma),
                "xi": int(xi),
            }
            try:
                solution = solver.solve_linear_at_shapes(shapes)
            except RuntimeError as error:
                message = str(error)
                is_cone_rejection = (
                    message == "SLSQP could not find a feasible cone-constrained solution"
                    or message.startswith(
                        "Cone-constrained solve returned infeasible coefficients:"
                    )
                )
                if not is_cone_rejection:
                    raise
                rejected_count += 1
                if len(rejected_examples) < 100:
                    rejected_examples.append({**shapes, "reason": message})
                continue
            if xi_best is None or solution.cost < xi_best.cost:
                xi_best = solution
            if len(retained) < retained_count or solution.cost < retained[-1].cost:
                retained.append(solution)
                retained.sort(key=lambda item: item.cost)
                if len(retained) > retained_count:
                    retained.pop()
        if xi_best is None:
            raise RuntimeError(f"No feasible SV_pl_polycon profile for xi={xi}")
        print(
            f"Mapped xi={xi}: mean cost={xi_best.mean_cost:.12g}, "
            f"beta={xi_best.shape_values['beta']}, "
            f"gamma={xi_best.shape_values['gamma']}, "
            f"b={xi_best.shape_values['b']:.8g}",
            flush=True,
        )

    if not retained:
        raise RuntimeError("SV_pl_polycon production search found no feasible profile")
    if config.optimization.refine_each_zeta:
        refined = [
            refine_profiled_shapes(
                solver,
                solution,
                maximum_iterations=config.optimization.refinement_max_iterations,
                parameter_tolerance=config.optimization.parameter_tolerance,
                cost_tolerance=config.optimization.cost_tolerance,
            )
            for solution in retained
        ]
        training_solution = min((*retained, *refined), key=lambda item: item.cost)
    else:
        training_solution = retained[0]
    elapsed = time.perf_counter() - start

    complete_shapes = training_solution.complete_shape_values()
    independent_shapes = {
        name: complete_shapes[name] for name in definition.shape_parameters
    }
    selected_xi = int(independent_shapes["xi"])
    coupling_rms = solver.basis_rms_at_shapes(independent_shapes)
    coupling_upper = training_solution.effective_linear_upper_bounds["K_coup"]
    coupling_bounds = {selected_xi: float(coupling_upper)}
    basis_rms = {selected_xi: float(coupling_rms)}
    solution_specs = specs_for_solution(
        parameter_specs,
        training_solution,
        coupling_upper=coupling_upper,
        fixed_zeta=False,
        coupling_exponent_name="xi",
    )
    candidate = candidate_from_solution(
        config,
        "training_optimum",
        "free_xi",
        solver,
        training_solution,
        solution_specs,
        elapsed,
        coupling_bounds,
        basis_rms,
    )
    diagnostics = {
        "algorithm": (
            "exhaustive beta/gamma/xi integer grid, nine-point b grid, "
            "bounded cone-constrained linear profiling, and Powell b "
            "refinement of the 20 best grid profiles"
        ),
        "fixed_reference_maxima": {
            str(axis): float(value) for axis, value in reference_maxima.items()
        },
        "integer_shape_counts": {
            "beta": len(beta_values),
            "gamma": len(gamma_values),
            "xi": len(xi_values),
        },
        "b_grid": [float(value) for value in b_grid],
        "xi_values": list(xi_values),
        "retained_for_b_refinement": retained_count,
        "coupling_profiled_evaluations": solver.profiled_evaluation_count(),
        "rejected_grid_evaluations": rejected_count,
        "rejected_grid_examples": rejected_examples,
        "linear_backend_counts": solver.linear_backend_counts(),
        "minimum_constraint_margins": solver.minimum_constraint_margins(),
        "coupling_elapsed_seconds": float(elapsed),
        "selected_basis_weighted_rms": float(coupling_rms),
        "selected_K_coup_upper": float(coupling_upper),
        "derived_parameter_relations": [
            {
                "target": rule.target,
                "offset": rule.offset,
                "coefficients": dict(rule.coefficients),
            }
            for rule in definition.derived_parameters
        ],
        "rotated_cone_constraints": [
            asdict(item) for item in definition.rotated_cone_constraints
        ],
    }
    return (candidate,), diagnostics


def candidate_from_solution(
    config: CaseConfiguration,
    role: str,
    model_label: str,
    solver: SeparableMaterialSolver,
    solution: ProfiledSolution,
    parameter_specs: dict[str, dict[str, Any]],
    elapsed_seconds: float,
    coupling_upper_bounds: Mapping[int | float, float],
    basis_weighted_rms: Mapping[int | float, float],
) -> ExportCandidate:
    """Complete one parameter set and validate it through the direct UMAT."""
    parameters = historical.complete_profiled_parameters(solution, parameter_specs)
    direct = historical.evaluate_full_parameter_set(
        solver.wrapper,
        parameters,
        config.fit_axes,
    )
    validation = solver.validate_solution(solution)
    if not np.isclose(direct.raw_cost, solution.cost, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"Profiled and direct costs disagree for {model_label}")
    if validation.existing_wrapper_cost_difference > 1.0e-10:
        raise ValueError(f"Wrapper objective disagrees for {model_label}")
    return ExportCandidate(
        roles=(role,),
        model_labels=(model_label,),
        solution=solution,
        parameters=parameters,
        parameter_specs=parameter_specs,
        direct_raw_cost=direct.raw_cost,
        direct_mean_cost=direct.mean_cost,
        maximum_prediction_difference=(
            validation.maximum_weighted_prediction_difference
        ),
        fitted_observation_count=int(solver.problem.target.size),
        profiled_evaluations=solver.profiled_evaluation_count(),
        elapsed_seconds=float(elapsed_seconds),
        coupling_upper_bounds={key: float(value) for key, value in coupling_upper_bounds.items()},
        basis_weighted_rms={key: float(value) for key, value in basis_weighted_rms.items()},
        validation=asdict(validation),
    )


def parameter_sets_equal(
    left: Mapping[str, float | int],
    right: Mapping[str, float | int],
) -> bool:
    """Return True only for numerically identical complete parameter sets."""
    if set(left) != set(right):
        return False
    return all(
        np.isclose(float(left[name]), float(right[name]), rtol=1.0e-12, atol=1.0e-14)
        for name in left
    )


def merge_identical_candidates(
    candidates: Sequence[ExportCandidate],
) -> tuple[ExportCandidate, ...]:
    """Share one code when training and predictive parameter vectors coincide."""
    merged: list[ExportCandidate] = []
    for candidate in candidates:
        matching_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if parameter_sets_equal(existing.parameters, candidate.parameters)
            ),
            None,
        )
        if matching_index is None:
            merged.append(candidate)
            continue
        existing = merged[matching_index]
        combined_specs = copy.deepcopy(existing.parameter_specs)
        for name, specification in candidate.parameter_specs.items():
            if bool(specification.get("fixed")):
                combined_specs[name]["fixed"] = True
                combined_specs[name]["initial_guess"] = specification["initial_guess"]
        merged[matching_index] = replace(
            existing,
            roles=existing.roles + candidate.roles,
            model_labels=existing.model_labels + candidate.model_labels,
            parameter_specs=combined_specs,
        )
    return tuple(merged)


def certified_parameter_specs_for_solution(
    base_specs: Mapping[str, Mapping[str, Any]],
    certified: CertifiedDiscreteSeparableSolver,
    solution: CertifiedDiscreteSolution,
) -> dict[str, dict[str, Any]]:
    """Record effective amplitude bounds and structurally inactive families."""
    specifications = copy.deepcopy(base_specs)
    effective_bounds = certified.effective_raw_upper_bounds(
        solution.amplitude_ceiling
    )
    for linear_name, exponent_bounds in effective_bounds.items():
        shape_name = certified.family_shapes[linear_name]
        exponent = int(solution.shape_values[shape_name])
        specifications[linear_name]["maximum"] = float(
            exponent_bounds[exponent]
        )

    # The reviewed reduced model fixes both second-family coefficients to zero
    # and stores their otherwise unidentifiable exponents canonically.
    for linear_name in certified.inactive_family_parameters:
        shape_name = certified.family_shapes[linear_name]
        specifications[linear_name].update(
            {"initial_guess": 0.0, "fixed": True}
        )
        specifications[shape_name].update(
            {
                "initial_guess": int(solution.shape_values[shape_name]),
                "fixed": True,
            }
        )
    for shape_name in solution.inactive_shapes:
        specifications[shape_name].update(
            {
                "initial_guess": int(solution.shape_values[shape_name]),
                "fixed": True,
            }
        )
    return specifications


def certified_solution_diagnostics(
    solution: CertifiedDiscreteSolution,
) -> dict[str, Any]:
    """Serialize certificate diagnostics without large prediction arrays."""
    return {
        "amplitude_ceiling": solution.amplitude_ceiling,
        "shape_values": dict(solution.shape_values),
        "linear_values": dict(solution.linear_values),
        "amplitudes": dict(solution.amplitudes),
        "inactive_shapes": list(solution.inactive_shapes),
        "active_bounds": dict(solution.active_bounds),
        "cost": solution.cost,
        "mean_cost": solution.mean_cost,
        "status": solution.status,
        "certificate_satisfied": solution.certificate_satisfied,
        "relative_gap": solution.relative_gap,
        "absolute_gap": solution.absolute_gap,
        "primal_bound": solution.primal_bound,
        "dual_bound": solution.dual_bound,
        "certificate_backend": solution.certificate_backend,
        "enumerated_shape_combinations": (
            solution.enumerated_shape_combinations
        ),
        "solve_seconds": solution.solve_seconds,
        "design_matrix_rank": solution.design_matrix_rank,
        "design_matrix_singular_values": list(
            solution.design_matrix_singular_values
        ),
        "formulation_prediction_difference": (
            solution.formulation_prediction_difference
        ),
        "direct_validation": asdict(solution.validation),
    }


def require_certified_production_solution(
    solution: CertifiedDiscreteSolution,
    basis_validation: Mapping[str, float],
) -> None:
    """Reject uncertified or direct-UMAT-inconsistent production solutions."""
    if not solution.certificate_satisfied:
        raise RuntimeError(
            "Certified production solve did not prove the requested optimum: "
            f"status={solution.status}, relative_gap={solution.relative_gap:g}"
        )
    cost_limit = 1.0e-8 * max(1.0, solution.cost)
    validation = solution.validation
    if (
        validation.absolute_cost_difference > cost_limit
        or validation.existing_wrapper_cost_difference > cost_limit
        or validation.maximum_weighted_prediction_difference >= 1.0e-8
        or solution.formulation_prediction_difference >= 1.0e-8
    ):
        raise RuntimeError("Certified production solution failed direct UMAT validation")
    if max(basis_validation.values(), default=0.0) >= 1.0e-8:
        raise RuntimeError("Certified production basis validation failed")


def certified_candidate_from_solution(
    config: CaseConfiguration,
    core: SeparableMaterialSolver,
    certified: CertifiedDiscreteSeparableSolver,
    solution: CertifiedDiscreteSolution,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> ExportCandidate:
    """Convert one certified discrete optimum into a standard fit candidate."""
    profiled = solution.as_profiled_solution()
    effective_specs = certified_parameter_specs_for_solution(
        parameter_specs,
        certified,
        solution,
    )
    parameters = historical.complete_profiled_parameters(
        profiled,
        effective_specs,
    )
    direct = historical.evaluate_full_parameter_set(
        core.wrapper,
        parameters,
        config.fit_axes,
    )
    if not np.isclose(direct.raw_cost, solution.cost, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError("Certified and direct production costs disagree")

    effective_bounds = certified.effective_raw_upper_bounds(
        solution.amplitude_ceiling
    )
    coupling_bounds = {
        int(exponent): float(value)
        for exponent, value in effective_bounds.get("K_coup", {}).items()
    }
    coupling_rms = {
        metric.exponent: metric.weighted_rms
        for metric in certified.basis_metrics
        if metric.linear_parameter == "K_coup" and metric.excited
    }
    return ExportCandidate(
        roles=("training_optimum",),
        model_labels=("certified_reduced_second_anisotropic_families",),
        solution=profiled,
        parameters=parameters,
        parameter_specs=effective_specs,
        direct_raw_cost=direct.raw_cost,
        direct_mean_cost=direct.mean_cost,
        maximum_prediction_difference=(
            solution.validation.maximum_weighted_prediction_difference
        ),
        fitted_observation_count=int(core.problem.target.size),
        profiled_evaluations=solution.enumerated_shape_combinations,
        elapsed_seconds=solution.solve_seconds,
        coupling_upper_bounds=coupling_bounds,
        basis_weighted_rms=coupling_rms,
        validation={
            **asdict(solution.validation),
            "certificate": certified_solution_diagnostics(solution),
        },
    )


def solve_certified_export_candidates(
    config: CaseConfiguration,
    model_spec: ModelSpec,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[ExportCandidate, ...], dict[str, Any]]:
    """Solve one reviewed certified ceiling before any fit code is reserved."""
    settings = config.certified_study
    if settings is None or settings.production_amplitude_ceiling is None:
        raise ValueError("Certified production configuration is incomplete")
    wrapper, reference_maxima = build_configured_wrapper(
        config,
        model_spec,
        parameter_specs,
    )
    definition = load_separable_definition(
        config.separable_definition,
        config.model_id,
    )
    core = SeparableMaterialSolver(
        wrapper,
        config.fit_axes,
        definition=definition,
        inactive_linear_parameters=settings.inactive_linear_parameters,
    )
    # The core owns the resolved active/inactive partition. The certified
    # backend consumes it directly so the two layers cannot silently diverge.
    certified = CertifiedDiscreteSeparableSolver(core)
    ceiling = settings.production_amplitude_ceiling
    if settings.certification_backend == "exhaustive_enumeration":
        solution = certified.solve_exhaustive(
            ceiling,
            time_limit_seconds=settings.total_time_limit_seconds,
        )
    else:
        solution = certified.solve(
            ceiling,
            relative_gap=settings.relative_gap,
            time_limit_seconds=settings.total_time_limit_seconds,
            display=False,
        )
    basis_validation = core.validate_basis_contributions(solution.shape_values)
    require_certified_production_solution(solution, basis_validation)
    candidate = certified_candidate_from_solution(
        config,
        core,
        certified,
        solution,
        parameter_specs,
    )
    effective_bounds = certified.effective_raw_upper_bounds(ceiling)
    diagnostics = {
        "algorithm": (
            "certified discrete exhaustive enumeration with bounded-linear "
            "observable-amplitude profiling"
            if settings.certification_backend == "exhaustive_enumeration"
            else "certified discrete SCIP SOS1 observable-amplitude optimization"
        ),
        "fixed_reference_maxima": {
            str(axis): float(value) for axis, value in reference_maxima.items()
        },
        "production_amplitude_ceiling": ceiling,
        "certification_backend": settings.certification_backend,
        "inactive_linear_parameters": list(
            certified.inactive_family_parameters
        ),
        "configured_inactive_linear_parameters": list(
            settings.inactive_linear_parameters
        ),
        "auto_inactive_linear_parameters": list(
            certified.auto_inactive_family_parameters
        ),
        "basis_metrics": [asdict(metric) for metric in certified.basis_metrics],
        "effective_raw_upper_bounds": {
            name: {str(exponent): value for exponent, value in values.items()}
            for name, values in effective_bounds.items()
        },
        "basis_contribution_validation": dict(basis_validation),
        "certified_solution": certified_solution_diagnostics(solution),
    }
    return (candidate,), diagnostics


def continuous_grids_for_solver(
    solver: SeparableMaterialSolver,
    settings: OptimizationConfiguration,
) -> dict[str, np.ndarray]:
    """Build declared log/linear grids, for example five uniform b values."""
    grids: dict[str, np.ndarray] = {}
    for name in solver.definition.continuous_shape_parameters:
        default_points = (
            DEFAULT_POLYCON_B_GRID_POINTS
            if solver.definition.model_id == "SV_exp_polycon" and name == "b"
            else settings.grid_points_per_axis
        )
        points = settings.grid_points_by_parameter.get(name, default_points)
        bounds = solver.parameter_bounds(name)
        grids[name] = (
            logarithmic_grid(bounds, points)
            if solver.definition.shape_scale(name) == "log"
            else linear_grid(bounds, points)
        )
    return grids


def solve_continuous_xi_export_candidates(
    config: CaseConfiguration,
    model_spec: ModelSpec,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    definition: SeparableModelDefinition,
) -> tuple[tuple[ExportCandidate, ...], dict[str, Any]]:
    """Map integer-spaced xi seed planes, then refine all four shapes jointly."""
    if config.result_roles != ("training_optimum",):
        raise ValueError(
            "SV_exp_pc_mo_xi_real production exports only training_optimum"
        )
    xi_spec = parameter_specs["xi"]
    if xi_spec["type"] != "float":
        raise ValueError("SV_exp_pc_mo_xi_real requires xi to be a float")
    if float(xi_spec["minimum"]) <= 2.0:
        raise ValueError("SV_exp_pc_mo_xi_real requires xi > 2")

    wrapper, reference_maxima = build_configured_wrapper(
        config,
        model_spec,
        parameter_specs,
    )
    solver = calibration.ShapeCouplingBoundSolver(
        wrapper,
        config.fit_axes,
        definition=definition,
    )
    solver.configure_coupling_bound(
        config.coupling_bound_mode,
        config.amplitude_ceiling,
    )
    complete_grids = continuous_grids_for_solver(solver, config.optimization)
    xi_seeds = tuple(float(value) for value in complete_grids.pop("xi"))
    if not xi_seeds:
        raise ValueError("The continuous-xi search requires at least one seed plane")

    mapped: list[ProfiledSolution] = []
    rejected: list[dict[str, float | int | str]] = []
    start = time.perf_counter()
    for xi_seed in xi_seeds:
        plane_grids = {**complete_grids, "xi": np.asarray([xi_seed], dtype=float)}
        plane = deterministic_shape_grid_search(
            solver,
            plane_grids,
            {},
            reject_infeasible_cone_points=True,
        )
        mapped.append(plane.best_solution)
        rejected.extend(plane.rejected_evaluations)
        shapes = plane.best_solution.complete_shape_values()
        print(
            f"Mapped xi seed={xi_seed:g}: mean cost={plane.best_solution.mean_cost:.12g}, "
            f"beta_1={float(shapes['beta_1']):.8g}, "
            f"gamma_1={float(shapes['gamma_1']):.8g}, "
            f"b={float(shapes['b']):.8g}",
            flush=True,
        )

    refined = refine_candidates(solver, mapped, config.optimization)
    training_solution = min(refined, key=lambda solution: solution.cost)
    elapsed = time.perf_counter() - start

    coupling_bounds: dict[int | float, float] = {}
    basis_rms: dict[int | float, float] = {}
    basis_metrics: list[dict[str, Any]] = []
    for xi_seed, solution in zip(xi_seeds, refined):
        shapes = solution.complete_shape_values()
        fitted_xi = float(shapes["xi"])
        rms = float(solver.basis_rms_at_shapes(shapes))
        upper = float(solution.effective_linear_upper_bounds["K_coup"])
        coupling_bounds[fitted_xi] = upper
        basis_rms[fitted_xi] = rms
        basis_metrics.append(
            {
                "xi_seed": xi_seed,
                "fitted_xi": fitted_xi,
                "b": float(shapes["b"]),
                "a": float(solution.derived_shape_values["a"]),
                "weighted_rms": rms,
                "coupling_upper_bound": upper,
                "mean_cost": float(solution.mean_cost),
            }
        )

    selected_xi = float(training_solution.complete_shape_values()["xi"])
    selected_upper = float(training_solution.effective_linear_upper_bounds["K_coup"])
    selected_rms = float(solver.basis_rms_at_shapes(training_solution.complete_shape_values()))
    coupling_bounds[selected_xi] = selected_upper
    basis_rms[selected_xi] = selected_rms
    specifications = specs_for_solution(
        parameter_specs,
        training_solution,
        coupling_upper=selected_upper,
        fixed_zeta=False,
        coupling_exponent_name="xi",
    )
    candidate = candidate_from_solution(
        config,
        "training_optimum",
        "free_xi",
        solver,
        training_solution,
        specifications,
        elapsed,
        coupling_bounds,
        basis_rms,
    )
    diagnostics = {
        "fixed_reference_maxima": {
            str(axis): float(value) for axis, value in reference_maxima.items()
        },
        "algorithm": "13 xi seed planes followed by joint Powell refinement",
        "xi_seed_values": list(xi_seeds),
        "basis_metrics": basis_metrics,
        "selected_xi": selected_xi,
        "selected_K_coup_upper": selected_upper,
        "selected_basis_weighted_rms": selected_rms,
        "coupling_profiled_evaluations": solver.profiled_evaluation_count(),
        "rejected_grid_evaluations": len(rejected),
        "rejected_grid_points": rejected,
        "linear_backend_counts": solver.linear_backend_counts(),
        "minimum_constraint_margins": solver.minimum_constraint_margins(),
        "coupling_elapsed_seconds": float(elapsed),
        "derived_parameter_relations": [
            {
                "target": rule.target,
                "offset": rule.offset,
                "coefficients": dict(rule.coefficients),
            }
            for rule in definition.derived_parameters
        ],
        "rotated_cone_constraints": [
            asdict(item) for item in definition.rotated_cone_constraints
        ],
    }
    return (candidate,), diagnostics


def solve_polycon_export_candidates(
    config: CaseConfiguration,
    model_spec: ModelSpec,
    parameter_specs: Mapping[str, Mapping[str, Any]],
    definition: SeparableModelDefinition,
) -> tuple[tuple[ExportCandidate, ...], dict[str, Any]]:
    """Solve the additive-coupling model without allocating or exporting codes."""
    zeta_values = selected_zeta_values(config, parameter_specs)
    wrapper, reference_maxima = build_configured_wrapper(
        config,
        model_spec,
        parameter_specs,
    )
    solver = calibration.ShapeCouplingBoundSolver(
        wrapper,
        config.fit_axes,
        definition=definition,
    )
    solver.configure_coupling_bound(
        config.coupling_bound_mode,
        config.amplitude_ceiling,
    )
    continuous_grids = continuous_grids_for_solver(solver, config.optimization)
    start = time.perf_counter()
    grid = deterministic_shape_grid_search(
        solver,
        continuous_grids,
        {"zeta": zeta_values},
        progress_callback=print_zeta_progress,
        reject_infeasible_cone_points=True,
    )
    final_by_zeta = refine_candidates(
        solver,
        grid.best_by_integer_combination,
        config.optimization,
    )
    elapsed = time.perf_counter() - start
    by_zeta = {solution.zeta: solution for solution in final_by_zeta}
    training_solution = min(final_by_zeta, key=lambda solution: solution.cost)

    coupling_bounds: dict[int, float] = {}
    basis_rms: dict[int, float] = {}
    basis_metrics: list[dict[str, Any]] = []
    for solution in final_by_zeta:
        shapes = {
            name: solution.complete_shape_values()[name]
            for name in definition.shape_parameters
        }
        rms = solver.basis_rms_at_shapes(shapes)
        upper = solution.effective_linear_upper_bounds["K_coup"]
        coupling_bounds[solution.zeta] = float(upper)
        basis_rms[solution.zeta] = float(rms)
        basis_metrics.append(
            {
                "zeta": int(solution.zeta),
                "b": float(shapes["b"]),
                "a": float(solution.derived_shape_values["a"]),
                "weighted_rms": float(rms),
                "coupling_upper_bound": float(upper),
            }
        )

    candidates: list[ExportCandidate] = []
    if "training_optimum" in config.result_roles:
        training_specs = specs_for_solution(
            parameter_specs,
            training_solution,
            coupling_upper=coupling_bounds[training_solution.zeta],
            fixed_zeta=False,
        )
        candidates.append(
            candidate_from_solution(
                config,
                "training_optimum",
                "free_zeta",
                solver,
                training_solution,
                training_specs,
                elapsed,
                coupling_bounds,
                basis_rms,
            )
        )

    if "predictive_model" in config.result_roles:
        label = config.predictive_structure
        if label == "free_zeta":
            predictive_solution = training_solution
            fixed_zeta = False
        elif label is not None and label.startswith("fixed_zeta_"):
            selected_zeta = int(label.rsplit("_", 1)[1])
            predictive_solution = by_zeta[selected_zeta]
            fixed_zeta = True
        else:
            raise ValueError(
                "SV_exp_polycon predictive_structure supports free_zeta or fixed_zeta_N"
            )
        predictive_specs = specs_for_solution(
            parameter_specs,
            predictive_solution,
            coupling_upper=coupling_bounds[predictive_solution.zeta],
            fixed_zeta=fixed_zeta,
        )
        candidates.append(
            candidate_from_solution(
                config,
                "predictive_model",
                str(label),
                solver,
                predictive_solution,
                predictive_specs,
                elapsed,
                coupling_bounds,
                basis_rms,
            )
        )

    diagnostics = {
        "fixed_reference_maxima": {
            str(axis): float(value) for axis, value in reference_maxima.items()
        },
        "basis_metrics": basis_metrics,
        "coupling_profiled_evaluations": solver.profiled_evaluation_count(),
        "rejected_grid_evaluations": len(grid.rejected_evaluations),
        "rejected_grid_points": list(grid.rejected_evaluations),
        "linear_backend_counts": solver.linear_backend_counts(),
        "minimum_constraint_margins": solver.minimum_constraint_margins(),
        "coupling_elapsed_seconds": float(elapsed),
        "zeta_values": list(zeta_values),
        "derived_parameter_relations": [
            {
                "target": rule.target,
                "offset": rule.offset,
                "coefficients": dict(rule.coefficients),
            }
            for rule in definition.derived_parameters
        ],
        "rotated_cone_constraints": [asdict(item) for item in definition.rotated_cone_constraints],
    }
    return merge_identical_candidates(candidates), diagnostics


def solve_export_candidates(
    config: CaseConfiguration,
    model_spec: ModelSpec,
    parameter_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[ExportCandidate, ...], dict[str, Any]]:
    """Run the configured coupling search and optional reduced model."""
    definition = load_separable_definition(
        config.separable_definition,
        config.model_id,
    )
    if definition.backend == "scip_sos1":
        return solve_certified_export_candidates(
            config,
            model_spec,
            parameter_specs,
        )
    if definition.model_id == "SV_pl_polycon":
        return solve_pl_polycon_export_candidates(
            config,
            model_spec,
            parameter_specs,
            definition,
        )
    if definition.model_id == "SV_exp_pc_mo_xi_real":
        return solve_continuous_xi_export_candidates(
            config,
            model_spec,
            parameter_specs,
            definition,
        )
    if definition.derived_parameters:
        return solve_polycon_export_candidates(
            config,
            model_spec,
            parameter_specs,
            definition,
        )
    if definition.backend != "grid_powell":
        raise ValueError(
            f"Production export is not enabled for backend {definition.backend!r}; "
            "review the certified study checkpoint first"
        )
    zeta_values = selected_zeta_values(config, parameter_specs)
    wrapper, reference_maxima = build_configured_wrapper(
        config,
        model_spec,
        parameter_specs,
    )
    coupling_solver = calibration.ZetaCouplingBoundSolver(
        wrapper,
        config.fit_axes,
        definition=definition,
    )
    basis_metrics = calibration.compute_coupling_basis_metrics(coupling_solver)
    basis_rms = {metric.zeta: metric.weighted_rms for metric in basis_metrics}
    if config.coupling_bound_mode == "amplitude_equalized":
        assert config.amplitude_ceiling is not None
        coupling_bounds = calibration.amplitude_equalized_bounds(
            basis_rms,
            config.amplitude_ceiling,
        )
    else:
        parameter_upper = float(parameter_specs["K_coup"]["maximum"])
        coupling_bounds = {
            zeta: parameter_upper for zeta in coupling_solver.integer_values("zeta")
        }
    coupling_solver.set_coupling_upper_bounds(coupling_bounds)

    beta_values = logarithmic_grid(
        coupling_solver.parameter_bounds("beta_1"),
        config.optimization.grid_points_per_axis,
    )
    gamma_values = logarithmic_grid(
        coupling_solver.parameter_bounds("gamma_1"),
        config.optimization.grid_points_per_axis,
    )
    start = time.perf_counter()
    grid = deterministic_grid_search(
        coupling_solver,
        beta_values,
        gamma_values,
        zeta_values,
        progress_callback=print_zeta_progress,
    )
    final_by_zeta = refine_candidates(
        coupling_solver,
        grid.best_by_zeta,
        config.optimization,
    )
    coupling_elapsed = time.perf_counter() - start
    by_zeta = {solution.zeta: solution for solution in final_by_zeta}
    training_solution = min(final_by_zeta, key=lambda solution: solution.cost)

    candidates: list[ExportCandidate] = []
    if "training_optimum" in config.result_roles:
        training_specs = specs_for_solution(
            parameter_specs,
            training_solution,
            coupling_upper=coupling_bounds[training_solution.zeta],
            fixed_zeta=False,
        )
        candidates.append(
            candidate_from_solution(
                config,
                "training_optimum",
                "free_zeta",
                coupling_solver,
                training_solution,
                training_specs,
                coupling_elapsed,
                coupling_bounds,
                basis_rms,
            )
        )

    no_coupling_diagnostics: dict[str, Any] = {}
    if "predictive_model" in config.result_roles:
        label = config.predictive_structure
        if label == "no_coupling":
            no_coupling_specs = copy.deepcopy(parameter_specs)
            no_coupling_specs["K_coup"].update(
                {"initial_guess": 0.0, "fixed": True}
            )
            no_coupling_specs["zeta"].update(
                {"initial_guess": int(zeta_values[0]), "fixed": True}
            )
            no_coupling_wrapper, _ = build_configured_wrapper(
                config,
                model_spec,
                no_coupling_specs,
                reference_maxima,
            )
            no_coupling_definition = SeparableModelDefinition(
                model_id=definition.model_id,
                linear_parameters=tuple(
                    name for name in definition.linear_parameters if name != "K_coup"
                ),
                continuous_shape_parameters=definition.continuous_shape_parameters,
                integer_shape_parameters=(),
            )
            no_coupling_solver = SeparableMaterialSolver(
                no_coupling_wrapper,
                config.fit_axes,
                definition=no_coupling_definition,
            )
            no_coupling_start = time.perf_counter()
            no_coupling_grid = deterministic_grid_search(
                no_coupling_solver,
                beta_values,
                gamma_values,
                (zeta_values[0],),
            )
            no_coupling_solution = refine_candidates(
                no_coupling_solver,
                no_coupling_grid.best_by_zeta,
                config.optimization,
            )[0]
            no_coupling_elapsed = time.perf_counter() - no_coupling_start
            candidates.append(
                candidate_from_solution(
                    config,
                    "predictive_model",
                    label,
                    no_coupling_solver,
                    no_coupling_solution,
                    no_coupling_specs,
                    no_coupling_elapsed,
                    {},
                    {},
                )
            )
            no_coupling_diagnostics = {
                "profiled_evaluations": no_coupling_solver.profiled_evaluation_count(),
                "elapsed_seconds": float(no_coupling_elapsed),
            }
        elif label == "free_zeta":
            predictive_specs = specs_for_solution(
                parameter_specs,
                training_solution,
                coupling_upper=coupling_bounds[training_solution.zeta],
                fixed_zeta=False,
            )
            candidates.append(
                candidate_from_solution(
                    config,
                    "predictive_model",
                    label,
                    coupling_solver,
                    training_solution,
                    predictive_specs,
                    coupling_elapsed,
                    coupling_bounds,
                    basis_rms,
                )
            )
        else:
            selected_zeta = int(label.rsplit("_", 1)[1])
            predictive_solution = by_zeta[selected_zeta]
            predictive_specs = specs_for_solution(
                parameter_specs,
                predictive_solution,
                coupling_upper=coupling_bounds[selected_zeta],
                fixed_zeta=True,
            )
            candidates.append(
                candidate_from_solution(
                    config,
                    "predictive_model",
                    label,
                    coupling_solver,
                    predictive_solution,
                    predictive_specs,
                    coupling_elapsed,
                    coupling_bounds,
                    basis_rms,
                )
            )

    diagnostics = {
        "fixed_reference_maxima": {
            str(axis): float(value) for axis, value in reference_maxima.items()
        },
        "basis_metrics": [asdict(metric) for metric in basis_metrics],
        "coupling_profiled_evaluations": coupling_solver.profiled_evaluation_count(),
        "coupling_elapsed_seconds": float(coupling_elapsed),
        "no_coupling": no_coupling_diagnostics,
        "zeta_values": list(zeta_values),
    }
    return merge_identical_candidates(candidates), diagnostics


def active_bound_for_parameter(name: str, candidate: ExportCandidate) -> str:
    """Return ``lower`` or ``upper`` when an optimum touches its bound."""
    if bool(candidate.parameter_specs[name].get("fixed")):
        return ""
    mask = candidate.solution.active_bounds.get(name)
    if mask == -1:
        return "lower"
    if mask == 1:
        return "upper"
    value = float(candidate.parameters[name])
    minimum = float(candidate.parameter_specs[name]["minimum"])
    maximum = float(candidate.parameter_specs[name]["maximum"])
    tolerance = 1.0e-6 * max(1.0, abs(minimum), abs(maximum))
    if abs(value - minimum) <= tolerance:
        return "lower"
    if abs(value - maximum) <= tolerance:
        return "upper"
    return ""


def identifiability_status(name: str, candidate: ExportCandidate) -> str:
    """Separate structural inactivity from ordinary numerical bound contact."""
    fixed = bool(candidate.parameter_specs[name].get("fixed"))
    if name in candidate.solution.derived_shape_values:
        return "derived_constraint"
    if name in {"theta1", "theta2"}:
        return "fixed_angle_policy"
    if name in {"C1", "C2"}:
        return "fixed_architecture"
    if name == "K_coup" and "no_coupling" in candidate.model_labels:
        return "excluded_no_coupling"
    if name == "zeta" and "no_coupling" in candidate.model_labels:
        return "inactive_no_coupling"
    if name == "zeta" and fixed and "fixed_zeta" in candidate.model_label:
        return "fixed_by_model_selection"
    shape_coefficients = {
        "beta_1": "K_1ani1",
        "beta_2": "K_2ani1",
        "gamma_1": "K_1ani2",
        "gamma_2": "K_2ani2",
        "zeta": "K_coup",
        "beta": "K_course",
        "gamma": "K_wale",
        "xi": "K_coup",
    }
    controlling_coefficient = shape_coefficients.get(name)
    if (
        controlling_coefficient in candidate.parameters
        and np.isclose(candidate.parameters[controlling_coefficient], 0.0)
    ):
        return f"inactive_zero_{controlling_coefficient}"
    if (
        name in {"K_2ani1", "K_2ani2"}
        and fixed
        and np.isclose(candidate.parameters[name], 0.0)
    ):
        return "structurally_inactive_reduced_model"
    if fixed:
        return "fixed"
    if active_bound_for_parameter(name, candidate):
        return "bound_limited"
    return "estimated"


def tuple_parameter_specs(
    specifications: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[Any, type, Any, Any, int, bool]]:
    """Convert JSON-like specifications to the established artifact interface."""
    return {
        name: (
            specification["initial_guess"],
            int if specification["type"] == "int" else float,
            specification["minimum"],
            specification["maximum"],
            0,
            bool(specification["fixed"]),
        )
        for name, specification in specifications.items()
    }


def input_hashes(config: CaseConfiguration) -> dict[str, str]:
    """Hash every direct input, for example the edited parameter `.dat`."""
    return {
        "case_config": historical.sha256_file(config.path),
        "material_parameters": historical.sha256_file(config.parameter_file),
        "experimental_data": historical.sha256_file(config.experimental_data),
        "umat_source": historical.sha256_file(config.umat_source),
    }


def require_unchanged_inputs(
    config: CaseConfiguration,
    expected_hashes: Mapping[str, str],
) -> None:
    """Abort export if an input was edited while the long solve was running."""
    current = input_hashes(config)
    changed = [name for name, digest in current.items() if digest != expected_hashes[name]]
    if changed:
        raise ValueError("Inputs changed during optimization: " + ", ".join(changed))


def parameter_specs_dat_text(
    specifications: Mapping[str, Mapping[str, Any]],
    values: Mapping[str, float | int] | None = None,
) -> str:
    """Serialize effective material specifications, for example zeta min 2."""
    lines = ["Name\tGuess\tType\tMin\tMax\tFix?"]
    for name, specification in specifications.items():
        value = (
            values[name]
            if values is not None and name in values
            else specification["initial_guess"]
        )
        lines.append(
            "\t".join(
                (
                    name,
                    str(value),
                    str(specification["type"]),
                    str(specification["minimum"]),
                    str(specification["maximum"]),
                    str(bool(specification["fixed"])),
                )
            )
        )
    return "\n".join(lines) + "\n"


def compact_shape_domains(
    config: CaseConfiguration,
    effective_parameter_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Record searched intervals and fixed singleton shapes without ambiguity."""
    domains = selected_shape_domains(config, effective_parameter_specs)
    compact: dict[str, dict[str, Any]] = {}
    for name, domain in domains.items():
        if effective_parameter_specs[name]["fixed"]:
            compact[name] = {
                "kind": "fixed_value",
                "value": domain[0],
                "count": 1,
            }
        elif len(domain) == 2 and any(isinstance(value, float) for value in domain):
            compact[name] = {
                "kind": "continuous_interval",
                "minimum": domain[0],
                "maximum": domain[1],
            }
        else:
            compact[name] = {
                "kind": "enumerated_integer_domain",
                "minimum": min(domain),
                "maximum": max(domain),
                "count": len(domain),
            }
    return compact


def active_shape_domains(
    config: CaseConfiguration,
    effective_parameter_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return only shape domains that the solver actually searches."""
    return {
        name: domain
        for name, domain in compact_shape_domains(
            config, effective_parameter_specs
        ).items()
        if not effective_parameter_specs[name]["fixed"]
    }


def fixed_parameter_values(
    effective_parameter_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | int]:
    """Return fixed material values, for example K_2ani1=0 and beta_2=4."""
    return {
        name: specification["initial_guess"]
        for name, specification in effective_parameter_specs.items()
        if specification["fixed"]
    }


def material_parameter_search_payload(
    config: CaseConfiguration,
    effective_parameter_specs: Mapping[str, Mapping[str, Any]],
    run_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the exact merged search policy beyond a traditional `.dat`."""
    return {
        "schema_version": 2,
        "description": (
            "Authoritative material-parameter search configuration after case "
            "overrides; certified exponent-dependent bounds are included below."
        ),
        "effective_parameter_specs_after_overrides": copy.deepcopy(
            dict(effective_parameter_specs)
        ),
        "shape_domains": compact_shape_domains(config, effective_parameter_specs),
        "active_shape_domains": active_shape_domains(
            config, effective_parameter_specs
        ),
        "fixed_parameters": fixed_parameter_values(effective_parameter_specs),
        "inactive_linear_parameters": list(
            ()
            if config.certified_study is None
            else config.certified_study.inactive_linear_parameters
        ),
        "inactive_shape_parameters": list(
            ()
            if config.certified_study is None
            else tuple(
                shape
                for coefficient in config.certified_study.inactive_linear_parameters
                for shape in load_separable_definition(
                    config.separable_definition, config.model_id
                ).basis_dependencies()[coefficient]
            )
        ),
        "parameter_overrides": copy.deepcopy(config.parameter_overrides),
        "certified_study": copy.deepcopy(config.payload.get("certified_study")),
        "algorithm": run_diagnostics.get("algorithm"),
        "effective_raw_upper_bounds": copy.deepcopy(
            run_diagnostics.get("effective_raw_upper_bounds", {})
        ),
    }


def snapshot_inputs(
    code: int,
    config: CaseConfiguration,
    candidate: ExportCandidate,
    effective_parameter_specs: Mapping[str, Mapping[str, Any]],
    run_diagnostics: Mapping[str, Any],
) -> dict[str, str]:
    """Store source inputs plus effective material information without ambiguity."""
    snapshot_directory = artifact_directory(LEGACY_RESULTS_CSV, code) / "inputs"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    sources = {
        "case_config": (config.path, "case.json"),
        "material_parameters": (
            config.parameter_file,
            "material_parameters_source.dat",
        ),
        "umat_source": (config.umat_source, f"umat_source{config.umat_source.suffix}"),
    }
    if config.snapshot_experimental_data:
        sources["experimental_data"] = (
            config.experimental_data,
            f"experimental_data{config.experimental_data.suffix}",
        )
    snapshots: dict[str, str] = {}
    for name, (source, filename) in sources.items():
        destination = snapshot_directory / filename
        shutil.copy2(source, destination)
        snapshots[name] = str(destination.resolve())

    effective_path = snapshot_directory / "material_parameters.dat"
    effective_path.write_text(
        parameter_specs_dat_text(candidate.parameter_specs, candidate.parameters),
        encoding="utf-8",
    )
    snapshots["effective_material_parameters"] = str(effective_path.resolve())

    search_path = snapshot_directory / "material_parameter_search.json"
    search_path.write_text(
        json.dumps(
            material_parameter_search_payload(
                config,
                effective_parameter_specs,
                run_diagnostics,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    snapshots["material_parameter_search"] = str(search_path.resolve())
    return snapshots


def effective_specs_from_provenance(
    raw_parameter_specs: Mapping[str, Mapping[str, Any]],
    parameter_overrides: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Rebuild merged static specifications, for example beta min 1 -> 3."""
    effective = copy.deepcopy(dict(raw_parameter_specs))
    for name, overrides in parameter_overrides.items():
        if name not in effective:
            raise ValueError(f"Override references missing parameter {name!r}")
        for field, value in overrides.items():
            target = "initial_guess" if field == "value" else field
            effective[name][target] = value
    return effective


def migrate_fit_material_parameter_snapshots(
    fit_json_path: pathlib.Path,
) -> dict[str, str]:
    """Upgrade one legacy fit so source, search, and terminal specs are explicit."""
    fit_json_path = pathlib.Path(fit_json_path).resolve()
    payload = json.loads(fit_json_path.read_text(encoding="utf-8"))
    input_files = payload.get("input_files")
    if not isinstance(input_files, dict):
        raise ValueError("Fit artifact has no input_files object")
    material_entry = input_files.get("material_parameters")
    if not isinstance(material_entry, dict):
        raise ValueError("Fit artifact has no material_parameters input entry")

    input_directory = fit_json_path.parent / "inputs"
    legacy_path = input_directory / "material_parameters.dat"
    source_path = input_directory / "material_parameters_source.dat"
    expected_source_hash = str(material_entry.get("sha256", ""))
    if not source_path.is_file():
        if not legacy_path.is_file():
            raise FileNotFoundError(f"Missing legacy material snapshot: {legacy_path}")
        if historical.sha256_file(legacy_path) != expected_source_hash:
            raise ValueError("Legacy material snapshot hash does not match fit provenance")
        shutil.copy2(legacy_path, source_path)
    if historical.sha256_file(source_path) != expected_source_hash:
        raise ValueError("Source material snapshot hash does not match fit provenance")

    terminal_specs = payload.get("parameter_specs")
    terminal_values = payload.get("parameters")
    provenance = payload.get("parameter_provenance")
    if not isinstance(terminal_specs, dict) or not isinstance(terminal_values, dict):
        raise ValueError("Fit artifact lacks terminal material specifications")
    if not isinstance(provenance, dict):
        raise ValueError("Fit artifact lacks parameter provenance")
    raw_specs = provenance.get("raw_parameter_specs")
    overrides = provenance.get("parameter_overrides", {})
    if not isinstance(raw_specs, dict) or not isinstance(overrides, dict):
        raise ValueError("Fit artifact lacks reconstructable raw specs or overrides")
    effective_search_specs = effective_specs_from_provenance(raw_specs, overrides)

    legacy_path.write_text(
        parameter_specs_dat_text(terminal_specs, terminal_values),
        encoding="utf-8",
    )
    case_path = pathlib.Path(input_files["case_config"]["snapshot_path"])
    config = load_case_configuration(case_path)
    run_diagnostics = payload.get("diagnostics", {}).get("run", {})
    search_path = input_directory / "material_parameter_search.json"
    search_path.write_text(
        json.dumps(
            material_parameter_search_payload(
                config,
                effective_search_specs,
                run_diagnostics,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    material_entry["snapshot_path"] = str(source_path.resolve())
    material_entry["role"] = "source_parameter_file_before_case_overrides"
    input_files["effective_material_parameters"] = {
        "source_path": None,
        "snapshot_path": str(legacy_path.resolve()),
        "sha256": historical.sha256_file(legacy_path),
        "role": "terminal_effective_parameter_specifications",
    }
    input_files["material_parameter_search"] = {
        "source_path": None,
        "snapshot_path": str(search_path.resolve()),
        "sha256": historical.sha256_file(search_path),
        "role": "authoritative_merged_search_configuration",
    }
    provenance["snapshot_schema_version"] = 2
    provenance["effective_search_parameter_specs"] = effective_search_specs
    artifact_provenance = payload.setdefault("provenance", {})
    updates = artifact_provenance.setdefault("artifact_updates", [])
    update_kind = "material_parameter_snapshot_schema_v2"
    if not any(update.get("kind") == update_kind for update in updates):
        updates.append(
            {
                "kind": update_kind,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "reason": (
                    "Separate the exact source parameter file from merged search "
                    "and terminal effective material specifications."
                ),
            }
        )
    fit_json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "fit_json": str(fit_json_path),
        "source_material_parameters": str(source_path.resolve()),
        "effective_material_parameters": str(legacy_path.resolve()),
        "material_parameter_search": str(search_path.resolve()),
    }


def artifact_results(
    code: int,
    candidate: ExportCandidate,
    config: CaseConfiguration,
) -> dict[str, list[Any]]:
    """Build metadata and parameters consumed by the reusable artifact writer."""
    names = list(candidate.parameters)
    common: dict[str, Any] = {
        "Case_ID": config.case_id,
        "Run_Label": config.run_label,
        "Input_Config_Path": str(config.path.resolve()),
        "Model_ID": config.model_id,
        "UMAT": config.umat_source.name,
        "Exp_Data": config.experimental_data.name,
        "Exp_Data_Path": str(config.experimental_data.resolve()),
        "Exp_Data_Type": config.data_type,
        "Yarn_Type": config.yarn_type,
        "Knit": config.knit,
        "Knit_Token": config.knit_token,
        "Sample_Folder": config.sample_folder,
        "Param_File": config.parameter_file.name,
        "Param_File_Path": str(config.parameter_file.resolve()),
        "Exp_Type": list(config.experimental_types),
        "UMAT_Stress": config.umat_types[1],
        "UMAT_Strain": config.umat_types[0],
        "Cost_Type": config.cost_type,
        "Normalized_Stress_Floor_Factor": config.normalized_stress_floor_factor,
        "Min_Cost": candidate.direct_mean_cost,
        "Code": code,
    }
    results = {key: [value] * len(names) for key, value in common.items()}
    results["Parameter"] = names
    results["Opt_Value"] = [candidate.parameters[name] for name in names]
    return results


def separable_rows(
    code: int,
    candidate: ExportCandidate,
    config: CaseConfiguration,
    run_group: str,
    created_at: str,
    hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Return one deterministic result-table row per material parameter."""
    rows = []
    for name, value in candidate.parameters.items():
        specification = candidate.parameter_specs[name]
        rows.append(
            {
                "Code": code,
                "Created_At": created_at,
                "Optimizer": OPTIMIZER_NAME,
                "Run_Group": run_group,
                "Case_ID": config.case_id,
                "Run_Label": config.run_label,
                "Result_Role": candidate.result_role,
                "Model_ID": config.model_id,
                "UMAT": config.umat_source.name,
                "Exp_Data": config.experimental_data.name,
                "Exp_Data_Path": str(config.experimental_data.resolve()),
                "Exp_Data_Type": config.data_type,
                "Yarn_Type": config.yarn_type,
                "Knit": config.knit,
                "Knit_Token": config.knit_token,
                "Sample_Folder": config.sample_folder,
                "Param_File": config.parameter_file.name,
                "Param_File_Path": str(config.parameter_file.resolve()),
                "Exp_Type": list(config.experimental_types),
                "UMAT_Stress": config.umat_types[1],
                "UMAT_Strain": config.umat_types[0],
                "Cost_Type": config.cost_type,
                "Normalized_Stress_Floor_Factor": config.normalized_stress_floor_factor,
                "Fit_Axes": list(config.fit_axes),
                "Parameter": name,
                "Initial_Guess": specification["initial_guess"],
                "Min": specification["minimum"],
                "Max": specification["maximum"],
                "Fixed": bool(specification["fixed"]),
                "Opt_Value": value,
                "Active_Bound": active_bound_for_parameter(name, candidate),
                "Identifiability_Status": identifiability_status(name, candidate),
                "Raw_Cost": candidate.direct_raw_cost,
                "Mean_Cost": candidate.direct_mean_cost,
                "Fitted_Observation_Count": candidate.fitted_observation_count,
                "Profiled_Evaluations": candidate.profiled_evaluations,
                "Elapsed_Seconds": candidate.elapsed_seconds,
                "Input_Config_Path": str(config.path.resolve()),
                "Input_Config_SHA256": hashes["case_config"],
                "Parameter_File_SHA256": hashes["material_parameters"],
                "Experimental_Data_SHA256": hashes["experimental_data"],
                "UMAT_Source_SHA256": hashes["umat_source"],
            }
        )
    return rows


def export_candidates(
    candidates: Sequence[ExportCandidate],
    config: CaseConfiguration,
    raw_parameter_specs: Mapping[str, Mapping[str, Any]],
    run_diagnostics: Mapping[str, Any],
    hashes: Mapping[str, str],
    effective_parameter_specs: Mapping[str, Mapping[str, Any]] | None = None,
    run_group: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Reserve codes, snapshot inputs, and complete both result stores."""
    if effective_parameter_specs is None:
        _, resolved_effective_specs = parameter_specs_from_file(
            config,
            load_model_spec(config.model_id),
        )
        effective_parameter_specs = resolved_effective_specs
    definition = load_separable_definition(
        config.separable_definition,
        config.model_id,
    )
    if definition.backend == "scip_sos1":
        assert config.certified_study is not None
        optimizer_payload = {
            "method": OPTIMIZER_NAME,
            "algorithm": run_diagnostics.get(
                "algorithm",
                "certified discrete observable-amplitude optimization",
            ),
            "certification_backend": (
                config.certified_study.certification_backend
            ),
            "production_amplitude_ceiling": (
                config.certified_study.production_amplitude_ceiling
            ),
            "relative_gap": config.certified_study.relative_gap,
            "time_limit_seconds": (
                config.certified_study.total_time_limit_seconds
            ),
        }
        selected_amplitude_ceiling = (
            config.certified_study.production_amplitude_ceiling
        )
        inactive_linear_parameters = list(
            config.certified_study.inactive_linear_parameters
        )
    else:
        optimizer_payload = {
            "method": OPTIMIZER_NAME,
            "algorithm": run_diagnostics.get(
                "algorithm",
                (
                    "deterministic beta/gamma grid, bounded linear profiling, "
                    "and Powell refinement"
                ),
            ),
            **asdict(config.optimization),
        }
        selected_amplitude_ceiling = config.amplitude_ceiling
        inactive_linear_parameters = []
    if run_group is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_group = f"separable_{config.case_id}_{timestamp}"
    elif re.fullmatch(r"[A-Za-z0-9_.-]+", run_group) is None:
        raise ValueError(
            "run_group may contain only letters, numbers, dots, dashes, and underscores"
        )
    summary_directory = RUN_SUMMARY_ROOT / run_group
    if summary_directory.exists():
        raise FileExistsError(
            f"The separable-solver run group already exists: {summary_directory}"
        )
    registry_rows = tuple(
        {
            "Model_ID": config.model_id,
            "Knit": config.knit,
            "Yarn_Type": config.yarn_type,
            "Sample_Folder": config.sample_folder,
            "Exp_Data_Type": config.data_type,
            "Exp_Data_Path": str(config.experimental_data.resolve()),
            "Cost_Type": config.cost_type,
            "Fit_Axes": list(config.fit_axes),
            "Objective_Value": candidate.direct_mean_cost,
            "Result_Role": candidate.result_role,
            "Run_Group": run_group,
            "Case_ID": config.case_id,
            "Run_Label": config.run_label,
            "Input_Config_Path": str(config.path.resolve()),
            "Input_Config_SHA256": hashes["case_config"],
        }
        for candidate in candidates
    )
    codes = reserve_fit_codes(
        len(candidates),
        OPTIMIZER_NAME,
        LEGACY_RESULTS_CSV,
        metadata=registry_rows,
    )
    completed: list[dict[str, Any]] = []
    for index, (code, candidate) in enumerate(zip(codes, candidates)):
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            result_columns = artifact_results(code, candidate, config)
            snapshot_paths = snapshot_inputs(
                code,
                config,
                candidate,
                effective_parameter_specs,
                run_diagnostics,
            )
            input_files = {
                "case_config": {
                    "source_path": str(config.path.resolve()),
                    "snapshot_path": snapshot_paths["case_config"],
                    "sha256": hashes["case_config"],
                },
                "material_parameters": {
                    "source_path": str(config.parameter_file.resolve()),
                    "snapshot_path": snapshot_paths["material_parameters"],
                    "sha256": hashes["material_parameters"],
                    "role": "source_parameter_file_before_case_overrides",
                },
                "effective_material_parameters": {
                    "source_path": None,
                    "snapshot_path": snapshot_paths["effective_material_parameters"],
                    "sha256": historical.sha256_file(
                        pathlib.Path(snapshot_paths["effective_material_parameters"])
                    ),
                    "role": "terminal_effective_parameter_specifications",
                },
                "material_parameter_search": {
                    "source_path": None,
                    "snapshot_path": snapshot_paths["material_parameter_search"],
                    "sha256": historical.sha256_file(
                        pathlib.Path(snapshot_paths["material_parameter_search"])
                    ),
                    "role": "authoritative_merged_search_configuration",
                },
                "experimental_data": {
                    "source_path": str(config.experimental_data.resolve()),
                    "snapshot_path": snapshot_paths["experimental_data"],
                    "sha256": hashes["experimental_data"],
                },
                "umat_source": {
                    "source_path": str(config.umat_source.resolve()),
                    "snapshot_path": snapshot_paths["umat_source"],
                    "sha256": hashes["umat_source"],
                },
            }
            additional_payload = {
                "optimizer": optimizer_payload,
                "objective": {
                    "cost_type": config.cost_type,
                    "normalized_stress_floor_factor": (
                        config.normalized_stress_floor_factor
                    ),
                    "fit_axes": list(config.fit_axes),
                    "raw_cost": candidate.direct_raw_cost,
                    "mean_cost": candidate.direct_mean_cost,
                    "fitted_observation_count": candidate.fitted_observation_count,
                },
                "run_group": run_group,
                "case_id": config.case_id,
                "run_label": config.run_label,
                "result_role": candidate.result_role,
                "model_selection": {
                    "model_label": candidate.model_label,
                    "configured_predictive_structure": config.predictive_structure,
                    "selection_basis": config.selection_basis,
                },
                "parameter_provenance": {
                    "snapshot_schema_version": 2,
                    "bounds_source": config.bounds_source,
                    "parameter_overrides": config.parameter_overrides,
                    "coupling_bound_mode": config.coupling_bound_mode,
                    "amplitude_ceiling": selected_amplitude_ceiling,
                    "inactive_linear_parameters": inactive_linear_parameters,
                    "inactive_shape_parameters": list(
                        ()
                        if config.certified_study is None
                        else tuple(
                            shape
                            for coefficient in config.certified_study.inactive_linear_parameters
                            for shape in load_separable_definition(
                                config.separable_definition, config.model_id
                            ).basis_dependencies()[coefficient]
                        )
                    ),
                    "shape_domains": compact_shape_domains(
                        config, effective_parameter_specs
                    ),
                    "active_shape_domains": active_shape_domains(
                        config, effective_parameter_specs
                    ),
                    "fixed_parameters": fixed_parameter_values(
                        effective_parameter_specs
                    ),
                    "raw_parameter_specs": raw_parameter_specs,
                    "effective_search_parameter_specs": effective_parameter_specs,
                },
                "diagnostics": {
                    "solution": asdict(candidate.solution),
                    "direct_validation": candidate.validation,
                    "maximum_prediction_difference": (
                        candidate.maximum_prediction_difference
                    ),
                    "profiled_evaluations": candidate.profiled_evaluations,
                    "elapsed_seconds": candidate.elapsed_seconds,
                    "coupling_upper_bounds": {
                        str(zeta): value
                        for zeta, value in candidate.coupling_upper_bounds.items()
                    },
                    "basis_weighted_rms": {
                        str(zeta): value
                        for zeta, value in candidate.basis_weighted_rms.items()
                    },
                    "run": dict(run_diagnostics),
                },
                "input_manifest": config.payload,
                "input_files": input_files,
                "provenance": {
                    "source_fit_code": None,
                    "theta_policy": "theta1=0 and theta2=90, both fixed",
                },
            }
            json_path, properties_path = export_fit_artifacts(
                code,
                result_columns,
                tuple_parameter_specs(candidate.parameter_specs),
                LEGACY_RESULTS_CSV,
                model_id=config.model_id,
                additional_payload=additional_payload,
            )
            append_separable_fit_results(
                separable_rows(
                    code,
                    candidate,
                    config,
                    run_group,
                    created_at,
                    hashes,
                ),
                SEPARABLE_RESULTS_CSV,
            )
            update_fit_registration(
                code,
                "complete",
                OPTIMIZER_NAME,
                LEGACY_RESULTS_CSV,
                values={
                    **registry_rows[index],
                    "Artifact_Path": str(json_path.resolve()),
                },
            )
            completed.append(
                {
                    "code": code,
                    "result_role": candidate.result_role,
                    "model_label": candidate.model_label,
                    "mean_cost": candidate.direct_mean_cost,
                    "fit_json": str(json_path.resolve()),
                    "optimized_properties": str(properties_path.resolve()),
                }
            )
        except BaseException:
            update_fit_registration(
                code,
                "failed",
                OPTIMIZER_NAME,
                LEGACY_RESULTS_CSV,
                values=registry_rows[index],
            )
            for unused_code, unused_row in zip(
                codes[index + 1 :], registry_rows[index + 1 :]
            ):
                update_fit_registration(
                    unused_code,
                    "failed",
                    OPTIMIZER_NAME,
                    LEGACY_RESULTS_CSV,
                    values=unused_row,
                )
            raise

    summary_directory.mkdir(parents=True, exist_ok=False)
    summary_path = summary_directory / "summary.json"
    summary = {
        "schema_version": 1,
        "run_group": run_group,
        "case_id": config.case_id,
        "run_label": config.run_label,
        "optimizer": OPTIMIZER_NAME,
        "input_config": str(config.path.resolve()),
        "input_hashes": dict(hashes),
        "outputs": completed,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")
    return tuple(completed)


def print_effective_inputs(
    config: CaseConfiguration,
    raw_specs: Mapping[str, Mapping[str, Any]],
    effective_specs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Print direct paths, precedence, and the complete effective parameter table."""
    print(f"Case: {config.case_id}")
    print(f"Run label: {config.run_label}")
    print(f"Model: {config.model_id} ({config.separable_definition})")
    print(f"Workflow: {config.workflow}")
    print(f"UMAT source: {config.umat_source}")
    print(f"Parameter file: {config.parameter_file}")
    print(f"Experimental data: {config.experimental_data}")
    print(f"Experimental/UMAT types: {config.experimental_types} / {config.umat_types}")
    print(f"Objective: {config.cost_type}; axes={config.fit_axes}")
    print(f"Bound mode: {config.coupling_bound_mode}")
    if config.amplitude_ceiling is not None:
        print(f"Coupling amplitude ceiling: {config.amplitude_ceiling:g}")
    if config.certified_study is not None:
        print(
            "Certification backend: "
            f"{config.certified_study.certification_backend}"
        )
        print(
            "Structurally inactive nonlinear families: "
            + (
                ", ".join(config.certified_study.inactive_linear_parameters)
                or "none"
            )
        )
        if config.certified_study.production_amplitude_ceiling is not None:
            print(
                "Production observable-amplitude ceiling: "
                f"{config.certified_study.production_amplitude_ceiling:g}"
            )
    print(
        "Predictive structure: "
        + (config.predictive_structure or "not requested")
    )
    print("\nEffective parameter specifications")
    print("Name        Guess          Minimum        Maximum        Fixed  Source")
    for name, specification in effective_specs.items():
        source = "JSON override" if specification != raw_specs[name] else "parameter .dat"
        print(
            f"{name:<11} {str(specification['initial_guess']):<14} "
            f"{str(specification['minimum']):<14} {str(specification['maximum']):<14} "
            f"{str(specification['fixed']):<6} {source}"
        )
    print(
        "\nFree-parameter initial guesses are recorded but do not seed the "
        "deterministic grid or bounded linear solve."
    )
    if config.coupling_bound_mode == "amplitude_equalized":
        definition = load_separable_definition(
            config.separable_definition,
            config.model_id,
        )
        dependency_count = len(definition.basis_dependencies().get("K_coup", ()))
        print(
            "K_coup maximum from the parameter .dat will be replaced by "
            + (
                "shape-specific amplitude-equalized bounds."
                if dependency_count > 1
                else "zeta-specific amplitude-equalized bounds."
            )
        )


def run_case(
    case_path: pathlib.Path = CASE_FILE,
    run_group: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Solve one direct-input case and return its completed fit records."""
    config = load_case_configuration(case_path)
    model_spec = load_model_spec(config.model_id)
    raw_specs, effective_specs = parameter_specs_from_file(config, model_spec)
    case_preflight(config, effective_specs, require_production=True)
    print_effective_inputs(config, raw_specs, effective_specs)
    hashes = input_hashes(config)
    initialize_separable_results_table(SEPARABLE_RESULTS_CSV)
    umat.compile_umat_meson(config.umat_source)
    candidates, diagnostics = solve_export_candidates(
        config,
        model_spec,
        effective_specs,
    )
    require_unchanged_inputs(config, hashes)
    outputs = export_candidates(
        candidates,
        config,
        raw_specs,
        diagnostics,
        hashes,
        effective_parameter_specs=effective_specs,
        run_group=run_group,
    )

    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    """Select one case, solve it, and print its exported global fit codes."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=pathlib.Path,
        default=CASE_FILE,
        help=f"Direct-input case JSON (default: {CASE_FILE}).",
    )
    parser.add_argument(
        "--run-group",
        help=(
            "Optional stable run-group identifier. Batch runners use this to "
            "recover completed exports safely."
        ),
    )
    args = parser.parse_args(argv)
    outputs = run_case(args.case, run_group=args.run_group)

    print("\nCompleted deterministic fits", flush=True)
    for output in outputs:
        print(
            f"- fit {output['code']}: {output['result_role']} / "
            f"{output['model_label']}, mean cost={output['mean_cost']:.12g}",
            flush=True,
        )
        print(
            f"  Plot by setting DEFAULT_FIT_CODE = {output['code']} in the numbered calibration plotters.",
            flush=True,
        )
    print(f"Deterministic table: {SEPARABLE_RESULTS_CSV}", flush=True)


if __name__ == "__main__":
    main()
