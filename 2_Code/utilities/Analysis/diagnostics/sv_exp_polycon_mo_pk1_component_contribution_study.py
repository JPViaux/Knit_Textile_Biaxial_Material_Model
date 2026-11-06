"""Calculate fixed-fit SV_exp_polycon_mo component contributions in PK1 and PK2.

Edit the USER CONFIGURATION block and press Run in VS Code. The script reads
the four reviewed all-nine-ratio fits, validates their immutable inputs,
decomposes their stresses without refitting, and exports reproducible tables.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

import argparse
import csv
from dataclasses import dataclass
import hashlib
from release_paths import json
import pathlib
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORT_DIR = next(
    parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis"
).parent
CODE_DIR = SUPPORT_DIR.parent
for directory in (SUPPORT_DIR, CODE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from Analysis.core import analyzer_common as analyzer  # noqa: E402
from Analysis.core import error_calculator as errors  # noqa: E402
from Analysis.core.fit_results import DEFAULT_RESULTS_CSV, FitResult, load_fit  # noqa: E402
from Python_Wrapper import mech, umat  # noqa: E402
from Python_Wrapper.model_registry import load_model_spec  # noqa: E402


# ---------------------------------------------------------------------------
# USER CONFIGURATION - edit these values before pressing Run in VS Code.
# ---------------------------------------------------------------------------

# Default plotting scope; both "per_ratio" and "all_ratios" are exported.
NORMALIZATION_SCOPE = "per_ratio"
# True checks the analytic PK1/PK2 totals against the compiled Fortran UMAT.
VALIDATE_WITH_COMPILED_UMAT = True
# True replaces this combined study's outputs; False protects a completed study.
OVERWRITE_EXISTING = False
# Agreement tolerance in model stress units; example: 1e-8.
UMAT_VALIDATION_TOLERANCE = 1.0e-8
# Identity-deformation tolerance used to exclude origins; example: 1e-12.
ORIGIN_TOLERANCE = 1.0e-12


# ---------------------------------------------------------------------------
# FIXED STUDY DEFINITIONS - normally no changes are needed below this line.
# ---------------------------------------------------------------------------

MODEL_ID = "SV_exp_polycon_mo"
STRESS_MEASURES = ("PK1", "PK2")
NORMALIZATION_SCOPES = ("per_ratio", "all_ratios")
KNIT_ORDER = ("Interlock", "Rib", "Purl", "SingleJersey")
FIT_CODES = {
    "Interlock": 1962,
    "Rib": 1961,
    "Purl": 1963,
    "SingleJersey": 1964,
}
SAMPLE_FOLDERS = {
    "Interlock": "BRT_15mm_Tri_Int_S06",
    "Rib": "BRT_15mm_Tri_Rib_S03",
    "Purl": "BRT_15mm_Tri_Purl_S02",
    "SingleJersey": "BRT_15mm_Tri_SJ_S02",
}
EXPECTED_CASE_IDS = {
    "Interlock": "sv_exp_polycon_mo_interlock_s06_raw_all",
    "Rib": "sv_exp_polycon_mo_rib_s03_raw_all",
    "Purl": "sv_exp_polycon_mo_purl_s02_raw_all",
    "SingleJersey": "sv_exp_polycon_mo_singlejersey_s02_raw_all",
}
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
# Source concatenation order used by the reviewed all-nine raw files.
TRAINING_ORDER = (
    "0.25:1",
    "0.5:1",
    "0.75:1",
    "0:1",
    "1:0",
    "1:0.25",
    "1:0.5",
    "1:0.75",
    "1:1",
)
YARN_TYPE = "Tripoly"
REGISTRY_PATH = RELEASE_ARCHIVE_ROOT / "fit_registry.csv"
OUTPUT_DIRECTORY = (
    RELEASE_ARCHIVE_ROOT / "model_component_contributions_sv_exp_polycon_mo"
)
OUTPUT_CSV = OUTPUT_DIRECTORY / "component_stress_contributions.csv"
OUTPUT_README = OUTPUT_DIRECTORY / "README.md"
OUTPUT_PAYLOAD = OUTPUT_DIRECTORY / "workbook_payload.json"
OUTPUT_WORKBOOK = OUTPUT_DIRECTORY / "component_stress_contributions.xlsx"

COMPONENT_ORDER = {
    "linear_course": 0,
    "linear_wales": 1,
    "linear_course_wales_coupling": 2,
    "nonlinear_course": 3,
    "nonlinear_wales": 4,
    "nonlinear_course_wales_coupling": 5,
}
CSV_COLUMNS = (
    "model_id",
    "fit_code",
    "knit",
    "ratio",
    "stress_measure",
    "normalization_scope",
    "course_reference_peak",
    "wales_reference_peak",
    "component_id",
    "component_name",
    "Linear/Nonlinear",
    "Direction",
    "component_status",
    "parameter_names",
    "stress_contribution_percent",
    "course_stress_contribution_percent",
    "wales_stress_contribution_percent",
    "normalized_absolute_stress_magnitude",
    "mean_abs_stress_course",
    "mean_abs_stress_wales",
    "peak_abs_stress_course",
    "peak_abs_stress_wales",
    "evaluated_row_count",
    "excluded_origin_row_count",
)
NUMERIC_OUTPUT_COLUMNS = CSV_COLUMNS[6:8] + CSV_COLUMNS[14:]


@dataclass(frozen=True)
class FitConfiguration:
    """Identify one reviewed fit; example: Rib fit 1961."""

    knit: str
    fit_code: int
    sample_folder: str
    case_id: str


@dataclass(frozen=True)
class RatioDataset:
    """Store one raw loading curve and its PK1/PK2-ready histories."""

    ratio: str
    path: pathlib.Path
    data: np.ndarray
    dfgrd_voigt: np.ndarray
    experimental_pk1: np.ndarray
    experimental_pk2: np.ndarray
    origin_mask: np.ndarray
    axis_directions: tuple[str, str]


@dataclass(frozen=True)
class ComponentDefinition:
    """Map one mathematical term to its physical contribution label."""

    term_key: str
    component_id: str
    component_name: str
    coefficient_name: str
    shape_names: tuple[str, ...] = ()

    @property
    def parameter_names(self) -> str:
        """Return stable parameter provenance; example: K_coup;zeta;a;b."""
        return ";".join((self.coefficient_name, *self.shape_names))


def validate_normalization_scope(scope: str) -> str:
    """Validate the editable normalization setting."""
    if scope not in NORMALIZATION_SCOPES:
        raise ValueError(
            f"NORMALIZATION_SCOPE must be one of {NORMALIZATION_SCOPES}, got {scope!r}"
        )
    return scope


def validate_stress_measure(stress_measure: str) -> str:
    """Validate a stress selector; examples: ``"PK1"`` or ``"PK2"``."""
    if stress_measure not in STRESS_MEASURES:
        raise ValueError(
            f"stress_measure must be one of {STRESS_MEASURES}, got {stress_measure!r}"
        )
    return stress_measure


def study_configurations() -> tuple[FitConfiguration, ...]:
    """Return the four requested fits in physical knit order."""
    return tuple(
        FitConfiguration(
            knit=knit,
            fit_code=FIT_CODES[knit],
            sample_folder=SAMPLE_FOLDERS[knit],
            case_id=EXPECTED_CASE_IDS[knit],
        )
        for knit in KNIT_ORDER
    )


def ratio_display_label(ratio: str) -> str:
    """Convert a compact ratio such as 0.25:1 to the raw-data label."""
    return ratio.replace(":", " to ")


def sha256_file(path: pathlib.Path) -> str:
    """Hash one immutable input; example: a saved fit snapshot."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registry_rows() -> dict[int, dict[str, str]]:
    """Load the global fit registry by integer fit code."""
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Fit registry was not found: {REGISTRY_PATH}")
    with REGISTRY_PATH.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        raw_code = str(row.get("Code", "")).strip()
        if not raw_code:
            continue
        code = int(float(raw_code))
        if code in indexed:
            raise ValueError(f"Fit registry contains duplicate code {code}")
        indexed[code] = row
    return indexed


def load_and_validate_fit(
    config: FitConfiguration,
    registry: Mapping[int, Mapping[str, str]],
) -> tuple[FitResult, dict[str, Any]]:
    """Validate one all-nine fit and its recorded immutable input hashes."""
    row = registry.get(config.fit_code)
    if row is None:
        raise ValueError(f"Fit {config.fit_code} is missing from the registry")
    expected_registry = {
        "Status": "complete",
        "Model_ID": MODEL_ID,
        "Knit": config.knit,
        "Case_ID": config.case_id,
    }
    for column, expected in expected_registry.items():
        if row.get(column) != expected:
            raise ValueError(
                f"Fit {config.fit_code} has {column}={row.get(column)!r}; expected {expected!r}"
            )

    fit = load_fit(config.fit_code, DEFAULT_RESULTS_CSV)
    if fit.source.name != "fit.json":
        raise ValueError(f"Fit {config.fit_code} must resolve to its JSON artifact")
    payload = json.loads(fit.source.read_text(encoding="utf-8"))
    if fit.model != MODEL_ID or payload.get("model_id") != MODEL_ID:
        raise ValueError(f"Fit {config.fit_code} does not belong to {MODEL_ID}")
    if fit.metadata.get("Knit") != config.knit:
        raise ValueError(f"Fit {config.fit_code} does not belong to {config.knit}")
    if payload.get("case_id") != config.case_id:
        raise ValueError(f"Fit {config.fit_code} is not the requested raw all-nine case")
    if payload.get("result_role") != "training_optimum":
        raise ValueError(f"Fit {config.fit_code} is not a training optimum")
    if payload.get("objective", {}).get("fit_axes") != [0, 1]:
        raise ValueError(f"Fit {config.fit_code} was not calibrated on axes 0 and 1")

    model_spec = load_model_spec(MODEL_ID)
    missing = [name for name in model_spec.parameter_names if name not in fit.parameters]
    if missing:
        raise ValueError(f"Fit {config.fit_code} is missing parameters: {missing}")
    fixed_values = (("C1", 1.0), ("C2", 1.0), ("theta1", 0.0), ("theta2", 90.0))
    for name, expected in fixed_values:
        if not np.isclose(float(fit.parameters[name]), expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Fit {config.fit_code} violates the fixed {name} policy")
    if not np.isclose(
        float(fit.parameters["a"]) + float(fit.parameters["b"]),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"Fit {config.fit_code} violates a+b=1")

    input_files = payload.get("input_files")
    if not isinstance(input_files, dict):
        raise ValueError(f"Fit {config.fit_code} has no input_files provenance")
    for label in ("case_config", "material_parameters", "experimental_data", "umat_source"):
        record = input_files.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"Fit {config.fit_code} is missing {label} provenance")
        expected_hash = str(record.get("sha256", ""))
        for path_key in ("source_path", "snapshot_path"):
            path = pathlib.Path(str(record.get(path_key, "")))
            if not path.is_file():
                raise FileNotFoundError(
                    f"Fit {config.fit_code} {label} {path_key} is missing: {path}"
                )
            if sha256_file(path) != expected_hash:
                raise ValueError(
                    f"Fit {config.fit_code} {label} {path_key} hash changed"
                )
    return fit, payload


def identity_origin_mask(
    dfgrd_voigt: np.ndarray,
    tolerance: float = ORIGIN_TOLERANCE,
) -> np.ndarray:
    """Identify all identity-deformation rows, including duplicate origins."""
    identity = np.eye(3)
    return np.asarray(
        [
            np.allclose(
                mech.voigt_to_matrix(f_voigt), identity, rtol=0.0, atol=tolerance
            )
            for f_voigt in dfgrd_voigt
        ],
        dtype=bool,
    )


def pk1_history_to_pk2(
    dfgrd_voigt: np.ndarray,
    pk1_voigt: np.ndarray,
) -> np.ndarray:
    """Convert one experimental PK1 history to PK2 at the measured states."""
    converted = []
    for f_voigt, p_voigt in zip(dfgrd_voigt, pk1_voigt, strict=True):
        F = mech.voigt_to_matrix(f_voigt)
        P = mech.voigt_to_matrix(p_voigt)
        converted.append(mech.matrix_to_voigt(mech.pk1_to_pk2(P, F)))
    return np.asarray(converted, dtype=float)


def cauchy_history_to_pk2(
    dfgrd_voigt: np.ndarray,
    cauchy_voigt: np.ndarray,
) -> np.ndarray:
    """Convert one compiled-UMAT Cauchy history to PK2."""
    converted = []
    for f_voigt, sigma_voigt in zip(dfgrd_voigt, cauchy_voigt, strict=True):
        F = mech.voigt_to_matrix(f_voigt)
        sigma = mech.voigt_to_matrix(sigma_voigt)
        converted.append(mech.matrix_to_voigt(mech.cauchystress_to_pk2(sigma, F)))
    return np.asarray(converted, dtype=float)


def load_ratio_datasets(config: FitConfiguration) -> dict[str, RatioDataset]:
    """Load all nine authoritative raw curves in experimental PK1 and PK2."""
    datasets: dict[str, RatioDataset] = {}
    expected_directions: tuple[str, str] | None = None
    for ratio in RATIO_ORDER:
        display_ratio = ratio_display_label(ratio)
        path = errors.experimental_path(
            config.knit,
            display_ratio,
            None,
            "raw",
            YARN_TYPE,
            config.sample_folder,
        )
        data = np.atleast_2d(np.loadtxt(path, dtype=float))
        if data.ndim != 2 or data.shape[1] != 12 or not np.all(np.isfinite(data)):
            raise ValueError(f"Expected a finite 12-column ratio file: {path}")
        directions = errors.axis_directions_for_ratio(
            config.knit, display_ratio, config.sample_folder
        )
        if set(directions) != {"course", "wales"}:
            raise ValueError(f"Ratio {ratio} has invalid directions {directions}")
        if expected_directions is None:
            expected_directions = directions
        elif directions != expected_directions:
            raise ValueError(f"Axis directions change within {config.knit}: {directions}")
        dfgrd_voigt = umat.strains_to_dfgrd(data[:, :6], "H")
        experimental_pk1 = np.asarray(data[:, 6:12], dtype=float)
        datasets[ratio] = RatioDataset(
            ratio=ratio,
            path=path,
            data=data,
            dfgrd_voigt=dfgrd_voigt,
            experimental_pk1=experimental_pk1,
            experimental_pk2=pk1_history_to_pk2(dfgrd_voigt, experimental_pk1),
            origin_mask=identity_origin_mask(dfgrd_voigt),
            axis_directions=directions,
        )
    return datasets


def verify_training_data(
    config: FitConfiguration,
    payload: Mapping[str, Any],
    datasets: Mapping[str, RatioDataset],
) -> None:
    """Match the saved all-nine training array to the nine source curves exactly."""
    source_path = pathlib.Path(
        str(payload["input_files"]["experimental_data"]["source_path"])
    )
    training = np.atleast_2d(np.loadtxt(source_path, dtype=float))
    expected = np.vstack([datasets[ratio].data for ratio in TRAINING_ORDER])
    if not np.array_equal(training, expected):
        raise ValueError(
            f"Fit {config.fit_code} training data do not match the nine raw curves"
        )


def axis_index_by_direction(axis_directions: tuple[str, str]) -> dict[str, int]:
    """Return the in-plane Voigt index for course and wales."""
    mapping = {direction: index for index, direction in enumerate(axis_directions)}
    if set(mapping) != {"course", "wales"}:
        raise ValueError(f"Expected one course and one wales axis: {axis_directions}")
    return mapping


def directional_component(
    term_key: str,
    coefficient_name: str,
    shape_names: tuple[str, ...],
    kind: str,
    direction: str,
) -> ComponentDefinition:
    """Build one component definition; example: nonlinear course."""
    if direction not in {"course", "wales"}:
        raise ValueError(f"Unknown material direction: {direction}")
    component_id = f"{kind}_{direction}"
    component_name = f"{kind.capitalize()} {direction}"
    return ComponentDefinition(
        term_key,
        component_id,
        component_name,
        coefficient_name,
        shape_names,
    )


def component_definitions(
    axis_directions: tuple[str, str],
) -> tuple[ComponentDefinition, ...]:
    """Return the six additive model terms labeled by physical direction."""
    family_1, family_2 = axis_directions
    definitions = (
        directional_component("mu_1", "mu", (), "linear", family_1),
        directional_component("mu_2", "mu2", (), "linear", family_2),
        ComponentDefinition(
            "mu_3",
            "linear_course_wales_coupling",
            "Linear course-wales coupling",
            "mu3",
        ),
        directional_component(
            "K_1ani1", "K_1ani1", ("beta_1",), "nonlinear", family_1
        ),
        directional_component(
            "K_1ani2", "K_1ani2", ("gamma_1",), "nonlinear", family_2
        ),
        ComponentDefinition(
            "K_coup",
            "nonlinear_course_wales_coupling",
            "Nonlinear course-wales coupling",
            "K_coup",
            ("zeta", "a", "b"),
        ),
    )
    return tuple(sorted(definitions, key=lambda item: COMPONENT_ORDER[item.component_id]))


def reference_peaks(
    datasets: Mapping[str, RatioDataset],
    scope: str,
    stress_measure: str,
) -> dict[str, dict[str, float]]:
    """Return directional experimental peaks for one measure and scope."""
    validate_normalization_scope(scope)
    validate_stress_measure(stress_measure)
    axes = axis_index_by_direction(datasets[RATIO_ORDER[0]].axis_directions)
    per_ratio: dict[str, dict[str, float]] = {}
    for ratio in RATIO_ORDER:
        dataset = datasets[ratio]
        included = ~dataset.origin_mask
        experimental = experimental_stress_history(dataset, stress_measure)
        per_ratio[ratio] = {
            direction: float(
                np.max(np.abs(experimental[included, axis]))
            )
            for direction, axis in axes.items()
        }
    if scope == "per_ratio":
        return per_ratio
    pooled = {
        direction: max(per_ratio[ratio][direction] for ratio in RATIO_ORDER)
        for direction in ("course", "wales")
    }
    return {ratio: dict(pooled) for ratio in RATIO_ORDER}


def experimental_stress_history(
    dataset: RatioDataset,
    stress_measure: str,
) -> np.ndarray:
    """Return the measured stress history selected for normalization."""
    validate_stress_measure(stress_measure)
    return (
        dataset.experimental_pk1
        if stress_measure == "PK1"
        else dataset.experimental_pk2
    )


def decompose_stress_history(
    dataset: RatioDataset,
    parameters: Mapping[str, float | int],
    definitions: Sequence[ComponentDefinition],
    stress_measure: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Evaluate six additive histories and their total in PK1 or PK2."""
    validate_stress_measure(stress_measure)
    histories = {definition.term_key: [] for definition in definitions}
    totals = []
    maximum_difference = 0.0
    for f_voigt in dataset.dfgrd_voigt:
        F = mech.voigt_to_matrix(f_voigt)
        pieces = (
            analyzer.stress_decomposition(F, parameters, MODEL_ID)
            if stress_measure == "PK1"
            else analyzer.stress_decomposition_pk2(F, parameters, MODEL_ID)
        )
        component_sum = np.zeros((3, 3), dtype=float)
        for definition in definitions:
            term = np.asarray(pieces[definition.term_key], dtype=float)
            histories[definition.term_key].append(mech.matrix_to_voigt(term))
            component_sum += term
        total = np.asarray(pieces["total"], dtype=float)
        maximum_difference = max(
            maximum_difference, float(np.max(np.abs(component_sum - total)))
        )
        totals.append(mech.matrix_to_voigt(total))
    return (
        {key: np.asarray(values, dtype=float) for key, values in histories.items()},
        np.asarray(totals, dtype=float),
        maximum_difference,
    )


def direct_umat_difference(
    dataset: RatioDataset,
    parameters: Mapping[str, float | int],
    decomposed_total: np.ndarray,
    stress_measure: str,
) -> float:
    """Return the maximum selected-stress difference from the compiled UMAT."""
    validate_stress_measure(stress_measure)
    model_spec = load_model_spec(MODEL_ID)
    props = np.asarray(model_spec.pack_parameters(parameters), dtype=float)
    cauchy = analyzer.call_umat_history(dataset.dfgrd_voigt, props)
    direct = (
        analyzer.cauchy_history_to_pk1(dataset.dfgrd_voigt, cauchy)
        if stress_measure == "PK1"
        else cauchy_history_to_pk2(dataset.dfgrd_voigt, cauchy)
    )
    return float(np.max(np.abs(direct - decomposed_total)))


def component_classifications(definition: ComponentDefinition) -> tuple[str, str]:
    """Return constitutive-family and physical-direction labels."""
    family = "Linear" if definition.component_id.startswith("linear_") else "Nonlinear"
    if "coupling" in definition.component_id:
        direction = "Coupling"
    elif definition.component_id.endswith("_course"):
        direction = "Course"
    else:
        direction = "Wales"
    return family, direction


def safe_percentage(numerator: float, denominator: float) -> float:
    """Return a finite percentage, using zero for a stress-free denominator."""
    return 0.0 if denominator == 0.0 else 100.0 * numerator / denominator


def normalized_sum(
    component_sums: Sequence[float],
    reference: float,
    experimental_values: np.ndarray,
    *,
    label: str,
) -> list[float]:
    """Normalize component sums while enforcing the zero-reference contract."""
    if reference > 0.0:
        return [value / reference for value in component_sums]
    model_zero = all(value == 0.0 for value in component_sums)
    experiment_zero = not np.any(experimental_values)
    if not (model_zero and experiment_zero):
        raise ValueError(
            f"{label} has a zero experimental reference with nonzero stress"
        )
    return [0.0 for _ in component_sums]


def contribution_rows(
    config: FitConfiguration,
    dataset: RatioDataset,
    parameters: Mapping[str, float | int],
    definitions: Sequence[ComponentDefinition],
    histories: Mapping[str, np.ndarray],
    peaks: Mapping[str, float],
    scope: str,
    stress_measure: str,
) -> list[dict[str, Any]]:
    """Aggregate one ratio into one output row per constitutive component."""
    validate_stress_measure(stress_measure)
    included = ~dataset.origin_mask
    evaluated_count = int(np.count_nonzero(included))
    if evaluated_count == 0:
        raise ValueError(f"Ratio {dataset.ratio} contains only identity rows")
    axes = axis_index_by_direction(dataset.axis_directions)
    experimental = experimental_stress_history(dataset, stress_measure)
    course_values = [
        histories[item.term_key][included, axes["course"]] for item in definitions
    ]
    wales_values = [
        histories[item.term_key][included, axes["wales"]] for item in definitions
    ]
    course_sums = [float(np.sum(np.abs(values))) for values in course_values]
    wales_sums = [float(np.sum(np.abs(values))) for values in wales_values]
    normalized_course = normalized_sum(
        course_sums,
        float(peaks["course"]),
        experimental[included, axes["course"]],
        label=f"{config.knit} {dataset.ratio} course",
    )
    normalized_wales = normalized_sum(
        wales_sums,
        float(peaks["wales"]),
        experimental[included, axes["wales"]],
        label=f"{config.knit} {dataset.ratio} wales",
    )
    normalized_combined = [
        course + wales for course, wales in zip(normalized_course, normalized_wales)
    ]
    combined_denominator = sum(normalized_combined)
    course_denominator = sum(course_sums)
    wales_denominator = sum(wales_sums)
    if combined_denominator <= 0.0:
        raise ValueError(f"Ratio {dataset.ratio} has no nonzero component stress")

    rows = []
    for index, definition in enumerate(definitions):
        family, direction = component_classifications(definition)
        coefficient = float(parameters[definition.coefficient_name])
        magnitude = normalized_combined[index]
        status = (
            "fitted_zero"
            if coefficient == 0.0
            else "inactive_on_ratio" if magnitude == 0.0 else "contributing"
        )
        rows.append(
            {
                "model_id": MODEL_ID,
                "fit_code": config.fit_code,
                "knit": config.knit,
                "ratio": dataset.ratio,
                "stress_measure": stress_measure,
                "normalization_scope": scope,
                "course_reference_peak": float(peaks["course"]),
                "wales_reference_peak": float(peaks["wales"]),
                "component_id": definition.component_id,
                "component_name": definition.component_name,
                "Linear/Nonlinear": family,
                "Direction": direction,
                "component_status": status,
                "parameter_names": definition.parameter_names,
                "stress_contribution_percent": safe_percentage(
                    magnitude, combined_denominator
                ),
                "course_stress_contribution_percent": safe_percentage(
                    course_sums[index], course_denominator
                ),
                "wales_stress_contribution_percent": safe_percentage(
                    wales_sums[index], wales_denominator
                ),
                "normalized_absolute_stress_magnitude": magnitude,
                "mean_abs_stress_course": float(np.mean(np.abs(course_values[index]))),
                "mean_abs_stress_wales": float(np.mean(np.abs(wales_values[index]))),
                "peak_abs_stress_course": float(np.max(np.abs(course_values[index]))),
                "peak_abs_stress_wales": float(np.max(np.abs(wales_values[index]))),
                "evaluated_row_count": evaluated_count,
                "excluded_origin_row_count": int(np.count_nonzero(dataset.origin_mask)),
            }
        )
    return rows


def row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    """Return the deterministic output order."""
    return (
        KNIT_ORDER.index(str(row["knit"])),
        STRESS_MEASURES.index(str(row["stress_measure"])),
        NORMALIZATION_SCOPES.index(str(row["normalization_scope"])),
        RATIO_ORDER.index(str(row["ratio"])),
        COMPONENT_ORDER[str(row["component_id"])],
    )


def validate_output_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate completeness, uniqueness, finiteness, and percentage totals."""
    expected_count = (
        len(KNIT_ORDER)
        * len(STRESS_MEASURES)
        * len(NORMALIZATION_SCOPES)
        * len(RATIO_ORDER)
        * len(COMPONENT_ORDER)
    )
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")
    keys = [
        (
            row["fit_code"],
            row["stress_measure"],
            row["normalization_scope"],
            row["ratio"],
            row["component_id"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate fit/stress/scope/ratio/component rows were produced"
        )
    for row in rows:
        for column in NUMERIC_OUTPUT_COLUMNS:
            if not np.isfinite(float(row[column])):
                raise ValueError(f"Nonfinite {column} in row {row}")
    grouped: dict[tuple[int, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                int(row["fit_code"]),
                str(row["stress_measure"]),
                str(row["normalization_scope"]),
                str(row["ratio"]),
            ),
            [],
        ).append(row)
    for key, group in grouped.items():
        if len(group) != len(COMPONENT_ORDER):
            raise ValueError(f"Group {key} does not contain six components")
        combined = sum(float(row["stress_contribution_percent"]) for row in group)
        if not np.isclose(combined, 100.0, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"Combined percentages for {key} sum to {combined}")
        for direction in ("course", "wales"):
            if any(float(row[f"mean_abs_stress_{direction}"]) > 0.0 for row in group):
                total = sum(
                    float(row[f"{direction}_stress_contribution_percent"])
                    for row in group
                )
                if not np.isclose(total, 100.0, rtol=0.0, atol=1.0e-10):
                    raise ValueError(f"{direction} percentages for {key} sum to {total}")


def readme_text(default_scope: str, qa_rows: Sequence[Mapping[str, Any]]) -> str:
    """Build methods, provenance, and validation documentation."""
    maxima = {
        stress_measure: {
            "reconstruction": max(
                float(row["component_reconstruction_difference"])
                for row in qa_rows
                if row["stress_measure"] == stress_measure
            ),
            "umat": max(
                float(row["umat_difference"])
                for row in qa_rows
                if row["stress_measure"] == stress_measure
            ),
        }
        for stress_measure in STRESS_MEASURES
    }
    return "\n".join(
        (
            "# SV_exp_polycon_mo PK1 and PK2 Component Contributions",
            "",
            "This fixed-fit study evaluates fits 1961--1964 without refitting or",
            "modifying production fit records. PK2 terms are evaluated directly;",
            "PK1 terms are converted independently at each state using P_i = F S_i.",
            "",
            "## Normalization",
            "",
            "The output contains both supported normalization scopes.",
            f"The default selector for plotting is `{default_scope}`.",
            "",
            "- `per_ratio` (default): course and wale use their respective absolute",
            "  experimental peaks in the selected stress measure within each ratio.",
            "- `all_ratios`: course and wale use their respective absolute",
            "  experimental peaks in the selected measure across all nine ratios.",
            "",
            "For every non-origin row, each component's absolute directional stress",
            "history is divided by the applicable experimental reference peak.",
            "The two normalized directional magnitudes are added, then divided by",
            "the sum over the six components to obtain the combined percentage.",
            "Course-only and wale-only percentages use their directional sums.",
            "",
            "## Fit inventory",
            "",
            "| Knit | Fit | Ratios |",
            "|---|---:|---:|",
            "| Interlock | 1962 | 9 |",
            "| Rib | 1961 | 9 |",
            "| Purl | 1963 | 9 |",
            "| Single Jersey | 1964 | 9 |",
            "",
            "## Validation",
            "",
            f"- Maximum additive PK1 reconstruction difference: `{maxima['PK1']['reconstruction']:.6e}`.",
            f"- Maximum compiled-UMAT PK1 difference: `{maxima['PK1']['umat']:.6e}`.",
            f"- Maximum additive PK2 reconstruction difference: `{maxima['PK2']['reconstruction']:.6e}`.",
            f"- Maximum compiled-UMAT PK2 difference: `{maxima['PK2']['umat']:.6e}`.",
            f"- Required tolerance: `{UMAT_VALIDATION_TOLERANCE:.6e}`.",
            "- All fit source/snapshot hashes and exact nine-curve training arrays",
            "  were verified before calculation.",
            "",
            "## Interpretation limit",
            "",
            "These percentages describe absolute, direction-normalized additive",
            "stress magnitude along each fixed fitted loading path. They are not",
            "variance explained, parameter-identifiability measures, or ablation",
            "results, and they do not establish that a component can be removed.",
            "",
        )
    )


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the contribution CSV atomically with full float precision."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text_atomic(path: pathlib.Path, content: str) -> None:
    """Write one text artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def execute_study(
    *,
    normalization_scope: str,
    overwrite: bool,
    validate_with_umat: bool,
) -> pathlib.Path:
    """Run the complete four-knit PK1/PK2 study and return its CSV path."""
    default_scope = validate_normalization_scope(normalization_scope)
    # Export both scopes together so the plotter can switch without recalculation.
    scopes = NORMALIZATION_SCOPES
    protected = (OUTPUT_CSV, OUTPUT_README, OUTPUT_PAYLOAD, OUTPUT_WORKBOOK)
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Combined study output already exists; set OVERWRITE_EXISTING = True or use "
            f"--overwrite: {existing}"
        )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if validate_with_umat:
        umat.compile_umat_meson(load_model_spec(MODEL_ID).fortran_source)
    registry = registry_rows()
    rows: list[dict[str, Any]] = []
    qa_rows: list[dict[str, Any]] = []
    fit_inventory: list[dict[str, Any]] = []
    for config in study_configurations():
        print(f"Analyzing {config.knit} fit {config.fit_code} across nine ratios")
        fit, payload = load_and_validate_fit(config, registry)
        datasets = load_ratio_datasets(config)
        verify_training_data(config, payload, datasets)
        peaks_by_measure_scope = {
            (stress_measure, scope): reference_peaks(
                datasets,
                scope,
                stress_measure,
            )
            for stress_measure in STRESS_MEASURES
            for scope in scopes
        }
        definitions = component_definitions(datasets[RATIO_ORDER[0]].axis_directions)
        parameters = {
            name: fit.parameters[name]
            for name in load_model_spec(MODEL_ID).parameter_names
        }
        fit_inventory.append(
            {
                "model_id": MODEL_ID,
                "fit_code": config.fit_code,
                "knit": config.knit,
                "sample_folder": config.sample_folder,
                "case_id": config.case_id,
                "ratio_count": len(RATIO_ORDER),
                "axis_1_direction": datasets[RATIO_ORDER[0]].axis_directions[0],
                "axis_2_direction": datasets[RATIO_ORDER[0]].axis_directions[1],
                "fit_artifact": str(fit.source),
            }
        )
        for ratio in RATIO_ORDER:
            dataset = datasets[ratio]
            for stress_measure in STRESS_MEASURES:
                histories, total, reconstruction_difference = decompose_stress_history(
                    dataset,
                    parameters,
                    definitions,
                    stress_measure,
                )
                if reconstruction_difference > UMAT_VALIDATION_TOLERANCE:
                    raise ValueError(
                        f"{config.knit} {ratio} {stress_measure} reconstruction "
                        f"difference {reconstruction_difference:.6e} exceeds tolerance"
                    )
                umat_difference = 0.0
                if validate_with_umat:
                    umat_difference = direct_umat_difference(
                        dataset,
                        parameters,
                        total,
                        stress_measure,
                    )
                    if umat_difference > UMAT_VALIDATION_TOLERANCE:
                        raise ValueError(
                            f"{config.knit} {ratio} {stress_measure} UMAT difference "
                            f"{umat_difference:.6e} exceeds tolerance"
                        )
                for scope in scopes:
                    peaks = peaks_by_measure_scope[(stress_measure, scope)][ratio]
                    qa_rows.append(
                        {
                            "model_id": MODEL_ID,
                            "fit_code": config.fit_code,
                            "knit": config.knit,
                            "ratio": ratio,
                            "stress_measure": stress_measure,
                            "normalization_scope": scope,
                            "course_reference_peak": peaks["course"],
                            "wales_reference_peak": peaks["wales"],
                            "component_reconstruction_difference": reconstruction_difference,
                            "umat_difference": umat_difference,
                            "validation_tolerance": UMAT_VALIDATION_TOLERANCE,
                        }
                    )
                    rows.extend(
                        contribution_rows(
                            config,
                            dataset,
                            parameters,
                            definitions,
                            histories,
                            peaks,
                            scope,
                            stress_measure,
                        )
                    )

    rows.sort(key=row_sort_key)
    validate_output_rows(rows)
    readme = readme_text(default_scope, qa_rows)
    write_csv(OUTPUT_CSV, rows)
    write_text_atomic(OUTPUT_README, readme)
    write_text_atomic(
        OUTPUT_PAYLOAD,
        json.dumps(
            {
                "schema_version": 2,
                "stress_measures": list(STRESS_MEASURES),
                "default_normalization_scope": default_scope,
                "normalization_scopes": list(scopes),
                "contribution_rows": rows,
                "fit_inventory": fit_inventory,
                "qa_rows": qa_rows,
                "readme_lines": readme.splitlines(),
            },
            indent=2,
        ),
    )
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    return OUTPUT_CSV


def parse_args() -> argparse.Namespace:
    """Parse VS Code-compatible optional command-line overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalization-scope",
        choices=NORMALIZATION_SCOPES,
        default=NORMALIZATION_SCOPE,
        help="Override the documented default; both scopes are still exported.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace this combined study's outputs."
    )
    parser.add_argument(
        "--skip-umat-validation",
        action="store_true",
        help="Debug only: skip compiled-Fortran validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete requested PK1/PK2 study."""
    args = parse_args()
    execute_study(
        normalization_scope=args.normalization_scope,
        overwrite=OVERWRITE_EXISTING or args.overwrite,
        validate_with_umat=(
            VALIDATE_WITH_COMPILED_UMAT and not args.skip_umat_validation
        ),
    )


if __name__ == "__main__":
    main()
