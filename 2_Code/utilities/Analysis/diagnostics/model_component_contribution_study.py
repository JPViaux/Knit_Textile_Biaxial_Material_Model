"""Quantify fixed-fit PK2 stress contributions for the two manuscript models.

Edit the USER CONFIGURATION block and press Run in VSCode. The study reads the
registered prescribed and adaptive all-ratio B09 fits, validates their immutable
inputs, decomposes the model stress on each individual raw loading curve, and
writes one comparison table. It never optimizes parameters or writes
production-fit records.
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
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORT_DIR = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis").parent
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
# USER CONFIGURATION - edit these values before pressing Run in VSCode.
# ---------------------------------------------------------------------------

# All-ratio fit-code groups; example: prescribed Rib SV_exp uses fit 722.
FIT_CODES: dict[str, dict[str, dict[str, int]]] = {
    "prescribed_b09": {
        "SV_exp": {
            "Interlock": 726,
            "Rib": 722,
            "Purl": 723,
            "SingleJersey": 725,
        },
        "StapletonViaux": {
            "Interlock": 724,
            "Rib": 729,
            "Purl": 727,
            "SingleJersey": 728,
        },
    },
    "adaptive_b09": {
        "SV_exp": {
            "Interlock": 1075,
            "Rib": 1072,
            "Purl": 1071,
            "SingleJersey": 1074,
        },
        "StapletonViaux": {
            "Interlock": 1073,
            "Rib": 1078,
            "Purl": 1076,
            "SingleJersey": 1077,
        },
    },
}

# Output folder; example: Outputs/model_component_contributions.
OUTPUT_DIRECTORY = RELEASE_ARCHIVE_ROOT / "model_component_contributions"
# UMAT agreement tolerance in model stress units; example: 1e-8.
UMAT_VALIDATION_TOLERANCE = 1.0e-8
# Identity-deformation tolerance used to exclude origin rows; example: 1e-12.
ORIGIN_TOLERANCE = 1.0e-12
# True validates every decomposition against compiled Fortran; use False only for debugging.
VALIDATE_WITH_COMPILED_UMAT = True
# True replaces this study's CSV/README; False protects an existing completed study.
OVERWRITE_EXISTING = False


# ---------------------------------------------------------------------------
# FIXED STUDY DEFINITIONS - normally no changes are needed below this line.
# ---------------------------------------------------------------------------

MODEL_ORDER = ("SV_exp", "StapletonViaux")
KNIT_ORDER = ("Interlock", "Rib", "Purl", "SingleJersey")
FIT_SERIES_ORDER = ("prescribed_b09", "adaptive_b09")
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
B09_TRAINING_ORDER = (
    "1:1",
    "1:0",
    "0:1",
    "0.5:1",
    "1:0.5",
    "0.25:1",
    "1:0.25",
    "0.75:1",
    "1:0.75",
)
ADAPTIVE_B09_TRAINING_ORDERS = {
    ("StapletonViaux", "Interlock"): (
        "0.25:1", "0.75:1", "1:0.25", "1:1", "1:0", "0:1", "1:0.75", "1:0.5", "0.5:1",
    ),
    ("StapletonViaux", "Purl"): (
        "1:1", "0:1", "1:0.75", "0.5:1", "1:0", "0.75:1", "1:0.5", "1:0.25", "0.25:1",
    ),
    ("StapletonViaux", "Rib"): (
        "1:1", "0.5:1", "1:0.75", "1:0.5", "0.75:1", "1:0.25", "1:0", "0.25:1", "0:1",
    ),
    ("StapletonViaux", "SingleJersey"): (
        "0.75:1", "1:0.75", "0.5:1", "1:0.5", "1:1", "0.25:1", "1:0", "1:0.25", "0:1",
    ),
    ("SV_exp", "Interlock"): (
        "0.75:1", "1:0.25", "1:1", "1:0.5", "1:0", "1:0.75", "0:1", "0.5:1", "0.25:1",
    ),
    ("SV_exp", "Purl"): (
        "1:0.75", "1:1", "1:0", "0.5:1", "0:1", "0.25:1", "1:0.25", "0.75:1", "1:0.5",
    ),
    ("SV_exp", "Rib"): (
        "1:1", "0.5:1", "1:0.75", "1:0", "1:0.5", "1:0.25", "0.75:1", "0:1", "0.25:1",
    ),
    ("SV_exp", "SingleJersey"): (
        "1:0.75", "0.5:1", "0:1", "1:0.25", "1:1", "1:0", "0.25:1", "1:0.5", "0.75:1",
    ),
}
SAMPLE_FOLDERS = {
    "Interlock": "BRT_15mm_Tri_Int_S06",
    "Rib": "BRT_15mm_Tri_Rib_S03",
    "Purl": "BRT_15mm_Tri_Purl_S02",
    "SingleJersey": "BRT_15mm_Tri_SJ_S02",
}
YARN_TYPE = "Tripoly"
REGISTRY_PATH = RELEASE_ARCHIVE_ROOT / "fit_registry.csv"
OUTPUT_CSV = OUTPUT_DIRECTORY / "component_stress_contributions.csv"
OUTPUT_README = OUTPUT_DIRECTORY / "README.md"

COMPONENT_ORDER = {
    "linear_course": 0,
    "linear_wales": 1,
    "linear_course_wales_coupling": 2,
    "nonlinear_course": 3,
    "secondary_nonlinear_course": 4,
    "nonlinear_wales": 5,
    "secondary_nonlinear_wales": 6,
    "nonlinear_course_wales_coupling": 7,
}
CSV_COLUMNS = (
    "model_id",
    "fit_code",
    "knit",
    "ratio",
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
    "mean_abs_pk2_course",
    "mean_abs_pk2_wales",
    "peak_abs_pk2_course",
    "peak_abs_pk2_wales",
    "evaluated_row_count",
    "excluded_origin_row_count",
)
NUMERIC_OUTPUT_COLUMNS = (
    "stress_contribution_percent",
    "course_stress_contribution_percent",
    "wales_stress_contribution_percent",
    "normalized_absolute_stress_magnitude",
    "mean_abs_pk2_course",
    "mean_abs_pk2_wales",
    "peak_abs_pk2_course",
    "peak_abs_pk2_wales",
    "evaluated_row_count",
    "excluded_origin_row_count",
)


@dataclass(frozen=True)
class FitConfiguration:
    """Identify one immutable fit, for example prescribed SV_exp Rib fit 722."""

    model_id: str
    knit: str
    fit_code: int
    sample_folder: str
    fit_series: str = "prescribed_b09"


@dataclass(frozen=True)
class RatioDataset:
    """Store one raw ratio curve and its mechanics conversions."""

    ratio: str
    path: pathlib.Path
    data: np.ndarray
    dfgrd_voigt: np.ndarray
    experimental_pk2: np.ndarray
    origin_mask: np.ndarray
    axis_directions: tuple[str, str]


@dataclass(frozen=True)
class ComponentDefinition:
    """Map one mathematical stress term to a physical material component."""

    term_key: str
    component_id: str
    component_name: str
    coefficient_name: str
    shape_name: str | None = None
    structurally_fixed_zero: bool = False

    @property
    def parameter_names(self) -> str:
        """Return stable semicolon-separated parameter names."""
        names = [self.coefficient_name]
        if self.shape_name is not None:
            names.append(self.shape_name)
        return ";".join(names)


def study_configurations(model_id: str | None = None) -> tuple[FitConfiguration, ...]:
    """Return both deterministic B09 series in stable output order."""
    configurations = []
    for selected_model in MODEL_ORDER:
        if model_id is not None and selected_model != model_id:
            continue
        for knit in KNIT_ORDER:
            for fit_series in FIT_SERIES_ORDER:
                configurations.append(
                    FitConfiguration(
                        model_id=selected_model,
                        knit=knit,
                        fit_code=FIT_CODES[fit_series][selected_model][knit],
                        sample_folder=SAMPLE_FOLDERS[knit],
                        fit_series=fit_series,
                    )
                )
    return tuple(configurations)


def training_order(config: FitConfiguration) -> tuple[str, ...]:
    """Return the exact ratio concatenation order recorded by one B09 fit."""
    if config.fit_series == "prescribed_b09":
        return B09_TRAINING_ORDER
    if config.fit_series == "adaptive_b09":
        return ADAPTIVE_B09_TRAINING_ORDERS[(config.model_id, config.knit)]
    raise ValueError(f"Unknown fit series: {config.fit_series}")


def ratio_display_label(ratio: str) -> str:
    """Convert a compact label such as 0.25:1 to the raw-data label."""
    return ratio.replace(":", " to ")


def sha256_file(path: pathlib.Path) -> str:
    """Hash one immutable study input, for example a fit snapshot."""
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
    """Validate one complete B09 fit and every recorded immutable input hash."""
    if config.fit_code not in registry:
        raise ValueError(f"Fit {config.fit_code} is missing from the global registry")
    registry_row = registry[config.fit_code]
    if registry_row.get("Status") != "complete":
        raise ValueError(f"Fit {config.fit_code} is not complete in the registry")
    if registry_row.get("Model_ID") != config.model_id:
        raise ValueError(f"Fit {config.fit_code} has the wrong registry model")
    if registry_row.get("Knit") != config.knit:
        raise ValueError(f"Fit {config.fit_code} has the wrong registry knit")

    fit = load_fit(config.fit_code, DEFAULT_RESULTS_CSV)
    if fit.source.name != "fit.json":
        raise ValueError(f"Fit {config.fit_code} must resolve to its JSON artifact")
    payload = json.loads(fit.source.read_text(encoding="utf-8"))
    if fit.model != config.model_id or payload.get("model_id") != config.model_id:
        raise ValueError(f"Fit {config.fit_code} does not belong to {config.model_id}")
    if fit.metadata.get("Knit") != config.knit:
        raise ValueError(f"Fit {config.fit_code} does not belong to {config.knit}")
    if payload.get("result_role") != "training_optimum":
        raise ValueError(f"Fit {config.fit_code} is not a training optimum")
    case_id = str(payload.get("case_id", ""))
    expected_case_token = {
        "prescribed_b09": "progressive_b09",
        "adaptive_b09": "adaptive_greedy_v1_b09",
    }[config.fit_series]
    if expected_case_token not in case_id:
        raise ValueError(
            f"Fit {config.fit_code} is not a {config.fit_series} all-ratio fit"
        )
    if payload.get("objective", {}).get("fit_axes") != [0, 1]:
        raise ValueError(f"Fit {config.fit_code} was not fitted on PK2 axes 0 and 1")

    model_spec = load_model_spec(config.model_id)
    missing = [name for name in model_spec.parameter_names if name not in fit.parameters]
    if missing:
        raise ValueError(f"Fit {config.fit_code} is missing parameters: {missing}")
    for name, expected in (("C1", 1.0), ("C2", 1.0), ("theta1", 0.0), ("theta2", 90.0)):
        if not np.isclose(float(fit.parameters[name]), expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Fit {config.fit_code} violates the fixed {name} policy")

    raw_inputs = payload.get("input_files")
    if not isinstance(raw_inputs, dict):
        raise ValueError(f"Fit {config.fit_code} has no input_files provenance")
    for label in ("case_config", "material_parameters", "experimental_data", "umat_source"):
        record = raw_inputs.get(label)
        if not isinstance(record, dict):
            raise ValueError(f"Fit {config.fit_code} is missing {label} provenance")
        expected_hash = str(record.get("sha256", ""))
        for path_key in ("source_path", "snapshot_path"):
            path = pathlib.Path(str(record.get(path_key, "")))
            if not path.is_file():
                raise FileNotFoundError(f"Fit {config.fit_code} {label} {path_key} is missing: {path}")
            if sha256_file(path) != expected_hash:
                raise ValueError(f"Fit {config.fit_code} {label} {path_key} hash changed")
    return fit, payload


def pk1_history_to_pk2(dfgrd_voigt: np.ndarray, pk1_voigt: np.ndarray) -> np.ndarray:
    """Convert one experimental first-PK history to second-PK stress."""
    converted = []
    for f_voigt, p_voigt in zip(dfgrd_voigt, pk1_voigt):
        F = mech.voigt_to_matrix(f_voigt)
        P = mech.voigt_to_matrix(p_voigt)
        converted.append(mech.matrix_to_voigt(mech.pk1_to_pk2(P, F)))
    return np.asarray(converted, dtype=float)


def cauchy_history_to_pk2(dfgrd_voigt: np.ndarray, cauchy_voigt: np.ndarray) -> np.ndarray:
    """Convert one compiled-UMAT Cauchy history to second-PK stress."""
    converted = []
    for f_voigt, sigma_voigt in zip(dfgrd_voigt, cauchy_voigt):
        F = mech.voigt_to_matrix(f_voigt)
        sigma = mech.voigt_to_matrix(sigma_voigt)
        converted.append(mech.matrix_to_voigt(mech.cauchystress_to_pk2(sigma, F)))
    return np.asarray(converted, dtype=float)


def identity_origin_mask(
    dfgrd_voigt: np.ndarray,
    tolerance: float = ORIGIN_TOLERANCE,
) -> np.ndarray:
    """Identify identity-deformation rows, including duplicate origins."""
    identity = np.eye(3)
    return np.asarray(
        [
            np.allclose(
                mech.voigt_to_matrix(f_voigt),
                identity,
                rtol=0.0,
                atol=tolerance,
            )
            for f_voigt in dfgrd_voigt
        ],
        dtype=bool,
    )


def load_ratio_datasets(config: FitConfiguration) -> dict[str, RatioDataset]:
    """Load and convert all nine authoritative individual raw curves."""
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
            config.knit,
            display_ratio,
            config.sample_folder,
        )
        if set(directions) != {"course", "wales"}:
            raise ValueError(f"Ratio {ratio} has invalid material directions {directions}")
        if expected_directions is None:
            expected_directions = directions
        elif directions != expected_directions:
            raise ValueError(f"Axis directions change within {config.knit}: {directions}")

        dfgrd_voigt = umat.strains_to_dfgrd(data[:, :6], "H")
        datasets[ratio] = RatioDataset(
            ratio=ratio,
            path=path,
            data=data,
            dfgrd_voigt=dfgrd_voigt,
            experimental_pk2=pk1_history_to_pk2(dfgrd_voigt, data[:, 6:12]),
            origin_mask=identity_origin_mask(dfgrd_voigt),
            axis_directions=directions,
        )
    return datasets


def verify_b09_training_data(
    config: FitConfiguration,
    payload: Mapping[str, Any],
    datasets: Mapping[str, RatioDataset],
) -> None:
    """Prove that the fit's B09 data are the exact nine source arrays once each."""
    source_path = pathlib.Path(
        str(payload["input_files"]["experimental_data"]["source_path"])
    )
    training = np.atleast_2d(np.loadtxt(source_path, dtype=float))
    expected = np.vstack([datasets[ratio].data for ratio in training_order(config)])
    if not np.array_equal(training, expected):
        raise ValueError(f"B09 training data do not exactly match the nine ratio files: {source_path}")


def axis_index_by_direction(axis_directions: tuple[str, str]) -> dict[str, int]:
    """Return the PK2 Voigt index for course and wales."""
    mapping = {direction: index for index, direction in enumerate(axis_directions)}
    if set(mapping) != {"course", "wales"}:
        raise ValueError(f"Expected one course and one wales axis: {axis_directions}")
    return mapping


def reference_maxima(
    datasets: Mapping[str, RatioDataset],
) -> dict[str, float]:
    """Calculate fixed full-nine-ratio experimental PK2 reference peaks."""
    first = datasets[RATIO_ORDER[0]]
    axes = axis_index_by_direction(first.axis_directions)
    maxima = {
        direction: max(
            float(np.max(np.abs(datasets[ratio].experimental_pk2[:, axis])))
            for ratio in RATIO_ORDER
        )
        for direction, axis in axes.items()
    }
    for direction, maximum in maxima.items():
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError(f"Invalid {direction} PK2 reference maximum: {maximum}")
    return maxima


def directional_component(
    term_key: str,
    coefficient_name: str,
    shape_name: str | None,
    kind: str,
    direction: str,
    structurally_fixed_zero: bool = False,
) -> ComponentDefinition:
    """Build one physical course/wales component definition."""
    if direction not in {"course", "wales"}:
        raise ValueError(f"Unknown material direction: {direction}")
    if kind == "linear":
        component_id = f"linear_{direction}"
        component_name = f"Linear {direction}"
    elif kind == "nonlinear":
        component_id = f"nonlinear_{direction}"
        component_name = f"Nonlinear {direction}"
    elif kind == "secondary_nonlinear":
        component_id = f"secondary_nonlinear_{direction}"
        component_name = f"Secondary nonlinear {direction}"
    else:
        raise ValueError(f"Unknown component kind: {kind}")
    return ComponentDefinition(
        term_key=term_key,
        component_id=component_id,
        component_name=component_name,
        coefficient_name=coefficient_name,
        shape_name=shape_name,
        structurally_fixed_zero=structurally_fixed_zero,
    )


def component_definitions(
    model_id: str,
    axis_directions: tuple[str, str],
) -> tuple[ComponentDefinition, ...]:
    """Return model terms labeled in physical course/wales directions."""
    family_1, family_2 = axis_directions
    definitions = [
        directional_component("mu_1", "mu", None, "linear", family_1),
        directional_component("mu_2", "mu2", None, "linear", family_2),
        ComponentDefinition(
            term_key="mu_3",
            component_id="linear_course_wales_coupling",
            component_name="Linear course-wales coupling",
            coefficient_name="mu3",
        ),
        directional_component(
            "K_1ani1", "K_1ani1", "beta_1", "nonlinear", family_1
        ),
        directional_component(
            "K_1ani2", "K_1ani2", "gamma_1", "nonlinear", family_2
        ),
    ]
    if model_id == "StapletonViaux":
        definitions.extend(
            (
                directional_component(
                    "K_2ani1",
                    "K_2ani1",
                    "beta_2",
                    "secondary_nonlinear",
                    family_1,
                    structurally_fixed_zero=True,
                ),
                directional_component(
                    "K_2ani2",
                    "K_2ani2",
                    "gamma_2",
                    "secondary_nonlinear",
                    family_2,
                    structurally_fixed_zero=True,
                ),
            )
        )
    elif model_id != "SV_exp":
        raise ValueError(f"This study does not support model {model_id!r}")
    definitions.append(
        ComponentDefinition(
            term_key="K_coup",
            component_id="nonlinear_course_wales_coupling",
            component_name="Nonlinear course-wales coupling",
            coefficient_name="K_coup",
            shape_name="zeta",
        )
    )
    return tuple(sorted(definitions, key=lambda item: COMPONENT_ORDER[item.component_id]))


def decompose_pk2_history(
    dataset: RatioDataset,
    parameters: Mapping[str, float | int],
    model_id: str,
    definitions: Sequence[ComponentDefinition],
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Evaluate the additive PK2 histories and their analytic reconstruction."""
    histories = {definition.term_key: [] for definition in definitions}
    totals = []
    maximum_reconstruction_difference = 0.0
    for f_voigt in dataset.dfgrd_voigt:
        F = mech.voigt_to_matrix(f_voigt)
        pieces = analyzer.stress_decomposition_pk2(F, parameters, model_id)
        component_sum = np.zeros((3, 3), dtype=float)
        for definition in definitions:
            term = np.asarray(pieces[definition.term_key], dtype=float)
            histories[definition.term_key].append(mech.matrix_to_voigt(term))
            component_sum += term
        total = np.asarray(pieces["total"], dtype=float)
        maximum_reconstruction_difference = max(
            maximum_reconstruction_difference,
            float(np.max(np.abs(component_sum - total))),
        )
        totals.append(mech.matrix_to_voigt(total))
    return (
        {key: np.asarray(values, dtype=float) for key, values in histories.items()},
        np.asarray(totals, dtype=float),
        maximum_reconstruction_difference,
    )


def direct_umat_difference(
    dataset: RatioDataset,
    parameters: Mapping[str, float | int],
    model_id: str,
    decomposed_total: np.ndarray,
) -> float:
    """Return maximum PK2 difference from the compiled Fortran UMAT."""
    model_spec = load_model_spec(model_id)
    props = np.asarray(model_spec.pack_parameters(parameters), dtype=float)
    cauchy = analyzer.call_umat_history(dataset.dfgrd_voigt, props)
    direct_pk2 = cauchy_history_to_pk2(dataset.dfgrd_voigt, cauchy)
    return float(np.max(np.abs(direct_pk2 - decomposed_total)))


def component_status(
    definition: ComponentDefinition,
    parameters: Mapping[str, float | int],
    normalized_magnitude: float,
) -> str:
    """Classify why one component contributes or remains zero on a ratio."""
    coefficient = float(parameters[definition.coefficient_name])
    if definition.structurally_fixed_zero:
        if coefficient != 0.0:
            raise ValueError(
                f"{definition.coefficient_name} must be zero in reduced StapletonViaux fits"
            )
        return "structurally_fixed_zero"
    if coefficient == 0.0:
        return "fitted_zero"
    if normalized_magnitude == 0.0:
        return "inactive_on_ratio"
    return "contributing"


def component_classifications(definition: ComponentDefinition) -> tuple[str, str]:
    """Return the requested constitutive-family and physical-direction labels."""
    linear_nonlinear = (
        "Linear" if definition.component_id.startswith("linear_") else "Nonlinear"
    )
    if "coupling" in definition.component_id:
        direction = "Coupling"
    elif definition.component_id.endswith("_course"):
        direction = "Course"
    elif definition.component_id.endswith("_wales"):
        direction = "Wales"
    else:
        raise ValueError(f"Cannot classify component {definition.component_id}")
    return linear_nonlinear, direction


def safe_percentage(numerator: float, denominator: float) -> float:
    """Return a finite percentage, using zero when the direction is stress-free."""
    if denominator == 0.0:
        return 0.0
    return 100.0 * numerator / denominator


def contribution_rows(
    config: FitConfiguration,
    dataset: RatioDataset,
    parameters: Mapping[str, float | int],
    definitions: Sequence[ComponentDefinition],
    histories: Mapping[str, np.ndarray],
    maxima: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Aggregate one ratio into one CSV row per constitutive component."""
    included = ~dataset.origin_mask
    evaluated_count = int(np.count_nonzero(included))
    if evaluated_count == 0:
        raise ValueError(f"Ratio {dataset.ratio} contains only identity rows")
    axes = axis_index_by_direction(dataset.axis_directions)
    metrics = []
    for definition in definitions:
        history = histories[definition.term_key]
        course_values = history[included, axes["course"]]
        wales_values = history[included, axes["wales"]]
        course_sum = float(np.sum(np.abs(course_values)))
        wales_sum = float(np.sum(np.abs(wales_values)))
        normalized_course = course_sum / float(maxima["course"])
        normalized_wales = wales_sum / float(maxima["wales"])
        metrics.append(
            {
                "definition": definition,
                "course_sum": course_sum,
                "wales_sum": wales_sum,
                "normalized_course": normalized_course,
                "normalized_wales": normalized_wales,
                "normalized_combined": normalized_course + normalized_wales,
                "mean_course": float(np.mean(np.abs(course_values))),
                "mean_wales": float(np.mean(np.abs(wales_values))),
                "peak_course": float(np.max(np.abs(course_values))),
                "peak_wales": float(np.max(np.abs(wales_values))),
            }
        )
    combined_denominator = sum(item["normalized_combined"] for item in metrics)
    course_denominator = sum(item["course_sum"] for item in metrics)
    wales_denominator = sum(item["wales_sum"] for item in metrics)
    if combined_denominator <= 0.0:
        raise ValueError(f"Ratio {dataset.ratio} has no nonzero model component stress")

    rows = []
    for item in metrics:
        definition = item["definition"]
        normalized_magnitude = float(item["normalized_combined"])
        linear_nonlinear, direction = component_classifications(definition)
        rows.append(
            {
                "model_id": config.model_id,
                "fit_code": config.fit_code,
                "knit": config.knit,
                "ratio": dataset.ratio,
                "component_id": definition.component_id,
                "component_name": definition.component_name,
                "Linear/Nonlinear": linear_nonlinear,
                "Direction": direction,
                "component_status": component_status(
                    definition, parameters, normalized_magnitude
                ),
                "parameter_names": definition.parameter_names,
                "stress_contribution_percent": safe_percentage(
                    normalized_magnitude, combined_denominator
                ),
                "course_stress_contribution_percent": safe_percentage(
                    float(item["course_sum"]), course_denominator
                ),
                "wales_stress_contribution_percent": safe_percentage(
                    float(item["wales_sum"]), wales_denominator
                ),
                "normalized_absolute_stress_magnitude": normalized_magnitude,
                "mean_abs_pk2_course": float(item["mean_course"]),
                "mean_abs_pk2_wales": float(item["mean_wales"]),
                "peak_abs_pk2_course": float(item["peak_course"]),
                "peak_abs_pk2_wales": float(item["peak_wales"]),
                "evaluated_row_count": evaluated_count,
                "excluded_origin_row_count": int(np.count_nonzero(dataset.origin_mask)),
            }
        )
    return rows


def analyze_model(
    model_id: str,
    *,
    validate_with_umat: bool,
    tolerance: float,
) -> dict[str, Any]:
    """Analyze all four knits for one model inside an isolated worker."""
    if validate_with_umat:
        model_spec = load_model_spec(model_id)
        umat.compile_umat_meson(model_spec.fortran_source)

    registry = registry_rows()
    all_rows: list[dict[str, Any]] = []
    qa_rows = []
    reference_rows = []
    for config in study_configurations(model_id):
        fit, payload = load_and_validate_fit(config, registry)
        datasets = load_ratio_datasets(config)
        verify_b09_training_data(config, payload, datasets)
        maxima = reference_maxima(datasets)
        reference_rows.append(
            {
                "model_id": model_id,
                "fit_code": config.fit_code,
                "knit": config.knit,
                "axis_1_direction": datasets[RATIO_ORDER[0]].axis_directions[0],
                "axis_2_direction": datasets[RATIO_ORDER[0]].axis_directions[1],
                "course_reference_maximum": maxima["course"],
                "wales_reference_maximum": maxima["wales"],
            }
        )
        definitions = component_definitions(
            model_id, datasets[RATIO_ORDER[0]].axis_directions
        )
        parameters = {
            name: fit.parameters[name]
            for name in load_model_spec(model_id).parameter_names
        }
        for ratio in RATIO_ORDER:
            dataset = datasets[ratio]
            histories, total, reconstruction_difference = decompose_pk2_history(
                dataset,
                parameters,
                model_id,
                definitions,
            )
            if reconstruction_difference > tolerance:
                raise ValueError(
                    f"{model_id} {config.knit} {ratio} component reconstruction "
                    f"difference {reconstruction_difference:.6e} exceeds {tolerance:.6e}"
                )
            umat_difference = 0.0
            if validate_with_umat:
                umat_difference = direct_umat_difference(
                    dataset, parameters, model_id, total
                )
                if umat_difference > tolerance:
                    raise ValueError(
                        f"{model_id} {config.knit} {ratio} UMAT difference "
                        f"{umat_difference:.6e} exceeds {tolerance:.6e}"
                    )
            qa_rows.append(
                {
                    "model_id": model_id,
                    "fit_code": config.fit_code,
                    "knit": config.knit,
                    "ratio": ratio,
                    "component_reconstruction_difference": reconstruction_difference,
                    "umat_difference": umat_difference,
                }
            )
            all_rows.extend(
                contribution_rows(
                    config,
                    dataset,
                    parameters,
                    definitions,
                    histories,
                    maxima,
                )
            )
    return {"rows": all_rows, "qa": qa_rows, "reference_maxima": reference_rows}


def row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    """Return the deterministic CSV sort key."""
    fit_order = {
        config.fit_code: index
        for index, config in enumerate(study_configurations())
    }
    return (
        MODEL_ORDER.index(str(row["model_id"])),
        KNIT_ORDER.index(str(row["knit"])),
        fit_order[int(row["fit_code"])],
        RATIO_ORDER.index(str(row["ratio"])),
        COMPONENT_ORDER[str(row["component_id"])],
    )


def validate_output_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate completeness, percentages, finite values, and row ordering."""
    expected_total = sum(
        len(RATIO_ORDER) * (8 if config.model_id == "StapletonViaux" else 6)
        for config in study_configurations()
    )
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} output rows, found {len(rows)}")
    if list(rows) != sorted(rows, key=row_sort_key):
        raise ValueError("Output rows are not in deterministic study order")

    grouped: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["model_id"]),
                int(row["fit_code"]),
                str(row["knit"]),
                str(row["ratio"]),
            ),
            [],
        ).append(row)
        for column in NUMERIC_OUTPUT_COLUMNS:
            if not np.isfinite(float(row[column])):
                raise ValueError(f"Output contains nonfinite {column}: {row}")

    expected_groups = len(study_configurations()) * len(RATIO_ORDER)
    if len(grouped) != expected_groups:
        raise ValueError(f"Expected {expected_groups} model/knit/ratio groups")
    for key, group in grouped.items():
        expected_components = 8 if key[0] == "StapletonViaux" else 6
        if len(group) != expected_components:
            raise ValueError(f"Group {key} has {len(group)} components")
        combined = sum(float(row["stress_contribution_percent"]) for row in group)
        if not np.isclose(combined, 100.0, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"Combined percentages for {key} sum to {combined}")
        for direction in ("course", "wales"):
            magnitude_column = f"mean_abs_pk2_{direction}"
            percentage_column = f"{direction}_stress_contribution_percent"
            if any(float(row[magnitude_column]) > 0.0 for row in group):
                total = sum(float(row[percentage_column]) for row in group)
                if not np.isclose(total, 100.0, rtol=0.0, atol=1.0e-10):
                    raise ValueError(f"{direction} percentages for {key} sum to {total}")


def run_worker(model_id: str, output_path: pathlib.Path, tolerance: float) -> None:
    """Run one compiled model in an isolated process and persist JSON results."""
    result = analyze_model(
        model_id,
        validate_with_umat=True,
        tolerance=tolerance,
    )
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def run_isolated_model_worker(model_id: str, tolerance: float) -> dict[str, Any]:
    """Compile and analyze one model without retaining its DLL in the parent."""
    temp_root = RELEASE_ARCHIVE_ROOT
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"component_{model_id.lower()}_",
        dir=temp_root,
    ) as directory:
        worker_directory = pathlib.Path(directory)
        output_path = worker_directory / "worker_result.json"
        command = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--worker-model",
            model_id,
            "--worker-output",
            str(output_path),
            "--validation-tolerance",
            repr(tolerance),
        ]
        result = subprocess.run(
            command,
            cwd=worker_directory,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode != 0:
            raise RuntimeError(
                f"Component worker for {model_id} failed with exit code "
                f"{result.returncode}"
            )
        if not output_path.is_file():
            raise RuntimeError(f"Component worker did not write {output_path}")
        return json.loads(output_path.read_text(encoding="utf-8"))


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the final CSV atomically with full Python float precision."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def readme_text(
    qa_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    compiled_umat_validated: bool,
) -> str:
    """Build concise methods, provenance, and interpretation documentation."""
    maximum_reconstruction = max(
        float(row["component_reconstruction_difference"]) for row in qa_rows
    )
    maximum_umat = max(float(row["umat_difference"]) for row in qa_rows)
    lines = [
        "# Fixed-Fit Model Component Stress Contributions",
        "",
        "This study decomposes the prescribed and adaptive all-ratio B09 fits",
        "without refitting,",
        "ablation, or production-result changes. The primary Excel workbook is",
        "`component_stress_contributions.xlsx`; the CSV is its machine-readable",
        "source table.",
        "",
        "## Contribution definition",
        "",
        "For each non-origin row and physical material direction, every additive",
        "PK2 component is divided by that direction's peak absolute experimental",
        "PK2 stress across all nine ratios. A component's normalized absolute",
        "magnitude is the sum of those absolute normalized values over the ratio.",
        "Its reported percentage is that magnitude divided by the sum over all",
        "model components. Percentages therefore sum to 100% within each",
        "model/knit/ratio group.",
        "",
        "## Fit inventory",
        "",
        "| B09 series | Model | Interlock | Rib | Purl | Single Jersey |",
        "|---|---|---:|---:|---:|---:|",
        "| Prescribed | SV_exp | 726 | 722 | 723 | 725 |",
        "| Prescribed | StapletonViaux | 724 | 729 | 727 | 728 |",
        "| Adaptive | SV_exp | 1075 | 1072 | 1071 | 1074 |",
        "| Adaptive | StapletonViaux | 1073 | 1078 | 1076 | 1077 |",
        "",
        "## Physical directions and reference maxima",
        "",
        "| Model | Knit | Axis 1 | Axis 2 | Course peak PK2 | Wales peak PK2 |",
        "|---|---|---|---|---:|---:|",
    ]
    seen_references: set[tuple[str, str]] = set()
    for row in sorted(
        reference_rows,
        key=lambda item: (
            MODEL_ORDER.index(str(item["model_id"])),
            KNIT_ORDER.index(str(item["knit"])),
        ),
    ):
        reference_key = (str(row["model_id"]), str(row["knit"]))
        if reference_key in seen_references:
            continue
        seen_references.add(reference_key)
        lines.append(
            "| {model_id} | {knit} | {axis_1_direction} | {axis_2_direction} | "
            "{course_reference_maximum:.17g} | {wales_reference_maximum:.17g} |".format(
                **row
            )
        )
    lines.extend(
        (
            "",
            "Interlock's family 1/axis 1 is wales; family 1 is course for the",
            "other three knits. Component names use these physical mappings.",
            "",
            "StapletonViaux secondary nonlinear terms are included as explicit",
            "`structurally_fixed_zero` rows. SV_exp does not contain those terms.",
            "`Linear/Nonlinear` classifies the constitutive family, while",
            "`Direction` labels each row as Course, Wales, or Coupling.",
            "",
            "## Validation",
            "",
            f"- Maximum additive reconstruction difference: `{maximum_reconstruction:.6e}`.",
            (
                f"- Maximum compiled-UMAT PK2 difference: `{maximum_umat:.6e}`."
                if compiled_umat_validated
                else "- Compiled-UMAT validation was skipped by explicit debug control."
            ),
            f"- Required validation tolerance: `{UMAT_VALIDATION_TOLERANCE:.6e}`.",
            "- All source and snapshot hashes from every fit artifact were checked.",
            "- Every B09 training array was matched exactly to its recorded ordering",
            "  of the nine individual curves.",
            "",
            "## Interpretation limit",
            "",
            "The percentage measures the share of the fitted model's absolute,",
            "direction-normalized PK2 component-stress magnitude along a loading",
            "path. It is not variance explained and does not establish parameter",
            "identifiability or whether a term can be removed after refitting.",
            "",
        )
    )
    return "\n".join(lines)


def write_readme(path: pathlib.Path, content: str) -> None:
    """Write the study README atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def execute_study(*, overwrite: bool, validate_with_umat: bool) -> pathlib.Path:
    """Run the complete two-series fixed-fit study and return its CSV path."""
    existing = [path for path in (OUTPUT_CSV, OUTPUT_README) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Study output already exists; set OVERWRITE_EXISTING = True or use "
            f"--overwrite: {existing}"
        )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    codes = [config.fit_code for config in study_configurations()]
    if len(codes) != 16 or len(set(codes)) != 16:
        raise ValueError("The study must select sixteen unique B09 fit codes")

    model_results = []
    for model_id in MODEL_ORDER:
        print(f"Analyzing {model_id} across two B09 series, four knits, and nine ratios")
        if validate_with_umat:
            model_results.append(
                run_isolated_model_worker(model_id, UMAT_VALIDATION_TOLERANCE)
            )
        else:
            model_results.append(
                analyze_model(
                    model_id,
                    validate_with_umat=False,
                    tolerance=UMAT_VALIDATION_TOLERANCE,
                )
            )

    rows = sorted(
        [row for result in model_results for row in result["rows"]],
        key=row_sort_key,
    )
    qa_rows = [row for result in model_results for row in result["qa"]]
    reference_rows = [
        row for result in model_results for row in result["reference_maxima"]
    ]
    validate_output_rows(rows)
    write_csv(OUTPUT_CSV, rows)
    write_readme(
        OUTPUT_README,
        readme_text(
            qa_rows,
            reference_rows,
            compiled_umat_validated=validate_with_umat,
        ),
    )
    print(f"Wrote {len(rows)} component rows to {OUTPUT_CSV}")
    return OUTPUT_CSV


def parse_args() -> argparse.Namespace:
    """Parse normal user controls and hidden isolated-worker controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this study's existing CSV and README.",
    )
    parser.add_argument(
        "--skip-umat-validation",
        action="store_true",
        help="Debug only: skip compiled-Fortran validation.",
    )
    parser.add_argument("--worker-model", choices=MODEL_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=pathlib.Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--validation-tolerance",
        type=float,
        default=UMAT_VALIDATION_TOLERANCE,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    """Run either one isolated worker or the complete user-facing study."""
    args = parse_args()
    if args.worker_model is not None:
        if args.worker_output is None:
            raise ValueError("--worker-output is required with --worker-model")
        run_worker(args.worker_model, args.worker_output, args.validation_tolerance)
        return
    execute_study(
        overwrite=OVERWRITE_EXISTING or args.overwrite,
        validate_with_umat=(
            VALIDATE_WITH_COMPILED_UMAT and not args.skip_umat_validation
        ),
    )


if __name__ == "__main__":
    main()
