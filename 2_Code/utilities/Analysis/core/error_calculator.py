from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (INPUT_ROOT as RELEASE_INPUT_ROOT,
    ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import pathlib
import sys
from typing import Any

import numpy as np

curr_file = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis")
code_dir = curr_file.parent
project_root = code_dir.parent
sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(project_root))

from Python_Wrapper import umat
from Python_Wrapper.model_registry import load_model_spec
from Analysis.core import analyzer_common as ac
from Analysis.core.fit_results import DEFAULT_RESULTS_CSV, FitResult, load_fit

ParameterMap = Mapping[str, float | int]

# Set this to the saved fit number to evaluate when pressing Play in VS Code.
# Example: 2166. The material model and knit type are read from this fit.
DEFAULT_FIT_CODE: int | None = 2166
# Physical Course:Wale ratios evaluated by default; examples: ("1 to 1",) or all five.
DEFAULT_RATIOS = (
    "0 to 1",
    "0.5 to 1",
    "1 to 0",
    "1 to 0.5",
    "1 to 1",
)
# Stress measure used for errors; examples: "PK1" or "PK2".
DEFAULT_STRESS_MEASURE = "PK2"
# Choose which experimental data product is used by default: "raw".
EXP_DATA_TYPE = "raw"
# Default CSV written when pressing Play; set to None to print results only.
# Example: code_dir / "Outputs" / "my_study" / "errors.csv".
DEFAULT_OUTPUT_CSV: pathlib.Path | None = None
ALL_RATIOS = (
    "0 to 1",
    "0.25 to 1",
    "0.5 to 1",
    "0.75 to 1",
    "1 to 0",
    "1 to 0.25",
    "1 to 0.5",
    "1 to 0.75",
    "1 to 1",
)
DEFAULT_RAW_SAMPLE_FOLDERS = {
    "Interlock": "BRT_15mm_Tri_Int_S06",
    "Purl": "BRT_15mm_Tri_Purl_S02",
    "Rib": "BRT_15mm_Tri_Rib_S03",
    "SingleJersey": "BRT_15mm_Tri_SJ_S02",
}
EXPERIMENTAL_DATA_ROOT = ac.EXPERIMENTAL_DATA_ROOT
EXP_DATA_TYPES = ("raw",)
EXP_DATA_TYPE_FOLDERS = {"raw": "Raw_data"}
DEFAULT_RAW_DATA_SUBDIR = "Compiled_Data"
FALLBACK_AXIS_DIRECTIONS = {
    "Interlock": ("wales", "course"),
    "Purl": ("course", "wales"),
    "Rib": ("course", "wales"),
    "SingleJersey": ("course", "wales"),
}
METADATA_PATH = RELEASE_INPUT_ROOT / "Experiment_Data" / "biaxial_test_metadata.csv"
EXP_TYPE = ["H", "P"]
UMAT_TYPE = ["F", "Sigma"]
STRESS_MEASURES = ("PK1", "PK2")
# Swap x/y and xz/yz in Voigt order; for example, machine Wale/Course to Course/Wale.
XY_SWAP_VOIGT = np.asarray([1, 0, 2, 4, 3, 5], dtype=int)


@dataclass(frozen=True)
class RatioErrorResult:
    ratio: str
    path: pathlib.Path
    errors: dict[str, float]
    r_squared: dict[str, float]
    r_squared_sse: dict[str, float]
    r_squared_sst: dict[str, float]
    observation_count: dict[str, int]
    mean_r_squared: float
    clipped_mean_r_squared: float
    pooled_r_squared: float
    axis_directions: tuple[str, str]
    # Optional context keeps legacy callers source-compatible.
    sample: str = ""
    physical_ratio: str = ""
    machine_ratio: str = ""
    stress_measure: str = "PK1"
    stress_unit: str = "MPa"




def main() -> None:
    """Evaluate one shipped fit; example: Interlock C063 fit 2166."""
    args = parse_args()
    selected_fit = load_fit(args.fit_code, args.results_csv)
    knit = selected_knit(selected_fit)
    yarn_type = selected_yarn_type(selected_fit, args.yarn_type)
    model_spec = load_model_spec(selected_fit.model)
    parameters = selected_parameters(model_spec.parameter_names, selected_fit.model, selected_fit)
    sample_folder = args.raw_sample_folder or DEFAULT_RAW_SAMPLE_FOLDERS[knit]
    umat.compile_umat_meson(model_spec.fortran_source)
    results = [evaluate_ratio(ratio, np.asarray(model_spec.pack_parameters(parameters)), knit,
               args.data_dir, normalize_exp_data_type(args.exp_data_type), yarn_type,
               sample_folder, stress_measure=normalize_stress_measure(args.stress_measure))
               for ratio in selected_ratios(args)]
    print_results(results)
    if args.output_csv is not None:
        write_results_csv(args.output_csv, results)



def parse_args() -> argparse.Namespace:
    """Parse calculator inputs while resolving model and knit from the saved fit."""
    parser = argparse.ArgumentParser(
        description="Calculate SSMCCF fit errors and per-ratio R squared values."
    )
    parser.add_argument(
        "--fit-code",
        "--fit",
        dest="fit_code",
        type=int,
        default=DEFAULT_FIT_CODE,
        help=(
            "Fit code to evaluate. Its saved metadata supplies the material model and knit "
            "type. Standard Outputs/fits artifacts are checked before the legacy "
            "results_final.csv fallback."
        ),
    )
    parser.add_argument(
        "--results-csv",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Fit-results CSV. Default: {DEFAULT_RESULTS_CSV}",
    )
    parser.add_argument(
        "--ratio",
        dest="ratio",
        action="append",
        help=(
            "Experimental loading ratio label, e.g. '1 to 0' or '0.75 to 1'. "
            "Repeat this option for multiple ratios."
        ),
    )
    parser.add_argument(
        "--ratios",
        dest="ratio_list",
        nargs="+",
        default=None,
        help="Experimental loading ratio labels. Quote labels that contain spaces.",
    )
    parser.add_argument(
        "--all-ratios",
        action="store_true",
        help="Evaluate all available ratios for the selected dataset.",
    )
    parser.add_argument(
        "--stress-measure",
        choices=STRESS_MEASURES,
        type=str.upper,
        default=DEFAULT_STRESS_MEASURE,
        help=f"Stress used for errors and R squared. Default: {DEFAULT_STRESS_MEASURE}.",
    )
    parser.add_argument(
        "--exp-data-type",
        "--data-type",
        "--data-collection",
        dest="exp_data_type",
        choices=EXP_DATA_TYPES,
        default=EXP_DATA_TYPE,
        help=(
            "Experimental data product to use. "
            "'raw' reads the shipped Compiled_Data curves."
        ),
    )
    parser.add_argument(
        "--yarn-type",
        choices=tuple(ac.YARN_FOLDERS.keys()),
        default=None,
        help="Optional yarn override. By default, use the saved fit metadata.",
    )
    parser.add_argument(
        "--raw-sample-folder",
        "--sample-folder",
        dest="raw_sample_folder",
        default=None,
        help="Sample folder to use inside the shipped specimen directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional manual directory override. If set, this directory is searched "
            "before the hardcoded experimental-data folders."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Optional CSV path for the calculated errors and R squared values.",
    )
    return parser.parse_args()


def selected_ratios(
    args: argparse.Namespace,
) -> tuple[str, ...]:
    """Return ratio labels requested by the user."""
    if getattr(args, "all_ratios", False):
        return ALL_RATIOS

    ratios: list[str] = []
    if getattr(args, "ratio", None):
        ratios.extend(args.ratio)
    if getattr(args, "ratio_list", None):
        ratios.extend(args.ratio_list)
    if not ratios:
        ratios = list(DEFAULT_RATIOS)
    selected = tuple(dict.fromkeys(normalize_ratio_label(ratio) for ratio in ratios))
    allowed = ALL_RATIOS
    invalid = [ratio for ratio in selected if ratio not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported ratios: {invalid}; "
            f"choose from {allowed}."
        )
    return selected


def selected_yarn_type(selected_fit: FitResult, override: str | None) -> str:
    """Use an explicit yarn override or recover it from the saved fit."""
    if override:
        return override
    nested_context = selected_fit.metadata.get("experiment_context", {})
    context = nested_context if isinstance(nested_context, Mapping) else {}
    yarn_type = str(
        context.get("yarn_type", selected_fit.metadata.get("Yarn_Type", ""))
    ).strip()
    return yarn_type or ac.DEFAULT_YARN_TYPE






def selected_knit(selected_fit: FitResult) -> str:
    """Return the knit recorded in a fit artifact or legacy result row."""
    nested_context = selected_fit.metadata.get("experiment_context", {})
    context = nested_context if isinstance(nested_context, Mapping) else {}
    knit = str(context.get("knit", selected_fit.metadata.get("Knit", ""))).strip()
    if knit not in DEFAULT_RAW_SAMPLE_FOLDERS:
        options = ", ".join(DEFAULT_RAW_SAMPLE_FOLDERS)
        raise ValueError(
            f"Fit {selected_fit.code} has unsupported or missing knit {knit!r}; "
            f"choose a fit recorded as one of: {options}."
        )
    return knit


def selected_parameters(
    parameter_names: Sequence[str],
    model_id: str,
    selected_fit: FitResult,
) -> dict[str, float | int]:
    """Return the saved parameters after checking compatibility with the saved model."""
    missing = [name for name in parameter_names if name not in selected_fit.parameters]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Fit {selected_fit.code} is incompatible with {model_id}; missing: {names}"
        )
    return {name: selected_fit.parameters[name] for name in parameter_names}


def evaluate_ratio(
    ratio: str,
    sim_par: np.ndarray,
    knit: str,
    data_dir: pathlib.Path | None,
    exp_data_type: str,
    yarn_type: str,
    raw_sample_folder: str,
    *,
    sample_id: str | None = None,
    stress_measure: str = "PK1",
) -> RatioErrorResult:
    """Calculate errors and R squared values for one experimental ratio."""
    stress_measure = normalize_stress_measure(stress_measure)
    if sample_id is not None:
        raise ValueError("Use the shipped specimen folder rather than a separate sample ID")
    exp_path = experimental_path(
        knit,
        ratio,
        data_dir,
        exp_data_type,
        yarn_type,
        raw_sample_folder,
    )
    exp_data = np.loadtxt(exp_path)
    axis_directions = axis_directions_for_ratio(knit, ratio, raw_sample_folder)
    physical_ratio = normalize_ratio_label(ratio)
    machine_ratio = normalize_ratio_label(ratio)

    exp_strains = exp_data[:, :6]
    sim_data = umat.generate_exp_data(
        10,
        0,
        10,
        EXP_TYPE,
        sim_par,
        UMAT_TYPE,
        strains=exp_strains,
    )
    exp_stress, stress_unit = stress_for_error(exp_data, stress_measure)
    sim_stress, _ = stress_for_error(sim_data, stress_measure)
    errors = calc_errors(exp_stress, sim_stress, axes=(0, 1))
    r_squared, r_squared_sse, r_squared_sst = r_squared_stats_by_direction(
        exp_stress, sim_stress, axis_directions
    )
    return RatioErrorResult(
        ratio=ratio,
        path=exp_path,
        errors=errors,
        r_squared=r_squared,
        r_squared_sse=r_squared_sse,
        r_squared_sst=r_squared_sst,
        # Each physical direction contributes one residual per experimental row.
        observation_count={"course": exp_data.shape[0], "wales": exp_data.shape[0]},
        mean_r_squared=float(np.nanmean(list(r_squared.values()))),
        clipped_mean_r_squared=calc_clipped_mean_r_squared(r_squared.values()),
        pooled_r_squared=calc_pooled_r_squared(
            r_squared_sse.values(), r_squared_sst.values()
        ),
        axis_directions=axis_directions,
        sample=sample_id or raw_sample_folder,
        physical_ratio=physical_ratio,
        machine_ratio=machine_ratio,
        stress_measure=stress_measure,
        stress_unit=stress_unit,
    )


def normalize_stress_measure(stress_measure: str) -> str:
    """Normalize the stress selector; examples: ``PK1`` or ``pk2``."""
    value = str(stress_measure).strip().upper()
    if value not in STRESS_MEASURES:
        options = ", ".join(STRESS_MEASURES)
        raise ValueError(f"Stress measure must be one of: {options}.")
    return value


def stress_for_error(data: np.ndarray, stress_measure: str) -> tuple[np.ndarray, str]:
    """Return PK1 in MPa or PK2 in kPa for all six Voigt components."""
    stress_measure = normalize_stress_measure(stress_measure)
    pk1 = np.asarray(data[:, 6:12], dtype=float)
    if stress_measure == "PK1":
        return pk1, "MPa"

    stretches = 1.0 + np.asarray(data[:, :6], dtype=float)
    if np.any(stretches[:, :3] <= 0.0):
        raise ValueError("PK2 conversion requires positive principal stretches.")
    # For diagonal loading, S_ii = P_ii / lambda_i; convert MPa to kPa.
    pk2 = np.array(pk1, copy=True)
    pk2[:, :3] = 1000.0 * pk1[:, :3] / stretches[:, :3]
    pk2[:, 3:] = 1000.0 * pk1[:, 3:]
    return pk2, "kPa"


def canonicalize_course_wales(
    data: np.ndarray,
    axis_directions: tuple[str, str],
) -> np.ndarray:
    """Return a copy ordered as Course/Wale before UMAT evaluation."""
    directions = tuple(normalize_direction(value) for value in axis_directions)
    if directions == ("course", "wales"):
        return np.array(data, dtype=float, copy=True)
    if directions != ("wales", "course"):
        raise ValueError(f"Cannot canonicalize axis directions {axis_directions!r}.")
    canonical = np.array(data, dtype=float, copy=True)
    canonical[:, :6] = canonical[:, :6][:, XY_SWAP_VOIGT]
    canonical[:, 6:12] = canonical[:, 6:12][:, XY_SWAP_VOIGT]
    return canonical




def physical_ratio_values(
    axis_values: tuple[float, float],
    axis_directions: tuple[str, str],
) -> tuple[float, float]:
    """Map machine-axis ratios to physical Course:Wale values."""
    by_direction = {
        normalize_direction(direction): float(value)
        for direction, value in zip(axis_directions, axis_values, strict=True)
    }
    if set(by_direction) != {"course", "wales"}:
        raise ValueError(f"Missing Course/Wale metadata in {axis_directions!r}.")
    return by_direction["course"], by_direction["wales"]


def ratio_values(ratio: str) -> tuple[float, float]:
    """Parse one ratio label; examples: ``0.5 to 1`` or ``0.5:1``."""
    left, right = split_ratio(ratio)
    return float(left), float(right)


def ratio_label_from_values(values: Sequence[float]) -> str:
    """Format numeric ratio values using the project's filename convention."""
    return " to ".join(f"{float(value):g}" for value in values)


def experimental_path(
    knit: str,
    ratio: str,
    data_dir: pathlib.Path | None,
    exp_data_type: str,
    yarn_type: str,
    sample_folder: str,
) -> pathlib.Path:
    """Locate one experimental data file under the selected data product."""
    candidates = experimental_path_candidates(
        knit,
        ratio,
        data_dir,
        exp_data_type,
        yarn_type,
        sample_folder,
    )
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find experiment file. Searched:\n{searched}")


def experimental_path_candidates(knit, ratio, data_dir, exp_data_type, yarn_type, sample_folder):
    """Locate a shipped raw curve; example: Rib/S03/Compiled_Data/1 to 1.dat."""
    normalize_exp_data_type(exp_data_type)
    raw_name = raw_experiment_filename(ratio)
    candidates = []
    if data_dir is not None:
        candidates.extend((data_dir / raw_name, data_dir / DEFAULT_RAW_DATA_SUBDIR / raw_name))
    candidates.append(EXPERIMENTAL_DATA_ROOT / knit / sample_folder / DEFAULT_RAW_DATA_SUBDIR / raw_name)
    return deduplicate_paths(candidates)


def raw_experiment_filename(ratio: str) -> str:
    """Return the ratio filename used in Raw_data Compiled_Data folders."""
    return f"{normalize_ratio_label(ratio).replace('.', 'p')}.dat"





def normalize_exp_data_type(exp_data_type: str) -> str:
    """Accept only the shipped raw product; example: raw_data aliases raw."""
    if exp_data_type.strip().lower() not in {"raw", "raw_data", "raw data"}:
        raise ValueError("EXP_DATA_TYPE must be raw; no other product is shipped")
    return "raw"


def deduplicate_paths(paths: Sequence[pathlib.Path]) -> list[pathlib.Path]:
    """Remove duplicate path candidates while preserving order."""
    unique_paths = []
    seen = set()
    for path in paths:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        unique_paths.append(path)
        seen.add(key)
    return unique_paths


def calc_errors(
    exp_stress: np.ndarray,
    model_stress: np.ndarray,
    axes: Sequence[int] | None = None,
) -> dict[str, float]:
    """Calculate the four existing error measures over the selected axes."""
    if axes is not None:
        exp_stress = exp_stress[:, axes]
        model_stress = model_stress[:, axes]
    return {
        "MSE": calc_MSE(exp_stress, model_stress),
        "RMSE": calc_RMSE(exp_stress, model_stress),
        "MAE": calc_MAE(exp_stress, model_stress),
        "Rel_Err": calc_RelErr(exp_stress, model_stress),
    }


def r_squared_by_direction(
    exp_stress: np.ndarray,
    model_stress: np.ndarray,
    axis_directions: tuple[str, str],
) -> dict[str, float]:
    """Calculate R squared for the physical course and wales directions."""
    values, _sse, _sst = r_squared_stats_by_direction(
        exp_stress, model_stress, axis_directions
    )
    return values


def r_squared_stats_by_direction(
    exp_stress: np.ndarray,
    model_stress: np.ndarray,
    axis_directions: tuple[str, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Calculate R squared and its SSE/SST parts by physical direction."""
    values: dict[str, float] = {}
    sse_values: dict[str, float] = {}
    sst_values: dict[str, float] = {}
    for axis, direction in enumerate(axis_directions):
        direction_key = normalize_direction(direction)
        if direction_key in {"course", "wales"}:
            r_squared, sse, sst = calc_r_squared_parts(
                exp_stress[:, axis], model_stress[:, axis]
            )
            values[direction_key] = r_squared
            sse_values[direction_key] = sse
            sst_values[direction_key] = sst
    for direction, axis in (("course", 0), ("wales", 1)):
        if direction not in values:
            r_squared, sse, sst = calc_r_squared_parts(
                exp_stress[:, axis], model_stress[:, axis]
            )
            values[direction] = r_squared
            sse_values[direction] = sse
            sst_values[direction] = sst
    return values, sse_values, sst_values


def calc_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate coefficient of determination between observed and predicted data."""
    r_squared, _sse, _sst = calc_r_squared_parts(observed, predicted)
    return r_squared


def calc_r_squared_parts(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    """Return R squared plus residual and total sums of squares."""
    check_inputs(observed, predicted)
    residual_sum_squares = np.sum((observed - predicted) ** 2)
    total_sum_squares = np.sum((observed - np.mean(observed)) ** 2)
    if np.isclose(total_sum_squares, 0.0):
        return np.nan, float(residual_sum_squares), float(total_sum_squares)
    return (
        float(1.0 - residual_sum_squares / total_sum_squares),
        float(residual_sum_squares),
        float(total_sum_squares),
    )


def calc_clipped_mean_r_squared(values: Sequence[float]) -> float:
    """Average max(R squared, 0), ignoring undefined R squared values."""
    finite_values = [max(float(value), 0.0) for value in values if not np.isnan(value)]
    if not finite_values:
        return np.nan
    return float(np.mean(finite_values))


def calc_pooled_r_squared(
    residual_sum_squares: Sequence[float],
    total_sum_squares: Sequence[float],
) -> float:
    """Calculate pooled R squared from summed SSE and SST."""
    total_sse = float(np.sum(list(residual_sum_squares)))
    total_sst = float(np.sum(list(total_sum_squares)))
    if np.isclose(total_sst, 0.0):
        return np.nan
    return float(1.0 - total_sse / total_sst)


def calc_pooled_rmse(
    residual_sum_squares: Sequence[float],
    observation_counts: Sequence[int],
) -> float:
    """Calculate sqrt(sum(SSE) / sum(N)) across any set of curves."""
    total_sse = float(np.sum(list(residual_sum_squares)))
    total_observations = int(np.sum(list(observation_counts)))
    if total_observations <= 0:
        raise ValueError("Pooled RMSE requires at least one observation.")
    return float(np.sqrt(total_sse / total_observations))


def axis_directions_for_ratio(
    knit: str,
    ratio: str,
    raw_sample_folder: str,
) -> tuple[str, str]:
    """Return physical directions for stress axes 0 and 1."""
    metadata_dirs = axis_directions_from_metadata(raw_sample_folder, ratio)
    if metadata_dirs is not None:
        return metadata_dirs
    return FALLBACK_AXIS_DIRECTIONS[knit]


def axis_directions_from_metadata(
    raw_sample_folder: str,
    ratio: str,
) -> tuple[str, str] | None:
    """Read axis directions from the biaxial metadata CSV when available."""
    if not METADATA_PATH.is_file():
        return None

    ratio_step = ratio_to_step_label(ratio)
    fallback: tuple[str, str] | None = None
    with METADATA_PATH.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row.get("canonical_sample_folder") != raw_sample_folder:
                continue
            directions = (
                normalize_direction(row.get("axis1_direction", "")),
                normalize_direction(row.get("axis2_direction", "")),
            )
            if all(directions):
                fallback = directions
            if row.get("step_label") == ratio_step and all(directions):
                return directions
    return fallback


def ratio_to_step_label(ratio: str) -> str:
    """Convert display ratio labels to metadata step labels."""
    left, right = split_ratio(ratio)
    return f"ratio_{ratio_token(left)}_{ratio_token(right)}"


def split_ratio(ratio: str) -> tuple[str, str]:
    parts = normalize_ratio_label(ratio).split(" to ")
    if len(parts) != 2:
        raise ValueError(f"Invalid ratio label {ratio!r}; expected e.g. '0.75 to 1'.")
    return parts[0], parts[1]


def ratio_token(value: str) -> str:
    """Return metadata token for one ratio side."""
    if "." not in value:
        return value
    whole, fractional = value.split(".", 1)
    if whole != "0":
        return f"{whole}{fractional.rstrip('0')}"
    return f"0{fractional.rstrip('0')}"


def normalize_ratio_label(ratio: str) -> str:
    """Normalize simple ratio spelling variants."""
    return ratio.strip().replace("_", " ").replace(":", " to ").replace("  ", " ")


def normalize_direction(direction: Any) -> str:
    """Normalize physical direction labels from metadata."""
    key = str(direction).strip().lower()
    if key in {"wale", "wales"}:
        return "wales"
    if key in {"course", "courses"}:
        return "course"
    return key


def print_results(results: Sequence[RatioErrorResult]) -> None:
    """Print per-ratio and selected-ratio summaries without pooling specimens."""
    for sample, sample_results in group_results_by_sample(results).items():
        print(f"\nSample {sample}: per-ratio errors and R squared")
        print_sample_results(sample_results)
        summary = selected_ratio_summary(sample_results)
        print(f"\nSample {sample}: selected-ratio R squared summary")
        print(f"course: {format_float(summary['R2_course'])}")
        print(f"wales:  {format_float(summary['R2_wales'])}")
        print(f"pooled: {format_float(summary['R2_pooled'])}")


def print_sample_results(results: Sequence[RatioErrorResult]) -> None:
    """Print the detailed rows for one specimen."""
    header = (
        "Ratio",
        "MSE",
        "RMSE",
        "MAE",
        "Rel_Err (%)",
        "R2 course",
        "R2 wales",
        "Mean R2",
        "Clipped R2",
        "Pooled R2",
    )
    print(format_row(header))
    print(format_row(("-" * len(item) for item in header)))
    for result in results:
        print(
            format_row(
                (
                    result.ratio,
                    format_float(result.errors["MSE"]),
                    format_float(result.errors["RMSE"]),
                    format_float(result.errors["MAE"]),
                    format_float(result.errors["Rel_Err"]),
                    format_float(result.r_squared["course"]),
                    format_float(result.r_squared["wales"]),
                    format_float(result.mean_r_squared),
                    format_float(result.clipped_mean_r_squared),
                    format_float(result.pooled_r_squared),
                )
            )
        )


def group_results_by_sample(
    results: Sequence[RatioErrorResult],
) -> dict[str, list[RatioErrorResult]]:
    """Group rows by specimen while preserving their first-seen order."""
    groups: dict[str, list[RatioErrorResult]] = {}
    for result in results:
        sample = result.sample or result.path.parent.parent.name or "legacy"
        groups.setdefault(sample, []).append(result)
    return groups


def selected_ratio_summary(results: Sequence[RatioErrorResult]) -> dict[str, Any]:
    """Pool stored SSE/SST parts across the selected ratios for one specimen."""
    if not results:
        raise ValueError("A selected-ratio summary requires at least one result.")
    directions = ("course", "wales")
    sse = {
        direction: float(
            np.sum([result.r_squared_sse[direction] for result in results])
        )
        for direction in directions
    }
    sst = {
        direction: float(
            np.sum([result.r_squared_sst[direction] for result in results])
        )
        for direction in directions
    }
    counts = {
        direction: int(
            np.sum([result.observation_count[direction] for result in results])
        )
        for direction in directions
    }
    total_sse = sse["course"] + sse["wales"]
    total_count = counts["course"] + counts["wales"]
    total_absolute_error = float(
        np.sum(
            [
                result.errors["MAE"]
                * (
                    result.observation_count["course"]
                    + result.observation_count["wales"]
                )
                for result in results
            ]
        )
    )
    all_r_squared = [
        result.r_squared[direction]
        for result in results
        for direction in directions
    ]
    return {
        "sample": results[0].sample or results[0].path.parent.parent.name or "legacy",
        "stress_measure": results[0].stress_measure,
        "stress_unit": results[0].stress_unit,
        "selected_ratios": ";".join(result.physical_ratio or result.ratio for result in results),
        "MSE": total_sse / total_count,
        "RMSE": float(np.sqrt(total_sse / total_count)),
        "MAE": total_absolute_error / total_count,
        "SSE_course": sse["course"],
        "SST_course": sst["course"],
        "N_course": counts["course"],
        "R2_course": calc_pooled_r_squared([sse["course"]], [sst["course"]]),
        "SSE_wales": sse["wales"],
        "SST_wales": sst["wales"],
        "N_wales": counts["wales"],
        "R2_wales": calc_pooled_r_squared([sse["wales"]], [sst["wales"]]),
        "R2_mean": float(np.nanmean(all_r_squared)),
        "R2_clipped_mean": calc_clipped_mean_r_squared(all_r_squared),
        "R2_pooled": calc_pooled_r_squared(list(sse.values()), list(sst.values())),
    }


def format_row(values: Sequence[Any]) -> str:
    return "  ".join(f"{str(value):>14}" for value in values)


def format_float(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:.6g}"


def write_results_csv(path: pathlib.Path, results: Sequence[RatioErrorResult]) -> None:
    """Write ratio and selected-ratio-summary rows in one long-format CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "aggregation_level",
        "sample",
        "stress_measure",
        "stress_unit",
        "ratio",
        "selected_ratios",
        "machine_ratio",
        "path",
        "axis1_direction",
        "axis2_direction",
        "MSE",
        "RMSE",
        "MAE",
        "Rel_Err",
        "SSE_course",
        "SST_course",
        "N_course",
        "R2_course",
        "SSE_wales",
        "SST_wales",
        "N_wales",
        "R2_wales",
        "R2_mean",
        "R2_clipped_mean",
        "R2_pooled",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "aggregation_level": "ratio",
                    "sample": result.sample,
                    "stress_measure": result.stress_measure,
                    "stress_unit": result.stress_unit,
                    "ratio": result.physical_ratio or result.ratio,
                    "machine_ratio": result.machine_ratio or result.ratio,
                    "path": str(result.path),
                    "axis1_direction": result.axis_directions[0],
                    "axis2_direction": result.axis_directions[1],
                    "MSE": result.errors["MSE"],
                    "RMSE": result.errors["RMSE"],
                    "MAE": result.errors["MAE"],
                    "Rel_Err": result.errors["Rel_Err"],
                    "SSE_course": result.r_squared_sse["course"],
                    "SST_course": result.r_squared_sst["course"],
                    "N_course": result.observation_count["course"],
                    "R2_course": result.r_squared["course"],
                    "SSE_wales": result.r_squared_sse["wales"],
                    "SST_wales": result.r_squared_sst["wales"],
                    "N_wales": result.observation_count["wales"],
                    "R2_wales": result.r_squared["wales"],
                    "R2_mean": result.mean_r_squared,
                    "R2_clipped_mean": result.clipped_mean_r_squared,
                    "R2_pooled": result.pooled_r_squared,
                }
            )
        for sample_results in group_results_by_sample(results).values():
            summary = selected_ratio_summary(sample_results)
            writer.writerow(
                {
                    "aggregation_level": "selected_ratios_summary",
                    **summary,
                }
            )


def calc_MSE(exp_vals: np.ndarray, model_vals: np.ndarray) -> float:
    """Calculates the mean squared error."""
    check_inputs(exp_vals, model_vals)
    squ_diff = (exp_vals - model_vals) ** 2
    n_elem = exp_vals.size
    mse = np.sum(squ_diff) / n_elem
    return float(mse)


def calc_RMSE(exp_vals: np.ndarray, model_vals: np.ndarray) -> float:
    """Calculates the root mean squared error."""
    return float(np.sqrt(calc_MSE(exp_vals, model_vals)))


def calc_MAE(exp_vals: np.ndarray, model_vals: np.ndarray) -> float:
    """Calculates the mean absolute error."""
    check_inputs(exp_vals, model_vals)
    abs_dif = np.abs(exp_vals - model_vals)
    n_elem = exp_vals.size
    mae = np.sum(abs_dif) / n_elem
    return float(mae)


def calc_RelErr(exp_vals: np.ndarray, model_vals: np.ndarray) -> float:
    """Returns the relative error as a percentage."""
    check_inputs(exp_vals, model_vals)
    non_zero_mask = exp_vals != 0
    exp_vals_n0 = exp_vals[non_zero_mask]
    model_vals_n0 = model_vals[non_zero_mask]
    if exp_vals_n0.size == 0:
        return np.nan
    differ = np.abs(exp_vals_n0 - model_vals_n0)
    relative_errors = differ / np.abs(exp_vals_n0)
    rel_err = np.mean(relative_errors) * 100
    return float(rel_err)


def check_inputs(exp_vals: np.ndarray, model_vals: np.ndarray) -> None:
    """Check that the inputs are valid."""
    if exp_vals.shape != model_vals.shape:
        raise ValueError("The inputs must have the same shape")


if __name__ == "__main__":
    main()
