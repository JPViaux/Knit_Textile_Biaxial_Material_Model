"""Calculate error summaries for multiple saved fits.

Edit FIT_CODES and FIT_OVERRIDES, then run this file. The calculator loads each
fit artifact, evaluates it against the selected experimental data product, and
writes a multi-sheet XLSX summary.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from release_paths import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter

curr_file = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis")
code_dir = curr_file.parent
project_root = code_dir.parent
sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(project_root))

from Python_Wrapper import umat
from Python_Wrapper.model_registry import load_model_spec
from Analysis.core import error_calculator as ec
from Analysis.core.fit_results import DEFAULT_RESULTS_CSV, FitResult, load_fit


# Additional fits to compare in the workbook. Example: {"Purl": 1963}.
# The four entries below are the complete-data reproducibility references.
FIT_CODES: dict[str, int] = {"Interlock": 1962, "Rib": 1961, "Purl": 1963, "SingleJersey": 1964}

# Include the fresh campaign fits recorded by the nine audited Batch states.
# Example: True evaluates B01 through B09; False evaluates only FIT_CODES.
INCLUDE_PROGRESSIVE_STUDY = False
# Folder containing the nine permanent Batch states. Example: states/B01.json.
PROGRESSIVE_STATE_DIRECTORY = project_root / "generated/results/greedy_refitting/states"

# Retry transient OneDrive/antivirus locks around workbook I/O. Example: 12 attempts.
WORKBOOK_IO_ATTEMPTS = 12
# Maximum wait between workbook I/O retries in seconds. Example: 2.0 seconds.
WORKBOOK_IO_MAX_DELAY_SECONDS = 2.0

# Optional old-fit overrides. Example: {1963: {"knit": "Purl", "training_ratios": ("1:1",)}}.
FIT_OVERRIDES: dict[str | int, dict[str, Any]] = {
}

# Experimental data used for error calculation. Example values: "raw".
EVALUATION_EXP_DATA_TYPE = "raw"
# Fallback yarn type when old fit metadata is incomplete. Example: "Tripoly".
DEFAULT_YARN_TYPE = "Tripoly"
# Ratios included in the error calculation. Example value: "0.5 to 1".
DEFAULT_RATIOS = (
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
# CSV index used to locate fit artifacts. Example: DEFAULT_RESULTS_CSV.
# Worker lookup; example: KNIT_RESULTS_CSV points to a newly generated results table.
DEFAULT_RESULTS_CSV_PATH = pathlib.Path(os.environ.get("KNIT_RESULTS_CSV", str(DEFAULT_RESULTS_CSV)))
# Workbook written by this script. Example: SSMCCF/Outputs/global_error_summary.xlsx.
# Optional staging override for a workbook currently open in Excel.
# Example: SSMCCF_GLOBAL_ERROR_OUTPUT=SSMCCF/Outputs/global_error_summary_staged.xlsx.
OUTPUT_XLSX = pathlib.Path(
    os.environ.get(
        "SSMCCF_GLOBAL_ERROR_OUTPUT",
        str(project_root / "generated/results/global_error_summary.xlsx"),
    )
)
# Internal subprocess flag used to release UMAT DLLs between fits. Example: "--worker".
WORKER_ARG = "--worker"
# Timestamp column added to every output row. Example: "2026-07-20T18:30:00.123456+00:00".
RUN_TIMESTAMP_COLUMN = "run_timestamp"
# Worksheet containing the fit-level summary data. Example: "Fit_Summary".
SUMMARY_SHEET_NAME = "Fit_Summary"
# Pivot table whose source should include all summary rows. Example: "PivotTable1".
SUMMARY_PIVOT_TABLE_NAME = "PivotTable1"
# Filename ratio tokens mapped to workbook labels. Example: "0p5_to_1" -> "0.5:1".
DICT_TRAINING_RATIOS: dict[str, str] = {
    "0_to_1": "0:1",
    "0p25_to_1": "0.25:1",
    "0p5_to_1": "0.5:1",
    "0p75_to_1": "0.75:1",
    "1_to_0": "1:0",
    "1_to_0p25": "1:0.25",
    "1_to_0p5": "1:0.5",
    "1_to_0p75": "1:0.75",
    "1_to_1": "1:1",
}


@dataclass(frozen=True)
class FitEvaluationConfig:
    job_name: str
    fit_code: int
    fit: FitResult
    model_id: str
    knit: str
    yarn_type: str
    sample_folder: str
    evaluation_exp_data_type: str
    ratios: tuple[str, ...]
    training_exp_data_type: str | None
    training_exp_data: str | None
    training_exp_data_path: str | None
    training_ratios: tuple[str, ...]
    fixed_theta: str


def main() -> None:
    """Evaluate all configured fits and write the XLSX workbook."""
    run_timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    per_ratio_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="global_error_") as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        for job_name, fit_code in configured_fit_codes().items():
            payload = evaluate_fit_in_worker(job_name, fit_code, temp_path)
            per_ratio_rows.extend(add_run_timestamp(payload["per_ratio_rows"], run_timestamp))
            summary_rows.append(add_run_timestamp(payload["summary_row"], run_timestamp))
            metadata_rows.append(add_run_timestamp(payload["metadata_row"], run_timestamp))

    write_xlsx(OUTPUT_XLSX, per_ratio_rows, summary_rows, metadata_rows)
    print(f"Global error summary written to {OUTPUT_XLSX}")


def configured_fit_codes() -> dict[str, int]:
    """Return reference fits plus the audited progressive-study mappings."""
    selected = dict(FIT_CODES)
    if INCLUDE_PROGRESSIVE_STUDY:
        for job_name, fit_code in progressive_fit_codes_from_states(
            PROGRESSIVE_STATE_DIRECTORY
        ).items():
            if job_name in selected:
                raise ValueError(f"Duplicate configured evaluation name: {job_name}")
            selected[job_name] = fit_code
    return selected


def progressive_fit_codes_from_states(state_directory: pathlib.Path) -> dict[str, int]:
    """Read exactly one complete fit code from each of 72 stored case records."""
    mappings: dict[str, int] = {}
    for batch_number in range(1, 10):
        state_path = state_directory / f"B{batch_number:02d}.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"Progressive Batch state was not found: {state_path}")
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        cases = payload.get("cases", [])
        if not isinstance(cases, list) or len(cases) != 8:
            raise ValueError(f"{state_path.name} must contain exactly eight cases")
        for case in cases:
            if case.get("status") != "completed":
                raise ValueError(
                    f"{state_path.name} contains an incomplete case: {case.get('case_id')}"
                )
            codes = case.get("fit_codes", [])
            if not isinstance(codes, list) or len(codes) != 1:
                raise ValueError(
                    f"{state_path.name} case {case.get('case_id')} must have one fit code"
                )
            case_id = str(case.get("case_id", "")).strip()
            if not case_id or case_id in mappings:
                raise ValueError(f"Duplicate or empty progressive case_id: {case_id!r}")
            mappings[case_id] = int(codes[0])
    if len(mappings) != 72 or len(set(mappings.values())) != 72:
        raise ValueError("Progressive states must resolve 72 unique cases and fit codes")
    return mappings


def run_worker(
    job_name: str,
    fit_code: int,
    output_path: pathlib.Path,
    evaluation_exp_data_type: str | None = None,
) -> None:
    """Evaluate one fit in an isolated process and write JSON rows."""
    config = build_fit_config(
        job_name,
        fit_code,
        evaluation_exp_data_type=evaluation_exp_data_type,
    )
    ratio_results = evaluate_fit(config)
    payload = {
        "per_ratio_rows": per_ratio_output_rows(config, ratio_results),
        "summary_row": fit_summary_row(config, ratio_results),
        "metadata_row": fit_metadata_row(config),
    }
    output_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def evaluate_fit_in_worker(
    job_name: str,
    fit_code: int,
    temp_path: pathlib.Path,
    isolated_cwd: bool = False,
    evaluation_exp_data_type: str | None = None,
    results_csv: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Evaluate one fit in a subprocess so Windows releases compiled UMAT DLLs."""
    worker_directory = project_root
    output_path = temp_path / f"{job_name}_{fit_code}.json"
    if isolated_cwd:
        # Parallel workers need separate Meson files and DLL build directories.
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{job_name}_{fit_code}")
        worker_directory = temp_path / f"worker_{safe_name}"
        worker_directory.mkdir(parents=True, exist_ok=False)
        output_path = worker_directory / "evaluation.json"
    command = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        WORKER_ARG,
        job_name,
        str(fit_code),
        str(output_path),
    ]
    if evaluation_exp_data_type is not None:
        command.append(ec.normalize_exp_data_type(evaluation_exp_data_type))
    environment = os.environ.copy()
    if results_csv is not None:
        environment["KNIT_RESULTS_CSV"] = str(results_csv)
    result = subprocess.run(
        command,
        env=environment,
        cwd=str(worker_directory),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"Global error worker failed for {job_name} / fit {fit_code} "
            f"with exit code {result.returncode}."
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def build_fit_config(
    job_name: str,
    fit_code: int,
    evaluation_exp_data_type: str | None = None,
) -> FitEvaluationConfig:
    """Load one fit and resolve evaluation context from JSON plus overrides."""
    fit = load_fit(fit_code, DEFAULT_RESULTS_CSV_PATH)
    metadata = fit.metadata
    context = experiment_context(metadata)
    override = merged_override(job_name, fit_code)

    model_id = str(override.get("model_id", fit.model))
    knit = str(
        override.get("knit")
        or context.get("knit")
        or infer_knit(metadata)
        or ""
    )
    if not knit:
        raise ValueError(
            f"Could not infer knit for fit {fit_code}. Add FIT_OVERRIDES[{fit_code}] "
            "with {'knit': 'Interlock'|'Purl'|'Rib'|'SingleJersey'}."
        )

    yarn_type = str(override.get("yarn_type") or context.get("yarn_type") or DEFAULT_YARN_TYPE)
    sample_folder = str(
        override.get("sample_folder")
        or context.get("sample_folder")
        or ec.DEFAULT_RAW_SAMPLE_FOLDERS[knit]
    )
    resolved_evaluation_type = ec.normalize_exp_data_type(
        str(
            evaluation_exp_data_type
            or override.get("evaluation_exp_data_type", EVALUATION_EXP_DATA_TYPE)
        )
    )
    ratios = tuple(
        ec.normalize_ratio_label(ratio)
        for ratio in override.get("ratios", DEFAULT_RATIOS)
    )

    training_exp_data_type = (
        override.get("training_exp_data_type")
        or context.get("exp_data_type")
        or infer_exp_data_type(metadata)
    )
    training_exp_data = optional_string(context.get("exp_data") or metadata.get("Exp_Data"))
    training_exp_data_path = optional_string(
        context.get("exp_data_path") or metadata.get("Exp_Data_Path")
    )
    return FitEvaluationConfig(
        job_name=job_name,
        fit_code=fit_code,
        fit=fit,
        model_id=model_id,
        knit=knit,
        yarn_type=yarn_type,
        sample_folder=sample_folder,
        evaluation_exp_data_type=resolved_evaluation_type,
        ratios=ratios,
        training_exp_data_type=str(training_exp_data_type) if training_exp_data_type else None,
        training_exp_data=training_exp_data,
        training_exp_data_path=training_exp_data_path,
        training_ratios=resolve_training_ratios(
            override.get("training_ratios"),
            training_exp_data,
            training_exp_data_path,
        ),
        fixed_theta=resolve_fixed_theta(fit),
    )


def evaluate_fit(config: FitEvaluationConfig) -> list[ec.RatioErrorResult]:
    """Evaluate one fit across the configured ratios."""
    model_spec = load_model_spec(config.model_id)
    parameters = {
        name: config.fit.parameters[name]
        for name in model_spec.parameter_names
    }
    sim_par = np.asarray(model_spec.pack_parameters(parameters), dtype=float)

    print(
        f"Evaluating fit {config.fit_code} ({config.model_id}, {config.knit}) "
        f"against {config.evaluation_exp_data_type} data"
    )
    umat.compile_umat_meson(model_spec.fortran_source)
    return [
        ec.evaluate_ratio(
            ratio,
            sim_par,
            config.knit,
            None,
            config.evaluation_exp_data_type,
            config.yarn_type,
            config.sample_folder,
        )
        for ratio in config.ratios
    ]


def per_ratio_output_rows(
    config: FitEvaluationConfig,
    results: Sequence[ec.RatioErrorResult],
) -> list[dict[str, Any]]:
    """Build one output row per fit/ratio."""
    rows: list[dict[str, Any]] = []
    batch_id = progressive_batch_id(config)
    training_ratios = set(config.training_ratios)
    for result in results:
        ratio_label = normalize_training_ratio(result.ratio)
        rows.append(
            {
                "job_name": config.job_name,
                "fit_code": config.fit_code,
                "model": config.model_id,
                "knit": config.knit,
                "yarn_type": config.yarn_type,
                "evaluation_exp_data_type": config.evaluation_exp_data_type,
                "sample_folder": config.sample_folder,
                "progressive_batch": batch_id,
                "ratio": result.ratio,
                "ratio_training_status": (
                    "trained" if ratio_label in training_ratios else "unseen"
                ),
                "axis1_direction": result.axis_directions[0],
                "axis2_direction": result.axis_directions[1],
                "MSE": result.errors["MSE"],
                "RMSE": result.errors["RMSE"],
                "MAE": result.errors["MAE"],
                "Rel_Err": result.errors["Rel_Err"],
                "R2_course": result.r_squared["course"],
                "R2_wales": result.r_squared["wales"],
                "R2_mean": result.mean_r_squared,
                "R2_clipped_mean": result.clipped_mean_r_squared,
                "R2_pooled": result.pooled_r_squared,
                "SSE_course": result.r_squared_sse["course"],
                "SSE_wales": result.r_squared_sse["wales"],
                "SST_course": result.r_squared_sst["course"],
                "SST_wales": result.r_squared_sst["wales"],
                "N_course": result.observation_count["course"],
                "N_wales": result.observation_count["wales"],
                "exp_path": str(result.path),
            }
        )
    return rows


def fit_summary_row(
    config: FitEvaluationConfig,
    results: Sequence[ec.RatioErrorResult],
) -> dict[str, Any]:
    """Build the fit-level summary row."""
    course_r_squared = [result.r_squared["course"] for result in results]
    wales_r_squared = [result.r_squared["wales"] for result in results]
    all_r_squared = course_r_squared + wales_r_squared
    course_sse = [result.r_squared_sse["course"] for result in results]
    course_sst = [result.r_squared_sst["course"] for result in results]
    wales_sse = [result.r_squared_sse["wales"] for result in results]
    wales_sst = [result.r_squared_sst["wales"] for result in results]
    course_count = [result.observation_count["course"] for result in results]
    wales_count = [result.observation_count["wales"] for result in results]
    pooled_r_squared_course = ec.calc_pooled_r_squared(course_sse, course_sst)
    pooled_r_squared_wales = ec.calc_pooled_r_squared(wales_sse, wales_sst)
    pooled_r_squared = ec.calc_pooled_r_squared(
        course_sse + wales_sse,
        course_sst + wales_sst,
    )
    pooled_rmse_course = ec.calc_pooled_rmse(course_sse, course_count)
    pooled_rmse_wales = ec.calc_pooled_rmse(wales_sse, wales_count)
    pooled_rmse = ec.calc_pooled_rmse(
        course_sse + wales_sse,
        course_count + wales_count,
    )
    return {
        "job_name": config.job_name,
        "fit_code": config.fit_code,
        "model": config.model_id,
        "knit": config.knit,
        "yarn_type": config.yarn_type,
        "evaluation_exp_data_type": config.evaluation_exp_data_type,
        "sample_folder": config.sample_folder,
        "progressive_batch": progressive_batch_id(config),
        "n_ratios": len(results),
        "mean_MSE": float(np.mean([result.errors["MSE"] for result in results])),
        "mean_RMSE": float(np.mean([result.errors["RMSE"] for result in results])),
        "pooled_RMSE": pooled_rmse,
        "pooled_RMSE_course": pooled_rmse_course,
        "pooled_RMSE_wales": pooled_rmse_wales,
        "pooled_SSE": float(np.sum(course_sse + wales_sse)),
        "pooled_N": int(np.sum(course_count + wales_count)),
        "mean_MAE": float(np.mean([result.errors["MAE"] for result in results])),
        "mean_Rel_Err": float(np.mean([result.errors["Rel_Err"] for result in results])),
        "mean_R2": float(np.nanmean(all_r_squared)),
        "clipped_mean_R2": ec.calc_clipped_mean_r_squared(all_r_squared),
        "pooled_R2": pooled_r_squared,
        "mean_R2_course": float(np.nanmean(course_r_squared)),
        "clipped_mean_R2_course": ec.calc_clipped_mean_r_squared(course_r_squared),
        "pooled_R2_course": pooled_r_squared_course,
        "mean_R2_wales": float(np.nanmean(wales_r_squared)),
        "clipped_mean_R2_wales": ec.calc_clipped_mean_r_squared(wales_r_squared),
        "pooled_R2_wales": pooled_r_squared_wales,
        "min_R2": float(np.nanmin(all_r_squared)),
        "max_R2": float(np.nanmax(all_r_squared)),
        "Training_ratios": ", ".join(config.training_ratios),
        "Fixed_theta?": config.fixed_theta,
    }


def fit_metadata_row(config: FitEvaluationConfig) -> dict[str, Any]:
    """Build traceability metadata for one fit."""
    metadata = config.fit.metadata
    return {
        "job_name": config.job_name,
        "fit_code": config.fit_code,
        "model": config.model_id,
        "progressive_batch": progressive_batch_id(config),
        "fit_json": str(config.fit.source),
        "parameter_count": len(config.fit.parameters),
        "created_at": metadata.get("created_at", ""),
        "training_exp_data_type": config.training_exp_data_type or "",
        "training_exp_data": config.training_exp_data or "",
        "training_exp_data_path": config.training_exp_data_path or "",
        "training_param_file": metadata.get("Param_File", ""),
        "training_param_file_path": metadata.get("Param_File_Path", ""),
        "fit_min_cost": metadata.get("Min_Cost", ""),
        "fit_avg_cost": metadata.get("Avg_Cost", ""),
        "fit_cost_type": metadata.get("Cost_Type", ""),
    }


def progressive_batch_id(config: FitEvaluationConfig) -> str:
    """Infer a progressive-study label such as B03 from stored traceability fields."""
    metadata = config.fit.metadata
    candidates = (
        config.job_name,
        metadata.get("Case_ID", ""),
        config.training_exp_data or "",
        config.training_exp_data_path or "",
    )
    for candidate in candidates:
        match = re.search(
            r"progressive[_-]?b(\d{2})",
            str(candidate),
            re.IGNORECASE,
        )
        if match:
            return f"B{match.group(1)}"
    return ""


def write_xlsx(
    path: pathlib.Path,
    per_ratio_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Append the multi-sheet XLSX output, preserving previous run rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        workbook = retry_permission_error(
            lambda: load_workbook(path),
            f"open {path}",
        )
    else:
        workbook = Workbook()
        default_sheet = workbook.active
        workbook.remove(default_sheet)

    append_sheet(workbook, "Per_Ratio", per_ratio_rows)
    append_sheet(workbook, SUMMARY_SHEET_NAME, summary_rows)
    append_sheet(workbook, "Fit_Metadata", metadata_rows)
    update_summary_pivot_source(workbook)
    save_workbook_atomic(workbook, path)


def retry_permission_error(operation: Any, description: str) -> Any:
    """Retry an operation when a transient Windows file lock raises PermissionError."""
    for attempt in range(1, WORKBOOK_IO_ATTEMPTS + 1):
        try:
            return operation()
        except PermissionError:
            if attempt == WORKBOOK_IO_ATTEMPTS:
                raise
            delay = min(0.1 * (2 ** (attempt - 1)), WORKBOOK_IO_MAX_DELAY_SECONDS)
            print(
                f"Workbook lock while attempting to {description}; "
                f"retrying in {delay:.1f} s ({attempt}/{WORKBOOK_IO_ATTEMPTS})."
            )
            time.sleep(delay)
    raise AssertionError("Workbook retry loop ended unexpectedly")


def save_workbook_atomic(workbook: Any, path: pathlib.Path) -> None:
    """Save to a sibling temporary file and atomically replace the workbook."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        delete=False,
    ) as temporary_file:
        temporary_path = pathlib.Path(temporary_file.name)
    try:
        workbook.save(temporary_path)
        retry_permission_error(
            lambda: temporary_path.replace(path),
            f"replace {path}",
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def append_sheet(workbook: Any, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append rows to one worksheet, adding new columns when needed."""
    worksheet = workbook[title] if title in workbook.sheetnames else workbook.create_sheet(title)
    if not rows:
        return
    headers = ensure_headers(worksheet, rows)
    for row in rows:
        worksheet.append([xlsx_value(row.get(header, "")) for header in headers])
    format_sheet(worksheet)


def ensure_headers(worksheet: Any, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return worksheet headers, appending any new row keys to the header row."""
    new_headers = ordered_headers(rows)
    if worksheet.max_row == 1 and worksheet.cell(row=1, column=1).value is None:
        for column, header in enumerate(new_headers, start=1):
            worksheet.cell(row=1, column=column, value=header)
        return new_headers

    headers = [
        worksheet.cell(row=1, column=column).value
        for column in range(1, worksheet.max_column + 1)
    ]
    headers = [str(header) for header in headers if header not in (None, "")]
    for header in new_headers:
        if header not in headers:
            headers.append(header)
            worksheet.cell(row=1, column=len(headers), value=header)
    return headers


def ordered_headers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return first-seen row keys in stable order."""
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(str(key))
    return headers


def format_sheet(worksheet: Any) -> None:
    """Resize columns and keep the header row visible."""
    for column_cells in worksheet.columns:
        max_length = max(len(str(cell.value or "")) for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(
            max(max_length + 2, 10),
            70,
        )
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions


def update_summary_pivot_source(workbook: Any) -> None:
    """Point PivotTable1 at the full used range on Fit_Summary."""
    if SUMMARY_SHEET_NAME not in workbook.sheetnames:
        return
    summary_worksheet = workbook[SUMMARY_SHEET_NAME]
    if summary_worksheet.max_row < 1 or summary_worksheet.max_column < 1:
        return

    source_ref = worksheet_dimensions_ref(summary_worksheet)
    for worksheet in workbook.worksheets:
        for pivot_table in getattr(worksheet, "_pivots", ()):
            if pivot_table.name != SUMMARY_PIVOT_TABLE_NAME:
                continue
            worksheet_source = pivot_table.cache.cacheSource.worksheetSource
            if worksheet_source is None:
                raise ValueError(
                    f"{SUMMARY_PIVOT_TABLE_NAME} does not use a worksheet source."
                )
            worksheet_source.sheet = SUMMARY_SHEET_NAME
            worksheet_source.ref = source_ref
            worksheet_source.name = None
            pivot_table.cache.refreshOnLoad = True
            pivot_table.cache.enableRefresh = True
            return


def worksheet_dimensions_ref(worksheet: Any) -> str:
    """Return an A1-style reference covering the used worksheet dimensions."""
    last_column = get_column_letter(worksheet.max_column)
    return f"A1:{last_column}{worksheet.max_row}"


def xlsx_value(value: Any) -> Any:
    """Convert arrays and structured metadata into Excel-safe scalar values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, default=str)
    return value


def experiment_context(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the structured experiment_context block when available."""
    context = metadata.get("experiment_context", {})
    return context if isinstance(context, dict) else {}


def merged_override(job_name: str, fit_code: int) -> dict[str, Any]:
    """Merge job-name and fit-code overrides."""
    override: dict[str, Any] = {}
    override.update(FIT_OVERRIDES.get(fit_code, {}))
    override.update(FIT_OVERRIDES.get(job_name, {}))
    return override


def infer_knit(metadata: Mapping[str, Any]) -> str | None:
    """Infer knit from historical fit metadata when explicit context is absent."""
    text = " ".join(
        str(metadata.get(key, ""))
        for key in ("Exp_Data", "Exp_Data_Path", "Param_File", "Param_File_Path")
    ).lower()
    tokens = {
        "interlock": "Interlock",
        "purl": "Purl",
        "rib": "Rib",
        "singlejersey": "SingleJersey",
        "single jersey": "SingleJersey",
        "_sj_": "SingleJersey",
    }
    for token, knit in tokens.items():
        if token in text:
            return knit
    return None


def infer_exp_data_type(metadata: Mapping[str, Any]) -> str | None:
    """The publication contains raw compiled data only."""
    return "raw"


def resolve_training_ratios(
    override: Any,
    training_exp_data: str | None,
    training_exp_data_path: str | None,
) -> tuple[str, ...]:
    """Resolve the ratios used to train one fit from an override, filename, or README."""
    if override not in (None, ""):
        values = [override] if isinstance(override, str) else list(override)
        return tuple(normalize_training_ratio(value) for value in values)

    training_file = training_exp_data_path or training_exp_data
    if not training_file:
        raise ValueError(
            "Training ratios cannot be resolved because the fit has no training-data "
            "filename. Add a 'training_ratios' entry to its FIT_OVERRIDES record."
        )

    filename_ratio = training_ratio_from_filename(training_file)
    if filename_ratio is not None:
        return (filename_ratio,)

    stem = normalized_filename_stem(training_file)
    # Complete raw products encode the knit instead of one ratio, e.g. biax_exp_data_rib.dat.
    if is_complete_raw_training_file(stem):
        return tuple(DICT_TRAINING_RATIOS.values())
    if not is_grouped_training_file(stem):
        raise ValueError(
            f"Could not find a recognized ratio in training filename {training_file!r}. "
            "Add its token to DICT_TRAINING_RATIOS or use FIT_OVERRIDES."
        )
    if not training_exp_data_path:
        raise ValueError(
            f"Grouped training file {training_file!r} requires its full path so "
            "Readme.txt can be read. Add 'training_ratios' to FIT_OVERRIDES."
        )

    return training_ratios_from_readme(pathlib.Path(training_exp_data_path))


def training_ratio_from_filename(training_file: str | pathlib.Path) -> str | None:
    """Return the single ratio encoded in a filename such as 0p5_to_1_clean.dat."""
    stem = normalized_filename_stem(training_file)
    for token, ratio in DICT_TRAINING_RATIOS.items():
        if stem == token or stem.startswith(f"{token}_"):
            return ratio
    return None


def training_ratios_from_readme(training_path: pathlib.Path) -> tuple[str, ...]:
    """Read the matching grouped-file ratio list from the adjacent Readme.txt."""
    readme_path = next(
        (
            path
            for path in training_path.parent.iterdir()
            if path.is_file() and path.name.lower() == "readme.txt"
        ),
        None,
    )
    if readme_path is None:
        raise FileNotFoundError(
            f"Readme.txt was not found beside grouped training file {training_path}."
        )

    target_stem = normalized_filename_stem(training_path)
    for line in readme_path.read_text(encoding="utf-8-sig").splitlines():
        entry_name, separator, ratio_text = line.partition(":")
        if separator and normalized_filename_stem(entry_name) == target_stem:
            ratios = tuple(
                normalize_training_ratio(f"{left}:{right}")
                for left, right in re.findall(
                    r"(?<![\d.])(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)(?![\d.])",
                    ratio_text,
                )
            )
            if ratios:
                return ratios
            raise ValueError(
                f"README entry for {training_path.name} contains no recognized ratios: "
                f"{line!r}"
            )

    # An explicit *_all filename means every ratio in the configured dictionary.
    if target_stem.endswith("_all"):
        return tuple(DICT_TRAINING_RATIOS.values())
    raise ValueError(
        f"Readme entry for grouped training file {training_path.name!r} was not found "
        f"in {readme_path}."
    )


def normalized_filename_stem(value: str | pathlib.Path) -> str:
    """Normalize a training filename for matching; for example, '0.5 to 1' -> '0p5_to_1'."""
    stem = pathlib.Path(str(value).strip()).stem.lower()
    return re.sub(r"_+", "_", stem.replace(" ", "_").replace(".", "p"))


def is_complete_raw_training_file(stem: str) -> bool:
    """Return whether a stem uses the complete raw-data convention, e.g. biax_exp_data_rib."""
    return bool(re.fullmatch(r"biax_exp_data_[a-z0-9]+", stem))


def is_grouped_training_file(stem: str) -> bool:
    """Return whether a filename represents several experiments instead of one ratio."""
    return (
        "training_batch" in stem
        or "adaptive_greedy" in stem
        or "three_ratio_" in stem
        or bool(re.match(r"^clean(?:ed)?_biax_", stem))
    )


def normalize_training_ratio(value: Any) -> str:
    """Normalize a ratio to colon notation; for example, '0.5 to 1' -> '0.5:1'."""
    text = str(value).strip().lower().replace("p", ".")
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(?::|\s+to\s+|_to_)(\d+(?:\.\d+)?)",
        text,
    )
    if not match:
        raise ValueError(f"Invalid training ratio {value!r}; expected a value such as '0.5:1'.")
    ratio = f"{match.group(1)}:{match.group(2)}"
    if ratio not in DICT_TRAINING_RATIOS.values():
        raise ValueError(
            f"Training ratio {ratio!r} is not configured in DICT_TRAINING_RATIOS."
        )
    return ratio


def resolve_fixed_theta(fit: FitResult) -> str:
    """Return Y only when every theta parameter was fixed for the fitting run."""
    theta_flags = theta_flags_from_fit_json(fit.source)
    if not theta_flags and fit.source.suffix.lower() == ".csv":
        theta_flags = theta_flags_from_results_csv(fit.source, fit.code)
    if not theta_flags:
        parameter_path = optional_string(fit.metadata.get("Param_File_Path"))
        if parameter_path:
            theta_flags = theta_flags_from_parameter_file(pathlib.Path(parameter_path))
    if not theta_flags:
        raise ValueError(
            f"Could not determine theta fixed/free settings for fit {fit.code}."
        )
    return "Y" if all(theta_flags) else "N"


def theta_flags_from_fit_json(source: pathlib.Path) -> tuple[bool, ...]:
    """Read theta fixed flags from a fit artifact when parameter_specs is available."""
    if source.suffix.lower() != ".json":
        return ()
    payload = json.loads(source.read_text(encoding="utf-8"))
    specifications = payload.get("parameter_specs", {})
    if not isinstance(specifications, Mapping):
        return ()
    return tuple(
        parse_boolean(specification.get("fixed"))
        for name, specification in specifications.items()
        if str(name).lower().startswith("theta") and isinstance(specification, Mapping)
    )


def theta_flags_from_results_csv(source: pathlib.Path, fit_code: int) -> tuple[bool, ...]:
    """Read theta fixed flags from historical results_final.csv parameter rows."""
    with source.open(newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        return tuple(
            parse_boolean(row.get("Fix?"))
            for row in rows
            if csv_fit_code(row.get("Code")) == fit_code
            and str(row.get("Parameter", "")).lower().startswith("theta")
        )


def theta_flags_from_parameter_file(path: pathlib.Path) -> tuple[bool, ...]:
    """Read theta fixed flags from a six-column material-parameter input file."""
    properties = umat.Load_props(path).get_start_parameters()
    return tuple(
        bool(specification[5])
        for name, specification in properties.items()
        if str(name).lower().startswith("theta")
    )


def parse_boolean(value: Any) -> bool:
    """Parse a stored boolean; for example, True, 'True', False, or 'False'."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value {value!r} in theta parameter metadata.")


def csv_fit_code(value: Any) -> int | None:
    """Parse a CSV fit code such as 544 or '544.0'."""
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def optional_string(value: Any) -> str | None:
    """Return a stripped string, or None for missing values."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def add_run_timestamp(rows: Any, run_timestamp: str) -> Any:
    """Add the run timestamp to one row or a list of rows."""
    if isinstance(rows, Mapping):
        return {RUN_TIMESTAMP_COLUMN: run_timestamp, **rows}
    return [{RUN_TIMESTAMP_COLUMN: run_timestamp, **row} for row in rows]


def json_ready(value: Any) -> Any:
    """Convert output values into strict JSON-friendly Python objects."""
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_ARG:
        if len(sys.argv) not in (5, 6):
            raise SystemExit(
                f"Usage: {pathlib.Path(__file__).name} {WORKER_ARG} "
                "<job_name> <fit_code> <output_json> [raw]"
            )
        run_worker(
            sys.argv[2],
            int(sys.argv[3]),
            pathlib.Path(sys.argv[4]),
            sys.argv[5] if len(sys.argv) == 6 else None,
        )
    else:
        main()
