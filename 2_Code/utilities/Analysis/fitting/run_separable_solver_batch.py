"""Run explicitly selected separable-solver case manifests as one batch.

Edit the USER CONFIGURATION block and press Run in VSCode. The default is a
safe eight-case preview; set ``PREVIEW_ONLY = False`` to launch production
fits. Each case runs in a fresh process and its own Meson working directory.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

import argparse
import csv
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from release_paths import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid


os.environ.setdefault("MPLBACKEND", "Agg")


# Allow direct VS Code execution after this module moved below Analysis/.
import sys

_analysis_directory = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis")
_ssmccf_directory = _analysis_directory.parent
_code_directory = _ssmccf_directory.parent
for _import_directory in (_ssmccf_directory, _code_directory):
    if str(_import_directory) not in sys.path:
        sys.path.insert(0, str(_import_directory))
from Analysis.fitting import separable_solver_fit as solver


PathInput = str | pathlib.Path
CaseJob = dict[str, Any]
BatchEntry = dict[str, Any]

# ---------------------------------------------------------------------------
# USER CONFIGURATION - edit these values before pressing Run in VSCode.
# ---------------------------------------------------------------------------

# Progressive Batch labels. Example: "b01" fits only the 1:1 curve.
PROGRESSIVE_BATCH_TOKENS = tuple(f"b{number:02d}" for number in range(1, 10))
# Model filename tokens. Example: "sv_exp_polycon_mo" selects the shipped model.
PROGRESSIVE_MODEL_TOKENS = ("sv_exp_polycon_mo",)
# Knit/sample filename tokens. Example: "interlock_s06" selects Interlock Sample 06.
PROGRESSIVE_KNIT_TOKENS = (
    "interlock_s06",
    "rib_s03",
    "purl_s02",
    "singlejersey_s02",
)

# Select the eight shipped calibration cases.
SELECTED_CASE_FILES: tuple[PathInput, ...] = tuple(path.name for path in sorted(solver.CASE_DIRECTORY.glob("*.json")))

# Select simultaneous case workers. Examples: 1 for sequential work or 5 for
# the recommended responsive parallel run on the current workstation.
DEFAULT_PHYSICAL_CORES = 5

# Keep this True while checking manifests. Set False to create production fits.
# Examples: True for validation only; False to execute the selected cases.
PREVIEW_ONLY = True

# Optional JSON case-list override used by generated studies. Example:
# Outputs/adaptive_ratio_study/case_lists/B01.json. Keep None for the tuple above.
DEFAULT_CASE_LIST_FILE: pathlib.Path | None = None

# ---------------------------------------------------------------------------
# INTERNAL SETTINGS - normally no changes are needed below this line.
# ---------------------------------------------------------------------------

# Store resumable state separately from the Monte Carlo batch state. Example:
# Outputs/separable_solver_batch_state.json.
DEFAULT_STATE_FILE = (
    solver.RUN_SUMMARY_ROOT / "batch_state.json"
)

# Retry transient OneDrive or antivirus locks. Example: 12 attempts allows
# roughly 20 seconds of bounded exponential backoff.
STATE_WRITE_MAX_ATTEMPTS = 12

# Initial state-write retry delay in seconds. Example: 0.25 seconds.
STATE_WRITE_RETRY_DELAY_SECONDS = 0.25


def physical_core_count() -> int | None:
    """Return the physical CPU core count when psutil is available."""
    try:
        import psutil
    except ImportError:
        return None
    count = psutil.cpu_count(logical=False)
    return int(count) if count else None


def resolve_case_path(value: PathInput) -> pathlib.Path:
    """Resolve a filename, code-relative path, or absolute manifest path."""
    candidate = pathlib.Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parent == pathlib.Path("."):
        return (solver.CASE_DIRECTORY / candidate).resolve()
    return (solver.SUPPORT_CODE_DIR / candidate).resolve()


def load_case_list(path: pathlib.Path) -> tuple[PathInput, ...]:
    """Read an ordered JSON case list, for example ``{"case_files": [...]}``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("case_files") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("case-list JSON must be a non-empty list or contain case_files")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("case-list entries must be non-empty path strings")
    return tuple(values)


def case_signature(job: Mapping[str, Any], index: int) -> str:
    """Hash one validated case and every direct input used by its solve."""
    config = job["config"]
    payload = {
        "index": index,
        "case_file": str(pathlib.Path(job["case_file"]).resolve()),
        "case_id": config.case_id,
        "model_id": config.model_id,
        "result_roles": list(config.result_roles),
        "input_hashes": dict(job["input_hashes"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_case_jobs(
    selected_files: Sequence[PathInput] | None = None,
) -> list[CaseJob]:
    """Resolve and fully preflight all configured direct-input cases."""
    selected = tuple(SELECTED_CASE_FILES if selected_files is None else selected_files)
    if not selected:
        raise ValueError("SELECTED_CASE_FILES must contain at least one manifest")

    paths = [resolve_case_path(value) for value in selected]
    if len(paths) != len(set(paths)):
        raise ValueError("SELECTED_CASE_FILES contains a duplicate manifest path")

    jobs: list[CaseJob] = []
    case_ids: set[str] = set()
    for index, path in enumerate(paths, start=1):
        try:
            config = solver.load_case_configuration(path)
            model_spec = solver.load_model_spec(config.model_id)
            raw_specs, effective_specs = solver.parameter_specs_from_file(
                config,
                model_spec,
            )
            preflight = solver.case_preflight(
                config,
                effective_specs,
                require_production=True,
            )
            hashes = solver.input_hashes(config)
        except Exception as exc:
            raise ValueError(
                f"case {index} ({path.name}) failed preflight: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if config.case_id in case_ids:
            raise ValueError(f"Duplicate case_id in selected manifests: {config.case_id}")
        case_ids.add(config.case_id)
        jobs.append(
            {
                "case_file": path,
                "config": config,
                "model_spec": model_spec,
                "raw_parameter_specs": raw_specs,
                "effective_parameter_specs": effective_specs,
                "preflight": preflight,
                "input_hashes": hashes,
            }
        )
    return jobs


def print_case_preview(jobs: Sequence[CaseJob], indices: range) -> None:
    """Print every resolved case without compiling a UMAT or assigning a code."""
    print("\nResolved separable-solver cases")
    print("#  Case ID | Knit | Model | Roles")
    for index in indices:
        job = jobs[index - 1]
        config = job["config"]
        print(
            f"{index}  {config.case_id} | {config.knit} | {config.model_id} | "
            f"{','.join(config.result_roles)}"
        )
        print(f"   Case manifest:     {job['case_file']}")
        print(f"   Experimental data: {config.experimental_data}")
        print(f"   Parameter file:    {config.parameter_file}")
        print(f"   UMAT source:       {config.umat_source}")
        print(f"   Fit axes:           {config.fit_axes}")
        print(
            f"   Backend / shapes:   {job['preflight']['backend']} / "
            f"{job['preflight']['shape_summary']}"
        )


def write_state(state_path: pathlib.Path, state: dict[str, Any]) -> None:
    """Write batch state atomically with bounded transient-lock retries."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(state, indent=2) + "\n"

    for attempt in range(STATE_WRITE_MAX_ATTEMPTS):
        temporary_path = state_path.with_name(
            f".{state_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                file.write(payload)
            os.replace(temporary_path, state_path)
            return
        except PermissionError:
            if attempt + 1 >= STATE_WRITE_MAX_ATTEMPTS:
                raise
            delay = min(STATE_WRITE_RETRY_DELAY_SECONDS * (2**attempt), 2.0)
            print(
                f"State file is temporarily locked; retrying in {delay:.2f} s "
                f"({attempt + 2}/{STATE_WRITE_MAX_ATTEMPTS})."
            )
            time.sleep(delay)
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def load_state(state_path: pathlib.Path) -> dict[str, Any]:
    """Load and minimally validate a separable-solver batch state file."""
    with state_path.open(encoding="utf-8") as file:
        state = json.load(file)
    if state.get("schema_version") != 1 or not isinstance(state.get("cases"), list):
        raise ValueError(f"Invalid separable-solver batch state: {state_path}")
    return state


def create_state(
    jobs: Sequence[CaseJob],
    indices: range,
    cores: int,
    state_path: pathlib.Path,
) -> dict[str, Any]:
    """Create case-level state without reserving any production fit codes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_dir = state_path.parent / "separable_solver_batch_runs" / timestamp
    batch_group = f"separable_batch_{timestamp}"
    entries: list[BatchEntry] = []
    for index in indices:
        job = jobs[index - 1]
        config = job["config"]
        run_group = f"{batch_group}_{index:02d}_{config.case_id}"
        entries.append(
            {
                "index": index,
                "name": config.run_label,
                "case_id": config.case_id,
                "case_file": str(pathlib.Path(job["case_file"]).resolve()),
                "signature": case_signature(job, index),
                "input_hashes": dict(job["input_hashes"]),
                "result_roles": list(config.result_roles),
                "worker_dir": str((batch_dir / f"case_{index:02d}").resolve()),
                "run_group": run_group,
                "summary_path": str(
                    (solver.RUN_SUMMARY_ROOT / run_group / "summary.json").resolve()
                ),
                "status": "pending",
                "return_code": None,
                "fit_codes": [],
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cores": cores,
        "batch_group": batch_group,
        "batch_dir": str(batch_dir.resolve()),
        "cases": entries,
    }


def validate_resumed_state(
    state: Mapping[str, Any],
    jobs: Sequence[CaseJob],
) -> None:
    """Reject resume when a selected case or any direct input has changed."""
    seen_indices: set[int] = set()
    for entry in state["cases"]:
        index = int(entry["index"])
        if index in seen_indices:
            raise ValueError(f"Batch state repeats case index {index}")
        seen_indices.add(index)
        if index < 1 or index > len(jobs):
            raise ValueError(f"Batch state references missing case index {index}")
        expected = case_signature(jobs[index - 1], index)
        if entry.get("signature") != expected:
            raise ValueError(
                f"Case {index} configuration or direct inputs changed since state creation"
            )


def _read_registry_rows() -> list[dict[str, str]]:
    """Read the method-neutral registry used to audit completed exports."""
    registry_path = solver.LEGACY_RESULTS_CSV.parent / "fit_registry.csv"
    if not registry_path.is_file():
        return []
    with registry_path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _parse_positive_code(value: Any) -> int | None:
    """Return a positive fit code or None for malformed registry text."""
    try:
        code = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return code if code > 0 else None


def _roles_from_labels(labels: Sequence[str]) -> set[str]:
    """Expand combined labels such as training_optimum+predictive_model."""
    roles: set[str] = set()
    for label in labels:
        roles.update(part for part in str(label).split("+") if part)
    return roles


def audit_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one case as complete, untouched, or partially exported."""
    run_group = str(entry["run_group"])
    registry_rows = [
        row for row in _read_registry_rows() if row.get("Run_Group") == run_group
    ]
    registry_codes = sorted(
        code
        for code in (_parse_positive_code(row.get("Code")) for row in registry_rows)
        if code is not None
    )
    summary_path = pathlib.Path(entry["summary_path"])
    if not summary_path.is_file():
        if registry_rows:
            return {
                "status": "partial",
                "fit_codes": registry_codes,
                "message": "registry entries exist but the run summary is missing",
            }
        return {"status": "untouched", "fit_codes": [], "message": ""}

    try:
        with summary_path.open(encoding="utf-8") as file:
            summary = json.load(file)
        if summary.get("schema_version") != 1:
            raise ValueError("summary schema_version is not 1")
        if summary.get("run_group") != run_group:
            raise ValueError("summary run_group does not match batch state")
        if summary.get("case_id") != entry.get("case_id"):
            raise ValueError("summary case_id does not match batch state")
        if pathlib.Path(summary.get("input_config", "")).resolve() != pathlib.Path(
            entry["case_file"]
        ).resolve():
            raise ValueError("summary input manifest does not match batch state")
        if summary.get("input_hashes") != entry.get("input_hashes"):
            raise ValueError("summary input hashes do not match batch state")
        outputs = summary.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("summary contains no completed outputs")

        summary_codes: list[int] = []
        output_roles: list[str] = []
        rows_by_code = {
            _parse_positive_code(row.get("Code")): row for row in registry_rows
        }
        for output in outputs:
            if not isinstance(output, dict):
                raise ValueError("summary output is not an object")
            code = _parse_positive_code(output.get("code"))
            if code is None:
                raise ValueError("summary contains an invalid fit code")
            expected_fit_json = solver.artifact_directory(
                solver.LEGACY_RESULTS_CSV,
                code,
            ) / "fit.json"
            fit_json = pathlib.Path(output.get("fit_json", ""))
            if fit_json.resolve() != expected_fit_json.resolve() or not fit_json.is_file():
                raise ValueError(f"fit {code} artifact is missing or at an unexpected path")
            with fit_json.open(encoding="utf-8") as file:
                artifact = json.load(file)
            if artifact.get("run_group") != run_group:
                raise ValueError(f"fit {code} artifact has a different run_group")
            if artifact.get("case_id") != entry.get("case_id"):
                raise ValueError(f"fit {code} artifact has a different case_id")

            row = rows_by_code.get(code)
            if row is None:
                raise ValueError(f"fit {code} has no registry row")
            if row.get("Status") != "complete":
                raise ValueError(f"fit {code} registry status is not complete")
            if row.get("Case_ID") != entry.get("case_id"):
                raise ValueError(f"fit {code} registry case_id does not match")
            if row.get("Input_Config_SHA256") != entry["input_hashes"]["case_config"]:
                raise ValueError(f"fit {code} registry input hash does not match")
            summary_codes.append(code)
            output_roles.append(str(output.get("result_role", "")))

        if sorted(summary_codes) != registry_codes:
            raise ValueError("summary and registry fit-code sets do not match")
        requested_roles = set(str(role) for role in entry["result_roles"])
        if _roles_from_labels(output_roles) != requested_roles:
            raise ValueError("completed output roles do not match the requested roles")
    except (OSError, ValueError, json.JSONDecodeError, TypeError, KeyError) as exc:
        return {
            "status": "partial",
            "fit_codes": registry_codes,
            "message": f"{type(exc).__name__}: {exc}",
        }

    return {
        "status": "complete",
        "fit_codes": sorted(summary_codes),
        "message": "",
    }


def repair_completed_artifact_export(entry: Mapping[str, Any]) -> bool:
    """Finish registration when a validated artifact exists but its summary is missing."""
    run_group = str(entry["run_group"])
    registry_rows = [
        row for row in _read_registry_rows() if row.get("Run_Group") == run_group
    ]
    if len(registry_rows) != 1:
        return False
    row = registry_rows[0]
    code = _parse_positive_code(row.get("Code"))
    if code is None or row.get("Status") not in {"failed", "complete"}:
        return False

    fit_json = solver.artifact_directory(solver.LEGACY_RESULTS_CSV, code) / "fit.json"
    properties_path = fit_json.parent / "optimized_properties.dat"
    if not fit_json.is_file() or not properties_path.is_file():
        return False
    with fit_json.open(encoding="utf-8") as file:
        artifact = json.load(file)
    if artifact.get("run_group") != run_group:
        return False
    if artifact.get("case_id") != entry.get("case_id"):
        return False
    input_files = artifact.get("input_files", {})
    if not isinstance(input_files, Mapping):
        return False
    for input_name, expected_hash in entry["input_hashes"].items():
        stored = input_files.get(input_name, {})
        if not isinstance(stored, Mapping) or stored.get("sha256") != expected_hash:
            return False

    diagnostics = artifact.get("diagnostics", {})
    validation = diagnostics.get("direct_validation", {})
    if not isinstance(validation, Mapping):
        return False
    if float(validation.get("absolute_cost_difference", float("inf"))) >= 1.0e-8:
        return False
    if float(validation.get("maximum_weighted_prediction_difference", float("inf"))) >= 1.0e-8:
        return False
    certificate = validation.get("certificate")
    if isinstance(certificate, Mapping) and not certificate.get("certificate_satisfied"):
        return False

    with solver.SEPARABLE_RESULTS_CSV.open(newline="", encoding="utf-8-sig") as file:
        result_rows = [
            result
            for result in csv.DictReader(file)
            if _parse_positive_code(result.get("Code")) == code
        ]
    parameter_count = len(artifact.get("parameters", {}))
    if parameter_count <= 0 or len(result_rows) != parameter_count:
        return False

    solver.update_fit_registration(
        code,
        "complete",
        solver.OPTIMIZER_NAME,
        solver.LEGACY_RESULTS_CSV,
        values={"Artifact_Path": str(fit_json.resolve())},
    )
    summary_path = pathlib.Path(entry["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "run_group": run_group,
        "case_id": entry["case_id"],
        "run_label": artifact.get("run_label", entry.get("name", "")),
        "optimizer": solver.OPTIMIZER_NAME,
        "input_config": str(pathlib.Path(entry["case_file"]).resolve()),
        "input_hashes": dict(entry["input_hashes"]),
        "outputs": [
            {
                "code": code,
                "result_role": artifact.get("result_role", "training_optimum"),
                "model_label": artifact.get("model_selection", {}).get("model_label", ""),
                "mean_cost": artifact.get("objective", {}).get("mean_cost"),
                "fit_json": str(fit_json.resolve()),
                "optimized_properties": str(properties_path.resolve()),
            }
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return True


def recover_case_states(state: dict[str, Any]) -> tuple[int, int, int]:
    """Audit all cases and return completed, restartable, and partial counts."""
    completed = 0
    restartable = 0
    partial = 0
    for entry in state["cases"]:
        audit = audit_entry(entry)
        entry["fit_codes"] = list(audit["fit_codes"])
        if audit["status"] == "complete":
            entry["status"] = "completed"
            entry["return_code"] = 0
            entry.pop("error", None)
            completed += 1
        elif audit["status"] == "untouched":
            entry["status"] = "pending"
            entry["return_code"] = None
            entry.pop("error", None)
            restartable += 1
        else:
            entry["status"] = "partial_export"
            entry["error"] = str(audit["message"])
            partial += 1
    return completed, restartable, partial


def run_case_subprocess(entry: Mapping[str, Any], total: int) -> int:
    """Launch one deterministic case with isolated compilation and one CPU thread."""
    index = int(entry["index"])
    worker_dir = pathlib.Path(entry["worker_dir"])
    worker_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n=== Launching case {index}/{total}: {entry['case_id']} ===\n"
        f"Worker directory: {worker_dir}\nRun group: {entry['run_group']}"
    )
    env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(pathlib.Path(solver.__file__).resolve()),
            "--case",
            str(pathlib.Path(entry["case_file"]).resolve()),
            "--run-group",
            str(entry["run_group"]),
        ],
        cwd=str(worker_dir),
        env=env,
    )
    return completed.returncode


def recover_batch_only(state: dict[str, Any], state_path: pathlib.Path) -> None:
    """Audit existing global outputs without starting any pending case."""
    completed, restartable, partial = recover_case_states(state)
    repaired = 0
    if partial:
        for entry in state["cases"]:
            if entry["status"] == "partial_export" and repair_completed_artifact_export(entry):
                repaired += 1
        if repaired:
            completed, restartable, partial = recover_case_states(state)
    write_state(state_path, state)
    print(f"Recovered complete artifact exports: {repaired}")
    print(f"Verified completed cases: {completed}")
    print(f"Restartable cases not started: {restartable}")
    print(f"Partial exports requiring attention: {partial}")
    for entry in state["cases"]:
        if entry["status"] == "partial_export":
            print(f"- {entry['case_id']}: {entry['error']}")


def run_parallel_batch(
    state: dict[str, Any],
    state_path: pathlib.Path,
    cores: int,
) -> None:
    """Run all safe pending cases and persist state after each completion."""
    recover_case_states(state)
    state["cores"] = cores
    write_state(state_path, state)
    pending = [entry for entry in state["cases"] if entry["status"] == "pending"]
    if not pending:
        partial = [
            entry for entry in state["cases"] if entry["status"] == "partial_export"
        ]
        if partial:
            raise SystemExit(
                "No cases were started because partial exports require manual attention."
            )
        print("All selected cases were already completed and verified.")
        return

    print(f"\nRunning {len(pending)} pending cases on up to {cores} physical cores.")
    print(f"Batch state: {state_path}")
    for entry in pending:
        entry["status"] = "running"
        entry["return_code"] = None
        entry.pop("error", None)
    write_state(state_path, state)

    futures: dict[Future[int], BatchEntry] = {}
    with ThreadPoolExecutor(max_workers=cores) as executor:
        for entry in pending:
            future = executor.submit(run_case_subprocess, entry, len(state["cases"]))
            futures[future] = entry

        for future in as_completed(futures):
            entry = futures[future]
            try:
                return_code = future.result()
                entry["return_code"] = return_code
                audit = audit_entry(entry)
                entry["fit_codes"] = list(audit["fit_codes"])
                if audit["status"] == "complete":
                    entry["status"] = "completed"
                    entry.pop("error", None)
                    print(
                        f"\nCOMPLETED: {entry['case_id']} "
                        f"(fits {entry['fit_codes']})"
                    )
                elif audit["status"] == "partial":
                    entry["status"] = "partial_export"
                    entry["error"] = str(audit["message"])
                    print(f"\nPARTIAL EXPORT: {entry['case_id']}: {entry['error']}")
                else:
                    entry["status"] = "failed"
                    entry["error"] = f"worker exit code {return_code}; no export found"
                    print(f"\nFAILED: {entry['case_id']} (exit code {return_code})")
            except BaseException as exc:
                entry["return_code"] = None
                audit = audit_entry(entry)
                entry["fit_codes"] = list(audit["fit_codes"])
                if audit["status"] == "complete":
                    entry["status"] = "completed"
                    entry.pop("error", None)
                elif audit["status"] == "partial":
                    entry["status"] = "partial_export"
                    entry["error"] = str(audit["message"])
                else:
                    entry["status"] = "failed"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                print(f"\nFAILED while handling {entry['case_id']}: {entry.get('error', exc)}")
            write_state(state_path, state)

    completed = [entry for entry in state["cases"] if entry["status"] == "completed"]
    failed = [entry for entry in state["cases"] if entry["status"] == "failed"]
    partial = [
        entry for entry in state["cases"] if entry["status"] == "partial_export"
    ]
    print("\nBatch summary")
    print(f"Completed cases: {len(completed)}/{len(state['cases'])}")
    for entry in completed:
        print(f"- {entry['case_id']}: fits {entry['fit_codes']}")
    print(f"Failed restartable cases: {len(failed)}")
    print(f"Partial exports requiring attention: {len(partial)}")
    for entry in failed + partial:
        print(f"- {entry['case_id']}: {entry.get('error', entry.get('return_code'))}")
    if failed or partial:
        print(f'Resume safe cases with: --resume --state-file "{state_path}"')
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    """Validate configuration, preview cases, or execute a resumable batch."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First 1-based configured case index to run.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Last 1-based configured case index to run.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=DEFAULT_PHYSICAL_CORES,
        help="Maximum number of physical case workers.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume safe pending cases from --state-file.",
    )
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help="Audit outputs and update state without starting cases.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Validate and print cases without compiling or assigning fit codes.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Explicitly execute a new batch even while PREVIEW_ONLY is True. "
            "This keeps the checked-in VSCode default safe."
        ),
    )
    parser.add_argument(
        "--state-file",
        type=pathlib.Path,
        default=DEFAULT_STATE_FILE,
        help=f"Batch state path (default: {DEFAULT_STATE_FILE}).",
    )
    parser.add_argument(
        "--case-list",
        type=pathlib.Path,
        default=DEFAULT_CASE_LIST_FILE,
        help=(
            "JSON list of case manifests, or an object with a case_files list. "
            "When omitted, SELECTED_CASE_FILES is used."
        ),
    )
    args = parser.parse_args(argv)

    try:
        selected_files = load_case_list(args.case_list.resolve()) if args.case_list else None
        jobs = build_case_jobs(selected_files)
    except Exception as exc:
        parser.error(f"invalid USER CONFIGURATION: {type(exc).__name__}: {exc}")
    if args.cores <= 0:
        parser.error("--cores must be a positive integer")
    available_cores = physical_core_count()
    if available_cores is not None and args.cores > available_cores:
        parser.error(
            f"--cores cannot exceed the {available_cores} detected physical cores"
        )
    if args.preview and args.execute:
        parser.error("--preview and --execute are mutually exclusive")
    if args.preview and (args.resume or args.recover_only):
        parser.error("--preview cannot be combined with --resume or --recover-only")

    state_path = args.state_file.resolve()
    if args.resume or args.recover_only:
        if not state_path.is_file():
            raise SystemExit(f"Batch state file was not found: {state_path}")
        state = load_state(state_path)
        validate_resumed_state(state, jobs)
        if args.recover_only:
            recover_batch_only(state, state_path)
            return
        run_parallel_batch(state, state_path, args.cores)
        return

    end_index = args.end_index if args.end_index is not None else len(jobs)
    if args.start_index < 1 or end_index > len(jobs) or args.start_index > end_index:
        raise SystemExit(f"Case range must be between 1 and {len(jobs)}")
    indices = range(args.start_index, end_index + 1)
    print_case_preview(jobs, indices)
    if (PREVIEW_ONLY and not args.execute) or args.preview:
        print("\nPreview complete. No fit codes were assigned and no UMAT was compiled.")
        return

    if state_path.is_file():
        existing = load_state(state_path)
        unfinished = [
            entry for entry in existing["cases"] if entry.get("status") != "completed"
        ]
        if unfinished:
            raise SystemExit(
                f"An unfinished separable batch exists at {state_path}. "
                "Use --resume, --recover-only, or select a different --state-file."
            )

    state = create_state(jobs, indices, args.cores, state_path)
    write_state(state_path, state)
    run_parallel_batch(state, state_path, args.cores)


if __name__ == "__main__":
    main()
