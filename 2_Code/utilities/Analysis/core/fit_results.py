"""Load fitted material parameters and export reusable fit artifacts."""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from release_paths import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


PROJECT_DIR = next(parent for parent in Path(__file__).resolve().parents if parent.name == "Analysis").parent
DEFAULT_RESULTS_CSV = RELEASE_ARCHIVE_ROOT / "results_final.csv"
DEFAULT_FIT_REGISTRY = RELEASE_ARCHIVE_ROOT / "fit_registry.csv"
DEFAULT_SEPARABLE_RESULTS_CSV = (
    RELEASE_ARCHIVE_ROOT / "separable_solver_results.csv"
)

# Retry transient OneDrive/antivirus locks during atomic CSV replacement.
# Example: 12 attempts covers roughly 20 seconds of bounded backoff.
_ATOMIC_WRITE_MAX_ATTEMPTS = 12
# Initial retry delay in seconds. Example: 0.25 seconds.
_ATOMIC_WRITE_RETRY_DELAY_SECONDS = 0.25

FIT_REGISTRY_FIELDS = (
    "Code",
    "Status",
    "Created_At",
    "Completed_At",
    "Optimizer",
    "Model_ID",
    "Knit",
    "Yarn_Type",
    "Sample_Folder",
    "Exp_Data_Type",
    "Exp_Data_Path",
    "Cost_Type",
    "Fit_Axes",
    "Objective_Value",
    "Result_Role",
    "Run_Group",
    "Case_ID",
    "Run_Label",
    "Input_Config_Path",
    "Input_Config_SHA256",
    "Artifact_Path",
)

# The deterministic table is deliberately method-specific. For example,
# ``Active_Bound`` records ``upper`` when beta_1 finishes at its ceiling.
SEPARABLE_RESULT_FIELDS = (
    "Code",
    "Created_At",
    "Optimizer",
    "Run_Group",
    "Case_ID",
    "Run_Label",
    "Result_Role",
    "Model_ID",
    "UMAT",
    "Exp_Data",
    "Exp_Data_Path",
    "Exp_Data_Type",
    "Yarn_Type",
    "Knit",
    "Knit_Token",
    "Sample_Folder",
    "Param_File",
    "Param_File_Path",
    "Exp_Type",
    "UMAT_Stress",
    "UMAT_Strain",
    "Cost_Type",
    "Normalized_Stress_Floor_Factor",
    "Fit_Axes",
    "Parameter",
    "Initial_Guess",
    "Min",
    "Max",
    "Fixed",
    "Opt_Value",
    "Active_Bound",
    "Identifiability_Status",
    "Raw_Cost",
    "Mean_Cost",
    "Fitted_Observation_Count",
    "Profiled_Evaluations",
    "Elapsed_Seconds",
    "Input_Config_Path",
    "Input_Config_SHA256",
    "Parameter_File_SHA256",
    "Experimental_Data_SHA256",
    "UMAT_Source_SHA256",
)

# Allow concurrent certified workers to queue behind slower OneDrive registry writes.
_REGISTRY_LOCK_TIMEOUT_SECONDS = 120.0
_REGISTRY_LOCK_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class FitResult:
    """One fitted parameter combination and the metadata that identifies it."""

    code: int
    parameters: dict[str, float | int]
    metadata: dict[str, Any]
    source: Path
    model_id: str | None = None

    @property
    def model(self) -> str:
        """Return the explicit model ID, falling back to the historical filename."""
        if self.model_id is not None:
            return self.model_id
        umat_name = str(self.metadata.get("UMAT", self.metadata.get("umat", "")))
        return Path(umat_name).stem


def load_fit(code: int, results_csv: Path = DEFAULT_RESULTS_CSV) -> FitResult:
    """Load a fit by code, preferring its JSON artifact over the summary CSV."""
    results_csv = Path(results_csv)
    artifact_path = artifact_directory(results_csv, code) / "fit.json"
    if artifact_path.is_file():
        return load_fit_json(artifact_path)
    return load_fit_csv(code, results_csv)


def load_fit_csv(code: int, results_csv: Path = DEFAULT_RESULTS_CSV) -> FitResult:
    """Reconstruct one historical fit from ``results_final.csv``."""
    results_csv = Path(results_csv)
    if not results_csv.is_file():
        raise FileNotFoundError(f"Fit-results CSV was not found: {results_csv}")

    with results_csv.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if _parse_code(row.get("Code")) == code]

    if not rows:
        raise ValueError(f"Fit code {code} was not found in {results_csv}")

    parameters: dict[str, float | int] = {}
    for row in rows:
        name = row.get("Parameter", "").strip()
        if not name:
            raise ValueError(f"Fit code {code} contains a row without a parameter name")
        if name in parameters:
            raise ValueError(f"Fit code {code} contains duplicate parameter {name!r}")
        parameters[name] = _parse_number(row.get("Opt_Value"), f"Opt_Value for {name}")

    metadata = {
        key: value
        for key, value in rows[0].items()
        if key not in _PARAMETER_COLUMNS and value not in (None, "")
    }
    metadata["Code"] = code
    raw_model_id = metadata.get("Model_ID", metadata.get("model_id"))
    model_id = str(raw_model_id).strip() if raw_model_id not in (None, "") else None
    return FitResult(
        code=code,
        parameters=parameters,
        metadata=metadata,
        source=results_csv,
        model_id=model_id,
    )


def load_fit_json(path: Path) -> FitResult:
    """Load a fit artifact written by :func:`export_fit_artifacts`."""
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    try:
        code = int(payload["fit_code"])
        raw_parameters = payload["parameters"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid fit artifact: {path}") from exc

    if not isinstance(raw_parameters, dict) or not raw_parameters:
        raise ValueError(f"Fit artifact has no parameters: {path}")
    parameters = {
        str(name): _parse_number(value, f"parameter {name}")
        for name, value in raw_parameters.items()
    }
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Fit artifact metadata must be an object: {path}")
    experiment_context = payload.get("experiment_context")
    if isinstance(experiment_context, dict):
        metadata.setdefault("experiment_context", experiment_context)
    raw_model_id = payload.get("model_id")
    if raw_model_id is not None and (
        not isinstance(raw_model_id, str) or not raw_model_id.strip()
    ):
        raise ValueError(f"Fit artifact model_id must be a non-empty string: {path}")
    model_id = raw_model_id.strip() if isinstance(raw_model_id, str) else None
    return FitResult(
        code=code,
        parameters=parameters,
        metadata=metadata,
        source=path,
        model_id=model_id,
    )


def export_fit_artifacts(
    code: int,
    results: Mapping[str, Sequence[Any]],
    parameter_specs: Mapping[str, Sequence[Any]],
    results_csv: Path = DEFAULT_RESULTS_CSV,
    model_id: str | None = None,
    additional_payload: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a human-readable JSON record and a loadable optimized ``.dat`` file."""
    parameters = _parameters_from_results(results)
    output_dir = artifact_directory(Path(results_csv), code)
    output_dir.mkdir(parents=True, exist_ok=True)

    first = {key: _json_value(values[0]) for key, values in results.items() if values}
    metadata = {
        key: value
        for key, value in first.items()
        if key not in _PARAMETER_COLUMNS
    }
    metadata["Code"] = code

    if model_id is None:
        umat_name = str(metadata.get("UMAT", metadata.get("umat", ""))).strip()
        model_id = Path(umat_name).stem if umat_name else None
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("A non-empty model_id is required to export fit artifacts")
    model_id = model_id.strip()

    specifications: dict[str, dict[str, Any]] = {}
    for name, value in parameters.items():
        if name not in parameter_specs:
            raise ValueError(f"Missing parameter specification for {name!r}")
        initial, data_type, minimum, maximum, _, fixed = parameter_specs[name]
        specifications[name] = {
            "type": "int" if data_type is int else "float",
            "initial_guess": _json_value(initial),
            "minimum": _json_value(minimum),
            "maximum": _json_value(maximum),
            "fixed": bool(fixed),
        }

    payload = {
        "schema_version": 2,
        "fit_code": code,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "metadata": metadata,
        "experiment_context": _experiment_context_from_metadata(metadata),
        "parameters": parameters,
        "parameter_specs": specifications,
    }
    if additional_payload:
        protected = set(payload)
        collisions = sorted(protected.intersection(additional_payload))
        if collisions:
            raise ValueError(
                "Additional fit payload cannot replace core fields: "
                + ", ".join(collisions)
            )
        payload.update({str(key): value for key, value in additional_payload.items()})

    json_path = output_dir / "fit.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")

    properties_path = output_dir / "optimized_properties.dat"
    with properties_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(("Name", "Guess", "Type", "Min", "Max", "Fix?"))
        for name, value in parameters.items():
            spec = specifications[name]
            writer.writerow(
                (
                    name,
                    value,
                    spec["type"],
                    spec["minimum"],
                    spec["maximum"],
                    spec["fixed"],
                )
            )

    return json_path, properties_path


def fit_registry_path(results_csv: Path = DEFAULT_RESULTS_CSV) -> Path:
    """Return the global registry beside a results table, for example Outputs/."""
    return Path(results_csv).parent / DEFAULT_FIT_REGISTRY.name


@contextmanager
def _registry_lock(registry_path: Path):
    """Serialize fit-code changes with an exclusive sidecar lock file."""
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _REGISTRY_LOCK_TIMEOUT_SECONDS
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for fit registry lock: {lock_path}")
            time.sleep(_REGISTRY_LOCK_POLL_SECONDS)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one CSV table, returning empty collections when it is absent."""
    if not path.is_file():
        return [], []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _write_table_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Replace one CSV table atomically, for example after reserving a code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(
            {
                field: "" if row.get(field) is None else row.get(field, "")
                for field in fieldnames
            }
            for row in rows
        )
    for attempt in range(_ATOMIC_WRITE_MAX_ATTEMPTS):
        try:
            os.replace(temporary_path, path)
            return
        except PermissionError:
            if attempt + 1 >= _ATOMIC_WRITE_MAX_ATTEMPTS:
                raise
            delay = min(_ATOMIC_WRITE_RETRY_DELAY_SECONDS * (2**attempt), 2.0)
            time.sleep(delay)


def _schema_can_expand(
    fields: Sequence[str],
    expected_fields: Sequence[str],
) -> bool:
    """Allow an older table to gain new columns without losing its rows."""
    if not fields or len(fields) != len(set(fields)):
        return False
    expected_positions = {field: index for index, field in enumerate(expected_fields)}
    try:
        positions = [expected_positions[field] for field in fields]
    except KeyError:
        return False
    return positions == sorted(positions)


def _registry_value(value: Any) -> str:
    """Convert one registry value to stable text, for example axes [0, 1]."""
    if value in (None, ""):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _normalized_registry_row(values: Mapping[str, Any]) -> dict[str, str]:
    """Return one complete registry row with the declared field order."""
    return {
        field: _registry_value(values.get(field, ""))
        for field in FIT_REGISTRY_FIELDS
    }


def _legacy_rows(results_csv: Path) -> list[dict[str, str]]:
    """Build one registry row per historical Monte Carlo code."""
    _, rows = _read_table(results_csv)
    first_by_code: dict[int, dict[str, str]] = {}
    for row in rows:
        code = _parse_code(row.get("Code"))
        if code is not None:
            first_by_code.setdefault(code, row)

    output: list[dict[str, str]] = []
    for code, row in sorted(first_by_code.items()):
        artifact_path = artifact_directory(results_csv, code) / "fit.json"
        created_at = ""
        if artifact_path.is_file():
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                created_at = _registry_value(payload.get("created_at"))
            except (OSError, json.JSONDecodeError):
                created_at = ""
        output.append(
            _normalized_registry_row(
                {
                    "Code": code,
                    "Status": "complete",
                    "Created_At": created_at,
                    "Completed_At": created_at,
                    "Optimizer": "monte_carlo",
                    "Model_ID": row.get("Model_ID") or Path(row.get("UMAT", "")).stem,
                    "Knit": row.get("Knit"),
                    "Yarn_Type": row.get("Yarn_Type"),
                    "Sample_Folder": row.get("Sample_Folder"),
                    "Exp_Data_Type": row.get("Exp_Data_Type"),
                    "Exp_Data_Path": row.get("Exp_Data_Path"),
                    "Cost_Type": row.get("Cost_Type"),
                    "Fit_Axes": "",
                    "Objective_Value": row.get("Min_Cost"),
                    "Result_Role": "training_optimum",
                    "Run_Group": f"legacy_{code:04d}",
                    "Artifact_Path": str(artifact_path.resolve()) if artifact_path.is_file() else "",
                }
            )
        )
    return output


def _artifact_only_rows(results_csv: Path) -> list[dict[str, str]]:
    """Recover registry rows for fit artifacts absent from a legacy CSV."""
    fit_root = Path(results_csv).parent / "fits"
    if not fit_root.is_dir():
        return []
    recovered: list[dict[str, str]] = []
    for directory in sorted(fit_root.iterdir()):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        artifact_path = directory / "fit.json"
        if not artifact_path.is_file():
            continue
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = payload.get("metadata", {})
        context = payload.get("experiment_context", {})
        optimizer = payload.get("optimizer", {})
        objective = payload.get("objective", {})
        recovered.append(
            _normalized_registry_row(
                {
                    "Code": int(directory.name),
                    "Status": "complete",
                    "Created_At": payload.get("created_at"),
                    "Completed_At": payload.get("created_at"),
                    "Optimizer": optimizer.get("method", "legacy_unknown"),
                    "Model_ID": payload.get("model_id"),
                    "Knit": context.get("knit", metadata.get("Knit")),
                    "Yarn_Type": context.get("yarn_type", metadata.get("Yarn_Type")),
                    "Sample_Folder": context.get(
                        "sample_folder", metadata.get("Sample_Folder")
                    ),
                    "Exp_Data_Type": context.get(
                        "exp_data_type", metadata.get("Exp_Data_Type")
                    ),
                    "Exp_Data_Path": context.get(
                        "exp_data_path", metadata.get("Exp_Data_Path")
                    ),
                    "Cost_Type": objective.get("cost_type", metadata.get("Cost_Type")),
                    "Fit_Axes": objective.get("fit_axes", ""),
                    "Objective_Value": objective.get("mean_cost", ""),
                    "Result_Role": payload.get("result_role", "training_optimum"),
                    "Run_Group": payload.get("run_group", f"artifact_{directory.name}"),
                    "Artifact_Path": str(artifact_path.resolve()),
                }
            )
        )
    return recovered


def _load_registry_with_legacy(
    results_csv: Path,
    registry_path: Path,
) -> list[dict[str, str]]:
    """Merge the persisted registry with historical CSV and artifact codes."""
    _, stored_rows = _read_table(registry_path)
    by_code: dict[int, dict[str, str]] = {}
    for row in stored_rows:
        code = _parse_code(row.get("Code"))
        if code is not None:
            by_code[code] = _normalized_registry_row(row)
    for row in _legacy_rows(results_csv) + _artifact_only_rows(results_csv):
        code = int(row["Code"])
        by_code.setdefault(code, row)
    return [by_code[code] for code in sorted(by_code)]


def initialize_fit_registry(
    results_csv: Path = DEFAULT_RESULTS_CSV,
    registry_path: Path | None = None,
) -> Path:
    """Create or update the global registry from existing fit outputs."""
    results_csv = Path(results_csv)
    registry_path = Path(registry_path or fit_registry_path(results_csv))
    with _registry_lock(registry_path):
        rows = _load_registry_with_legacy(results_csv, registry_path)
        _write_table_atomic(registry_path, FIT_REGISTRY_FIELDS, rows)
    return registry_path


def reserve_fit_codes(
    count: int,
    optimizer: str,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    registry_path: Path | None = None,
    metadata: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[int, ...]:
    """Atomically reserve globally unique fit codes, for example one or twelve."""
    if int(count) <= 0:
        raise ValueError("Fit-code reservation count must be positive")
    if not isinstance(optimizer, str) or not optimizer.strip():
        raise ValueError("Optimizer must be a non-empty string")
    if metadata is not None and len(metadata) != int(count):
        raise ValueError("Fit-code metadata count must match the reservation count")

    results_csv = Path(results_csv)
    registry_path = Path(registry_path or fit_registry_path(results_csv))
    with _registry_lock(registry_path):
        rows = _load_registry_with_legacy(results_csv, registry_path)
        used = {int(row["Code"]) for row in rows}
        next_code = max(used, default=0) + 1
        codes = tuple(range(next_code, next_code + int(count)))
        created_at = datetime.now(timezone.utc).isoformat()
        for index, code in enumerate(codes):
            values = dict(metadata[index]) if metadata is not None else {}
            values.update(
                {
                    "Code": code,
                    "Status": "reserved",
                    "Created_At": created_at,
                    "Optimizer": optimizer.strip(),
                    "Artifact_Path": str(
                        (artifact_directory(results_csv, code) / "fit.json").resolve()
                    ),
                }
            )
            rows.append(_normalized_registry_row(values))
        rows.sort(key=lambda row: int(row["Code"]))
        _write_table_atomic(registry_path, FIT_REGISTRY_FIELDS, rows)
    return codes


def update_fit_registration(
    code: int,
    status: str,
    optimizer: str,
    results_csv: Path = DEFAULT_RESULTS_CSV,
    registry_path: Path | None = None,
    values: Mapping[str, Any] | None = None,
) -> Path:
    """Mark one reserved code complete or failed and update its summary fields."""
    if int(code) <= 0:
        raise ValueError("Fit code must be positive")
    if status not in {"reserved", "complete", "failed"}:
        raise ValueError("Fit status must be reserved, complete, or failed")
    if not isinstance(optimizer, str) or not optimizer.strip():
        raise ValueError("Optimizer must be a non-empty string")

    results_csv = Path(results_csv)
    registry_path = Path(registry_path or fit_registry_path(results_csv))
    with _registry_lock(registry_path):
        rows = _load_registry_with_legacy(results_csv, registry_path)
        target = next((row for row in rows if int(row["Code"]) == int(code)), None)
        if target is None:
            target = _normalized_registry_row(
                {
                    "Code": int(code),
                    "Status": "reserved",
                    "Created_At": datetime.now(timezone.utc).isoformat(),
                    "Optimizer": optimizer.strip(),
                }
            )
            rows.append(target)
        existing_optimizer = target.get("Optimizer", "")
        if existing_optimizer not in ("", optimizer.strip(), "legacy_unknown"):
            raise ValueError(
                f"Fit code {code} belongs to optimizer {existing_optimizer!r}, "
                f"not {optimizer.strip()!r}"
            )
        if values:
            for key, value in values.items():
                if key in FIT_REGISTRY_FIELDS and key != "Code":
                    target[key] = _registry_value(value)
        target["Optimizer"] = optimizer.strip()
        target["Status"] = status
        if status == "complete":
            target["Completed_At"] = datetime.now(timezone.utc).isoformat()
        rows.sort(key=lambda row: int(row["Code"]))
        _write_table_atomic(registry_path, FIT_REGISTRY_FIELDS, rows)
    return registry_path


def append_separable_fit_results(
    rows: Sequence[Mapping[str, Any]],
    results_csv: Path = DEFAULT_SEPARABLE_RESULTS_CSV,
) -> Path:
    """Append one complete deterministic fit without duplicating its code."""
    if not rows:
        raise ValueError("At least one separable-solver result row is required")
    normalized = [
        {
            field: _registry_value(row.get(field, ""))
            for field in SEPARABLE_RESULT_FIELDS
        }
        for row in rows
    ]
    codes = {_parse_code(row["Code"]) for row in normalized}
    if None in codes or len(codes) != 1:
        raise ValueError("One append operation must contain exactly one positive fit code")
    code = next(iter(codes))
    if code is None or code <= 0:
        raise ValueError("Separable-solver fit code must be positive")
    if any(not row["Parameter"].strip() for row in normalized):
        raise ValueError("Every separable-solver result row needs a parameter name")
    parameter_names = [row["Parameter"] for row in normalized]
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError(f"Fit code {code} contains duplicate parameter rows")

    results_csv = Path(results_csv)
    # Reuse the sidecar-lock protocol; for example ``separable_solver_results.csv.lock``.
    with _registry_lock(results_csv):
        fields, existing = _read_table(results_csv)
        if fields and fields != list(SEPARABLE_RESULT_FIELDS):
            if not _schema_can_expand(fields, SEPARABLE_RESULT_FIELDS):
                raise ValueError(
                    f"Unexpected separable-solver results schema in {results_csv}"
                )
            # Preserve completed legacy rows while adding newly introduced provenance.
            _write_table_atomic(results_csv, SEPARABLE_RESULT_FIELDS, existing)
        same_code = [row for row in existing if _parse_code(row.get("Code")) == code]
        if same_code:
            expected = [
                {field: row.get(field, "") for field in SEPARABLE_RESULT_FIELDS}
                for row in normalized
            ]
            actual = [
                {field: row.get(field, "") for field in SEPARABLE_RESULT_FIELDS}
                for row in same_code
            ]
            if actual == expected:
                return results_csv
            raise ValueError(
                f"Fit code {code} already has different rows in {results_csv}"
            )
        _write_table_atomic(
            results_csv,
            SEPARABLE_RESULT_FIELDS,
            [*existing, *normalized],
        )
    return results_csv


def initialize_separable_results_table(
    results_csv: Path = DEFAULT_SEPARABLE_RESULTS_CSV,
) -> Path:
    """Create the deterministic table header, for example before a long solve."""
    results_csv = Path(results_csv)
    with _registry_lock(results_csv):
        fields, rows = _read_table(results_csv)
        if (
            fields
            and fields != list(SEPARABLE_RESULT_FIELDS)
            and not _schema_can_expand(fields, SEPARABLE_RESULT_FIELDS)
        ):
            raise ValueError(
                f"Unexpected separable-solver results schema in {results_csv}"
            )
        # Expand older tables in place; missing provenance remains blank by design.
        if fields != list(SEPARABLE_RESULT_FIELDS):
            _write_table_atomic(results_csv, SEPARABLE_RESULT_FIELDS, rows)
    return results_csv


def validate_parameters(parameters: Mapping[str, Any], required: Sequence[str], model: str) -> None:
    """Raise a useful error when a selected fit cannot drive a material model."""
    missing = [name for name in required if name not in parameters]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"The selected fit is incompatible with {model}; missing: {names}")


def artifact_directory(results_csv: Path, code: int) -> Path:
    """Return the stable output directory for a fit code."""
    return Path(results_csv).parent / "fits" / f"{code:04d}"


def _parameters_from_results(results: Mapping[str, Sequence[Any]]) -> dict[str, float | int]:
    names = results.get("Parameter", ())
    values = results.get("Opt_Value", ())
    if not names or len(names) != len(values):
        raise ValueError("Results must contain equally sized Parameter and Opt_Value columns")
    return {
        str(name): _parse_number(_json_value(value), f"Opt_Value for {name}")
        for name, value in zip(names, values)
    }


def _parse_code(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_number(value: Any, label: str) -> float | int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid numeric {label}: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric {label}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Invalid numeric {label}: {value!r}")
    return int(number) if number.is_integer() and not any(ch in text.lower() for ch in (".", "e")) else number


def _json_value(value: Any) -> Any:
    """Convert NumPy-style scalar objects to JSON-native values."""
    item = getattr(value, "item", None)
    converted = item() if callable(item) else value
    if isinstance(converted, float) and not math.isfinite(converted):
        return None
    return converted


def _experiment_context_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract explicit experiment-selection metadata for fit reuse."""
    field_map = {
        "model_id": ("Model_ID", "model_id"),
        "exp_data": ("Exp_Data",),
        "exp_data_path": ("Exp_Data_Path",),
        "exp_data_type": ("Exp_Data_Type", "exp_data_type"),
        "yarn_type": ("Yarn_Type", "yarn_type"),
        "knit": ("Knit", "knit"),
        "knit_token": ("Knit_Token", "knit_token"),
        "sample_folder": ("Sample_Folder", "sample_folder"),
        "param_file": ("Param_File",),
        "param_file_path": ("Param_File_Path",),
        "exp_type": ("Exp_Type",),
        "umat_strain": ("UMAT_Strain",),
        "umat_stress": ("UMAT_Stress",),
    }
    context: dict[str, Any] = {}
    for output_key, metadata_keys in field_map.items():
        for metadata_key in metadata_keys:
            value = metadata.get(metadata_key)
            if value not in (None, ""):
                context[output_key] = value
                break
    return context


_PARAMETER_COLUMNS = {
    "Parameter",
    "Ini_Guess",
    "Min",
    "Max",
    "Fix?",
    "Mean",
    "Std_Dev",
    "Opt_Value",
}
