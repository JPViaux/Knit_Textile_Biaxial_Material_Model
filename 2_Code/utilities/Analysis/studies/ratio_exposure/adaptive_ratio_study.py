"""Prepare, run, evaluate, and select the adaptive ratio-exposure study.

Press Run in VSCode with ``ACTION = "preview"`` to inspect B01 safely. Change
``ACTION`` to ``"all"`` and ``TARGET_BATCH`` to, for example, ``"B03"`` to
prepare, fit, evaluate, and freeze one production Batch. The command-line
``--all-steps --action all`` form is intended for a fully supervised run.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT,
    GENERATED_ROOT as RELEASE_GENERATED_ROOT)

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from datetime import datetime, timezone
from functools import cmp_to_key
import hashlib
from release_paths import json, resolve_string
import math
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from openpyxl import load_workbook


ANALYSIS_DIRECTORY = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis")
SUPPORT_DIRECTORY = ANALYSIS_DIRECTORY.parent
for import_directory in (SUPPORT_DIRECTORY, ANALYSIS_DIRECTORY):
    sys.path.insert(0, str(import_directory))

from Analysis.diagnostics import global_error_calculator as global_errors  # noqa: E402
from Analysis.studies.ratio_exposure import prepare_progressive_ratio_study as progressive  # noqa: E402
from Analysis.fitting import run_separable_solver_batch as batch_runner  # noqa: E402
from Analysis.fitting import separable_solver_fit as solver  # noqa: E402


# ---------------------------------------------------------------------------
# USER CONFIGURATION - edit these values before pressing Run in VSCode.
# ---------------------------------------------------------------------------

# Safe default action. Examples: "preview", "prepare", "run", "evaluate", "all".
ACTION = "preview"
# Batch to process in VSCode. Examples: "B01" for single-ratio candidates.
TARGET_BATCH = "B01"
# Simultaneous production workers. Examples: 1 sequentially or 5 recommended.
DEFAULT_PHYSICAL_CORES = 5
# Append audited candidate rows to the shared workbook. Example: True.
APPEND_GLOBAL_WORKBOOK = False

# ---------------------------------------------------------------------------
# FIXED STUDY CONFIGURATION - normally no changes are needed below this line.
# ---------------------------------------------------------------------------

STUDY_ID = "adaptive_ratio_greedy_v1"
CASE_ID_PREFIX = "adaptive_greedy_v1"
OUTPUT_DIRECTORY = RELEASE_GENERATED_ROOT / "results/greedy_refitting"
DEFINITION_DIRECTORY = OUTPUT_DIRECTORY / "definitions"
CASE_LIST_DIRECTORY = OUTPUT_DIRECTORY / "case_lists"
STATE_DIRECTORY = OUTPUT_DIRECTORY / "states"
FIT_MAP_DIRECTORY = OUTPUT_DIRECTORY / "fit_maps"
EVALUATION_DIRECTORY = OUTPUT_DIRECTORY / "evaluations"
SELECTION_DIRECTORY = OUTPUT_DIRECTORY / "selections"
REUSE_DIRECTORY = OUTPUT_DIRECTORY / "reuse_certificates"
REGISTRY_PATH = RELEASE_ARCHIVE_ROOT / "fit_registry.csv"
GLOBAL_WORKBOOK = RELEASE_ARCHIVE_ROOT / "global_error_summary.xlsx"
ADJACENT_CATALOG_FILENAME = "adaptive_ratio_study_v1.json"
MODEL_IDS = ("SV_exp_polycon_mo",)
RATIO_ORDER = tuple(progressive.RATIO_FILENAMES)
RATIO_SET = frozenset(RATIO_ORDER)
TIE_TOLERANCE = 1.0e-10
VALIDATION_TOLERANCE = 1.0e-8


def utc_now() -> str:
    """Return a stable UTC timestamp, for example 2026-07-30T12:00:00+00:00."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def batch_id(number: int) -> str:
    """Return a two-digit Batch identifier, for example 1 becomes B01."""
    if number < 1 or number > 9:
        raise ValueError("Adaptive Batch number must be between 1 and 9")
    return f"B{number:02d}"


def batch_number(value: str) -> int:
    """Parse a Batch identifier, for example B09 becomes 9."""
    match = re.fullmatch(r"B(0[1-9])", str(value).strip().upper())
    if match is None:
        raise ValueError("Adaptive Batch must be B01 through B09")
    return int(match.group(1))


def model_token(model_id: str) -> str:
    """The publication ships one model; example: SV_exp_polycon_mo."""
    if model_id != "SV_exp_polycon_mo":
        raise ValueError(f"Unsupported publication model: {model_id}")
    return "sv_exp_polycon_mo"



def ratio_token(ratio: str) -> str:
    """Return a filename-safe ratio token, for example 0.25:1 -> 0p25to1."""
    if ratio not in RATIO_SET:
        raise ValueError(f"Unknown loading ratio: {ratio}")
    return ratio.replace(".", "p").replace(":", "to")


def pair_id(model_id: str, config: progressive.KnitConfiguration) -> str:
    """Return one model-knit search key, for example sv_exp_interlock_s06."""
    return f"{model_token(model_id)}_{config.sample_token}"


def candidate_id(
    step: str,
    added_ratio: str,
    model_id: str,
    config: progressive.KnitConfiguration,
) -> str:
    """Return a permanent candidate case ID encoding its added ratio."""
    return (
        f"{CASE_ID_PREFIX}_{step.lower()}_add_{ratio_token(added_ratio)}_"
        f"{model_token(model_id)}_{config.sample_token}_raw"
    )


def candidate_filename(
    step: str,
    added_ratio: str,
    model_id: str,
    config: progressive.KnitConfiguration,
) -> str:
    """Return a permanent raw candidate filename, for example one B02 addition."""
    return (
        f"{config.token}_adaptive_greedy_v1_{model_token(model_id)}_"
        f"{step.lower()}_add_{ratio_token(added_ratio)}.dat"
    )


def sha256_bytes(data: bytes) -> str:
    """Hash generated bytes, for example a cumulative training candidate."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    """Hash one existing file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    """Serialize stable indented JSON with a final newline."""
    return (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json_atomic(path: pathlib.Path, payload: Any) -> None:
    """Atomically write JSON, including across transient OneDrive activity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        delete=False,
    ) as temporary:
        temporary_path = pathlib.Path(temporary.name)
        temporary.write(json_bytes(payload))
    try:
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_or_verify_json(path: pathlib.Path, payload: Any, write: bool) -> str:
    """Create immutable JSON or verify an existing file byte-for-byte."""
    return progressive.write_or_verify(path, json_bytes(payload), write)


def stable_created_at(path: pathlib.Path) -> str:
    """Preserve an existing artifact timestamp during idempotent regeneration."""
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing = str(payload.get("created_at", "")).strip()
        if existing:
            return existing
    return utc_now()


def canonical_manifest(
    payload: Mapping[str, Any],
    direct_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Remove identity-only fields and replace direct paths with content hashes."""
    canonical = copy.deepcopy(dict(payload))
    canonical.pop("case_id", None)
    canonical.pop("run_label", None)
    canonical["model"]["umat_source"] = f"sha256:{direct_hashes['umat_source']}"
    canonical["files"]["material_parameters"] = (
        f"sha256:{direct_hashes['material_parameters']}"
    )
    canonical["files"]["experimental_data"] = (
        f"sha256:{direct_hashes['experimental_data']}"
    )
    if "coupling" in canonical:
        canonical["coupling"].pop("selection_basis", None)
    if "certified_study" in canonical:
        canonical["certified_study"].pop("selection_basis", None)
    return canonical


def semantic_fingerprint(
    payload: Mapping[str, Any],
    direct_hashes: Mapping[str, str],
    effective_parameter_specs: Mapping[str, Any],
) -> str:
    """Hash every fitting-relevant setting while excluding labels and paths."""
    fingerprint_payload = {
        "schema_version": 1,
        "manifest": canonical_manifest(payload, direct_hashes),
        "effective_parameter_specs": effective_parameter_specs,
    }
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def build_case_payload(
    baseline: Mapping[str, Any],
    step: str,
    added_ratio: str,
    model_id: str,
    config: progressive.KnitConfiguration,
    training_path: pathlib.Path,
) -> dict[str, Any]:
    """Derive one adaptive training-only production case from a baseline."""
    payload = progressive.build_case_payload(
        baseline,
        step,
        model_id,
        config,
        training_path,
    )
    identifier = candidate_id(step, added_ratio, model_id, config)
    payload["case_id"] = identifier
    payload["run_label"] = (
        f"{config.knit} adaptive {step} add {added_ratio} {model_id} training optimum"
    )
    if model_id == "SV_exp":
        payload["coupling"]["selection_basis"] = (
            f"Adaptive greedy {step}; candidate addition {added_ratio}"
        )
    else:
        payload["certified_study"]["selection_basis"] = (
            f"Adaptive greedy {step}; candidate addition {added_ratio}"
        )
    return payload


def expected_step_candidate_count(step_number: int) -> int:
    """Return candidates across eight paths, for example B01 has 72."""
    return 8 * (10 - step_number)


def expected_total_candidate_count() -> int:
    """Return all candidates in eight nine-ratio greedy searches: 360."""
    return sum(expected_step_candidate_count(step) for step in range(1, 10))


def load_prefixes(step_number: int) -> dict[str, tuple[str, ...]]:
    """Load each pair's frozen preceding winner sequence."""
    prefixes = {
        pair_id(model, config): ()
        for model in MODEL_IDS
        for config in progressive.KNIT_CONFIGURATIONS
    }
    if step_number == 1:
        return prefixes
    previous_path = SELECTION_DIRECTORY / f"{batch_id(step_number - 1)}.json"
    if not previous_path.is_file():
        raise FileNotFoundError(
            f"Previous adaptive selection is not frozen: {previous_path}"
        )
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    selected = previous.get("selected_by_pair", [])
    if len(selected) != 8:
        raise ValueError(f"{previous_path.name} must contain eight selected paths")
    for row in selected:
        key = str(row["pair_id"])
        ratios = tuple(row["training_ratios"])
        if key not in prefixes or len(ratios) != step_number - 1:
            raise ValueError(f"Invalid preceding adaptive prefix for {key}")
        if len(set(ratios)) != len(ratios) or not set(ratios) <= RATIO_SET:
            raise ValueError(f"Invalid ratio sequence in {previous_path}: {ratios}")
        prefixes[key] = ratios
    return prefixes


def effective_specs_for_baseline(case_path: pathlib.Path) -> dict[str, Any]:
    """Resolve parameter domains from one validated baseline manifest."""
    config = solver.load_case_configuration(case_path)
    model_spec = solver.load_model_spec(config.model_id)
    _, effective = solver.parameter_specs_from_file(config, model_spec)
    return effective


def build_step_candidates(step_number: int) -> list[dict[str, Any]]:
    """Build all in-memory candidate definitions for one adaptive Batch."""
    step = batch_id(step_number)
    prefixes = load_prefixes(step_number)
    candidates: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for config in progressive.KNIT_CONFIGURATIONS:
            key = pair_id(model_id, config)
            prefix = prefixes[key]
            curves = progressive.source_curve_paths(config)
            baseline_path = config.baseline_cases[model_id]
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            effective_specs = effective_specs_for_baseline(baseline_path)
            for added_ratio in RATIO_ORDER:
                if added_ratio in prefix:
                    continue
                training_ratios = prefix + (added_ratio,)
                training_bytes = progressive.build_batch_bytes(training_ratios, curves)
                prefix_bytes = progressive.build_batch_bytes(prefix, curves) if prefix else b""
                if prefix_bytes and not training_bytes.startswith(prefix_bytes):
                    raise ValueError(f"{key} {step} candidate lost its byte prefix")
                training_path = progressive.batch_output_directory(config) / candidate_filename(
                    step, added_ratio, model_id, config
                )
                payload = build_case_payload(
                    baseline,
                    step,
                    added_ratio,
                    model_id,
                    config,
                    training_path,
                )
                identifier = str(payload["case_id"])
                case_path = OUTPUT_DIRECTORY / "cases" / f"{identifier}.json"
                direct_hashes = {
                    "material_parameters": progressive.sha256_file(
                        progressive.PROJECT_ROOT / payload["files"]["material_parameters"]
                    ),
                    "experimental_data": sha256_bytes(training_bytes),
                    "umat_source": progressive.sha256_file(
                        progressive.PROJECT_ROOT / payload["model"]["umat_source"]
                    ),
                }
                candidates.append(
                    {
                        "candidate_id": identifier,
                        "pair_id": key,
                        "batch": step,
                        "step_number": step_number,
                        "model": model_id,
                        "knit": config.knit,
                        "sample_folder": config.sample_token,
                        "added_ratio": added_ratio,
                        "training_ratios": list(training_ratios),
                        "training_data": str(training_path.resolve()),
                        "training_data_project_relative": progressive.project_relative(
                            training_path
                        ),
                        "case_file": str(case_path.resolve()),
                        "case_file_project_relative": progressive.project_relative(case_path),
                        "row_count": int(
                            sum(
                                np.atleast_2d(np.loadtxt(curves[ratio])).shape[0]
                                for ratio in training_ratios
                            )
                        ),
                        "training_bytes": training_bytes,
                        "case_payload": payload,
                        "case_bytes": progressive.json_bytes(payload),
                        "direct_hashes": direct_hashes,
                        "semantic_fingerprint": semantic_fingerprint(
                            payload, direct_hashes, effective_specs
                        ),
                        "sources": [
                            {
                                "ratio": ratio,
                                "path": progressive.project_relative(curves[ratio]),
                                "row_count": int(
                                    np.atleast_2d(np.loadtxt(curves[ratio])).shape[0]
                                ),
                                "sha256": progressive.sha256_file(curves[ratio]),
                            }
                            for ratio in training_ratios
                        ],
                    }
                )
    expected = expected_step_candidate_count(step_number)
    if len(candidates) != expected or len({row["candidate_id"] for row in candidates}) != expected:
        raise RuntimeError(f"{step} expected {expected} unique candidates")
    return candidates


def public_candidate_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Remove in-memory bytes and payloads from a stored candidate record."""
    excluded = {"training_bytes", "case_bytes", "case_payload"}
    return {key: value for key, value in candidate.items() if key not in excluded}


def artifact_is_reusable(artifact: Mapping[str, Any]) -> bool:
    """Require the deterministic validation evidence used by production audits."""
    if artifact.get("result_role") != "training_optimum":
        return False
    diagnostics = artifact.get("diagnostics", {})
    direct = diagnostics.get("direct_validation", {})
    differences = (
        direct.get("absolute_cost_difference"),
        direct.get("existing_wrapper_cost_difference"),
        direct.get("maximum_weighted_prediction_difference"),
        diagnostics.get("maximum_prediction_difference"),
    )
    try:
        if any(abs(float(value)) > VALIDATION_TOLERANCE for value in differences):
            return False
    except (TypeError, ValueError):
        return False
    if artifact.get("model_id") == "StapletonViaux":
        certificate = direct.get("certificate", {})
        if (
            certificate.get("certificate_satisfied") is not True
            or certificate.get("status") != "enumeration_complete"
        ):
            return False
    input_files = artifact.get("input_files", {})
    if set(input_files) != {
        "case_config",
        "material_parameters",
        "experimental_data",
        "umat_source",
    }:
        return False
    for record in input_files.values():
        snapshot = pathlib.Path(str(record.get("snapshot_path", "")))
        if not snapshot.is_file() or sha256_file(snapshot) != record.get("sha256"):
            return False
    return True


def reusable_fit_index() -> dict[str, list[dict[str, Any]]]:
    """Index complete audited registry artifacts by semantic fitting fingerprint."""
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with REGISTRY_PATH.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row.get("Status") != "complete" or not row.get("Artifact_Path"):
            continue
        artifact_path = pathlib.Path(resolve_string(row["Artifact_Path"]))
        if not artifact_path.is_file():
            continue
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        # Reuse is decided against fits that predate this study.  Excluding the
        # study's own cases prevents a newly completed candidate (for example,
        # B01 fit 788) from changing from ``new`` to ``reused`` on regeneration.
        if str(artifact.get("case_id", "")).startswith(f"{CASE_ID_PREFIX}_"):
            continue
        if not artifact_is_reusable(artifact):
            continue
        inputs = artifact["input_files"]
        case_snapshot = pathlib.Path(inputs["case_config"]["snapshot_path"])
        try:
            config = solver.load_case_configuration(case_snapshot)
            model_spec = solver.load_model_spec(config.model_id)
            _, effective_specs = solver.parameter_specs_from_file(config, model_spec)
        except Exception:
            continue
        direct_hashes = {
            name: str(inputs[name]["sha256"])
            for name in ("material_parameters", "experimental_data", "umat_source")
        }
        fingerprint = semantic_fingerprint(config.payload, direct_hashes, effective_specs)
        index[fingerprint].append(
            {
                "fit_code": int(artifact["fit_code"]),
                "artifact_path": str(artifact_path.resolve()),
                "source_case_id": str(artifact.get("case_id", "")),
                "source_case_snapshot": str(case_snapshot.resolve()),
                "direct_hashes": direct_hashes,
            }
        )
    for matches in index.values():
        matches.sort(key=lambda record: int(record["fit_code"]))
    return dict(index)


def attach_reuse_records(
    candidates: list[dict[str, Any]],
    reuse_index: Mapping[str, Sequence[Mapping[str, Any]]],
    write: bool,
) -> None:
    """Attach the lowest-code exact match and create an immutable certificate."""
    for candidate in candidates:
        matches = list(reuse_index.get(str(candidate["semantic_fingerprint"]), ()))
        if not matches:
            candidate["reuse"] = None
            continue
        chosen = dict(matches[0])
        certificate_path = REUSE_DIRECTORY / f"{candidate['candidate_id']}.json"
        certificate = {
            "schema_version": 1,
            "study_id": STUDY_ID,
            "created_at": stable_created_at(certificate_path),
            "candidate_id": candidate["candidate_id"],
            "semantic_fingerprint": candidate["semantic_fingerprint"],
            "fit_code": chosen["fit_code"],
            "source_case_id": chosen["source_case_id"],
            "source_case_snapshot": chosen["source_case_snapshot"],
            "artifact_path": chosen["artifact_path"],
            "candidate_direct_hashes": candidate["direct_hashes"],
            "matched_direct_hashes": chosen["direct_hashes"],
            "equivalent_fit_codes": [int(match["fit_code"]) for match in matches],
            "validation": {
                "registry_status": "complete",
                "artifact_deterministic_audit": True,
                "identity_fields_excluded": [
                    "case_id",
                    "run_label",
                    "selection_basis",
                    "filesystem_paths",
                    "timestamps",
                ],
            },
        }
        write_or_verify_json(certificate_path, certificate, write)
        candidate["reuse"] = {
            "fit_code": chosen["fit_code"],
            "artifact_path": chosen["artifact_path"],
            "source_case_id": chosen["source_case_id"],
            "certificate_path": str(certificate_path.resolve()),
        }


def update_adjacent_catalogs(candidates: Sequence[Mapping[str, Any]]) -> None:
    """Merge permanent candidate provenance into each knit's raw-data catalog."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["knit"])].append(candidate)
    for config in progressive.KNIT_CONFIGURATIONS:
        current = grouped.get(config.knit, [])
        catalog_path = progressive.batch_output_directory(config) / ADJACENT_CATALOG_FILENAME
        if catalog_path.is_file():
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        else:
            catalog = {
                "schema_version": 1,
                "study_id": STUDY_ID,
                "knit": config.knit,
                "created_at": utc_now(),
                "candidates": [],
            }
        existing = {row["candidate_id"]: row for row in catalog.get("candidates", [])}
        for candidate in current:
            record = public_candidate_record(candidate)
            previous = existing.get(str(candidate["candidate_id"]))
            if previous is not None and previous != record:
                raise FileExistsError(
                    f"Conflicting adaptive catalog record: {candidate['candidate_id']}"
                )
            existing[str(candidate["candidate_id"])] = record
        catalog["updated_at"] = utc_now()
        catalog["candidates"] = [existing[key] for key in sorted(existing)]
        write_json_atomic(catalog_path, catalog)


def prepare_step(step_number: int, write: bool) -> dict[str, Any]:
    """Preview or create all candidate data, manifests, reuse records, and lists."""
    step = batch_id(step_number)
    candidates = build_step_candidates(step_number)
    reuse_index = reusable_fit_index()
    attach_reuse_records(candidates, reuse_index, write=False)
    if write:
        readme_entries: dict[pathlib.Path, dict[str, Sequence[str]]] = defaultdict(dict)
        for candidate in candidates:
            training_path = pathlib.Path(candidate["training_data"])
            case_path = pathlib.Path(candidate["case_file"])
            progressive.write_or_verify(
                training_path,
                candidate["training_bytes"],
                True,
            )
            progressive.write_or_verify(case_path, candidate["case_bytes"], True)
            readme_entries[training_path.parent][training_path.name] = tuple(
                candidate["training_ratios"]
            )
        for folder, entries in readme_entries.items():
            progressive.update_readme(folder / "Readme.txt", entries, True)

        # All manifests, including reusable candidates, must pass production preflight.
        batch_runner.build_case_jobs(tuple(candidate["case_file"] for candidate in candidates))
        attach_reuse_records(candidates, reuse_index, write=True)
        update_adjacent_catalogs(candidates)

    definition_path = DEFINITION_DIRECTORY / f"{step}.json"
    definition = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_at": stable_created_at(definition_path),
        "batch": step,
        "step_number": step_number,
        "candidate_count": len(candidates),
        "candidates": [public_candidate_record(candidate) for candidate in candidates],
    }
    write_or_verify_json(definition_path, definition, write)
    unresolved = [
        candidate["case_file"] for candidate in candidates if candidate["reuse"] is None
    ]
    case_list_path = CASE_LIST_DIRECTORY / f"{step}.json"
    case_list = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "batch": step,
        "case_count": len(unresolved),
        "case_files": unresolved,
    }
    write_or_verify_json(case_list_path, case_list, write)
    return {
        "batch": step,
        "candidate_count": len(candidates),
        "reuse_count": sum(candidate["reuse"] is not None for candidate in candidates),
        "new_fit_count": len(unresolved),
        "definition_path": str(definition_path.resolve()),
        "case_list_path": str(case_list_path.resolve()),
        "write": write,
    }


def run_command(command: Sequence[str]) -> None:
    """Run one visible child command and stop on any nonzero exit code."""
    result = subprocess.run(command, cwd=ANALYSIS_DIRECTORY)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")


def runner_state_complete(path: pathlib.Path, expected: int) -> bool:
    """Return True only when a runner state contains expected completed cases."""
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    cases = state.get("cases", [])
    return (
        len(cases) == expected
        and all(case.get("status") == "completed" for case in cases)
        and all(len(case.get("fit_codes", [])) == 1 for case in cases)
    )


def registry_rows() -> list[dict[str, str]]:
    """Read the shared method-neutral fit registry."""
    with REGISTRY_PATH.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    generated_registry = solver.LEGACY_RESULTS_CSV.parent / "fit_registry.csv"
    if generated_registry.is_file():
        with generated_registry.open(newline="", encoding="utf-8-sig") as file:
            rows.extend(csv.DictReader(file))
    return [{key: resolve_string(value) for key, value in row.items()} for row in rows]


def run_step(step_number: int, cores: int) -> dict[str, Any]:
    """Preview, execute/resume, audit, and map all candidates in one Batch."""
    step = batch_id(step_number)
    definition_path = DEFINITION_DIRECTORY / f"{step}.json"
    if not definition_path.is_file():
        raise FileNotFoundError(f"Prepare {step} before running it")
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    candidates = definition["candidates"]
    unresolved = [row for row in candidates if row.get("reuse") is None]
    case_list_path = CASE_LIST_DIRECTORY / f"{step}.json"
    state_path = STATE_DIRECTORY / f"{step}_runner.json"

    if unresolved and not runner_state_complete(state_path, len(unresolved)):
        command = [
            sys.executable,
            str((ANALYSIS_DIRECTORY / "fitting" / "run_separable_solver_batch.py").resolve()),
            "--case-list",
            str(case_list_path.resolve()),
            "--state-file",
            str(state_path.resolve()),
            "--cores",
            str(cores),
        ]
        if state_path.is_file():
            run_command(command + ["--recover-only"])
            if not runner_state_complete(state_path, len(unresolved)):
                run_command(command + ["--resume"])
        else:
            run_command(command + ["--preview"])
            run_command(command + ["--execute"])
    if unresolved and not runner_state_complete(state_path, len(unresolved)):
        raise RuntimeError(f"{step} runner state is not complete")

    new_codes: dict[str, int] = {}
    if unresolved:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        new_codes = {
            str(case["case_id"]): int(case["fit_codes"][0]) for case in state["cases"]
        }
    registry = {int(row["Code"]): row for row in registry_rows() if row.get("Code")}
    mappings: list[dict[str, Any]] = []
    for candidate in candidates:
        reused = candidate.get("reuse") is not None
        code = (
            int(candidate["reuse"]["fit_code"])
            if reused
            else int(new_codes[candidate["candidate_id"]])
        )
        row = registry.get(code)
        if row is None or row.get("Status") != "complete":
            raise ValueError(f"Adaptive candidate {candidate['candidate_id']} fit is incomplete")
        if not reused and row.get("Case_ID") != candidate["candidate_id"]:
            raise ValueError(f"Registry case ID differs for new adaptive fit {code}")
        mappings.append(
            {
                "candidate_id": candidate["candidate_id"],
                "pair_id": candidate["pair_id"],
                "fit_code": code,
                "reused_fit": reused,
                "registry_case_id": row.get("Case_ID", ""),
                "artifact_path": row.get("Artifact_Path", ""),
            }
        )
    if len(mappings) != len(candidates) or len({row["fit_code"] for row in mappings}) != len(mappings):
        raise ValueError(f"{step} must map each candidate to one unique registered fit")

    fit_map_path = FIT_MAP_DIRECTORY / f"{step}.json"
    fit_map = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_at": stable_created_at(fit_map_path),
        "batch": step,
        "definition_path": str(definition_path.resolve()),
        "runner_state": str(state_path.resolve()) if unresolved else None,
        "candidate_count": len(mappings),
        "reused_count": sum(row["reused_fit"] for row in mappings),
        "new_fit_count": sum(not row["reused_fit"] for row in mappings),
        "mappings": mappings,
    }
    write_or_verify_json(fit_map_path, fit_map, True)
    return fit_map


def pooled_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Reproduce pooled R2 and RMSE from per-ratio SSE, SST, and N."""
    sse = sum(float(row["SSE_course"]) + float(row["SSE_wales"]) for row in rows)
    sst = sum(float(row["SST_course"]) + float(row["SST_wales"]) for row in rows)
    count = sum(int(row["N_course"]) + int(row["N_wales"]) for row in rows)
    if sst <= 0 or count <= 0:
        raise ValueError("Pooled metrics require positive SST and observation count")
    return {
        "pooled_SSE": sse,
        "pooled_SST": sst,
        "pooled_N": count,
        "pooled_R2": 1.0 - sse / sst,
        "pooled_RMSE": math.sqrt(sse / count),
    }


def normalize_ratio(value: Any) -> str:
    """Normalize workbook labels, for example 0.5 to 1 becomes 0.5:1."""
    return global_errors.normalize_training_ratio(value)


def candidate_score(
    per_ratio_rows: Sequence[Mapping[str, Any]],
    training_ratios: Sequence[str],
) -> dict[str, Any]:
    """Calculate fixed all-ratio and descriptive transfer-only pooled metrics."""
    if len(per_ratio_rows) != 9 or {normalize_ratio(row["ratio"]) for row in per_ratio_rows} != RATIO_SET:
        raise ValueError("Every adaptive candidate must be evaluated on all nine ratios")
    all_metrics = pooled_metrics(per_ratio_rows)
    trained = set(training_ratios)
    transfer_rows = [
        row for row in per_ratio_rows if normalize_ratio(row["ratio"]) not in trained
    ]
    transfer = pooled_metrics(transfer_rows) if transfer_rows else None
    return {
        "all_ratio_pooled_R2": all_metrics["pooled_R2"],
        "all_ratio_pooled_RMSE": all_metrics["pooled_RMSE"],
        "all_ratio_pooled_SSE": all_metrics["pooled_SSE"],
        "all_ratio_pooled_SST": all_metrics["pooled_SST"],
        "all_ratio_pooled_N": all_metrics["pooled_N"],
        "transfer_ratio_count": len(transfer_rows),
        "transfer_ratios": [normalize_ratio(row["ratio"]) for row in transfer_rows],
        "transfer_pooled_R2": None if transfer is None else transfer["pooled_R2"],
        "transfer_pooled_RMSE": None if transfer is None else transfer["pooled_RMSE"],
        "transfer_pooled_SSE": None if transfer is None else transfer["pooled_SSE"],
        "transfer_pooled_SST": None if transfer is None else transfer["pooled_SST"],
        "transfer_pooled_N": None if transfer is None else transfer["pooled_N"],
    }


def compare_candidates(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    """Order candidates by the registered deterministic selection rule."""
    delta = float(left["all_ratio_pooled_R2"]) - float(right["all_ratio_pooled_R2"])
    if abs(delta) > TIE_TOLERANCE:
        return -1 if delta > 0 else 1
    left_transfer = left.get("transfer_pooled_R2")
    right_transfer = right.get("transfer_pooled_R2")
    left_value = -math.inf if left_transfer is None else float(left_transfer)
    right_value = -math.inf if right_transfer is None else float(right_transfer)
    if left_value != right_value:
        return -1 if left_value > right_value else 1
    left_rmse = float(left["all_ratio_pooled_RMSE"])
    right_rmse = float(right["all_ratio_pooled_RMSE"])
    if left_rmse != right_rmse:
        return -1 if left_rmse < right_rmse else 1
    return RATIO_ORDER.index(str(left["added_ratio"])) - RATIO_ORDER.index(
        str(right["added_ratio"])
    )


def rank_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a uniquely ranked candidate list with the winner first."""
    ordered = sorted((dict(row) for row in rows), key=cmp_to_key(compare_candidates))
    for rank, row in enumerate(ordered, start=1):
        row["candidate_rank"] = rank
        row["selected_at_step"] = rank == 1
    return ordered


def previous_selected_score(step_number: int, key: str) -> float | None:
    """Return the preceding winner's fixed all-ratio R2, or None for B01."""
    if step_number == 1:
        return None
    path = SELECTION_DIRECTORY / f"{batch_id(step_number - 1)}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    match = next(row for row in payload["selected_by_pair"] if row["pair_id"] == key)
    return float(match["all_ratio_pooled_R2"])


def adaptive_fields(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return workbook context common to every row for one candidate."""
    return {
        "adaptive_study_id": STUDY_ID,
        "adaptive_batch": candidate["batch"],
        "adaptive_candidate_id": candidate["candidate_id"],
        "adaptive_pair_id": candidate["pair_id"],
        "adaptive_added_ratio": candidate["added_ratio"],
        "adaptive_training_ratios": ", ".join(candidate["training_ratios"]),
    }


def workbook_evaluation_count(path: pathlib.Path, sheet: str, evaluation_id: str) -> int:
    """Count one evaluation ID in a workbook sheet for idempotent appends."""
    if not path.is_file():
        return 0
    workbook = load_workbook(path, read_only=True, data_only=True)
    if sheet not in workbook.sheetnames:
        workbook.close()
        return 0
    worksheet = workbook[sheet]
    headers = [cell.value for cell in next(worksheet.iter_rows(min_row=1, max_row=1))]
    if "adaptive_evaluation_id" not in headers:
        workbook.close()
        return 0
    column = headers.index("adaptive_evaluation_id") + 1
    count = sum(
        cell.value == evaluation_id
        for row in worksheet.iter_rows(min_row=2, min_col=column, max_col=column)
        for cell in row
    )
    workbook.close()
    return count


def append_global_rows_once(evaluation: Mapping[str, Any]) -> None:
    """Append adaptive rows exactly once and store a verified receipt."""
    evaluation_id = str(evaluation["evaluation_id"])
    receipt_path = EVALUATION_DIRECTORY / f"{evaluation['batch']}_global_append.json"
    expected_ratio = len(evaluation["per_ratio_rows"])
    expected_summary = len(evaluation["summary_rows"])
    ratio_count = workbook_evaluation_count(
        GLOBAL_WORKBOOK, "Per_Ratio", evaluation_id
    )
    summary_count = workbook_evaluation_count(
        GLOBAL_WORKBOOK, "Fit_Summary", evaluation_id
    )
    if ratio_count == expected_ratio and summary_count == expected_summary:
        if not receipt_path.is_file():
            write_json_atomic(
                receipt_path,
                {
                    "schema_version": 1,
                    "evaluation_id": evaluation_id,
                    "verified_at": utc_now(),
                    "per_ratio_rows": ratio_count,
                    "summary_rows": summary_count,
                },
            )
        return
    if ratio_count or summary_count:
        raise ValueError(
            f"Partial adaptive workbook append for {evaluation_id}: "
            f"{ratio_count}/{expected_ratio}, {summary_count}/{expected_summary}"
        )
    global_errors.write_xlsx(
        GLOBAL_WORKBOOK,
        evaluation["per_ratio_rows"],
        evaluation["summary_rows"],
        evaluation["metadata_rows"],
    )
    append_global_rows_once(evaluation)


def evaluate_step(
    step_number: int,
    append_global: bool,
    cores: int = 1,
) -> dict[str, Any]:
    """Evaluate, rank, freeze, and optionally append every candidate in one Batch."""
    step = batch_id(step_number)
    evaluation_path = EVALUATION_DIRECTORY / f"{step}.json"
    selection_path = SELECTION_DIRECTORY / f"{step}.json"
    if evaluation_path.is_file() and selection_path.is_file():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if append_global:
            append_global_rows_once(evaluation)
        return evaluation

    definition = json.loads(
        (DEFINITION_DIRECTORY / f"{step}.json").read_text(encoding="utf-8")
    )
    fit_map = json.loads(
        (FIT_MAP_DIRECTORY / f"{step}.json").read_text(encoding="utf-8")
    )
    candidate_by_id = {row["candidate_id"]: row for row in definition["candidates"]}
    mapping_by_id = {row["candidate_id"]: row for row in fit_map["mappings"]}
    if set(candidate_by_id) != set(mapping_by_id):
        raise ValueError(f"{step} definition and fit map do not agree")

    run_timestamp = utc_now()
    summaries: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"adaptive_error_{step.lower()}_") as temporary:
        temporary_path = pathlib.Path(temporary)
        total = len(candidate_by_id)
        ordered_candidates = [
            (identifier, candidate_by_id[identifier], mapping_by_id[identifier])
            for identifier in candidate_by_id
        ]
        # Each subprocess receives an isolated Meson/DLL directory, which makes
        # the requested five-worker evaluation safe on Windows.
        with ThreadPoolExecutor(max_workers=min(cores, total)) as executor:
            futures = {
                identifier: executor.submit(
                    global_errors.evaluate_fit_in_worker,
                    identifier,
                    int(mapping["fit_code"]),
                    temporary_path,
                    True,
                    results_csv=(global_errors.DEFAULT_RESULTS_CSV_PATH if mapping["reused_fit"] else solver.LEGACY_RESULTS_CSV),
                )
                for identifier, _candidate, mapping in ordered_candidates
            }
            for index, (identifier, candidate, mapping) in enumerate(
                ordered_candidates,
                start=1,
            ):
                print(
                    f"[{step} evaluation {index}/{total}] {identifier} / "
                    f"fit {mapping['fit_code']}",
                    flush=True,
                )
                payload = futures[identifier].result()
                common = {
                "run_timestamp": run_timestamp,
                **adaptive_fields(candidate),
                "adaptive_reused_fit": bool(mapping["reused_fit"]),
                }
                candidate_ratio_rows = []
                for raw_row in payload["per_ratio_rows"]:
                    row = {**common, **raw_row}
                    ratio = normalize_ratio(row["ratio"])
                    row["progressive_batch"] = step
                    row["ratio_training_status"] = (
                        "trained"
                        if ratio in candidate["training_ratios"]
                        else "unseen"
                    )
                    candidate_ratio_rows.append(row)
                score = {
                    **adaptive_fields(candidate),
                    "fit_code": int(mapping["fit_code"]),
                    "reused_fit": bool(mapping["reused_fit"]),
                    **candidate_score(
                        candidate_ratio_rows,
                        candidate["training_ratios"],
                    ),
                }
                scores.append(score)
                ratio_rows.extend(candidate_ratio_rows)
                summary = {**common, **payload["summary_row"]}
                summary["progressive_batch"] = step
                summary["Training_ratios"] = ", ".join(candidate["training_ratios"])
                summary.update(score)
                summaries.append(summary)
                metadata = {**common, **payload["metadata_row"]}
                metadata["progressive_batch"] = step
                metadata_rows.append(metadata)

    ranked_by_pair: dict[str, list[dict[str, Any]]] = {}
    for key, rows in grouped(scores, "adaptive_pair_id").items():
        ranked_by_pair[key] = rank_candidates(rows)
    ranking_by_candidate = {
        row["adaptive_candidate_id"]: row
        for rows in ranked_by_pair.values()
        for row in rows
    }
    for row in ratio_rows + summaries + metadata_rows:
        rank = ranking_by_candidate[row["adaptive_candidate_id"]]
        row["adaptive_candidate_rank"] = rank["candidate_rank"]
        row["adaptive_selected_at_step"] = rank["selected_at_step"]

    selected_rows: list[dict[str, Any]] = []
    for key in sorted(ranked_by_pair):
        ranking = ranked_by_pair[key]
        winner = dict(ranking[0])
        candidate = candidate_by_id[winner["adaptive_candidate_id"]]
        previous_score = previous_selected_score(step_number, key)
        improvement = (
            None
            if previous_score is None
            else float(winner["all_ratio_pooled_R2"]) - previous_score
        )
        winner.update(
            {
                "pair_id": key,
                "candidate_id": candidate["candidate_id"],
                "model": candidate["model"],
                "knit": candidate["knit"],
                "sample_folder": candidate["sample_folder"],
                "batch": step,
                "added_ratio": candidate["added_ratio"],
                "training_ratios": candidate["training_ratios"],
                "previous_selected_pooled_R2": previous_score,
                "incremental_pooled_R2": improvement,
                "negative_increment": improvement is not None and improvement < 0.0,
                "runner_up_margin": (
                    None
                    if len(ranking) == 1
                    else float(winner["all_ratio_pooled_R2"])
                    - float(ranking[1]["all_ratio_pooled_R2"])
                ),
            }
        )
        selected_rows.append(winner)

    fit_map_hash = sha256_file(FIT_MAP_DIRECTORY / f"{step}.json")
    evaluation_id = f"{STUDY_ID}_{step.lower()}_{fit_map_hash[:12]}"
    for row in ratio_rows + summaries + metadata_rows:
        row["adaptive_evaluation_id"] = evaluation_id
    evaluation = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_at": run_timestamp,
        "evaluation_id": evaluation_id,
        "batch": step,
        "fit_map_sha256": fit_map_hash,
        "candidate_count": len(summaries),
        "per_ratio_count": len(ratio_rows),
        "selection_metric": "fixed_all_nine_ratio_overall_pooled_R2",
        "tie_tolerance": TIE_TOLERANCE,
        "candidate_scores": scores,
        "rankings_by_pair": ranked_by_pair,
        "per_ratio_rows": ratio_rows,
        "summary_rows": summaries,
        "metadata_rows": metadata_rows,
    }
    selection = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_at": run_timestamp,
        "batch": step,
        "evaluation_id": evaluation_id,
        "selected_by_pair": selected_rows,
    }
    write_or_verify_json(evaluation_path, evaluation, True)
    write_or_verify_json(selection_path, selection, True)
    if append_global:
        append_global_rows_once(evaluation)
    return evaluation


def grouped(
    rows: Iterable[Mapping[str, Any]], key: str
) -> dict[str, list[Mapping[str, Any]]]:
    """Group dictionaries by one text field, for example adaptive_pair_id."""
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row[key])].append(row)
    return dict(output)


def process_step(
    step_number: int,
    action: str,
    cores: int,
    append_global: bool,
) -> dict[str, Any]:
    """Run the requested safe lifecycle action for one Batch."""
    if action == "preview":
        return prepare_step(step_number, write=False)
    preparation = prepare_step(step_number, write=True)
    if action == "prepare":
        return preparation
    fit_map = run_step(step_number, cores)
    if action == "run":
        return fit_map
    evaluation = evaluate_step(step_number, append_global, cores)
    return evaluation


def print_result(result: Mapping[str, Any]) -> None:
    """Print a compact lifecycle result for VSCode users."""
    print(json.dumps(result, indent=2, allow_nan=False)[:8000])


def main(argv: Sequence[str] | None = None) -> None:
    """Parse VSCode/CLI controls and process one or all adaptive Batches."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("preview", "prepare", "run", "evaluate", "all"),
        default=ACTION,
        help=f"Lifecycle action; checked-in default is {ACTION!r}.",
    )
    parser.add_argument(
        "--batch",
        default=TARGET_BATCH,
        help=f"One Batch B01-B09; checked-in default is {TARGET_BATCH}.",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Process B01 through B09 sequentially, freezing each winner first.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=DEFAULT_PHYSICAL_CORES,
        help=f"Production case workers; default {DEFAULT_PHYSICAL_CORES}.",
    )
    parser.add_argument(
        "--no-global-workbook",
        action="store_true",
        help="Keep study-local evaluations without appending the shared workbook.",
    )
    args = parser.parse_args(argv)
    if args.cores <= 0:
        parser.error("--cores must be positive")
    steps = range(1, 10) if args.all_steps else (batch_number(args.batch),)
    for step_number in steps:
        print(f"\n=== Adaptive {batch_id(step_number)} / {args.action} ===", flush=True)
        result = process_step(
            step_number,
            args.action,
            args.cores,
            APPEND_GLOBAL_WORKBOOK and not args.no_global_workbook,
        )
        print_result(
            {
                key: value
                for key, value in result.items()
                if key not in {"per_ratio_rows", "summary_rows", "metadata_rows"}
            }
        )


if __name__ == "__main__":
    main()
