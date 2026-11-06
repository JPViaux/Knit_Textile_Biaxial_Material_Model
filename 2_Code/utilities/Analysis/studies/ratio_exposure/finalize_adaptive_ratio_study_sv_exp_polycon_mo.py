"""Audit and stage the complete SV_exp_polycon_mo greedy ratio study.

Run this file from VSCode after B01--B09 finish. It reads frozen campaign
artifacts, checks the requested affine and cone constraints, and writes the
JSON payload consumed by the artifact-tool workbook builder.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (ARCHIVE_ROOT as RELEASE_ARCHIVE_ROOT)

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from release_paths import json
import math
import pathlib
import sys
from typing import Any, Mapping


ANALYSIS_DIRECTORY = next(
    parent for parent in pathlib.Path(__file__).resolve().parents
    if parent.name == "Analysis"
)
SUPPORT_DIRECTORY = ANALYSIS_DIRECTORY.parent
CODE_DIRECTORY = SUPPORT_DIRECTORY.parent
for import_directory in (CODE_DIRECTORY, SUPPORT_DIRECTORY, ANALYSIS_DIRECTORY):
    if str(import_directory) not in sys.path:
        sys.path.insert(0, str(import_directory))

from Analysis.studies.ratio_exposure import (  # noqa: E402
    adaptive_ratio_study_sv_exp_polycon_mo as wrapper,
)


# Editable output names. Example: use a version suffix for an alternate report.
PAYLOAD_FILENAME = "full_workbook_payload.json"
AUDIT_FILENAME = "full_study_audit.json"
REPORT_FILENAME = "sv_exp_polycon_mo_greedy_study.xlsx"


def utc_now() -> str:
    """Return an ISO UTC timestamp, for example 2026-08-26T09:00:00Z."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    """Read a frozen JSON artifact, for example selections/B09.json."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: pathlib.Path) -> str:
    """Hash a file in chunks, for example the shared global-error workbook."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_campaign() -> dict[str, list[dict[str, Any]]]:
    """Load all nine production definitions, maps, evaluations, and selections."""
    wrapper.configure_engine(full_campaign=True)
    engine = wrapper.engine
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in range(1, 10):
        batch = engine.batch_id(step)
        definition = read_json(engine.DEFINITION_DIRECTORY / f"{batch}.json")
        mapping = read_json(engine.FIT_MAP_DIRECTORY / f"{batch}.json")
        evaluation = read_json(engine.EVALUATION_DIRECTORY / f"{batch}.json")
        selection = read_json(engine.SELECTION_DIRECTORY / f"{batch}.json")
        output["definitions"].extend(definition["candidates"])
        output["mappings"].extend(mapping["mappings"])
        output["summaries"].extend(evaluation["summary_rows"])
        output["ratios"].extend(evaluation["per_ratio_rows"])
        output["metadata"].extend(evaluation["metadata_rows"])
        output["selections"].extend(selection["selected_by_pair"])
        receipt = engine.EVALUATION_DIRECTORY / f"{batch}_global_append.json"
        output["receipts"].append(read_json(receipt))
    return dict(output)


def registry_complete_codes() -> set[int]:
    """Return complete registry codes, for example 1984 through 2163."""
    path = RELEASE_ARCHIVE_ROOT / "fit_registry.csv"
    with path.open(newline="", encoding="utf-8-sig") as file:
        return {
            int(row["Code"])
            for row in csv.DictReader(file)
            if row.get("Code") and row.get("Status") == "complete"
        }


def consensus_rows(selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count cross-knit agreement on each added ratio within a batch."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        grouped[(str(row["batch"]), str(row["added_ratio"]))].append(row)
    return [
        {
            "batch": batch,
            "added_ratio": ratio,
            "selected_knit_count": len(rows),
            "selected_knits": ", ".join(sorted(str(row["knit"]) for row in rows)),
            "mean_selected_all_ratio_pooled_R2": sum(
                float(row["all_ratio_pooled_R2"]) for row in rows
            ) / len(rows),
        }
        for (batch, ratio), rows in sorted(grouped.items())
    ]


def build_payload(records: Mapping[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconcile campaign totals and build flat, filterable report tables."""
    definitions = records["definitions"]
    mappings = records["mappings"]
    summaries = records["summaries"]
    ratios = records["ratios"]
    selections = records["selections"]
    expected = {"candidates": 180, "selected": 36, "fit_ratio": 1620, "parameters": 2700}
    if len(definitions) != expected["candidates"] or len(mappings) != expected["candidates"]:
        raise ValueError("Expected 180 definitions and 180 mappings")
    if len(summaries) != expected["candidates"] or len(ratios) != expected["fit_ratio"]:
        raise ValueError("Evaluation tables do not reconcile to 180 x 9 rows")
    if len(selections) != expected["selected"]:
        raise ValueError("Expected 36 frozen selected-path records")
    if len(records["receipts"]) != 9:
        raise ValueError("Expected nine global-workbook append receipts")

    definition_by_id = {str(row["candidate_id"]): row for row in definitions}
    mapping_by_id = {str(row["candidate_id"]): row for row in mappings}
    summary_by_id = {str(row["adaptive_candidate_id"]): row for row in summaries}
    if not (len(definition_by_id) == len(mapping_by_id) == len(summary_by_id) == 180):
        raise ValueError("Campaign candidate identifiers are not unique")
    fit_codes = {int(row["fit_code"]) for row in mappings}
    if len(fit_codes) != 180 or not fit_codes <= registry_complete_codes():
        raise ValueError("Every campaign fit must be unique and registry-complete")

    ratio_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ratios:
        ratio_groups[str(row["adaptive_candidate_id"])].append(row)

    candidate_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    maximum_difference = 0.0
    minimum_margin_minus_tolerance = math.inf
    maximum_affine_error = 0.0
    backend_counts: Counter[str] = Counter()
    for identifier, definition in definition_by_id.items():
        mapping = mapping_by_id[identifier]
        summary = summary_by_id[identifier]
        artifact_path = pathlib.Path(str(mapping["artifact_path"]))
        artifact = read_json(artifact_path)
        if len(ratio_groups[identifier]) != 9:
            raise ValueError(f"{identifier} does not have nine evaluation rows")
        diagnostic = artifact["diagnostics"]
        solution = diagnostic["solution"]
        constraint_name = "4*mu*mu2-mu3^2"
        margin = float(solution["constraint_margins"][constraint_name])
        tolerance = float(solution["constraint_tolerances"][constraint_name])
        affine_error = abs(float(artifact["parameters"]["a"]) + float(artifact["parameters"]["b"]) - 1.0)
        difference = abs(float(diagnostic["maximum_prediction_difference"]))
        if margin < -tolerance:
            raise ValueError(f"{identifier} violates the declared cone tolerance")
        if affine_error > 1.0e-12:
            raise ValueError(f"{identifier} violates a = 1 - b")
        maximum_difference = max(maximum_difference, difference)
        minimum_margin_minus_tolerance = min(minimum_margin_minus_tolerance, margin + tolerance)
        maximum_affine_error = max(maximum_affine_error, affine_error)
        backend = str(solution["linear_solver_backend"])
        backend_counts[backend] += 1

        candidate_rows.append({
            "candidate_id": identifier,
            "batch": definition["batch"],
            "knit": definition["knit"],
            "added_ratio": definition["added_ratio"],
            "training_ratios": ", ".join(definition["training_ratios"]),
            "fit_code": int(mapping["fit_code"]),
            "selected_at_step": bool(summary["adaptive_selected_at_step"]),
            "candidate_rank": int(summary["adaptive_candidate_rank"]),
            "all_ratio_pooled_R2": float(summary["all_ratio_pooled_R2"]),
            "all_ratio_pooled_RMSE": float(summary["all_ratio_pooled_RMSE"]),
            "transfer_ratio_count": int(summary["transfer_ratio_count"]),
            "transfer_pooled_R2": summary["transfer_pooled_R2"],
            "artifact_path": str(artifact_path),
        })
        for name, value in artifact["parameters"].items():
            spec = artifact["parameter_specs"][name]
            parameter_rows.append({
                "candidate_id": identifier,
                "batch": definition["batch"],
                "knit": definition["knit"],
                "fit_code": int(mapping["fit_code"]),
                "selected_at_step": bool(summary["adaptive_selected_at_step"]),
                "parameter": name,
                "value": value,
                "minimum": spec["minimum"],
                "maximum": spec["maximum"],
                "fixed": bool(spec["fixed"]),
            })
        constraint_rows.append({
            "candidate_id": identifier,
            "batch": definition["batch"],
            "knit": definition["knit"],
            "fit_code": int(mapping["fit_code"]),
            "linear_solver_backend": backend,
            "cone_margin": margin,
            "cone_tolerance": tolerance,
            "cone_feasible": margin >= -tolerance,
            "a": artifact["parameters"]["a"],
            "b": artifact["parameters"]["b"],
            "a_plus_b_minus_1": float(artifact["parameters"]["a"]) + float(artifact["parameters"]["b"]) - 1.0,
            "maximum_prediction_difference": difference,
            "effective_K_coup_upper_bound": solution["effective_linear_upper_bounds"]["K_coup"],
        })
        provenance_rows.append({
            "candidate_id": identifier,
            "fit_code": int(mapping["fit_code"]),
            "semantic_fingerprint": definition["semantic_fingerprint"],
            "case_file": definition["case_file"],
            "training_data": definition["training_data"],
            "artifact_path": str(artifact_path),
            "artifact_sha256": sha256(artifact_path),
            "input_snapshot_hashes": {
                name: record["sha256"] for name, record in artifact["input_files"].items()
            },
        })
    if len(parameter_rows) != expected["parameters"]:
        raise ValueError("Expected 2,700 parameter records")

    workbook_path = wrapper.engine.GLOBAL_WORKBOOK
    audit = {
        "schema_version": 1,
        "study_id": wrapper.engine.STUDY_ID,
        "created_at": utc_now(),
        "status": "passed",
        "candidate_count": len(candidate_rows),
        "selected_record_count": len(selections),
        "per_ratio_row_count": len(ratios),
        "parameter_record_count": len(parameter_rows),
        "unique_fit_code_count": len(fit_codes),
        "fit_code_minimum": min(fit_codes),
        "fit_code_maximum": max(fit_codes),
        "maximum_prediction_difference": maximum_difference,
        "minimum_cone_margin_plus_tolerance": minimum_margin_minus_tolerance,
        "maximum_a_plus_b_error": maximum_affine_error,
        "linear_backend_counts": dict(backend_counts),
        "global_append_receipt_count": len(records["receipts"]),
        "global_workbook": str(workbook_path.resolve()),
        "global_workbook_sha256": sha256(workbook_path),
    }
    payload = {
        "schema_version": 1,
        "study_id": wrapper.engine.STUDY_ID,
        "report_status": "Complete SV_exp_polycon_mo study; B01--B09 for four knits",
        "created_at": utc_now(),
        "report_filename": REPORT_FILENAME,
        "expected_full_counts": expected,
        "audit": audit,
        "sheets": {
            "Candidates": candidate_rows,
            "Fit Ratio": ratios,
            "Selected Path": selections,
            "Parameters": parameter_rows,
            "Constraint Audit": constraint_rows,
            "Consensus": consensus_rows(selections),
            "Provenance": provenance_rows,
        },
    }
    return payload, audit


def write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Write indented JSON for human review, for example full_study_audit.json."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Create the audited workbook payload without modifying any Excel file."""
    records = load_campaign()
    payload, audit = build_payload(records)
    output = wrapper.engine.OUTPUT_DIRECTORY
    write_json(output / PAYLOAD_FILENAME, payload)
    write_json(output / AUDIT_FILENAME, audit)
    print(json.dumps({
        "payload": str((output / PAYLOAD_FILENAME).resolve()),
        "audit": str((output / AUDIT_FILENAME).resolve()),
        "report": str((output / REPORT_FILENAME).resolve()),
        "audit_summary": audit,
    }, indent=2))


if __name__ == "__main__":
    main()
