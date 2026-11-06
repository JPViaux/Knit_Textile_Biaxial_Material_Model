"""Prepare cumulative raw-data Batches and direct production case manifests.

Edit ``BATCH_DEFINITIONS`` or ``WRITE_OUTPUTS`` and press Run in VSCode. The
default preview validates every source and reports the files that would be
created. Existing conflicting files are rejected instead of overwritten.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (INPUT_ROOT as RELEASE_INPUT_ROOT,
    EXPERIMENT_ROOT as RELEASE_EXPERIMENT_ROOT,
    GENERATED_ROOT as RELEASE_GENERATED_ROOT)

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from release_paths import json
import os
import pathlib
import re
from typing import Any, Mapping, Sequence

import numpy as np


ANALYSIS_DIR = next(parent for parent in pathlib.Path(__file__).resolve().parents if parent.name == "Analysis")
SUPPORT_CODE_DIR = ANALYSIS_DIR.parent
CODE_ROOT = SUPPORT_CODE_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent
CASE_DIRECTORY = RELEASE_INPUT_ROOT / "Separable_Solver" / "Cases"

# ---------------------------------------------------------------------------
# USER CONFIGURATION - edit these values before pressing Run in VSCode.
# ---------------------------------------------------------------------------

# False previews and verifies; True creates the Batches, catalogs, README
# entries, and cases. Example: use False before every new study extension.
WRITE_OUTPUTS = False

# Each Batch is cumulative. Example: B02 contains every B01 row plus 1:0.
BATCH_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "B01": ("1:1",),
    "B02": ("1:1", "1:0"),
    "B03": ("1:1", "1:0", "0:1"),
    "B04": ("1:1", "1:0", "0:1", "0.5:1"),
    "B05": ("1:1", "1:0", "0:1", "0.5:1", "1:0.5"),
    "B06": ("1:1", "1:0", "0:1", "0.5:1", "1:0.5", "0.25:1"),
    "B07": (
        "1:1",
        "1:0",
        "0:1",
        "0.5:1",
        "1:0.5",
        "0.25:1",
        "1:0.25",
    ),
    "B08": (
        "1:1",
        "1:0",
        "0:1",
        "0.5:1",
        "1:0.5",
        "0.25:1",
        "1:0.25",
        "0.75:1",
    ),
    "B09": (
        "1:1",
        "1:0",
        "0:1",
        "0.5:1",
        "1:0.5",
        "0.25:1",
        "1:0.25",
        "0.75:1",
        "1:0.75",
    ),
}

# ---------------------------------------------------------------------------
# FIXED STUDY INPUTS - normally no changes are needed below this line.
# ---------------------------------------------------------------------------

RATIO_FILENAMES: dict[str, str] = {
    "0:1": "0 to 1.dat",
    "0.25:1": "0p25 to 1.dat",
    "0.5:1": "0p5 to 1.dat",
    "0.75:1": "0p75 to 1.dat",
    "1:0": "1 to 0.dat",
    "1:0.25": "1 to 0p25.dat",
    "1:0.5": "1 to 0p5.dat",
    "1:0.75": "1 to 0p75.dat",
    "1:1": "1 to 1.dat",
}
EXPECTED_RATIO_SET = frozenset(RATIO_FILENAMES)
CATALOG_FILENAME = "progressive_training_batches.json"


@dataclass(frozen=True)
class KnitConfiguration:
    """Describe one raw sample, for example Interlock S06."""

    knit: str
    token: str
    sample_token: str
    compiled_directory: pathlib.Path
    baseline_cases: Mapping[str, pathlib.Path]


def _raw_directory(knit_folder: str, sample_folder: str) -> pathlib.Path:
    """Return one raw Compiled_Data folder, for example Rib S03."""
    return (
        RELEASE_EXPERIMENT_ROOT
        / knit_folder
        / sample_folder
        / "Compiled_Data"
    )


KNIT_CONFIGURATIONS: tuple[KnitConfiguration, ...] = (
    KnitConfiguration(
        "Interlock",
        "interlock",
        "interlock_s06",
        _raw_directory("Interlock", "BRT_15mm_Tri_Int_S06"),
        {
            "SV_exp_polycon_mo": CASE_DIRECTORY / "sv_exp_polycon_mo_interlock_s06_raw_all.json",
        },
    ),
    KnitConfiguration(
        "Rib",
        "rib",
        "rib_s03",
        _raw_directory("Rib", "BRT_15mm_Tri_Rib_S03"),
        {
            "SV_exp_polycon_mo": CASE_DIRECTORY / "sv_exp_polycon_mo_rib_s03_raw_all.json",
        },
    ),
    KnitConfiguration(
        "Purl",
        "purl",
        "purl_s02",
        _raw_directory("Purl", "BRT_15mm_Tri_Purl_S02"),
        {
            "SV_exp_polycon_mo": CASE_DIRECTORY / "sv_exp_polycon_mo_purl_s02_raw_all.json",
        },
    ),
    KnitConfiguration(
        "SingleJersey",
        "singlejersey",
        "singlejersey_s02",
        _raw_directory("SingleJersey", "BRT_15mm_Tri_SJ_S02"),
        {
            "SV_exp_polycon_mo": CASE_DIRECTORY / "sv_exp_polycon_mo_singlejersey_s02_raw_all.json",
        },
    ),
)


def sha256_bytes(data: bytes) -> str:
    """Hash bytes for provenance, for example one generated Batch."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    """Hash one source file without changing it."""
    return sha256_bytes(path.read_bytes())


def normalized_text_block(data: bytes) -> bytes:
    """Ensure a source curve ends with one line break before concatenation."""
    return data if data.endswith((b"\n", b"\r")) else data + b"\n"


def validate_batch_definitions(
    definitions: Mapping[str, Sequence[str]] = BATCH_DEFINITIONS,
) -> None:
    """Require ordered, one-ratio-at-a-time cumulative Batches."""
    expected_ids = [f"B{index:02d}" for index in range(1, len(definitions) + 1)]
    if list(definitions) != expected_ids:
        raise ValueError(f"Batch IDs must be consecutive and ordered: {expected_ids}")
    previous: tuple[str, ...] = ()
    for batch_id, raw_ratios in definitions.items():
        ratios = tuple(raw_ratios)
        unknown = sorted(set(ratios) - EXPECTED_RATIO_SET)
        if unknown:
            raise ValueError(f"{batch_id} contains unknown ratios: {unknown}")
        if len(ratios) != len(set(ratios)):
            raise ValueError(f"{batch_id} contains a duplicate ratio")
        if ratios[:-1] != previous:
            raise ValueError(f"{batch_id} must retain the preceding Batch as a prefix")
        previous = ratios
    if frozenset(previous) != EXPECTED_RATIO_SET:
        raise ValueError("The final Batch must contain all nine ratios exactly once")


def source_curve_paths(config: KnitConfiguration) -> dict[str, pathlib.Path]:
    """Resolve and validate all nine individual raw ratio curves."""
    paths = {
        ratio: config.compiled_directory / filename
        for ratio, filename in RATIO_FILENAMES.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing individual ratio curves: {missing}")
    return paths


def build_batch_bytes(
    ratios: Sequence[str],
    curve_paths: Mapping[str, pathlib.Path],
) -> bytes:
    """Concatenate complete source curves in the declared exposure order."""
    return b"".join(normalized_text_block(curve_paths[ratio].read_bytes()) for ratio in ratios)


def _numeric_rows(path: pathlib.Path) -> np.ndarray:
    """Load a numeric curve as a 2-D array, for example 61 by 12."""
    return np.atleast_2d(np.loadtxt(path, dtype=float))


def verify_numeric_batch(
    output_path: pathlib.Path,
    ratios: Sequence[str],
    curve_paths: Mapping[str, pathlib.Path],
) -> int:
    """Prove the output is the exact numeric concatenation of its ratio curves."""
    expected = np.vstack([_numeric_rows(curve_paths[ratio]) for ratio in ratios])
    actual = _numeric_rows(output_path)
    if not np.array_equal(actual, expected):
        raise ValueError(f"Generated Batch does not match its sources: {output_path}")
    return int(actual.shape[0])


def write_or_verify(path: pathlib.Path, data: bytes, write: bool) -> str:
    """Create an absent file or verify an existing file byte-for-byte."""
    digest = sha256_bytes(data)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite conflicting file: {path}")
        return digest
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return digest


def batch_filename(config: KnitConfiguration, batch_id: str) -> str:
    """Return a non-conflicting study filename, for example rib progressive B01."""
    number = int(batch_id[1:])
    return f"{config.token}_progressive_training_batch_{number:02d}.dat"


def _normalized_stem(text: str) -> str:
    """Normalize a README entry name the same way as the error calculator."""
    return re.sub(r"_+", "_", pathlib.Path(text.strip()).stem.lower().replace(" ", "_").replace(".", "p"))


def update_readme(
    readme_path: pathlib.Path,
    entries: Mapping[str, Sequence[str]],
    write: bool,
) -> None:
    """Append missing provenance entries and reject contradictory definitions."""
    original = readme_path.read_text(encoding="utf-8-sig") if readme_path.is_file() else ""
    lines = original.splitlines()
    indexed = {
        _normalized_stem(line.partition(":")[0]): line
        for line in lines
        if line.partition(":")[1]
    }
    additions: list[str] = []
    for filename, ratios in entries.items():
        stem = pathlib.Path(filename).stem
        expected = f"{stem}: {', '.join(ratios)}"
        existing = indexed.get(_normalized_stem(stem))
        if existing is not None and existing.strip() != expected:
            raise ValueError(f"Conflicting README entry for {filename}: {existing!r}")
        if existing is None:
            additions.append(expected)
    if write and additions:
        prefix = original
        if prefix and not prefix.endswith(("\n", "\r")):
            prefix += "\n"
        readme_path.write_text(prefix + "\n".join(additions) + "\n", encoding="utf-8")


def project_relative(path: pathlib.Path) -> str:
    """Store a portable path relative to the project root."""
    return os.path.relpath(path.resolve(), PROJECT_ROOT.resolve()).replace("\\", "/")


def progressive_case_name(
    batch_id: str,
    model_id: str,
    config: KnitConfiguration,
) -> str:
    """Return one permanent case filename and matching case ID."""
    model_token = "sv_exp_polycon_mo"
    return f"progressive_{batch_id.lower()}_{model_token}_{config.sample_token}_raw"


def build_case_payload(
    baseline: Mapping[str, Any],
    batch_id: str,
    model_id: str,
    config: KnitConfiguration,
    training_path: pathlib.Path,
) -> dict[str, Any]:
    """Derive one training-only production case from a validated baseline."""
    payload = copy.deepcopy(dict(baseline))
    case_id = progressive_case_name(batch_id, model_id, config)
    payload["case_id"] = case_id
    payload["run_label"] = (
        f"{config.knit} raw progressive {batch_id} {model_id} training optimum"
    )
    payload.setdefault("workflow", "production")
    payload["files"]["experimental_data"] = project_relative(training_path)
    payload["outputs"]["result_roles"] = ["training_optimum"]
    payload["outputs"]["snapshot_experimental_data"] = True
    return payload


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize stored provenance deterministically."""
    return (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8")


def prepare_knit(config: KnitConfiguration, write: bool) -> dict[str, Any]:
    """Prepare or verify all Batches, catalog entries, and cases for one knit."""
    curves = source_curve_paths(config)
    readme_entries: dict[str, Sequence[str]] = {}
    batch_records: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    previous_bytes = b""

    baselines = {
        model_id: json.loads(path.read_text(encoding="utf-8"))
        for model_id, path in config.baseline_cases.items()
    }
    for batch_id, ratios in BATCH_DEFINITIONS.items():
        output_path = batch_output_directory(config) / batch_filename(config, batch_id)
        batch_bytes = build_batch_bytes(ratios, curves)
        if previous_bytes and not batch_bytes.startswith(previous_bytes):
            raise ValueError(f"{batch_id} does not retain the previous Batch byte prefix")
        digest = write_or_verify(output_path, batch_bytes, write)
        if output_path.is_file():
            row_count = verify_numeric_batch(output_path, ratios, curves)
        else:
            row_count = int(sum(_numeric_rows(curves[ratio]).shape[0] for ratio in ratios))
        readme_entries[output_path.name] = ratios
        batch_records.append(
            {
                "batch_id": batch_id,
                "ratios": list(ratios),
                "path": project_relative(output_path),
                "row_count": row_count,
                "sha256": digest,
                "sources": [
                    {
                        "ratio": ratio,
                        "path": project_relative(curves[ratio]),
                        "row_count": int(_numeric_rows(curves[ratio]).shape[0]),
                        "sha256": sha256_file(curves[ratio]),
                    }
                    for ratio in ratios
                ],
            }
        )
        for model_id, baseline in baselines.items():
            case_id = progressive_case_name(batch_id, model_id, config)
            case_path = batch_output_directory(config) / "cases" / f"{case_id}.json"
            payload = build_case_payload(
                baseline,
                batch_id,
                model_id,
                config,
                output_path,
            )
            case_digest = write_or_verify(case_path, json_bytes(payload), write)
            case_records.append(
                {
                    "batch_id": batch_id,
                    "model_id": model_id,
                    "case_id": case_id,
                    "case_path": project_relative(case_path),
                    "sha256": case_digest,
                }
            )
        previous_bytes = batch_bytes

    update_readme(batch_output_directory(config) / "Readme.txt", readme_entries, write)
    catalog_path = batch_output_directory(config) / CATALOG_FILENAME
    existing_created_at: str | None = None
    if catalog_path.is_file():
        existing = json.loads(catalog_path.read_text(encoding="utf-8"))
        existing_created_at = str(existing.get("created_at", "")) or None
    catalog = {
        "schema_version": 1,
        "study_id": "progressive_ratio_exposure_v1",
        "knit": config.knit,
        "created_at": existing_created_at or datetime.now(timezone.utc).isoformat(),
        "batch_order": list(BATCH_DEFINITIONS),
        "batches": batch_records,
        "cases": case_records,
    }
    write_or_verify(catalog_path, json_bytes(catalog), write)
    return {
        "knit": config.knit,
        "compiled_directory": str(config.compiled_directory.resolve()),
        "catalog_path": str(catalog_path.resolve()),
        "batches": batch_records,
        "cases": case_records,
    }


def prepare_study(write: bool = WRITE_OUTPUTS) -> dict[str, Any]:
    """Prepare or preview the complete four-knit, nine-Batch study."""
    validate_batch_definitions()
    results = [prepare_knit(config, write) for config in KNIT_CONFIGURATIONS]
    case_count = sum(len(result["cases"]) for result in results)
    batch_count = sum(len(result["batches"]) for result in results)
    if case_count != 72 or batch_count != 36:
        raise RuntimeError(f"Expected 72 cases and 36 Batches; got {case_count} and {batch_count}")
    return {
        "write_outputs": write,
        "batch_files": batch_count,
        "case_files": case_count,
        "knits": results,
    }


def print_summary(summary: Mapping[str, Any]) -> None:
    """Print a concise VSCode preview or completion report."""
    action = "Prepared" if summary["write_outputs"] else "Previewed"
    print(f"{action} {summary['batch_files']} Batch files and {summary['case_files']} cases")
    for knit in summary["knits"]:
        first = knit["batches"][0]
        last = knit["batches"][-1]
        print(
            f"- {knit['knit']}: {first['row_count']} rows in B01, "
            f"{last['row_count']} rows in B09"
        )
        print(f"  Catalog: {knit['catalog_path']}")


def main(argv: Sequence[str] | None = None) -> None:
    """Preview by default; pass --write to create the permanent study inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Create verified outputs")
    args = parser.parse_args(argv)
    summary = prepare_study(write=WRITE_OUTPUTS or args.write)
    print_summary(summary)



def batch_output_directory(config: KnitConfiguration) -> pathlib.Path:
    """Separate new batch products from measurements; example: generated/results/batches/rib_s03."""
    return RELEASE_GENERATED_ROOT / "results/batches" / config.sample_token

if __name__ == "__main__":
    main()
