"""Resolve experimental datasets and their physical-axis metadata.

The editable catalog in this module is the single code-facing selector for
plotting and separable-solver case generation.  The CSV remains the scientific
metadata authority; the Word summary is provenance and is never parsed at run
time.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (INPUT_ROOT as RELEASE_INPUT_ROOT,
    EXPERIMENT_ROOT as RELEASE_EXPERIMENT_ROOT)

import csv
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


SUPPORT_DIR = pathlib.Path(__file__).resolve().parents[2]
CODE_ROOT = SUPPORT_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent
EXPERIMENTAL_DATA_ROOT = RELEASE_EXPERIMENT_ROOT
METADATA_PATH = RELEASE_INPUT_ROOT / "Experiment_Data" / "biaxial_test_metadata.csv"

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

RAW_RATIO_FILENAMES = {
    "0 to 1": "0 to 1.dat",
    "0.25 to 1": "0p25 to 1.dat",
    "0.5 to 1": "0p5 to 1.dat",
    "0.75 to 1": "0p75 to 1.dat",
    "1 to 0": "1 to 0.dat",
    "1 to 0.25": "1 to 0p25.dat",
    "1 to 0.5": "1 to 0p5.dat",
    "1 to 0.75": "1 to 0p75.dat",
    "1 to 1": "1 to 1.dat",
}



def ratio_to_step_label(ratio: str) -> str:
    """Convert ``0.25 to 1`` to the metadata label ``ratio_025_1``."""
    normalized = " ".join(str(ratio).strip().replace(":", " to ").split())
    try:
        left, right = normalized.split(" to ")
    except ValueError as error:
        raise ValueError(f"Invalid ratio label {ratio!r}") from error

    def token(value: str) -> str:
        return value.replace(".", "")

    return f"ratio_{token(left)}_{token(right)}"


def normalize_direction(value: str) -> str:
    """Normalize common direction spellings to ``course`` or ``wales``."""
    normalized = str(value).strip().lower()
    aliases = {
        "course": "course",
        "courses": "course",
        "wale": "wales",
        "wales": "wales",
    }
    return aliases.get(normalized, normalized)


@dataclass(frozen=True)
class ExperimentRecord:
    """One experimental step and its protocol/quality metadata."""

    source_record_id: str
    source_sample_folder: str
    canonical_sample_folder: str
    step_label: str
    test_order: int
    axis_directions: tuple[str, str]
    preload_status: str
    dic_status: str
    quality_flag: str
    analysis_recommendation: str
    notes: str
    metadata_source: str


@dataclass(frozen=True)
class ExperimentalDataset:
    """One coherent filesystem dataset selectable by a stable catalog key."""

    key: str
    yarn_type: str
    knit: str
    knit_token: str
    sample_folder: str
    data_type: str
    combined_filename: str
    # Optional exact source rows for datasets assembled from multiple sessions.
    # Example: Purl uses S02 for most ratios and the preloaded S01 0:1 record.
    metadata_record_by_ratio: Mapping[str, str] = field(default_factory=dict)

    @property
    def data_directory(self) -> pathlib.Path:
        """Return the shipped specimen's Compiled_Data directory."""
        if self.data_type != "raw":
            raise ValueError(f"Unsupported experimental data type {self.data_type!r}; use raw")
        return EXPERIMENTAL_DATA_ROOT / self.knit / self.sample_folder / "Compiled_Data"

    @property
    def combined_data_path(self) -> pathlib.Path:
        """Return the solver-ready combined experimental file."""
        return self.data_directory / self.combined_filename

    def ratio_paths(self) -> dict[str, pathlib.Path]:
        """Return the nine authoritative curves; example: 0 to 1.dat."""
        return {ratio: self.data_directory / RAW_RATIO_FILENAMES[ratio] for ratio in ALL_RATIOS}


def _purl_record(source_sample: str, ratio: str) -> str:
    """Return the stable Purl source-record identifier used in the CSV."""
    return f"{source_sample}::{ratio_to_step_label(ratio)}"


# ---------------------------------------------------------------------------
# EDITABLE DATASET CATALOG
# Add one entry per selectable sample/data product. Examples below cover every
# four shipped Tripoly specimens; only raw compiled data are selectable.
# ---------------------------------------------------------------------------
EXPERIMENTAL_DATASETS: dict[str, ExperimentalDataset] = {
    "tripoly_interlock_s06_raw": ExperimentalDataset(
        "tripoly_interlock_s06_raw", "Tripoly", "Interlock", "interlock",
        "BRT_15mm_Tri_Int_S06", "raw", "biax_exp_data_interlock.dat",
    ),
    "tripoly_rib_s03_raw": ExperimentalDataset(
        "tripoly_rib_s03_raw", "Tripoly", "Rib", "rib",
        "BRT_15mm_Tri_Rib_S03", "raw", "biax_exp_data_rib.dat",
    ),
    "tripoly_purl_s02_raw": ExperimentalDataset(
        "tripoly_purl_s02_raw", "Tripoly", "Purl", "purl",
        "BRT_15mm_Tri_Purl_S02", "raw", "biax_exp_data_purl.dat",
        metadata_record_by_ratio={
            **{
                ratio: _purl_record("BRT_15mm_Tri_Purl_S02", ratio)
                for ratio in ALL_RATIOS
                if ratio != "0 to 1"
            },
            # The current folder README states that only 0:1 has pretensioning.
            "0 to 1": _purl_record("BRT_15mm_Tri_Purl_S01", "0 to 1"),
        },
    ),
    "tripoly_singlejersey_s02_raw": ExperimentalDataset(
        "tripoly_singlejersey_s02_raw", "Tripoly", "SingleJersey",
        "singlejersey", "BRT_15mm_Tri_SJ_S02", "raw",
        "biax_exp_data_singlejersey.dat",
    ),

}


def load_metadata(path: pathlib.Path = METADATA_PATH) -> tuple[ExperimentRecord, ...]:
    """Load and validate every experimental metadata row."""
    if not path.is_file():
        raise FileNotFoundError(f"Experimental metadata was not found: {path}")
    records: list[ExperimentRecord] = []
    seen_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            # Legacy CSV identifiers are derived without relabeling specimens.
            # Example: BRT_15mm_Tri_Purl_S02::ratio_1_1.
            source_sample = str(row.get("source_sample_folder") or row.get("sample_id", "")).strip()
            record_id = str(row.get("source_record_id") or f"{source_sample}::{row.get('step_label', '')}").strip()
            if not row.get("metadata_source"):
                row["metadata_source"] = "Current metadata CSV; source identifiers derived from sample_id and step_label"

            if not record_id or not source_sample:
                raise ValueError(
                    f"Metadata row {row_number} lacks source provenance fields"
                )
            if record_id in seen_ids:
                raise ValueError(f"Duplicate source_record_id {record_id!r}")
            seen_ids.add(record_id)
            directions = (
                normalize_direction(row.get("axis1_direction", "")),
                normalize_direction(row.get("axis2_direction", "")),
            )
            if set(directions) != {"course", "wales"}:
                raise ValueError(
                    f"Metadata row {row_number} must define one course and one wales axis"
                )
            records.append(
                ExperimentRecord(
                    source_record_id=record_id,
                    source_sample_folder=source_sample,
                    canonical_sample_folder=str(
                        row.get("canonical_sample_folder", "")
                    ).strip(),
                    step_label=str(row.get("step_label", "")).strip(),
                    test_order=int(str(row.get("test_order", "0")).strip() or 0),
                    axis_directions=directions,
                    preload_status=str(row.get("preload_status", "")).strip(),
                    dic_status=str(row.get("dic_status", "")).strip(),
                    quality_flag=str(row.get("quality_flag", "")).strip(),
                    analysis_recommendation=str(
                        row.get("analysis_recommendation", "")
                    ).strip(),
                    notes=str(row.get("notes", "")).strip(),
                    metadata_source=str(row.get("metadata_source", "")).strip(),
                )
            )
    return tuple(records)


def validate_dataset(
    dataset: ExperimentalDataset,
    *,
    require_files: bool = True,
) -> ExperimentalDataset:
    """Validate catalog identity, metadata, and expected files."""
    if dataset.key not in EXPERIMENTAL_DATASETS:
        raise KeyError(f"Dataset key {dataset.key!r} is not registered")
    if dataset.data_type not in {"raw"}:
        raise ValueError(f"Dataset {dataset.key} has invalid data_type")
    records = metadata_records_for_dataset(dataset)
    for ratio in ALL_RATIOS:
        record_for_ratio(dataset, ratio, records)
    if require_files:
        missing = [
            path
            for path in (*dataset.ratio_paths().values(), dataset.combined_data_path)
            if not path.is_file()
        ]
        if missing:
            formatted = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"Dataset {dataset.key} is missing files:\n{formatted}")
    return dataset


def get_dataset(key: str, *, require_files: bool = True) -> ExperimentalDataset:
    """Return one validated catalog entry."""
    try:
        dataset = EXPERIMENTAL_DATASETS[str(key).strip()]
    except KeyError as error:
        options = ", ".join(EXPERIMENTAL_DATASETS)
        raise KeyError(f"Unknown experimental dataset {key!r}; choose from: {options}") from error
    return validate_dataset(dataset, require_files=require_files)


def find_dataset(
    yarn_type: str,
    knit: str,
    sample_folder: str,
    data_type: str,
    *,
    require_files: bool = True,
) -> ExperimentalDataset:
    """Resolve saved-fit metadata to exactly one catalog entry."""
    matches = [
        dataset
        for dataset in EXPERIMENTAL_DATASETS.values()
        if (
            dataset.yarn_type == yarn_type
            and dataset.knit == knit
            and dataset.sample_folder == sample_folder
            and dataset.data_type == data_type
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one catalog dataset for "
            f"{yarn_type}/{knit}/{sample_folder}/{data_type}; found {len(matches)}"
        )
    return validate_dataset(matches[0], require_files=require_files)


def metadata_records_for_dataset(
    dataset: ExperimentalDataset,
) -> tuple[ExperimentRecord, ...]:
    """Return all CSV rows associated with a canonical sample folder."""
    records = tuple(
        record
        for record in load_metadata()
        if record.canonical_sample_folder == dataset.sample_folder
        or record.source_record_id in dataset.metadata_record_by_ratio.values()
    )
    if not records:
        raise ValueError(f"No metadata rows found for {dataset.sample_folder}")
    return records


def record_for_ratio(
    dataset: ExperimentalDataset,
    ratio: str,
    records: Sequence[ExperimentRecord] | None = None,
) -> ExperimentRecord:
    """Resolve the exact metadata record used by one ratio file."""
    if ratio not in ALL_RATIOS:
        raise ValueError(f"Unsupported ratio {ratio!r}")
    available = tuple(records) if records is not None else metadata_records_for_dataset(dataset)
    exact_id = dataset.metadata_record_by_ratio.get(ratio)
    if exact_id:
        matches = [record for record in available if record.source_record_id == exact_id]
    else:
        step_label = ratio_to_step_label(ratio)
        matches = [record for record in available if record.step_label == step_label]
    if len(matches) != 1:
        raise ValueError(
            f"Dataset {dataset.key} ratio {ratio} resolves to {len(matches)} metadata rows"
        )
    return matches[0]


def axis_directions(dataset: ExperimentalDataset, ratio: str) -> tuple[str, str]:
    """Return physical directions for array axes 0 and 1."""
    directions = record_for_ratio(dataset, ratio).axis_directions
    if set(directions) != {"course", "wales"}:
        raise ValueError(f"Invalid axis directions for {dataset.key}/{ratio}: {directions}")
    return directions


def physical_ratio_label(
    ratio: str,
    directions: tuple[str, str],
) -> str:
    """Return the loading ratio in the display convention ``Course:Wales``."""
    left, right = " ".join(ratio.replace(":", " to ").split()).split(" to ")
    by_direction = {directions[0]: left, directions[1]: right}
    return f"{by_direction['course']}:{by_direction['wales']}"


def quality_records(dataset: ExperimentalDataset) -> tuple[ExperimentRecord, ...]:
    """Return the exact non-conditioning records used by the nine ratio files."""
    records = metadata_records_for_dataset(dataset)
    return tuple(record_for_ratio(dataset, ratio, records) for ratio in ALL_RATIOS)


def quality_warnings(dataset: ExperimentalDataset) -> tuple[ExperimentRecord, ...]:
    """Return selected records marked caution or exclude."""
    return tuple(
        record
        for record in quality_records(dataset)
        if record.analysis_recommendation.lower() in {"caution", "exclude"}
    )


def confirm_dataset_override(
    original: ExperimentalDataset,
    selected: ExperimentalDataset,
    *,
    input_func: Callable[[str], str] = input,
) -> None:
    """Require an explicit terminal confirmation for a changed dataset."""
    if original.key == selected.key:
        return
    prompt = (
        "The explicit experimental dataset differs from the saved fit.\n"
        f"Fit dataset:      {original.key} ({original.sample_folder}, {original.data_type})\n"
        f"Selected dataset: {selected.key} ({selected.sample_folder}, {selected.data_type})\n"
        "Type yes to continue: "
    )
    try:
        response = input_func(prompt)
    except EOFError as error:
        raise RuntimeError("Dataset override was not confirmed") from error
    if response.strip().lower() != "yes":
        raise RuntimeError("Dataset override was not confirmed")
