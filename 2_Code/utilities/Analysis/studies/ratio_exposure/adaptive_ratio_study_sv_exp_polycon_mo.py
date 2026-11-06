"""Run the greedy ratio-exposure study for SV_exp_polycon_mo.

The checked-in VSCode settings intentionally run only the Rib B01 pilot. Each
candidate uses the full reviewed material-model search, but the pilot limits
study scope to one knit and one batch and never appends the shared workbook.
After review, set ``PILOT_MODE = False`` and select all four knits for the
production B01--B09 campaign.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (GENERATED_ROOT as RELEASE_GENERATED_ROOT)

import argparse
import copy
from dataclasses import replace
from release_paths import json
import pathlib
import sys
from typing import Any, Mapping, Sequence


ANALYSIS_DIRECTORY = next(
    parent for parent in pathlib.Path(__file__).resolve().parents
    if parent.name == "Analysis"
)
SUPPORT_DIRECTORY = ANALYSIS_DIRECTORY.parent
for import_directory in (SUPPORT_DIRECTORY, ANALYSIS_DIRECTORY):
    sys.path.insert(0, str(import_directory))

from Analysis.studies.ratio_exposure import adaptive_ratio_study as engine  # noqa: E402
from Analysis.studies.ratio_exposure import prepare_progressive_ratio_study as progressive  # noqa: E402


# ---------------------------------------------------------------------------
# USER SETTINGS - examples: ACTION = "preview" or KNITS_TO_RUN = ("Rib",).
# ---------------------------------------------------------------------------

ACTION = "all"
TARGET_BATCH = "B01"
KNITS_TO_RUN = ("Rib",)
DEFAULT_PHYSICAL_CORES = 5
PILOT_MODE = True
APPEND_GLOBAL_WORKBOOK = False
# Full production can be selected here in VSCode or with ``--full-campaign``.
# Example: FULL_CAMPAIGN = True runs all four knits through B01--B09.
FULL_CAMPAIGN = False


# ---------------------------------------------------------------------------
# FIXED MODEL/STUDY SETTINGS - normally no changes are required below.
# ---------------------------------------------------------------------------

MODEL_ID = "SV_exp_polycon_mo"
MODEL_TOKEN = "sv_exp_polycon_mo"
ALL_KNITS = ("Interlock", "Rib", "Purl", "SingleJersey")
ACTIVE_KNITS_TO_RUN = KNITS_TO_RUN
BASELINE_CASE_NAMES = {
    "Interlock": "sv_exp_polycon_mo_interlock_s06_raw_all.json",
    "Rib": "sv_exp_polycon_mo_rib_s03_raw_all.json",
    "Purl": "sv_exp_polycon_mo_purl_s02_raw_all.json",
    "SingleJersey": "sv_exp_polycon_mo_singlejersey_s02_raw_all.json",
}


def model_token(model_id: str) -> str:
    """Return the isolated filename token for the configured material model."""
    if model_id != MODEL_ID:
        raise ValueError(f"Unsupported model for this study: {model_id}")
    return MODEL_TOKEN


def selected_knit_configurations() -> tuple[progressive.KnitConfiguration, ...]:
    """Attach the polycon baseline case and apply the editable knit filter."""
    unknown = set(ACTIVE_KNITS_TO_RUN) - set(ALL_KNITS)
    if unknown:
        raise ValueError(f"Unknown KNITS_TO_RUN entries: {sorted(unknown)}")
    selected: list[progressive.KnitConfiguration] = []
    for config in progressive.KNIT_CONFIGURATIONS:
        if config.knit not in ACTIVE_KNITS_TO_RUN:
            continue
        baseline_cases = dict(config.baseline_cases)
        baseline_cases[MODEL_ID] = (
            progressive.CASE_DIRECTORY / BASELINE_CASE_NAMES[config.knit]
        )
        selected.append(replace(config, baseline_cases=baseline_cases))
    if not selected:
        raise ValueError("KNITS_TO_RUN selected no knit configurations")
    return tuple(selected)


def build_case_payload(
    baseline: Mapping[str, Any],
    step: str,
    added_ratio: str,
    model_id: str,
    config: progressive.KnitConfiguration,
    training_path: pathlib.Path,
) -> dict[str, Any]:
    """Derive one training-only polycon candidate from its reviewed baseline."""
    if model_id != MODEL_ID:
        raise ValueError(f"Unexpected adaptive-study model: {model_id}")
    payload = copy.deepcopy(dict(baseline))
    identifier = engine.candidate_id(step, added_ratio, model_id, config)
    payload["case_id"] = identifier
    payload["run_label"] = (
        f"{config.knit} {MODEL_ID} greedy {step}; add {added_ratio}"
    )
    payload["workflow"] = "production"
    payload["files"]["experimental_data"] = progressive.project_relative(
        training_path
    )
    payload["outputs"]["result_roles"] = ["training_optimum"]
    payload["outputs"]["snapshot_experimental_data"] = True
    payload["coupling"]["selection_basis"] = (
        f"{MODEL_ID} greedy {step}; candidate addition {added_ratio}; "
        "rank by fixed all-nine pooled R2"
    )
    return payload


def expected_step_candidate_count(step_number: int) -> int:
    """Return selected-knit candidates, for example Rib B01 has nine."""
    return len(ACTIVE_KNITS_TO_RUN) * (10 - step_number)


def expected_total_candidate_count() -> int:
    """Return the full selected-knit B01--B09 candidate count."""
    return len(ACTIVE_KNITS_TO_RUN) * sum(10 - step for step in range(1, 10))


def load_prefixes(step_number: int) -> dict[str, tuple[str, ...]]:
    """Load each selected knit's preceding frozen greedy prefix."""
    configurations = selected_knit_configurations()
    prefixes = {
        engine.pair_id(MODEL_ID, config): () for config in configurations
    }
    if step_number == 1:
        return prefixes
    previous_path = engine.SELECTION_DIRECTORY / engine.batch_id(step_number - 1)
    previous_path = previous_path.with_suffix(".json")
    if not previous_path.is_file():
        raise FileNotFoundError(
            f"Previous greedy selection is not frozen: {previous_path}"
        )
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    selected = previous.get("selected_by_pair", [])
    if len(selected) != len(prefixes):
        raise ValueError(
            f"{previous_path.name} must contain {len(prefixes)} selected paths"
        )
    for row in selected:
        key = str(row["pair_id"])
        ratios = tuple(row["training_ratios"])
        if key not in prefixes or len(ratios) != step_number - 1:
            raise ValueError(f"Invalid preceding greedy prefix for {key}")
        if len(set(ratios)) != len(ratios) or not set(ratios) <= engine.RATIO_SET:
            raise ValueError(f"Invalid ratio sequence in {previous_path}: {ratios}")
        prefixes[key] = ratios
    return prefixes


def configure_engine(full_campaign: bool | None = None) -> None:
    """Isolate engine state and install the model-specific routing hooks."""
    global ACTIVE_KNITS_TO_RUN
    run_full_campaign = FULL_CAMPAIGN if full_campaign is None else full_campaign
    pilot_mode = PILOT_MODE and not run_full_campaign
    ACTIVE_KNITS_TO_RUN = ALL_KNITS if run_full_campaign else KNITS_TO_RUN
    suffix = "_pilot" if pilot_mode else ""
    study_id = f"adaptive_ratio_greedy_{MODEL_TOKEN}_v1{suffix}_release"
    output_root = (
        RELEASE_GENERATED_ROOT
        / "results/greedy_refitting"
    )
    if pilot_mode:
        output_root = output_root / "pilot"

    engine.STUDY_ID = study_id
    engine.CASE_ID_PREFIX = study_id
    engine.MODEL_IDS = (MODEL_ID,)
    engine.OUTPUT_DIRECTORY = output_root
    engine.DEFINITION_DIRECTORY = output_root / "definitions"
    engine.CASE_LIST_DIRECTORY = output_root / "case_lists"
    engine.STATE_DIRECTORY = output_root / "states"
    engine.FIT_MAP_DIRECTORY = output_root / "fit_maps"
    engine.EVALUATION_DIRECTORY = output_root / "evaluations"
    engine.SELECTION_DIRECTORY = output_root / "selections"
    engine.REUSE_DIRECTORY = output_root / "reuse_certificates"
    engine.ACTION = ACTION
    engine.TARGET_BATCH = TARGET_BATCH
    engine.DEFAULT_PHYSICAL_CORES = DEFAULT_PHYSICAL_CORES
    engine.APPEND_GLOBAL_WORKBOOK = (
        False  # Published workbooks remain immutable; new evaluations stay in generated.
    )

    progressive.KNIT_CONFIGURATIONS = selected_knit_configurations()
    engine.model_token = model_token
    engine.build_case_payload = build_case_payload
    engine.expected_step_candidate_count = expected_step_candidate_count
    engine.expected_total_candidate_count = expected_total_candidate_count
    engine.load_prefixes = load_prefixes


def main(argv: Sequence[str] | None = None) -> None:
    """Configure the generic engine, then run the selected safe lifecycle action."""
    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument(
        "--full-campaign",
        action="store_true",
        help="Run all four knits sequentially from B01 through B09.",
    )
    wrapper_args, engine_args = wrapper_parser.parse_known_args(argv)
    run_full_campaign = FULL_CAMPAIGN or wrapper_args.full_campaign
    configure_engine(full_campaign=run_full_campaign)
    if run_full_campaign:
        if "--action" not in engine_args:
            engine_args.extend(("--action", "all"))
        if "--all-steps" not in engine_args and "--batch" not in engine_args:
            engine_args.append("--all-steps")
    engine.main(engine_args)


if __name__ == "__main__":
    main()
