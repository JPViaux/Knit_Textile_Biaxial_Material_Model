"""Evaluate the four all-nine SV_exp_polycon_mo fits without editing Excel.

Edit ``FIT_CODES`` below and press Run in VSCode. The established global error
calculator evaluates all nine ratios and stages workbook-ready rows as JSON.
"""

from __future__ import annotations

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (GENERATED_ROOT as RELEASE_GENERATED_ROOT)

from datetime import datetime, timezone
from release_paths import json
import pathlib
import sys
import tempfile


ANALYSIS_DIR = pathlib.Path(__file__).resolve().parents[1]
SUPPORT_CODE_DIR = ANALYSIS_DIR.parent
sys.path.insert(0, str(SUPPORT_CODE_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from Analysis.diagnostics import global_error_calculator as global_error  # noqa: E402


# USER SETTINGS - example: replace 1962 after generating a newer Interlock fit.
FIT_CODES = {
    "interlock_sv_exp_polycon_mo_raw_all_nine": 1962,
    "rib_sv_exp_polycon_mo_raw_all_nine": 1961,
    "purl_sv_exp_polycon_mo_raw_all_nine": 1963,
    "singlejersey_sv_exp_polycon_mo_raw_all_nine": 1964,
}
EVALUATION_EXP_DATA_TYPE = "raw"  # Examples: "raw".
OUTPUT_JSON = (
    RELEASE_GENERATED_ROOT
    / "results/sv_exp_polycon_mo_all_nine"
    / "global_error_rows.json"
)


def main() -> None:
    """Evaluate all nine ratios and save workbook-ready rows with one timestamp."""
    run_timestamp = datetime.now(timezone.utc).isoformat()
    per_ratio_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="sv_exp_polycon_mo_global_error_") as temp:
        temp_path = pathlib.Path(temp)
        for job_name, fit_code in FIT_CODES.items():
            payload = global_error.evaluate_fit_in_worker(
                job_name,
                fit_code,
                temp_path,
                evaluation_exp_data_type=EVALUATION_EXP_DATA_TYPE,
            )
            per_ratio_rows.extend(
                global_error.add_run_timestamp(payload["per_ratio_rows"], run_timestamp)
            )
            summary = global_error.add_run_timestamp(
                payload["summary_row"], run_timestamp
            )
            summary_rows.append(summary)
            metadata_rows.append(
                global_error.add_run_timestamp(payload["metadata_row"], run_timestamp)
            )
            print(
                f"fit {fit_code}: pooled_R2={summary['pooled_R2']:.12g}, "
                f"course={summary['pooled_R2_course']:.12g}, "
                f"wales={summary['pooled_R2_wales']:.12g}"
            )

    output = {
        "schema_version": 1,
        "run_timestamp": run_timestamp,
        "evaluation_exp_data_type": EVALUATION_EXP_DATA_TYPE,
        "fit_codes": FIT_CODES,
        "per_ratio_rows": per_ratio_rows,
        "summary_rows": summary_rows,
        "metadata_rows": metadata_rows,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(global_error.json_ready(output), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved global-error rows: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
