"""Paper-release entry point: edit settings below and run in VSCode."""
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]  # Example: GitHub_package/2_Code.
sys.path.insert(0, str(CODE_ROOT))
from release_bootstrap import prepare
prepare()

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (GENERATED_ROOT as RELEASE_GENERATED_ROOT)
from Analysis.studies.model_selection import sv_exp_polycon_mo_tolerance_sensitivity as workflow

# Output location; example: a different filename keeps an earlier analysis.
workflow.OUTPUT_CSV_PATH = RELEASE_GENERATED_ROOT / 'results/component_tolerance_sensitivity.csv'
if __name__ == '__main__':
    workflow.main()
