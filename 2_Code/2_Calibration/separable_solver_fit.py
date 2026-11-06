"""Paper-release entry point: edit settings below and run in VSCode."""
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]  # Example: GitHub_package/2_Code.
sys.path.insert(0, str(CODE_ROOT))
from release_bootstrap import prepare
prepare()

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (INPUT_ROOT as RELEASE_INPUT_ROOT,
    GENERATED_ROOT as RELEASE_GENERATED_ROOT)
from Analysis.fitting import separable_solver_fit as workflow

# Case selection; example: three_ratio_raw_c063_sv_exp_polycon_mo_interlock_s06.json.
CASE_FILE = RELEASE_INPUT_ROOT / 'Separable_Solver/Cases/three_ratio_raw_c063_sv_exp_polycon_mo_interlock_s06.json'

# Fresh output files; example: change calibration_refitting to retain another run.
workflow.LEGACY_RESULTS_CSV = RELEASE_GENERATED_ROOT / 'results/calibration_refitting/results_final.csv'
workflow.SEPARABLE_RESULTS_CSV = RELEASE_GENERATED_ROOT / 'results/calibration_refitting/separable_solver_results.csv'
workflow.RUN_SUMMARY_ROOT = RELEASE_GENERATED_ROOT / 'results/calibration_refitting/summaries'
if __name__ == '__main__':
    workflow.main(['--case', str(CASE_FILE)])
