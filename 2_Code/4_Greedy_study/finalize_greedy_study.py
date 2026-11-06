"""Finalize fresh paper-release fits; run after the corresponding campaign."""
from pathlib import Path
import sys
CODE_ROOT = Path(__file__).resolve().parents[1]  # Example: GitHub_package/2_Code.
sys.path.insert(0, str(CODE_ROOT))
from release_bootstrap import prepare
prepare()
from Analysis.studies.ratio_exposure import finalize_adaptive_ratio_study_sv_exp_polycon_mo as workflow
if __name__ == '__main__':
    workflow.main()
