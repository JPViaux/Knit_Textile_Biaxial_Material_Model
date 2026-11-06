"""Paper-release entry point: edit settings below and run in VSCode."""
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]  # Example: GitHub_package/2_Code.
sys.path.insert(0, str(CODE_ROOT))
from release_bootstrap import prepare
prepare()
from Analysis.studies.ratio_exposure import adaptive_ratio_study_sv_exp_polycon_mo as workflow

# Lifecycle action; example: 'prepare' checks inputs, 'all' runs the campaign.
ACTION = 'prepare'
# All four knits and nine steps; example: False uses the source pilot settings.
FULL_CAMPAIGN = False
workflow.ACTION = ACTION
workflow.FULL_CAMPAIGN = FULL_CAMPAIGN
if __name__ == '__main__':
    workflow.main(['--action', ACTION])
