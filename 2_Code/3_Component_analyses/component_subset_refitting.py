"""Paper-release entry point: edit settings below and run in VSCode."""
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]  # Example: GitHub_package/2_Code.
sys.path.insert(0, str(CODE_ROOT))
from release_bootstrap import prepare
prepare()

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (GENERATED_ROOT as RELEASE_GENERATED_ROOT)
from Analysis.studies.model_selection import sv_exp_polycon_mo_component_necessity_study as workflow

# Run scale; example: 'pilot' is a coarse check, 'full' refits all 156 subsets.
RUN_MODE = 'pilot'
workflow.RUN_MODE = RUN_MODE
# Fresh refits; example: change the suffix to retain multiple campaigns.
workflow.OUTPUT_ROOT = RELEASE_GENERATED_ROOT / 'results/component_subset_refitting'
workflow.PILOT_OUTPUT = workflow.OUTPUT_ROOT / 'pilot'
workflow.CHECKPOINT_DIR = workflow.PILOT_OUTPUT / 'checkpoints'
# Admitted coarse exponents; example: (3, 6, 10, 15) avoids the old pilot-only 2.
workflow.PILOT_ZETA_VALUES = (3, 6, 10, 15)
workflow.FULL_CHECKPOINT_DIR = workflow.OUTPUT_ROOT / 'checkpoints'
if __name__ == '__main__':
    workflow.main()
