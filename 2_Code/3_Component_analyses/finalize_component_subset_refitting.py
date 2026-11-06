"""Finalize fresh paper-release fits; run after the corresponding campaign."""
from pathlib import Path
import sys
CODE_ROOT = Path(__file__).resolve().parents[1]  # Example: GitHub_package/2_Code.
sys.path.insert(0, str(CODE_ROOT))
from release_bootstrap import prepare
prepare()

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (GENERATED_ROOT as RELEASE_GENERATED_ROOT)
from Analysis.studies.model_selection import finalize_sv_exp_polycon_mo_component_necessity as workflow
# Refit directory; example: the output selected in component_subset_refitting.py.
OUTPUT_DIRECTORY = RELEASE_GENERATED_ROOT / 'results/component_subset_refitting'
# Include the original stability screens; example: False finalizes calibration only.
RUN_STABILITY = True
workflow.study.OUTPUT_ROOT = OUTPUT_DIRECTORY
workflow.study.FULL_CHECKPOINT_DIR = OUTPUT_DIRECTORY / 'checkpoints'
workflow.OUTPUT = OUTPUT_DIRECTORY
workflow.FIGURES = OUTPUT_DIRECTORY / 'figures'
workflow.RUN_STABILITY = RUN_STABILITY
if __name__ == '__main__':
    workflow.main()
