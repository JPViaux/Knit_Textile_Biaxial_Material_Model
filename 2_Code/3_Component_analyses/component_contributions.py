"""Paper-release entry point: edit settings below and run in VSCode."""
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]  # Example: GitHub_package/2_Code.
sys.path.insert(0, str(CODE_ROOT))
from release_bootstrap import prepare
prepare()

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (GENERATED_ROOT as RELEASE_GENERATED_ROOT)
from Analysis.diagnostics import sv_exp_polycon_mo_pk1_component_contribution_study as workflow

# Validate analytic decomposition against Fortran; example: True requires a compiler.
VALIDATE_WITH_COMPILED_UMAT = True
# Protect archived reports; example: new tables go into generated/component_contributions.
OUTPUT_DIRECTORY = RELEASE_GENERATED_ROOT / 'results/component_contributions'
workflow.VALIDATE_WITH_COMPILED_UMAT = VALIDATE_WITH_COMPILED_UMAT
workflow.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
workflow.OUTPUT_CSV = OUTPUT_DIRECTORY / 'component_stress_contributions.csv'
workflow.OUTPUT_README = OUTPUT_DIRECTORY / 'README.md'
workflow.OUTPUT_PAYLOAD = OUTPUT_DIRECTORY / 'workbook_payload.json'
workflow.OUTPUT_WORKBOOK = OUTPUT_DIRECTORY / 'component_stress_contributions.xlsx'
if __name__ == '__main__':
    workflow.main()
