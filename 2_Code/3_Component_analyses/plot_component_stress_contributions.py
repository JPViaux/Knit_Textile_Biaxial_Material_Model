"""Plot the archived component percentages; edit settings and run in VSCode."""
from pathlib import Path
import sys
CODE_ROOT = Path(__file__).resolve().parents[1]  # Example: GitHub_package/2_Code.
sys.path.insert(0, str(CODE_ROOT))
from release_bootstrap import prepare
prepare()

# Release locations; example: move GitHub_package without editing absolute paths.
from release_paths import (GENERATED_ROOT as RELEASE_GENERATED_ROOT)
from Analysis.plotting import plot_component_stress_contributions as workflow
# Stress measure; examples: 'PK2' for the paper figure, 'PK1' for nominal stress.
STRESS_MEASURE = 'PK2'
# Scope; examples: 'all_ratios' for the paper figure, 'per_ratio' for each path.
NORMALIZATION_SCOPE = 'all_ratios'
# Display setting; example: False saves files without opening a window.
SHOW_PLOT = True
workflow.STRESS_MEASURE = STRESS_MEASURE
workflow.NORMALIZATION_SCOPE = NORMALIZATION_SCOPE
workflow.SHOW_PLOT = SHOW_PLOT
workflow.OUTPUT_PATH = RELEASE_GENERATED_ROOT / 'figures/component_stress_contributions.svg'
if __name__ == '__main__':
    workflow.main()
