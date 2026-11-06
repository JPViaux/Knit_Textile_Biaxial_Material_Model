"""Paper-release entry point: edit settings below and run in VSCode."""
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]  # Example: GitHub_package/2_Code.
sys.path.insert(0, str(CODE_ROOT))
from release_bootstrap import prepare
prepare()
from Analysis.plotting import plotter_fig_5 as workflow

# Display setting; example: False for unattended figure export.
SHOW_PLOT = True
workflow.SHOW_PLOT = SHOW_PLOT
workflow.SHOW_FIGURES = SHOW_PLOT
if __name__ == '__main__':
    workflow.main()
