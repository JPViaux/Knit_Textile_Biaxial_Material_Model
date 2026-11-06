"""Activate bundled utilities when running a numbered script from VSCode."""
from pathlib import Path
import os
import sys

# Code root; example: keep this beside the numbered workflow folders.
CODE_ROOT = Path(__file__).resolve().parent

def prepare():
    """Set imports for this process and fitting workers, then enter the build folder."""
    support = str(CODE_ROOT / 'utilities')
    sys.path.insert(0, support)
    # Worker imports; example: a greedy fitting subprocess inherits this path.
    os.environ['PYTHONPATH'] = os.pathsep.join([support] + ([os.environ['PYTHONPATH']] if os.environ.get('PYTHONPATH') else []))
    # Legacy compiler working directory; example: generated/build/builddir.
    build = CODE_ROOT / 'generated/build'
    build.mkdir(parents=True, exist_ok=True)
    os.chdir(build)
