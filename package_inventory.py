"""Shared inventory rules for package checking and reviewed manifest rebuilds."""
from pathlib import Path
import hashlib

# Excluded local artifacts; example: generated/build and .artifact_work/node_modules.
IGNORED_DIRECTORIES = {'.git', '__pycache__', 'generated', 'builddir', '.pytest_cache',
                       '.artifact_work', 'node_modules', '.venv'}
IGNORED_SUFFIXES = {'.pyc', '.pyo', '.dll', '.so', '.dylib'}

def package_files(root):
    """Return stable relative paths for all intended public files."""
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if (path.name == 'release_manifest.json' or '-DESKTOP-' in path.name
                or path.name in {'.DS_Store', 'Thumbs.db'} or path.name.startswith('~$')
                or path.suffix in IGNORED_SUFFIXES):
            continue
        yield path

def checksum(path):
    """Hash file contents in bounded chunks; example: large stability CSV."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
