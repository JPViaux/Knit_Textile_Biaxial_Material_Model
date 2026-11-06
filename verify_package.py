"""Verify publication data/code checksums and mechanics; run directly in VSCode."""
from pathlib import Path
import json
import runpy
import sys
from package_inventory import package_files, checksum

# Validation options; example: False checks only the code/data inventory.
RUN_SCIENTIFIC_CHECKS = True
# Compiler check; example: False skips compiled replay but keeps input/mechanics checks.
CHECK_COMPILED_UMAT = True
ROOT = Path(__file__).resolve().parent

def main():
    """Fail on missing, changed or unlisted public files before scientific checks."""
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    expected = {item['file']: item for item in manifest['files']}
    actual = {path.relative_to(ROOT).as_posix(): path for path in package_files(ROOT)}
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise AssertionError(f'Missing files: {sorted(missing)}; unlisted files: {sorted(extra)}')
    changed = [name for name, path in actual.items()
               if path.stat().st_size != expected[name]['bytes']
               or checksum(path) != expected[name]['sha256']]
    if changed:
        raise AssertionError(f'Changed files: {changed}')
    print(f'PASS: SHA-256 inventory for {len(actual)} data/code/documentation files', flush=True)
    if RUN_SCIENTIFIC_CHECKS:
        sys.path.insert(0, str(ROOT / '2_Code'))
        loaded = runpy.run_path(str(ROOT / '2_Code/verify_release.py'))
        loaded['main'].__globals__['CHECK_COMPILED_UMAT'] = CHECK_COMPILED_UMAT
        loaded['main']()

if __name__ == '__main__':
    main()
