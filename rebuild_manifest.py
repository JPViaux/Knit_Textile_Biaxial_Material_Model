"""Rebuild reviewed publication checksums; run directly in VSCode."""
from pathlib import Path
import json
from package_inventory import package_files, checksum

# Package root; example: this script beside 1_Experimental_data and 2_Code.
ROOT = Path(__file__).resolve().parent

def main():
    """Record package files, preserving original-source provenance when available."""
    manifest_path = ROOT / 'release_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': []}
    records = {item['file']: item for item in previous['files']}
    files = []
    for path in package_files(ROOT):
        name = path.relative_to(ROOT).as_posix()
        item = dict(records.get(name, {}))
        item.update(file=name, sha256=checksum(path), bytes=path.stat().st_size)
        files.append(item)
    manifest_path.write_text(json.dumps({'schema_version': 1, 'algorithm': 'sha256',
        'files': files}, indent=2) + '\n', encoding='utf-8')
    print(f'Rebuilt reviewed manifest: {len(files)} files')

if __name__ == '__main__':
    main()
