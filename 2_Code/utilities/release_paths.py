"""Resolve historical paths without modifying archived fit-input bytes."""
from pathlib import Path
import json as _json
import os

# Release roots; example: moving GitHub_package preserves these relationships.
CODE_ROOT = Path(__file__).resolve().parents[1]
UTILITIES_ROOT = CODE_ROOT / 'utilities'
INPUT_ROOT = CODE_ROOT / 'inputs'
ARCHIVE_ROOT = CODE_ROOT / 'archived_results'
EXPERIMENT_ROOT = CODE_ROOT.parent / '1_Experimental_data'
GENERATED_ROOT = CODE_ROOT / 'generated'
FIGURE_ROOT = GENERATED_ROOT / 'figures'

def resolve_string(value):
    """Translate current and historical paths without changing archived bytes."""
    normalized = value.replace('\\', '/')
    # Historical folder names exist only in provenance compatibility; example: saved fits.
    for marker in ('2_Code/_shared/SSMCCF/', '2_Code/SSMCCF/'):
        if marker in normalized:
            tail = normalized.split(marker, 1)[1]
            for prefix, root in (('Inputs/', INPUT_ROOT), ('Outputs/', ARCHIVE_ROOT),
                                 ('Material_Model/SV_exp_polycon_mo/', CODE_ROOT / '1_Material_model')):
                if tail.startswith(prefix):
                    return str(root / tail[len(prefix):])
            return str(UTILITIES_ROOT / tail)
    for marker, root in (('2_Code/utilities/', UTILITIES_ROOT),
                         ('2_Code/inputs/', INPUT_ROOT),
                         ('2_Code/archived_results/', ARCHIVE_ROOT),
                         ('2_Code/1_Material_model/', CODE_ROOT / '1_Material_model'),
                         ('2_Code/generated/', GENERATED_ROOT),
                         ('1_Experimental_data/', EXPERIMENT_ROOT),
                         ('4_Experimental_Data/Raw_data/Tripoly/', EXPERIMENT_ROOT)):
        if marker in normalized:
            return str(root / normalized.split(marker, 1)[1])
    return value

def rebase(value):
    """Resolve path strings recursively while preserving scientific values."""
    if isinstance(value, str):
        return resolve_string(value)
    if isinstance(value, list):
        return [rebase(item) for item in value]
    if isinstance(value, dict):
        result = {key: rebase(item) for key, item in value.items()}
        # Original source hashes identify the immutable snapshot, not a relocated
        # executable configuration. Example: archived case_config uses case.json.
        if isinstance(result.get('input_files'), dict):
            for record in result['input_files'].values():
                if isinstance(record, dict) and record.get('snapshot_path') and record.get('source_path'):
                    record['source_path'] = record['snapshot_path']
        return result
    return value

class ReleaseJSON:
    """Use standard JSON with release-local path resolution on reads only."""
    def loads(self, *args, **kwargs):
        return rebase(_json.loads(*args, **kwargs))
    def load(self, *args, **kwargs):
        return rebase(_json.load(*args, **kwargs))
    def __getattr__(self, name):
        return getattr(_json, name)

json = ReleaseJSON()
