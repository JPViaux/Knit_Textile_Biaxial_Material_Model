# Experimental data used in the paper

This folder contains the four specimens used by the published workflows:

| Architecture | Publication specimen folder |
|---|---|
| Interlock | `Interlock/BRT_15mm_Tri_Int_S06` |
| Rib | `Rib/BRT_15mm_Tri_Rib_S03` |
| Purl | `Purl/BRT_15mm_Tri_Purl_S02` |
| Single jersey | `SingleJersey/BRT_15mm_Tri_SJ_S02` |

Each contains `Load_Disp_Data` (instrument exports), `DIC` (strain exports), and `Compiled_Data` (curves and training batches). Existing specimen notes are retained. The compiled `.dat` files and immutable input snapshots in `2_Code/archived_results/fits` are the reproduction starting point. A complete instrument/DIC recompilation workflow is not included; raw exports are supplied for inspection and provenance.

Nine machine-axis ratios per specimen were measured: 0:1, 0.25:1, 0.5:1, 0.75:1, 1:1, 1:0.75, 1:0.5, 1:0.25, 1:0. These are displacement ratios, not stress ratios. Follow metadata for the correspondence between machine axes and course/wale directions. The three-ratio C063 calibration uses 0.5:1, 1:0.5 and 1:1.

## Metadata and source identity

`../2_Code/inputs/Experiment_Data/biaxial_test_metadata.csv` contains 40 protocol records: nine ratios plus one conditioning record for each specimen. Conditioning records are excluded from fitting. Sample/step identifiers, axes, preload, quality and recommendations accompany the supplied data; [DATA_DICTIONARY.md](DATA_DICTIONARY.md) explains the conventions.

Only the selected Purl 0:1 record was relabeled S02 for publication to match the compiled data used by the code. Its `source_sample_folder`, `source_step_folder` and `source_record_id` retain the original S01 source. The reader's source override selects that record. The unused historical S02 0:1 metadata row is not included. The selected curve's preload values (axis 1: 0.1 N; axis 2: 0.2 N), `bad_data` flag and `exclude` recommendation remain unchanged. A publication label does not certify curve quality. All other original source identities and flags are retained.

## Processing and units

The compilation convention converts force to nominal stress using 75 mm reference width and 1.59 mm reference thickness: `P_ii = Load_i / (75 * 1.59)` in N/mm² (MPa). The common thickness is a normalization convention, not a measurement of each architecture's thickness. Paper force/reference-width plots use `Load_i / 75 = 1.59 * P_ii` in N/mm.

Compiled axial strains use the supplied DIC means; the model interprets them as displacement-gradient entries `H_ii = lambda_i - 1`. The through-thickness entry enforces `lambda_3 = 1/(lambda_1*lambda_2)`. Instrument and DIC streams were combined into headerless twelve-column files; preprocessing can include axis swaps, preload/unloading filtering and reference adjustments. Those settings are not reconstructed independently for every raw export here. Use the actual compiled curves and archived snapshots to replay published fits; do not infer preprocessing solely from a raw filename.

Each `Compiled_Data` folder contains nine authoritative individual ratio curves, one all-nine combined file, and one C063 combined file: **44 compiled files in total**. The short folder inventory names only shipped files. Unused development batches/catalogs were removed from the publication copy; all 188 archived fits retain their original byte-preserved training snapshots. Instrument and DIC exports remain unchanged. New derived batches and catalogs are written under `2_Code/generated/results/batches`.

Supplied measurement arrays and fit snapshots were not altered during publication cleanup. Code/data licensing remains undecided.
