# Data dictionary

## Compiled `.dat` files

Tab-separated numeric text, no header or index. Each row is one paired strain/stress state. The loader uses `typ=['H', 'P']`; filenames identify machine-axis loading ratios or concatenated calibration batches. Six-component ordering is **11, 22, 33, 23, 13, 12**.

| Column (1-based) | Quantity | Units |
|---|---|---|
| 1 | H11 = lambda1 - 1 | Dimensionless |
| 2 | H22 = lambda2 - 1 | Dimensionless |
| 3 | H33 = lambda3 - 1 | Dimensionless |
| 4 | H23 | Dimensionless |
| 5 | H13 | Dimensionless |
| 6 | H12 | Dimensionless |
| 7 | P11, nominal first Piola–Kirchhoff stress | N/mm² = MPa |
| 8 | P22, nominal first Piola–Kirchhoff stress | N/mm² = MPa |
| 9 | P33 | N/mm² = MPa |
| 10 | P23 | N/mm² = MPa |
| 11 | P13 | N/mm² = MPa |
| 12 | P12 | N/mm² = MPa |

For on-axis compiled curves, off-diagonal entries and P33 are zero. `F = I + H` and `S = F^{-1} P` are the wrapper's conversions; its calibration objective uses second Piola–Kirchhoff stress S (PK2). The historical compiler may call stress columns `Sigma`; these force/reference-area values are nominal stress, not Cauchy stress. kPa displays multiply MPa by 1000. N/mm force/reference-width displays multiply nominal stress by 1.59 mm.

## Instrument and DIC CSV exports

| Field or field family | Interpretation |
|---|---|
| `Load (N)`, `Load2 (N)` | Machine-axis 1 and 2 forces, N |
| `Position (mm)`, `Position2 (mm)` | Machine displacement channels, mm |
| `Time`, `Time(1d)` | Export time channels; preserve export labels and specimen notes when synchronizing streams |
| `Ratio` | Machine loading-path label |
| `mean(exx)`, `mean(eyy)` | Mean supplied axial DIC strain quantities; dimensionless; used to construct compiled H11/H22 |
| `mean(ezz)` | Through-thickness quantity when present; compiled H33 uses the incompressibility convention |
| Standard deviation fields | Spatial strain variability in the corresponding DIC export; not an additional independent specimen |
| Unnamed leading column | Exported row index, not a physical measurement |

Raw export headers vary. The twelve-column definition applies to compiled `.dat` files; do not apply it positionally to raw CSVs.

## Publication metadata CSV

The authoritative header is in `2_Code/inputs/Experiment_Data/biaxial_test_metadata.csv`. Records describe protocols and interpretation, not extra numerical curves.

| Field or field family | Meaning |
|---|---|
| `sample_id`, `sample_number`, `canonical_sample_folder`, `canonical_step_folder` | Publication identity and canonical data location |
| `source_sample_folder`, `source_step_folder`, `source_record_id` | Original identity before publication relabeling; source_record_id is unique |
| `metadata_source` | Provenance of the supplied metadata record |
| Ratio/test order fields | Machine loading path and sequential protocol order |
| Axis/orientation fields | Mapping from machine axes to physical course/wale directions; consult these before relabeling plots |
| Preload fields | Recorded pretension/preload information; force values use N |
| Quality and inclusion fields | Original assessments/recommendations, preserved even for the selected Purl 0:1 curve |
| Conditioning fields/records | Protocol preparation; not a calibration loading ratio |

Canonical and source identities deliberately differ for selected Purl 0:1: S02 publication label, original `BRT_15mm_Tri_Purl_S01::ratio_0_1` source. Its `bad_data`/`exclude` status is retained. See README.md for the selection decision and scope.
