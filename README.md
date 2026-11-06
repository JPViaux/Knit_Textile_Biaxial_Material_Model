# Weft-knitted textile hyperelastic model

**A hyperelastic material model for weft-knitted textiles under biaxial tension calibrated across four knit architectures**

Juan Pablo Viaux, Caitlyn MacKay, Ihsan Uluturk, Scott E. Stapleton.

Publication package v1.0.0 for submission to *Mechanics of Materials*.
Repository: https://github.com/JPViaux/Knit_Textile_Biaxial_Material_Model
Journal publication details and article/archive DOIs are pending.

## Choose a starting point

- **View the results:** open the [archive index](2_Code/archived_results/README.md), which identifies the eight main fits and the greedy candidates. No installation is needed to inspect the saved tables and figures.
- **Reproduce a figure:** set up Python below and run `2_Code/4_Greedy_study/plotter_fig_9_candidates.py` in VSCode for a first compiler-free example. The workflow table lists the other figures and their requirements.
- **Run a new calibration:** use `2_Code/2_Calibration/separable_solver_fit.py`. Edit `CASE_FILE` to choose one of the eight bundled cases. New calibration and campaign outputs go under `2_Code/generated`.

## Set up in VSCode

1. Open the repository root. Keep `1_Experimental_data` and `2_Code` together; the enclosing folder may have any name.
2. On Windows, run `conda env create -f 2_Code/environment.yml` in the VSCode terminal and select the `knit-paper` interpreter.
3. Open the desired entry point, edit its commented settings, and choose **Run Python File**.
4. Run `verify_package.py` to check the package and scientific results. Set `RUN_SCIENTIFIC_CHECKS=False` for checksums only, or `CHECK_COMPILED_UMAT=False` for scientific checks without compilation.

For exact tested Windows builds, replace step 2 with `conda create -n knit-paper --file 2_Code/environment-win-64.lock.txt`. `tested_environment_win64.json` records the full tested inventory. Other operating systems have not been verified. `requirements.txt` supplies Python pins for an existing environment; GNU Fortran and Ninja must be installed separately for compiled workflows. The published fitting backend is grid/Powell. No Abaqus installation is required for the Python workflows; the supplied Fortran source also contains a UMAT interface, but Abaqus model setup is outside this package.

## Contents

| Location | Purpose |
|---|---|
| [Experimental data guide](1_Experimental_data/README.md) and [data dictionary](1_Experimental_data/DATA_DICTIONARY.md) | Four specimens, instrument/DIC exports, compiled curves, units, axes and metadata provenance |
| `2_Code/1_Material_model` | Final `SV_exp_polycon_mo.f` and `model.json`, including the canonical PROPS order |
| `2_Code/2_Calibration`, `2_Code/3_Component_analyses`, `2_Code/4_Greedy_study` | Editable calibration, component and ratio-selection entry points |
| `2_Code/utilities` | Shared Analysis/Python_Wrapper modules, mechanics tests, plotting style and path helpers |
| `2_Code/inputs` | Cases, parameter bounds, experimental metadata and manuscript table workbook |
| `2_Code/archived_results` | Immutable fit snapshots, reports, campaign evidence and bundled figures |
| `2_Code/generated` | New `build/`, `results/` and `figures/` outputs; created on demand and excluded from Git |
| [Verification record](2_Code/VERIFICATION.md) | Completed checks and their limits; `verify_package.py` checks package hashes and calls `2_Code/verify_release.py` for mechanics checks |
| [Citation metadata](CITATION.cff) and [release guide](RELEASE.md) | Citation, versioning, upload and permanent archival instructions |
| [release_manifest.json](release_manifest.json) | SHA-256 inventory of data, code and documentation; rebuild explicitly with `rebuild_manifest.py` after reviewing intentional changes |

## Paper-to-workflow guide

Script/input paths in this table are relative to `2_Code`. Figure numbering follows the submission manuscript. Exports reproduce numerical plots; assembled manuscript panel layouts may differ.

| Paper content | Entry point | Authoritative inputs | Compiler? |
|---|---|---|---|
| Figures 1–3: knit/test/protocol context | Experimental illustration/layout; no automated figure workflow is included | Instrument/DIC exports in the sibling `1_Experimental_data` folder | — |
| Figure 4: three-ratio calibration | `2_Calibration/plotter_fig_4.py` | Fits 2164–2167 and C063 compiled curves | Yes |
| Figure 5: six held-out paths | `2_Calibration/plotter_fig_5.py` | Same fits and nine-ratio curves | Yes |
| Figure 6: pooled agreement | `2_Calibration/plot_pooled_r2_by_calibration_pair.py` | `inputs/Biaxial_paper_tables_exp_polycon.xlsx`, worksheet `Data` | No |
| Figure 7: component shares | `3_Component_analyses/plot_component_stress_contributions.py` | `archived_results/model_component_contributions_sv_exp_polycon_mo/component_stress_contributions.csv`; PK2, all-ratio normalization | No |
| Figure 8: single-ratio calibration | `4_Greedy_study/plot_single_ratio_calibration.py` | Archived greedy campaign and B01 fits | No |
| Figure 9: candidates/selections | `4_Greedy_study/plotter_fig_9_candidates.py` | Canonical archived `full_workbook_payload.json` and candidate fits | No |
| Three-ratio/all-nine parameter tables | Stored `fit.json` parameters; refit with `2_Calibration/separable_solver_fit.py` | Fits 2164–2167 / 1961–1964 | Only for refitting |
| Component omission/cost comparison | `3_Component_analyses/component_tolerance_sensitivity.py`; refit with `component_subset_refitting.py` | `archived_results/sv_exp_polycon_mo_component_necessity/all_nine_v1` | Only for refitting |

To recompute fixed-fit stress partitions, run `3_Component_analyses/component_contributions.py`. To prepare or run new ratio-selection fits, use `4_Greedy_study/run_greedy_study.py`.

## Calibration and production settings

Edit `CASE_FILE` in `2_Code/2_Calibration/separable_solver_fit.py` to choose a JSON under `2_Code/inputs/Separable_Solver/Cases`. The default is Interlock C063; architecture-specific all-nine cases use all measured ratios. Cases retain their historical objective, grid sizes and Powell settings. The all-nine cases reproduce the historical zeta=2..15 domain; C063 and production component-refitting searches use zeta=3..15. These are distinct search domains, not interchangeable reruns. Archived parameters and snapshots are unchanged.

| Specimen | C063 fit | All-nine fit |
|---|---:|---:|
| Interlock S06 | 2166 | 1962 |
| Rib S03 | 2167 | 1961 |
| Purl S02 dataset | 2165 | 1963 |
| Single jersey S02 | 2164 | 1964 |

C063 uses machine-axis displacement ratios 0.5:1, 1:0.5 and 1:1. Calibration minimizes normalized PK2 residuals; agreement/nominal-stress plots use PK1. Force/reference-width plots use N/mm with the common 1.59 mm reference-thickness convention. PK1 agreement metrics and the PK2 fitting objective are different quantities.

The component-refitting pilot uses a coarse grid and integer exponents (3,6,10,15). Set `RUN_MODE='full'` for the production search over integer zeta=3..15 and 39 admissible subsets per knit. Greedy fitting defaults to preparation; choose the action and `FULL_CAMPAIGN` explicitly for a full recalibration. These production searches are expensive and were not rerun during packaging.

Figure exports are written below `2_Code/generated/figures`; Figure 6 uses `figure_06`, Figure 8 uses `figure_08`, and Figure 9 uses `greedy`. Calibration outputs and derived study batches are below `generated/results`; compiler products are below `generated/build`.

## Archived results and provenance

`2_Code/archived_results/fits` contains eight final fits and 180 greedy candidate fits, including original input snapshots, parameters and fit metadata. Only those fits appear in the retained registry/results tables. Component reports are in `model_component_contributions_sv_exp_polycon_mo` and `sv_exp_polycon_mo_component_necessity/all_nine_v1`; greedy evidence is in `adaptive_ratio_study_sv_exp_polycon_mo_v1`, all under `archived_results`. Reports include CSV/JSON and original XLSX files; the pooled-R2 source workbook is under `inputs`.

Measurements, constitutive equations and archived scientific results are preserved. Historical paths remain in archived JSON; `2_Code/utilities/release_paths.py` resolves them locally on reads, and input hashes are checked against the original snapshots. `.gitattributes` preserves exact bytes across Git checkouts. Release adapters handle imports, paths, outputs and library filenames; the wrapper also accepts the directional normalization scales used by the final fitting code, reproducing all eight saved costs.

Only `SV_exp_polycon_mo` is registered and shipped. Legacy model tokens in filenames/helper names identify provenance or reusable support functions. Use the numbered entry points for the supported paper workflows. Fresh greedy campaign IDs end in `_release`; derived batches and catalogs are created under `generated/results/batches`. Published measurements and archived reports are read-only inputs. Default plotting entry points preserve archived inputs/reports.

The publication metadata CSV under `2_Code/inputs/Experiment_Data` contains 40 records: nine ratios plus one conditioning record for each of the four specimens above. Conditioning is excluded from fitting. Selected Purl 0:1 has publication label S02 and original S01 identity in `source_sample_folder`, `source_step_folder` and `source_record_id`. The reader selects that original source record; the unused historical S02 0:1 record is omitted. Its preload, `bad_data` flag and `exclude` recommendation remain unchanged despite its use by the paper workflow. See the data guide for the complete source-label decision.

One specimen per architecture was used; held-out ratios assess within-specimen transfer and greedy selection is retrospective. Component percentages describe fixed-fit stress shares; subset refitting supplies conditional calibration support. These tests establish neither unrestricted three-dimensional stability nor parameter uniqueness.

## Citation and licensing

Use `CITATION.cff` to cite the package, and cite the paper once publication details are available. Code/data licenses are not yet selected; public access alone grants no open-source/open-data license. Licensing and permanent Zenodo archival remain pending the authors' decision.
