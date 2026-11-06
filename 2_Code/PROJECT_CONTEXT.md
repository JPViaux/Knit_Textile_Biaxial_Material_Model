# Paper-release project context

## Current package

This is the publication copy for *A hyperelastic material model for weft-knitted textiles under biaxial tension calibrated across four knit architectures*, by Juan Pablo Viaux, Caitlyn MacKay, Ihsan Uluturk, and Scott E. Stapleton. Repository: https://github.com/JPViaux/Knit_Textile_Biaxial_Material_Model.git. Read the [root guide](../README.md) first. Development sources outside this package are not modified.

The user authorized the full implementation and readability cleanup. Numbered workflows cover the final SV_exp_polycon_mo model, calibration, component analyses, and retrospective greedy selection. Shared support lives in utilities, editable cases/bounds/metadata in inputs, immutable scientific evidence in archived_results, and new outputs in generated/build, generated/results, or generated/figures. The archive index lists eight final fits and 180 greedy candidates; all input snapshots retain their original bytes and hashes.

## Data and scientific decisions

Only four raw compiled datasets are selectable: Interlock S06, Rib S03, Purl S02, and SingleJersey S02. The experimental folders retain instrument/DIC exports and 44 compiled files (36 individual curves, four all-nine batches, four C063 batches). Unused development batches/catalogs and standalone tools that require absent models/data were removed. Archived training snapshots remain intact. A complete instrument/DIC recompilation pipeline is not supplied.

The metadata has 40 rows: nine ratios plus conditioning for each specimen. The selected Purl 0:1 record has publication label S02 and explicit original S01 source identifiers. Its preload, bad_data flag, and exclude recommendation are retained. No other specimen relabeling or quality change is authorized. Machine axes must be interpreted using metadata; Interlock orientation differs from other knits.

C063 fits: Interlock 2166, Rib 2167, Purl 2165, SingleJersey 2164. All-nine fits: Interlock 1962, Rib 1961, Purl 1963, SingleJersey 1964. Historical all-nine cases use integer zeta=2..15; C063/component production use 3..15. Scientific bounds and archived results were preserved rather than silently harmonized. Calibration minimizes normalized PK2 residuals; nominal plots use PK1. Force/width uses the common 1.59 mm thickness convention. Held-out ratios assess within-specimen transfer, and component shares are descriptive fixed-fit quantities.

## Release adaptations and maintenance

release_bootstrap.py prepares imports/worker paths and the generated build directory. release_paths.py translates historical archived paths on reads and checks original source hashes against immutable snapshots. Active cases now use current public paths. The release UMAT wrapper accepts the directional reference maxima required by the saved fitting objective. Only the final model is registered. Fresh fitting workers write generated/results/calibration_refitting; diagnostic worker lookups accept an explicit generated results table. Active campaign cases, batches, catalogs, evaluations, and state files are generated products. The retained compiled Readme files preserve C063 ratio membership because diagnostic readers parse those entries. The supported workflows use grid/Powell; the packaged Windows environment is verified with GNU Fortran 15.2.0, Meson, and Ninja.

Figures 8 and 9 read one canonical archived greedy payload. All figure exports go to generated/figures; Figure 8 no longer overwrites its archived SVG/PDF. New greedy batches/catalogs go to generated/results/batches, and fresh campaign IDs end in _release. Published workbooks are not appended by the release campaign. The component deliverable ledger covers shipped files and omits two unshipped temporary QA exports. The root manifest is authoritative; rebuild_manifest.py refreshes it only after reviewed intentional edits. .gitattributes preserves exact snapshot bytes in Git checkouts.

See [VERIFICATION.md](VERIFICATION.md) for current tests and limitations. Licensing and Zenodo remain deferred by the user. GitHub upload is pending authentication; do not claim a push, online release, or DOI. The local v1.0.0 tag predates this cleanup and must not be represented as containing uncommitted changes.

## Current verification

The staged cleanup passed the complete compiled scientific verifier, all nine workflow smoke checks, a separate final greedy preparation check, and a bounded new calibration plus isolated evaluation-worker check. Measurement/archive hashes were checked across runs, and active cases were checked during fresh preparation. The verifier now checks every selectable dataset, shipped component-deliverable hashes, Figure 8 output location, and preservation of published measurement/archive files. Full production optimizations were not rerun. Backups of replaced/removed publication files are kept outside this package in development tmp/publication_simplification_backup.

The final destination also passed verify_package.py in the clean Windows environment: all 2,070 public checksums, every scientific check above, all eight compiled saved costs, shipped deliverable hashes, and measurement/archive preservation. Local build/figure outputs and Python caches were moved outside the upload folder after verification. Changes are local and uncommitted; the existing unpublished tag predates this cleanup.
