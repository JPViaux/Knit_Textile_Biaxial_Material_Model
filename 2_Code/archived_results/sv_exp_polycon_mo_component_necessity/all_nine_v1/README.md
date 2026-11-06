# SV_exp_polycon_mo All-Nine Component-Necessity Study

## Scope

This isolated study calibrates `SV_exp_polycon_mo` independently to all nine biaxial loading ratios for Interlock S06, Rib S03, Purl S02, and Single Jersey S02. It evaluates all 39 structurally admissible nonempty subsets of six physical components. Linear coupling is included only when both directional linear components are active.

The classifications quantify support within these four all-nine, within-specimen calibration datasets. They are not held-out prediction, external validation, or evidence that an excluded physical mechanism is absent.

## Method

- Reference fits: Interlock 1962, Rib 1961, Purl 1963, and Single Jersey 1964.
- Fixed quantities: `C1=C2=1`, `theta1=0 deg`, `theta2=90 deg`, and fit axes `[0,1]`.
- Production search: bounded amplitude profiling, the rotated-cone linear constraint, 17-point logarithmic beta/gamma grids, a 9-point `b` grid with `a=1-b`, Powell refinement, and exhaustive integer `zeta=3..15` search. The admissible production domain is strictly `zeta > 2`.
- Complexity excludes every amplitude or shape parameter whose parent component is inactive. The six-component model has 10 effective fitted parameters.
- Primary score: observation-weighted normalized PK2 all-nine calibration cost. Sensitivity score: equally weighted mean of the nine ratio costs.
- Adequate set: candidates no more than 2% above the corresponding winner. Selection chooses the lowest effective complexity in that set, followed by lower cost and deterministic mask order.
- `Required`: every adequate model contains the component. `Optional/substitutable`: adequate models exist with and without it. `Unsupported`: every adequate model excludes it. `Weighting-sensitive`: the status changes under equal-ratio weighting.

## Selected structures

| Knit | Selected | Parameters | Adequate models | Required components | Optional/substitutable |
|---|---:|---:|---:|---|---|
| Interlock | M63 | 10 | 1 | all six | none |
| Rib | M31 | 7 | 2 | both linear directions, linear coupling, both nonlinear directions | nonlinear coupling |
| Purl | M39 | 6 | 4 | both linear directions, linear coupling, nonlinear coupling | nonlinear course, nonlinear wale |
| Single Jersey | M55 | 8 | 2 | both linear directions, linear coupling, nonlinear wale, nonlinear coupling | nonlinear course |

No component was classified as unsupported, and no classification changed under equal-ratio weighting. Interlock retains the verified reversed physical-axis mapping when translating physical course/wale components to model parameters.

## Calibration comparison

| Knit | Selected normalized PK2 cost | Selected pooled PK1 R2 | Full pooled PK1 R2 |
|---|---:|---:|---:|
| Interlock | 0.000768569 | 0.938667 | 0.938667 |
| Rib | 0.001311737 | 0.906934 | 0.908892 |
| Purl | 0.001249923 | 0.922159 | 0.922196 |
| Single Jersey | 0.001328507 | 0.906434 | 0.906434 |

These are like-for-like calibration metrics on the same nine ratios. The observed/selected/full curves and per-curve metrics are retained in `predictions.csv` and `curve_metrics.csv`.

## Stability and UMAT checks

Every selected model was screened along the measured paths and over its 121x121 knit-specific incompressible stretch envelope using 721 propagation directions and 32 local refinements. The screens handle the combined-max activation boundary explicitly. Negative counterexamples occur in every knit-specific envelope; Interlock also has a negative measured-path counterexample. Rib, Purl, and Single Jersey are positive on the tested smooth measured-path membrane states, while their full-3D tests retain the exact structural zero mode.

All 156 candidate predictions agree with the complete compiled UMAT within the `1e-10` cost tolerance; the largest recorded difference is `2.84e-14`. Stability is separate from calibration necessity. Finite positive screens do not prove ellipticity, the model name does not prove global polyconvexity, and component removal does not resolve the formulation's structural 3D limitation.

## Outputs

- `summary.json`, `study_definition.json`, and `calibration_summary.json`: study definition and machine-readable results.
- `candidate_scores.csv`, `component_necessity.csv`, `parameters.csv`, `curve_metrics.csv`, `predictions.csv`, `conditioning.csv`, `stability.csv`, `stability_measured.csv`, `stability_envelope.csv`, and `umat_validation.csv`: auditable tables.
- `sv_exp_polycon_mo_component_necessity.xlsx`: formula-backed nine-sheet workbook.
- `figures/`: matching publication SVG and PDF figures.
- `deliverable_hashes.csv`: SHA-256 hashes for the final study deliverables.
- `checkpoints/`: atomic candidate checkpoints supporting safe resumption.

The study does not modify the UMAT, source parameter files, reference fits, existing fit artifacts, or shared fit registry. No selected structure is registered as a production fit.
