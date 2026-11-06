# SV_exp_polycon_mo PK1 and PK2 Component Contributions

This fixed-fit study evaluates fits 1961--1964 without refitting or
modifying production fit records. PK2 terms are evaluated directly;
PK1 terms are converted independently at each state using P_i = F S_i.

## Normalization

The output contains both supported normalization scopes.
The default selector for plotting is `per_ratio`.

- `per_ratio` (default): course and wale use their respective absolute
  experimental peaks in the selected stress measure within each ratio.
- `all_ratios`: course and wale use their respective absolute
  experimental peaks in the selected measure across all nine ratios.

For every non-origin row, each component's absolute directional stress
history is divided by the applicable experimental reference peak.
The two normalized directional magnitudes are added, then divided by
the sum over the six components to obtain the combined percentage.
Course-only and wale-only percentages use their directional sums.

## Fit inventory

| Knit | Fit | Ratios |
|---|---:|---:|
| Interlock | 1962 | 9 |
| Rib | 1961 | 9 |
| Purl | 1963 | 9 |
| Single Jersey | 1964 | 9 |

## Validation

- Maximum additive PK1 reconstruction difference: `1.387779e-17`.
- Maximum compiled-UMAT PK1 difference: `2.640567e-09`.
- Maximum additive PK2 reconstruction difference: `6.938894e-18`.
- Maximum compiled-UMAT PK2 difference: `2.243490e-09`.
- Required tolerance: `1.000000e-08`.
- All fit source/snapshot hashes and exact nine-curve training arrays
  were verified before calculation.

## Interpretation limit

These percentages describe absolute, direction-normalized additive
stress magnitude along each fixed fitted loading path. They are not
variance explained, parameter-identifiability measures, or ablation
results, and they do not establish that a component can be removed.
