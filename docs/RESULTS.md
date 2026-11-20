# Results

All values are from the paper (Table B1 and the abstract). Displacement is total
Manhattan (L1) displacement in microns. "Gap" is how far a method is above the
certified optimum: `gap = D / D* - 1`.

## The 21 benchmark instances

Instances are built from Open3DBench placement data. `n` is the number of
terminals, `m` the number of legal sites, and utilization is `n / m`.
"Rel. gap" is the relative primal–dual gap of the independent LP dual
certificate that proves the GEO-Exact assignment optimal.

| Instance | n | m | Util. (%) | Certified optimum (µm) | Rel. gap | SW (%) | BTAssign-style (%) | DGHL (%) | Greedy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ariane133@4 | 4,934 | 61,256 | 8.1 | 19,471.02 | 1.9e-16 | 0.000 | 0.36 | 1.01 | 43.6 |
| ariane133@6 | 4,934 | 27,225 | 18.1 | 51,511.04 | 2.8e-16 | 0.000 | 0.16 | 5.01 | 52.8 |
| bp@4 | 7,667 | 49,506 | 15.5 | 24,858.73 | 0 | 0.000 | 0.40 | 5.31 | 43.9 |
| bp@6 | 7,667 | 22,052 | 34.8 | 103,511.77 | 2.8e-16 | 0.119 | 5.86 | 14.85 | 94.9 |
| bp_be@4 | 2,702 | 18,056 | 15.0 | 11,981.19 | 3.0e-16 | 0.000 | 0.52 | 4.10 | 66.0 |
| bp_be@6 | 2,702 | 8,118 | 33.3 | 67,910.17 | 2.1e-16 | 0.000 | 2.47 | 5.83 | 70.7 |
| swerv_wrapper@4 | 5,755 | 34,056 | 16.9 | 24,572.34 | 0 | 0.000 | 5.09 | 9.51 | 57.5 |
| swerv_wrapper@6 | 5,755 | 15,180 | 37.9 | 149,801.24 | 7.8e-16 | 0.009 | 9.13 | 12.68 | 84.2 |
| ariane136@4 | 4,793 | 61,256 | 7.8 | 15,319.95 | 0 | 0.000 | 0.03 | 0.69 | 33.6 |
| ariane136@6 | 4,793 | 27,225 | 17.6 | 44,668.69 | 1.6e-16 | 0.000 | 0.04 | 5.07 | 64.2 |
| bp_fe@4 | 2,790 | 13,775 | 20.3 | 31,949.09 | 3.4e-16 | 0.000 | 0.80 | 6.34 | 79.6 |
| bp_fe@6 | 2,790 | 6,111 | 45.7 | 123,512.27 | 8.2e-16 | 0.000 | 2.48 | 4.41 | 59.5 |
| bp_multi@4 | 7,151 | 39,204 | 18.2 | 31,816.96 | 1.1e-16 | 0.000 | 1.16 | 10.95 | 59.5 |
| bp_multi@6 | 7,151 | 17,424 | 41.0 | 151,740.50 | 3.8e-16 | 0.024 | 9.20 | 15.14 | 102.0 |
| bp_quad@4 | 39,814 | 387,506 | 10.3 | 213,777.81 | 8.2e-16 | 0.000 | 3.13 | 3.29 | 58.0 |
| bp_quad@6 | 39,814 | 172,225 | 23.1 | 676,286.39 | 1.7e-16 | 0.076 | 6.42 | 6.97 | 62.3 |
| aes@2 | 1,492 | 27,060 | 5.5 | 1,439.86 | 0 | 0.000 | 0.00 | 0.00 | 0.0 |
| dynamic_node@2 | 3,083 | 5,625 | 54.8 | 2,629.24 | 0 | 0.000 | 0.00 | 0.00 | 0.0 |
| ibex@2 | 1,600 | 5,625 | 28.4 | 1,537.78 | 0 | 0.000 | 0.00 | 0.00 | 0.0 |
| jpeg@2 | 13,458 | 22,500 | 59.8 | 13,537.65 | 0 | 0.000 | 0.55 | 0.00 | 0.0 |
| swerv@2 | 11,285 | 30,625 | 36.8 | 11,294.12 | 0 | 0.000 | 0.00 | 0.00 | 0.0 |

## Summary

| | SW (ViaLegal defaults) | BTAssign-style | DGHL | Greedy |
|---|---:|---:|---:|---:|
| Mean gap, 16 contended instances | 0.014% | 3.0% | 6.9% | 64.5% |
| Mean gap, all 21 instances | 0.011% | | | |
| Worst gap | 0.119% (bp@6) | | | |

The 16 contended instances are the Train, Test A and scalability instances
(the first 16 rows). On the five Test B instances at 2 µm, the methods mostly
reach the optimum.

## Where the headroom is

At the nominal pitches, the sliding-window (SW) baseline with ViaLegal's
default settings leaves almost nothing to recover. Larger gaps appear outside
that regime:

- **SW configuration:** with 2×2 windows, SW reaches 34.1% above optimal; with
  no cross-window reassignment, 20.9%.
- **High utilization:** in pitch sweeps (217 further certified instances), SW
  gaps above 1% appear only at 58.5% utilization or more. They reach 21.1% on
  bp, 27.5% on swerv and 8.9% on jpeg. aes, dynamic_node and ibex stay at zero.

## Speed

Geometric-mean speed-ups of GEO-Exact over the 21 instances: 8.1× over DGHL and
40× over SW (run 1). This compares a compiled C core with Python baselines on a
shared server, so treat it as an order of magnitude, not a precise ratio.

## Exactness checks in this repository

- `tests/test_geo_exact.py` compares GEO-Exact with SciPy's dense Hungarian
  solver on randomized small instances (full and holed grids, off-grid
  terminals, exact ties, every solver variant and insertion order). It checks
  1,148 solves and expects 0 mismatches.
- `scripts/check_exactness.py` (needs the benchmark data) checks that the block
  and ring variants reach the same total on every full instance, and that
  GEO-Exact equals dense Hungarian on random 300-terminal subsets of the Train
  and Test A instances.
