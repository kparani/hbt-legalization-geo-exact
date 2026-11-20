# GEO-Exact: Certified-Optimal HBT Legalization

**Exact, provably optimal legalization of hybrid bonding terminals (HBTs) for
face-to-face 3D ICs: full Open3DBench instances of up to 39,814 terminals
solved in seconds, with an independent optimality certificate.**

> Paper: https://doi.org/10.48550/arXiv.2611.14928

---

## The problem in one paragraph

In a face-to-face 3D IC, signals cross between the two dies through hybrid
bonding terminals: tiny pads that must sit on a regular grid. The placer
suggests where each terminal should go, but those spots are not on the grid.
**Legalization** gives every terminal its own grid site while moving the
terminals as little as possible in total. It is a seating problem: many
terminals want the same nearby seats, and the best arrangement can need a
chain of small moves rather than one big one.

Published legalizers are heuristics, and none of the Open3DBench legalizers is
released, so nobody could say how far any of them is from the best possible
answer. GEO-Exact computes that best answer exactly, which turns "method A
beats method B" into "method A is X% above the optimum".

## Highlights

- **Exact at full scale.** On all 21 benchmark instances (1,492 to 39,814
  terminals), an independent LP dual certificate proves the GEO-Exact
  assignment optimal. The largest relative primal–dual gap is 8.2×10⁻¹⁶.
- **Measured headroom, not guessed.** Against the certified optima, the
  sliding-window (SW) baseline with ViaLegal's default settings is on average
  **0.011%** above optimal (worst 0.119%). On the 16 contended instances,
  BTAssign-style, DGHL and greedy are **3.0%, 6.9% and 64.5%** above.
- **Where the gap really is.** SW can be far from optimal outside its default
  regime: up to **34.1%** with 2×2 windows, and up to **27.5%** at high
  utilization in pitch sweeps (217 further certified instances).
- **Fast.** About an order of magnitude faster than the baselines in our
  harness (geometric means **8.1×** over DGHL and **40×** over SW; a compiled
  C core against Python baselines on a shared server).

Full per-instance results: [docs/RESULTS.md](docs/RESULTS.md).

## Quick start

Requirements: Python 3, `numpy`, `scipy` and `gcc`, on Linux, macOS or Windows
with WSL. The C core is compiled automatically the first time it is used.

```bash
git clone https://github.com/kparani/hbt-legalization-geo-exact.git
cd hbt-legalization-geo-exact
pip install -r requirements.txt

# Exactness test: GEO-Exact vs dense Hungarian on randomized small instances
PYTHONPATH=src python tests/test_geo_exact.py
# -> checked 1148 solves, failures 0

# Demo on a synthetic instance (no benchmark data needed)
PYTHONPATH=src python examples/quickstart.py
```

Example output of the demo (times depend on the machine; the GEO-Exact time
includes compiling the C core on the first run):

```text
3000 terminals, 12100 sites, pitch 4.0 um

method             total L1 (um)  gap to optimum  time (s)
GEO-Exact              37,669.71         optimal      0.79
greedy                 56,602.10          50.26%      1.99
sliding window         37,669.71           0.00%      5.21
BTAssign-style         37,734.66           0.17%     10.98
DGHL                   42,346.85          12.42%      0.44
```

## Using it in your own code

```python
import numpy as np
from geo_exact import geo_exact   # with src/ on your PYTHONPATH

terminals = np.array([...])   # (n, 2) ideal terminal positions, microns
sites = np.array([...])       # (m, 2) legal grid sites, microns, m >= n

assignment, total = geo_exact(terminals, sites, pitch_um=4.0)
# assignment[i] = index of the site given to terminal i
# total        = minimum total Manhattan (L1) displacement
```

Every legalizer in `src/` has the same interface and returns
`(assignment, total)`:

| Function | Method |
|---|---|
| `geo_exact.geo_exact` | GEO-Exact, the exact global optimum |
| `baselines.sliding_window` | Sliding window, ViaLegal-style (defaults: tile 30× pitch, 3×3 window, stride 2) |
| `btassign.btassign` | BTAssign-style quadtree bipartite matching |
| `dghl.dghl` | DGHL, a density-guided heuristic |
| `baselines.greedy_nearest` | Greedy nearest free site |
| `fast_methods.sparse_exact` | k-nearest-neighbour sparse exact matching |

## How GEO-Exact works

**In simple words:** it seats terminals one at a time. When a terminal's
preferred seat is taken, it searches for the cheapest chain of moves that frees
a seat, and it keeps price tags on terminals and seats so it never has to
reconsider choices that cannot pay off. It only looks at nearby seats, and it
widens the search ring by ring, stopping as soon as it can prove nothing
farther away could help.

**Technically:** the problem is a minimum-weight injective bipartite assignment
with L1 cost. GEO-Exact runs successive shortest augmenting paths with dual
potentials (Hungarian / Jonker–Volgenant style) and a heap-based Dijkstra on
reduced costs. The complete bipartite graph is never built. Sites are bucketed
in a uniform grid, and each terminal's edges are generated lazily, one block
(or ring) of cells at a time. Opening a block is itself a heap event, keyed by
a lower bound on the reduced cost of every site inside it, so Dijkstra stays
exact. Terminals are inserted sparse-first; the insertion order changes the run
time and which optimum is returned among ties, never the optimal value. The
core is C (`src/geo_lap.c`), called from Python through `ctypes`, and no n×m
matrix is stored.

## Repository layout

```text
src/
  geo_exact.py      GEO-Exact Python interface (compiles and loads the C core)
  geo_lap.c         C core: lazy-edge shortest augmenting paths
  fast_methods.py   indexed sliding window and k-NN sparse exact baselines
  baselines.py      greedy and sliding-window baselines
  btassign.py       BTAssign-style baseline
  dghl.py           DGHL heuristic
tests/
  test_geo_exact.py randomized exactness test against dense Hungarian
examples/
  quickstart.py     synthetic demo comparing every method with the optimum
scripts/
  run_geo_study.py  benchmark driver for the 21 instances
  run_one.py        runs one (design, pitch, method) trial in a fresh process
  check_exactness.py block-vs-ring and subset-vs-Hungarian checks
docs/
  RESULTS.md        per-instance results from the paper
```

## Reproducing the benchmark study

The scripts in `scripts/` reproduce the 21-instance study. They need:

1. **Open3DBench placement data**, which is not included here. Get it from the
   [Open3DBench repository](https://github.com/lamda-bbo/Open3DBench).
2. **A `study_loader` module** that provides `load_case(design, pitch, source)`
   and returns the ideal terminal positions and legal sites for one instance.
   It is not part of this release.

```bash
PYTHONPATH=src python scripts/run_geo_study.py run      # run all cases
PYTHONPATH=src python scripts/run_geo_study.py summary  # rebuild the report
PYTHONPATH=src python scripts/check_exactness.py
```

`run_one.py` uses the Unix `resource` module, so run the study on Linux, macOS
or WSL.

## Scope and limitations

- The baselines (SW, BTAssign-style, greedy) are reconstructions from the
  published descriptions, not the original authors' code. DGHL is our own
  heuristic.
- The objective is total L1 displacement. Downstream routing or timing is not
  evaluated.
- The dual-certificate checker used in the paper is not part of this release;
  this repository includes the randomized exactness test and the cross-checks
  described in [docs/RESULTS.md](docs/RESULTS.md).
- Run times in the paper compare a compiled C core with Python baselines on a
  shared server.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{geoexact_hbt,
  title  = {[paper title]},
  author = {[authors]},
  year   = {2026},
  eprint = {[arXiv ID]},
  archivePrefix = {arXiv}
}
```
