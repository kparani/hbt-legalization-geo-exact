#!/usr/bin/env python3
"""Independent exactness checks for GEO-Exact on the real benchmark cases.

1. Full instances: the block solver and the ring solver (different edge
   generation, different insertion order) must reach the same optimal total.
2. Subsets: for random 300-terminal subsets of each Train/Test-A case, GEO-Exact
   must equal SciPy's dense Hungarian (linear_sum_assignment) over ALL sites.

Writes results/GEO_EXACTNESS.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from study_loader import load_case  # noqa: E402
from geo_exact import geo_exact  # noqa: E402
from run_geo_study import CASES, RESULTS  # noqa: E402

SUBSET = 300
SUBSETS_PER_CASE = 2


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    lines = ["# GEO-Exact exactness checks", "",
             "## 1. Block solver vs ring solver on full instances", "",
             "| Design | Pitch | n | block total | ring total | diff | block s | ring s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fails = 0
    for split, design, pitch, source in CASES:
        ideal, sites = load_case(design, pitch, source)
        ideal = np.asarray(ideal, float)
        sites = np.asarray(sites, float)
        t = time.perf_counter(); _, tb = geo_exact(ideal, sites, pitch_um=pitch); t1 = time.perf_counter() - t
        t = time.perf_counter(); _, tr = geo_exact(ideal, sites, pitch_um=pitch, variant="ring", order="input")
        t2 = time.perf_counter() - t
        diff = tb - tr
        if abs(diff) > 1e-6 * max(1.0, abs(tr)):
            fails += 1
        lines.append(f"| {design} | {pitch:g} | {len(ideal):,} | {tb:,.4f} | {tr:,.4f} | {diff:+.2e} | {t1:.2f} | {t2:.2f} |")
        print(lines[-1], flush=True)
    lines += ["", f"## 2. Random {SUBSET}-terminal subsets vs dense Hungarian over all sites", "",
              "| Design | Pitch | subset seed | Hungarian | GEO | diff |", "|---|---:|---:|---:|---:|---:|"]
    for split, design, pitch, source in CASES:
        if split not in ("Train", "Test A"):
            continue
        ideal, sites = load_case(design, pitch, source)
        ideal = np.asarray(ideal, float)
        sites = np.asarray(sites, float)
        for seed in range(SUBSETS_PER_CASE):
            rng = np.random.default_rng(seed)
            # a contiguous dense patch is a harder test than a uniform sample
            centre = ideal[rng.integers(len(ideal))]
            near = np.argsort(np.abs(ideal - centre).sum(axis=1))[:SUBSET]
            sub = ideal[near]
            cost = np.abs(sub[:, None, :] - sites[None, :, :]).sum(axis=2)
            r, c = linear_sum_assignment(cost)
            hung = float(cost[r, c].sum())
            del cost
            _, geo = geo_exact(sub, sites, pitch_um=pitch)
            diff = geo - hung
            if abs(diff) > 1e-6 * max(1.0, hung):
                fails += 1
            lines.append(f"| {design} | {pitch:g} | {seed} | {hung:,.4f} | {geo:,.4f} | {diff:+.2e} |")
            print(lines[-1], flush=True)
    lines += ["", f"**Failures: {fails}**"]
    (RESULTS / "GEO_EXACTNESS.md").write_text("\n".join(lines) + "\n")
    print(f"FAILURES: {fails}")


if __name__ == "__main__":
    main()
