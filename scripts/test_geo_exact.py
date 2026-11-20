#!/usr/bin/env python3
"""Randomised exactness test: GEO-Exact vs dense Hungarian on small instances.

Covers full and holed grids, terminals off the grid, exact ties, and every
solver variant / insertion order.  Run: python3 test_geo_exact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from geo_exact import geo_exact  # noqa: E402


def main(trials: int = 300) -> None:
    rng = np.random.default_rng(12345)
    checked = fails = 0
    for t in range(trials):
        n = int(rng.integers(5, 150))
        p = float(rng.choice([1.0, 2.0, 4.0, 6.0]))
        side = int(np.ceil(np.sqrt(n / rng.uniform(0.2, 0.95))))
        g = np.arange(side) * p
        gx, gy = np.meshgrid(g, g)
        sites = np.c_[gx.ravel(), gy.ravel()]
        if rng.random() < 0.5:
            sites = sites[rng.random(len(sites)) > 0.2]
        if len(sites) < n:
            continue
        span = side * p
        pts = np.r_[rng.normal(span / 2, span / 6, (n // 2, 2)),
                    rng.uniform(-p, span + p, (n - n // 2, 2))]
        if rng.random() < 0.3:
            pts = np.round(pts / (p / 2)) * (p / 2)
        cost = np.abs(pts[:, None, :] - sites[None, :, :]).sum(axis=2)
        r, c = linear_sum_assignment(cost)
        opt = float(cost[r, c].sum())
        for variant, order, block in (("block", "sparse_first", 8), ("block", "input", 2),
                                      ("block", "raster", 1), ("ring", "input", 8)):
            _, got = geo_exact(pts, sites, pitch_um=p, variant=variant, order=order, block=block)
            checked += 1
            if abs(got - opt) > 1e-6 * max(1.0, opt):
                fails += 1
                print(f"MISMATCH trial={t} variant={variant} order={order} opt={opt} got={got}")
    print(f"checked {checked} solves, failures {fails}")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
