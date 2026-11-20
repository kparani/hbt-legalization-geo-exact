#!/usr/bin/env python3
"""Quick demo on a synthetic instance (no Open3DBench data needed).

Builds a pitch grid of legal sites and a cloud of terminals with a dense
hotspot, then compares every legalizer in src/ against the GEO-Exact optimum.

Run from the repository root:
    PYTHONPATH=src python examples/quickstart.py
"""
from __future__ import annotations

import time

import numpy as np

from baselines import greedy_nearest, sliding_window
from btassign import btassign
from dghl import dghl
from geo_exact import geo_exact


def make_instance(n: int = 3000, pitch: float = 4.0, side: int = 110, seed: int = 0):
    rng = np.random.default_rng(seed)
    g = np.arange(side) * pitch
    gx, gy = np.meshgrid(g, g)
    sites = np.c_[gx.ravel(), gy.ravel()]
    span = side * pitch
    background = rng.uniform(0, span, (n // 2, 2))
    hotspot = rng.normal(span * 0.4, span * 0.08, (n - n // 2, 2))
    terminals = np.clip(np.r_[background, hotspot], 0, span)
    return terminals, sites, pitch


def main() -> None:
    terminals, sites, pitch = make_instance()
    print(f"{len(terminals)} terminals, {len(sites)} sites, pitch {pitch} um\n")

    t0 = time.perf_counter()
    _, opt = geo_exact(terminals, sites, pitch_um=pitch)
    t_opt = time.perf_counter() - t0  # includes compiling geo_lap.c on first run

    methods = {
        "greedy": lambda: greedy_nearest(terminals, sites),
        "sliding window": lambda: sliding_window(terminals, sites, pitch_um=pitch),
        "BTAssign-style": lambda: btassign(terminals, sites, pitch_um=pitch),
        "DGHL": lambda: dghl(terminals, sites, pitch_um=pitch),
    }
    print(f"{'method':<16}{'total L1 (um)':>16}{'gap to optimum':>16}{'time (s)':>10}")
    print(f"{'GEO-Exact':<16}{opt:>16,.2f}{'optimal':>16}{t_opt:>10.2f}")
    for name, run in methods.items():
        t0 = time.perf_counter()
        _, total = run()
        dt = time.perf_counter() - t0
        gap = 100 * (total / opt - 1)
        if abs(gap) < 5e-6:  # equal to the optimum up to floating-point rounding
            gap = 0.0
        print(f"{name:<16}{total:>16,.2f}{gap:>15.2f}%{dt:>10.2f}")


if __name__ == "__main__":
    main()
