#!/usr/bin/env python3
"""Run ONE (design, pitch, method) trial in a fresh process and print a JSON line.

Usage: python3 run_one.py <design> <pitch_um> <source> <method>

Methods:
  sw          baselines.sliding_window (the paper's ViaLegal-style baseline)
  sw_indexed  fast_methods.sliding_window_indexed (same algorithm, tile index)
  dghl        dghl.dghl with the paper's locked configuration
  geo         geo_exact.geo_exact (exact global optimum, block variant)
  geo_ring    geo_exact.geo_exact(variant="ring") (independent exact cross-check)
  knn8/16/32  fast_methods.sparse_exact with k = 8/16/32 (the review's 3.2 baseline)

Only the legalizer call is timed.  Peak RSS is the whole process (data
loading included), read from getrusage at exit.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # hbt_legalization/
sys.path.insert(0, str(HERE))

from study_loader import load_case  # noqa: E402  (provided by the harness owner)
from baselines import sliding_window  # noqa: E402
from dghl import dghl  # noqa: E402
from fast_methods import sliding_window_indexed, sparse_exact  # noqa: E402
from geo_exact import geo_exact  # noqa: E402

DGHL_LOCKED = dict(max_passes=3, max_local_hungarian_hbts=576,
                   shrink_factor=0.5, max_local_sites=10000)


def run(method: str, ideal: np.ndarray, sites: np.ndarray, pitch: float, stats: dict):
    if method == "sw":
        return sliding_window(ideal, sites, pitch_um=pitch)
    if method == "sw_indexed":
        return sliding_window_indexed(ideal, sites, pitch_um=pitch, stats=stats)
    if method == "dghl":
        return dghl(ideal, sites, pitch_um=pitch, stats=stats, **DGHL_LOCKED)
    if method == "geo":
        return geo_exact(ideal, sites, pitch_um=pitch, stats=stats)
    if method == "geo_ring":
        return geo_exact(ideal, sites, pitch_um=pitch, variant="ring", order="input", stats=stats)
    if method.startswith("knn"):
        # Review 3.2 baseline: global sparse exact matching (SciPy LAPJVsp, compiled)
        # on each terminal's k nearest sites; k is doubled globally until a full
        # matching exists (k_used is reported), capped at 1024.
        return sparse_exact(ideal, sites, k=int(method[3:]), max_k=1024, stats=stats)
    raise SystemExit(f"unknown method {method}")


def main() -> None:
    design, pitch, source, method = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
    ideal, sites = load_case(design, pitch, source)
    ideal = np.asarray(ideal, dtype=float)
    sites = np.asarray(sites, dtype=float)
    if method.startswith("geo"):
        from geo_exact import _load
        _load()  # make sure the C library is built before timing
    stats: dict = {}
    t0 = time.perf_counter()
    assignment, total = run(method, ideal, sites, pitch, stats)
    runtime = time.perf_counter() - t0
    assignment = np.asarray(assignment)
    injective = bool(np.all(assignment >= 0) and len(np.unique(assignment)) == len(assignment))
    disp = np.abs(ideal - sites[assignment]).sum(axis=1)
    rec = {
        "design": design, "pitch": pitch, "source": source, "method": method,
        "n": int(len(ideal)), "m": int(len(sites)),
        "total": float(total), "total_recomputed": float(disp.sum()),
        "max_disp": float(disp.max()), "p95_disp": float(np.percentile(disp, 95)),
        "runtime_s": runtime, "maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "injective": injective,
        "stats": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in stats.items()},
    }
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
