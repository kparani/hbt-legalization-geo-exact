#!/usr/bin/env python3
"""Exact geometric HBT legalization (global L1 optimum).

GEO-Exact solves the full injective assignment problem

    minimise  sum_i |x_i - s_sigma(i)|_1   over injective sigma: terminals -> sites

exactly, using successive shortest augmenting paths (Hungarian /
Jonker-Volgenant) with Dijkstra on reduced costs.  The complete bipartite
graph is never built: legal sites are bucketed in a uniform grid and each
terminal's edges are generated lazily, block by block, behind lower bounds
that keep Dijkstra exact.  See geo_lap.c for the algorithm and invariants.

The C core is compiled on first import with the system C compiler into the
same directory as this module; no third-party packages are needed.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

try:
    from .baselines import Assignment, _validate, _finish
except ImportError:
    from baselines import Assignment, _validate, _finish

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "geo_lap.c"
_LIB = _HERE / "libgeolap.so"
_lib = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if not _LIB.exists() or _LIB.stat().st_mtime < _SRC.stat().st_mtime:
        cc = os.environ.get("CC", "gcc")
        subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", str(_LIB), str(_SRC), "-lm"],
                       check=True)
    lib = ctypes.CDLL(str(_LIB))
    dp = ctypes.POINTER(ctypes.c_double)
    ip = ctypes.POINTER(ctypes.c_int)
    llp = ctypes.POINTER(ctypes.c_longlong)
    lib.geo_lap.argtypes = [ctypes.c_int, dp, dp, ctypes.c_int, dp, dp, ctypes.c_double,
                            ip, ip, llp]
    lib.geo_lap.restype = ctypes.c_int
    lib.geo_lap2.argtypes = [ctypes.c_int, dp, dp, ctypes.c_int, dp, dp, ctypes.c_double,
                             ctypes.c_int, ip, ip, llp]
    lib.geo_lap2.restype = ctypes.c_int
    _lib = lib
    return lib


def _pitch_from_sites(sites: np.ndarray) -> float:
    diffs = []
    for axis in (0, 1):
        vals = np.unique(sites[:, axis])
        d = np.diff(vals)
        d = d[d > 1e-12]
        if len(d):
            diffs.append(float(np.min(d)))
    return float(min(diffs)) if diffs else 1.0


_ERRORS = {-1: "more terminals than sites", -2: "bad cell size", -3: "out of memory",
           -4: "no augmenting path (should be impossible)"}


def insertion_order(ideal: np.ndarray, pitch: float, order: str) -> np.ndarray | None:
    """Terminal insertion order.  Never changes the optimal total."""
    n = len(ideal)
    if order == "input":
        return None
    if order == "raster":
        return np.lexsort((ideal[:, 0], ideal[:, 1])).astype(np.int32)
    if order == "sparse_first":
        # Insert terminals in low-density areas first: they settle without
        # long augmenting paths, which shortens the later dense-core searches.
        counts = cKDTree(ideal).query_ball_point(ideal, 5.0 * pitch, p=1, return_length=True)
        return np.argsort(np.asarray(counts), kind="stable").astype(np.int32)
    raise ValueError("order must be 'input', 'raster' or 'sparse_first'")


def geo_exact(ideal_positions: np.ndarray, legal_sites: np.ndarray,
              pitch_um: float | None = None, cell_um: float | None = None,
              block: int = 8, order: str = "sparse_first", variant: str = "block",
              stats: dict[str, Any] | None = None) -> Assignment:
    """Globally optimal injective assignment minimising total L1 displacement.

    ``cell_um``: grid cell size (default: the site pitch, about one site per
    cell).  ``block``: cells per block side for the block bound.  ``variant``:
    "block" (default, geo_lap2) or "ring" (geo_lap, cell-ring generation, no
    block bound or pruning; kept as an independent cross-check).
    Insertion order and variant change run time and, when several optimal
    assignments exist, which one is returned -- never the optimal total.
    """
    ideal, sites = _validate(ideal_positions, legal_sites)
    lib = _load()
    n, m = len(ideal), len(sites)
    if cell_um is None:
        cell_um = float(pitch_um) if pitch_um else _pitch_from_sites(sites)
    px = np.ascontiguousarray(ideal[:, 0], dtype=np.float64)
    py = np.ascontiguousarray(ideal[:, 1], dtype=np.float64)
    sx = np.ascontiguousarray(sites[:, 0], dtype=np.float64)
    sy = np.ascontiguousarray(sites[:, 1], dtype=np.float64)
    ordr = insertion_order(ideal, float(cell_um), order)
    if ordr is not None:
        ordr = np.ascontiguousarray(ordr, dtype=np.int32)
    out = np.empty(n, dtype=np.int32)
    st = np.zeros(8, dtype=np.int64)
    dp = ctypes.POINTER(ctypes.c_double)
    ip = ctypes.POINTER(ctypes.c_int)
    llp = ctypes.POINTER(ctypes.c_longlong)
    optr = ordr.ctypes.data_as(ip) if ordr is not None else None
    if variant == "block":
        rc = lib.geo_lap2(n, px.ctypes.data_as(dp), py.ctypes.data_as(dp), m,
                          sx.ctypes.data_as(dp), sy.ctypes.data_as(dp), float(cell_um),
                          int(block), optr, out.ctypes.data_as(ip), st.ctypes.data_as(llp))
    elif variant == "ring":
        rc = lib.geo_lap(n, px.ctypes.data_as(dp), py.ctypes.data_as(dp), m,
                         sx.ctypes.data_as(dp), sy.ctypes.data_as(dp), float(cell_um),
                         optr, out.ctypes.data_as(ip), st.ctypes.data_as(llp))
    else:
        raise ValueError("variant must be 'block' or 'ring'")
    if rc != 0:
        raise RuntimeError(f"geo_lap failed: {_ERRORS.get(rc, rc)}")
    assignment = out.astype(int)
    if np.any(assignment < 0) or len(np.unique(assignment)) != n:
        raise RuntimeError("geo_exact produced a non-injective assignment")
    if stats is not None:
        stats.clear()
        stats.update({"heap_pops": int(st[0]), "relaxations": int(st[1]),
                      "rows_scanned": int(st[2]), "ring_events": int(st[3]),
                      "max_heap": int(st[4]), "augment_edges": int(st[5]),
                      "cell_um": float(cell_um), "block": int(block),
                      "order": order, "variant": variant})
    return _finish(ideal, sites, assignment)
