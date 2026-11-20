#!/usr/bin/env python3
"""Fast HBT legalization methods for the "SW accuracy at DGHL speed" study.

Two methods, both with the same signature style as ``baselines.py``:

``sliding_window_indexed``
    The exact ViaLegal-style sliding-window algorithm from ``baselines.py``,
    re-engineered with a tile index so each window touches only nearby
    terminals and sites instead of scanning every terminal and site.  It is
    meant to return a bit-identical assignment to ``baselines.sliding_window``
    (same rows, same candidate order, same LAP inputs).  This is the
    "fair-speed" baseline requested in the review.

``sparse_exact``
    One global min-cost bipartite matching restricted to each terminal's k
    nearest legal sites (Manhattan metric), solved with SciPy's sparse LAPJV
    (``min_weight_full_bipartite_matching``).  If the restricted graph has no
    matching that covers every terminal, k is doubled for the whole problem
    until it does.  The result is optimal on the restricted graph and becomes
    the global optimum once k is large enough.

All coordinates are microns.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

try:
    from .baselines import Assignment, _validate, _finish, _bucket_cell_size, _nearest_free
except ImportError:  # direct execution
    from baselines import Assignment, _validate, _finish, _bucket_cell_size, _nearest_free


# ---------------------------------------------------------------------------
# Indexed sliding window
# ---------------------------------------------------------------------------

class _TileIndex:
    """Static or dynamic tile buckets keyed by integer (ix, iy)."""

    def __init__(self, origin: np.ndarray, tile: float):
        self.origin = origin
        self.tile = tile

    def keys(self, points: np.ndarray) -> np.ndarray:
        return np.floor((points - self.origin) / self.tile).astype(np.int64)

    def key_range(self, lower: np.ndarray, upper: np.ndarray) -> tuple[range, range]:
        lo = np.floor((lower - self.origin) / self.tile).astype(np.int64) - 1
        hi = np.floor((upper - self.origin) / self.tile).astype(np.int64) + 1
        return range(int(lo[0]), int(hi[0]) + 1), range(int(lo[1]), int(hi[1]) + 1)


def _components(n_rows: int, nbr: np.ndarray, n_cols: int) -> np.ndarray:
    """Connected-component label per row of the row-site bipartite kNN graph."""
    from scipy.sparse.csgraph import connected_components
    k = nbr.shape[1]
    r = np.repeat(np.arange(n_rows), k)
    c = nbr.ravel() + n_rows
    g = csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)),
                   shape=(n_rows + n_cols, n_rows + n_cols))
    _, lab = connected_components(g, directed=False)
    return lab[:n_rows]


def _pruned_window_lap(P: np.ndarray, S: np.ndarray, K: int,
                       stats: dict[str, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """LAP for one window using each row's K nearest sites, solved per component.

    If a component has fewer distinct sites than rows, K is doubled for that
    component's rows until it is feasible (or the full window is used).
    Returns (local_row_indices, local_site_indices) like linear_sum_assignment.
    """
    r = len(P)
    m = len(S)
    if r == 0:
        return np.empty(0, int), np.empty(0, int)
    tree = cKDTree(S)
    kk = min(K, m)
    _, nbr = tree.query(P, k=kk, p=1, workers=-1)
    nbr = np.asarray(nbr).reshape(r, kk)
    lab = _components(r, nbr, m)
    out_r: list[np.ndarray] = []
    out_c: list[np.ndarray] = []
    for comp in np.unique(lab):
        rows = np.flatnonzero(lab == comp)
        k2 = kk
        while True:
            if k2 >= m:
                cols = np.arange(m)
            else:
                if k2 == kk:
                    cols = np.unique(nbr[rows])
                else:
                    _, nb2 = tree.query(P[rows], k=k2, p=1, workers=-1)
                    cols = np.unique(np.asarray(nb2).reshape(len(rows), k2))
            if len(cols) >= len(rows):
                break
            k2 = min(k2 * 2, m)
            if stats is not None:
                stats["prune_widen"] = stats.get("prune_widen", 0) + 1
        cost = np.abs(P[rows, None, :] - S[cols][None, :, :]).sum(axis=2)
        mr, mc = linear_sum_assignment(cost)
        out_r.append(rows[mr])
        out_c.append(cols[mc])
        if stats is not None:
            stats["prune_components"] = stats.get("prune_components", 0) + 1
            stats["prune_max_rows"] = max(stats.get("prune_max_rows", 0), len(rows))
    rr = np.concatenate(out_r)
    cc = np.concatenate(out_c)
    o = np.argsort(rr, kind="stable")
    return rr[o], cc[o]


def sliding_window_indexed(ideal_positions: np.ndarray, legal_sites: np.ndarray,
                           window_size_um: float | None = None,
                           stride_um: float | None = None,
                           pitch_um: float | None = None,
                           stats: dict[str, int] | None = None,
                           prune_k: int | None = None) -> Assignment:
    """Tile-indexed version of ``baselines.sliding_window``.

    With ``prune_k=None`` this reproduces ``baselines.sliding_window``.  With
    ``prune_k=K`` each window's LAP is restricted to each terminal's K nearest
    available sites and split into independent connected components (see
    ``_pruned_window_lap``); this is the fast variant under study.

    Same window schedule, same inclusive boundaries, same rule that sites held
    by terminals outside the window are unavailable, same single sweep and
    same fallback.  Row and candidate index arrays are kept in ascending
    order so ``linear_sum_assignment`` sees exactly the same cost matrix.
    """
    ideal, sites = _validate(ideal_positions, legal_sites)
    if window_size_um is None:
        if pitch_um is None:
            raise ValueError("provide pitch_um or window_size_um")
        window_size_um = 3.0 * 30.0 * pitch_um
    if stride_um is None:
        if pitch_um is None:
            raise ValueError("provide pitch_um or stride_um")
        stride_um = 2.0 * 30.0 * pitch_um
    window = float(window_size_um)
    stride = float(stride_um)

    origin = sites.min(axis=0).astype(float)
    max_site = sites.max(axis=0)
    x_origins = np.arange(origin[0], max_site[0] + stride * 0.5, stride)
    y_origins = np.arange(origin[1], max_site[1] + stride * 0.5, stride)

    # Tile size: one third of the window keeps each query to ~5x5 tiles.
    index = _TileIndex(origin, window / 3.0)

    # Static site buckets (ascending site ids inside each bucket).
    site_keys = index.keys(sites)
    site_buckets: dict[tuple[int, int], np.ndarray] = {}
    order = np.lexsort((np.arange(len(sites)), site_keys[:, 1], site_keys[:, 0]))
    sk = site_keys[order]
    if len(order):
        breaks = np.flatnonzero(np.any(np.diff(sk, axis=0) != 0, axis=1)) + 1
        for chunk in np.split(order, breaks):
            kx, ky = site_keys[chunk[0]]
            site_buckets[(int(kx), int(ky))] = np.sort(chunk)

    # Dynamic row buckets over "current" positions.
    current = ideal.copy()
    row_key = index.keys(current)
    row_buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for r, (kx, ky) in enumerate(row_key):
        row_buckets[(int(kx), int(ky))].add(r)

    def move_row(r: int, new_pos: np.ndarray) -> None:
        current[r] = new_pos
        nk = np.floor((new_pos - origin) / index.tile).astype(np.int64)
        ok = row_key[r]
        if nk[0] != ok[0] or nk[1] != ok[1]:
            row_buckets[(int(ok[0]), int(ok[1]))].discard(r)
            row_buckets[(int(nk[0]), int(nk[1]))].add(r)
            row_key[r] = nk

    assignment = np.full(len(ideal), -1, dtype=int)
    owner = np.full(len(sites), -1, dtype=int)
    if stats is not None:
        stats.clear()
        stats.update({"windows_considered": 0, "windows_with_rows": 0,
                      "local_matches": 0, "reassignment_changes": 0})

    for y0 in y_origins:
        for x0 in x_origins:
            if stats is not None:
                stats["windows_considered"] += 1
            lower = np.array([x0, y0])
            upper = lower + window
            xr, yr = index.key_range(lower, upper)

            gathered: list[int] = []
            site_parts: list[np.ndarray] = []
            for ix in xr:
                for iy in yr:
                    bucket = row_buckets.get((ix, iy))
                    if bucket:
                        gathered.extend(bucket)
                    sb = site_buckets.get((ix, iy))
                    if sb is not None:
                        site_parts.append(sb)
            if not gathered:
                continue
            cand_rows = np.fromiter(gathered, dtype=np.int64, count=len(gathered))
            cand_rows.sort()
            pts = current[cand_rows]
            rows = cand_rows[np.all((pts >= lower) & (pts <= upper), axis=1)]
            if len(rows) == 0:
                continue
            if stats is not None:
                stats["windows_with_rows"] += 1

            old_assignments = assignment[rows].copy()

            if site_parts:
                cand_sites = np.concatenate(site_parts)
                cand_sites.sort()
            else:
                cand_sites = np.empty(0, dtype=np.int64)
            sp = sites[cand_sites]
            in_box = np.all((sp >= lower) & (sp <= upper), axis=1)
            cand_sites = cand_sites[in_box]
            own = owner[cand_sites]
            inside = np.zeros(len(ideal), dtype=bool)
            inside[rows] = True
            occupied_outside = (own >= 0) & ~inside[np.maximum(own, 0)]
            candidates = cand_sites[~occupied_outside]
            if len(candidates) == 0:
                continue

            if prune_k is None:
                cost = np.abs(ideal[rows, None, :] - sites[candidates][None, :, :]).sum(axis=2)
                matched_rows, matched_sites = linear_sum_assignment(cost)
            else:
                matched_rows, matched_sites = _pruned_window_lap(
                    ideal[rows], sites[candidates], prune_k, stats)
            for row in rows:
                old = assignment[row]
                if old >= 0 and owner[old] == row:
                    owner[old] = -1
                assignment[row] = -1
                move_row(int(row), ideal[row])
            for local_row, local_site in zip(matched_rows, matched_sites):
                row = int(rows[local_row])
                site = int(candidates[local_site])
                old = int(old_assignments[local_row])
                if stats is not None:
                    stats["local_matches"] += 1
                    if old >= 0 and old != site:
                        stats["reassignment_changes"] += 1
                if old >= 0 and owner[old] == row:
                    owner[old] = -1
                assignment[row] = site
                owner[site] = row
                move_row(row, sites[site])

    leftovers = np.flatnonzero(assignment < 0)
    if len(leftovers):
        free = set(np.flatnonzero(owner < 0).tolist())
        cell = _bucket_cell_size(sites, stride)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, point in enumerate(sites):
            buckets[tuple(np.floor(point / cell).astype(int))].append(idx)
        for row in leftovers:
            idx = _nearest_free(ideal[row], sites, free, cell, buckets)
            assignment[row] = idx
            owner[idx] = row
            free.remove(idx)
    return _finish(ideal, sites, assignment)


# ---------------------------------------------------------------------------
# Sparse exact matching on a k-nearest-site graph
# ---------------------------------------------------------------------------

def _knn_graph(ideal: np.ndarray, tree: cKDTree, n_sites: int, k: int) -> csr_matrix:
    k = min(k, n_sites)
    dist, idx = tree.query(ideal, k=k, p=1, workers=-1)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    n = len(ideal)
    # +1 keeps zero-distance edges as real (non-implicit) entries; adding the
    # same constant to every edge does not change which full matching is
    # optimal, because every full matching uses exactly n edges.
    data = (dist + 1.0).ravel()
    indptr = np.arange(0, n * k + 1, k)
    return csr_matrix((data, idx.ravel(), indptr), shape=(n, n_sites))


def _seed_nearest_free(ideal: np.ndarray, sites: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Fast injective seed: nearest free site in raster order, widening k as needed."""
    n = len(ideal)
    owner = np.full(len(sites), -1, dtype=int)
    assignment = np.full(n, -1, dtype=int)
    order = np.argsort(ideal[:, 0] + ideal[:, 1], kind="stable")
    remaining = order
    width = 16
    while len(remaining):
        width = min(width, len(sites))
        _, nb = tree.query(ideal[remaining], k=width, p=1, workers=-1)
        nb = np.atleast_2d(nb)
        nxt = []
        for local, row in enumerate(remaining):
            for s in nb[local]:
                if owner[s] < 0:
                    owner[s] = row
                    assignment[row] = s
                    break
            else:
                nxt.append(row)
        remaining = np.asarray(nxt, dtype=int)
        if width == len(sites) and len(remaining):
            raise RuntimeError("seed: ran out of sites")
        width *= 4
    return assignment


def sparse_exact_adaptive(ideal_positions: np.ndarray, legal_sites: np.ndarray,
                          k0: int = 8, max_k: int = 4096, max_rounds: int = 12,
                          stats: dict[str, Any] | None = None) -> Assignment:
    """Global optimal matching on a per-terminal adaptive k-nearest-site graph.

    Every terminal starts with its ``k0`` nearest sites, plus one guaranteed
    edge to a feasible seed site so a full matching always exists.  After each
    solve, any terminal whose matched site is not strictly closer than the
    farthest of its k candidates (it "hit the edge" of its list) has k doubled.
    The loop stops when no terminal hits the edge, or at ``max_rounds``.
    """
    ideal, sites = _validate(ideal_positions, legal_sites)
    n, m = len(ideal), len(sites)
    tree = cKDTree(sites)
    seed = _seed_nearest_free(ideal, sites, tree)
    seed_cost = np.abs(ideal - sites[seed]).sum(axis=1)

    ks = np.full(n, min(k0, m), dtype=int)
    cache_k = 0
    cache_d = cache_i = None
    rounds = 0
    edges_last = 0
    while True:
        rounds += 1
        kmax = int(ks.max())
        if kmax > cache_k:
            cache_d, cache_i = tree.query(ideal, k=kmax, p=1, workers=-1)
            cache_d = np.atleast_2d(cache_d).reshape(n, kmax)
            cache_i = np.atleast_2d(cache_i).reshape(n, kmax)
            cache_k = kmax
        cols_mask = np.arange(cache_k)[None, :] < ks[:, None]
        lens = ks + 1
        indptr = np.concatenate(([0], np.cumsum(lens)))
        idx = np.empty(indptr[-1], dtype=np.int64)
        dat = np.empty(indptr[-1], dtype=float)
        # kNN part
        flat_i = cache_i[cols_mask]
        flat_d = cache_d[cols_mask]
        starts = indptr[:-1]
        pos = np.repeat(starts, ks) + (np.arange(len(flat_i)) - np.repeat(np.cumsum(ks) - ks, ks))
        idx[pos] = flat_i
        dat[pos] = flat_d + 1.0
        # seed edge last in each row (duplicates of a kNN edge are harmless:
        # same column twice is summed by csr, so drop dup by sending it to the
        # kNN edge's own cost instead)
        idx[indptr[1:] - 1] = seed
        dat[indptr[1:] - 1] = seed_cost + 1.0
        graph = csr_matrix((dat, idx, indptr), shape=(n, m))
        graph.sum_duplicates()  # merges seed edge with identical kNN edge
        # sum_duplicates doubled the cost of merged pairs; restore exact costs.
        graph.data = np.abs(ideal[np.repeat(np.arange(n), np.diff(graph.indptr))]
                            - sites[graph.indices]).sum(axis=1) + 1.0
        edges_last = graph.nnz
        rows, cols = min_weight_full_bipartite_matching(graph)
        assignment = np.full(n, -1, dtype=int)
        assignment[rows] = cols
        matched = np.abs(ideal - sites[assignment]).sum(axis=1)
        radius = cache_d[np.arange(n), ks - 1]
        hit = (matched >= radius - 1e-9) & (ks < min(max_k, m))
        if not hit.any() or rounds >= max_rounds:
            break
        ks[hit] = np.minimum(ks[hit] * 2, min(max_k, m))
    if np.any(assignment < 0) or len(np.unique(assignment)) != n:
        raise RuntimeError("sparse_exact_adaptive produced a non-injective assignment")
    if stats is not None:
        stats.clear()
        stats.update({"rounds": rounds, "k_max": int(ks.max()), "k_mean": float(ks.mean()),
                      "edges_last": int(edges_last), "edge_rows_left": int(hit.sum())})
    return _finish(ideal, sites, assignment)


def sparse_exact(ideal_positions: np.ndarray, legal_sites: np.ndarray,
                 k: int = 16, max_k: int = 4096,
                 stats: dict[str, Any] | None = None) -> Assignment:
    """Global optimal matching restricted to each terminal's k nearest sites."""
    ideal, sites = _validate(ideal_positions, legal_sites)
    tree = cKDTree(sites)
    tries = 0
    kk = int(k)
    while True:
        tries += 1
        graph = _knn_graph(ideal, tree, len(sites), kk)
        try:
            rows, cols = min_weight_full_bipartite_matching(graph)
            break
        except ValueError:
            if kk >= min(max_k, len(sites)):
                raise RuntimeError(f"sparse_exact: no full matching even at k={kk}")
            kk = min(kk * 2, max_k, len(sites))
    assignment = np.full(len(ideal), -1, dtype=int)
    assignment[rows] = cols
    if np.any(assignment < 0) or len(np.unique(assignment)) != len(assignment):
        raise RuntimeError("sparse_exact produced a non-injective assignment")
    if stats is not None:
        stats.clear()
        stats.update({"k_requested": int(k), "k_used": kk, "tries": tries})
    return _finish(ideal, sites, assignment)
