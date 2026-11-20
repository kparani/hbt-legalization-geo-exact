#!/usr/bin/env python3
"""Standalone injective HBT-to-grid assignment baselines."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


Assignment = tuple[np.ndarray, float]


def _validate(ideal_positions: np.ndarray, legal_sites: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ideal = np.asarray(ideal_positions, dtype=float).reshape(-1, 2)
    sites = np.asarray(legal_sites, dtype=float).reshape(-1, 2)
    if len(ideal) > len(sites):
        raise ValueError(f"not enough legal sites ({len(sites)}) for HBTs ({len(ideal)})")
    return ideal, sites


def _finish(ideal: np.ndarray, sites: np.ndarray, assignment: np.ndarray) -> Assignment:
    distances = np.abs(ideal - sites[assignment]).sum(axis=1)
    return assignment, float(distances.sum())


def exact_hungarian(ideal_positions: np.ndarray, legal_sites: np.ndarray,
                    max_points: int = 2000) -> Assignment | None:
    """Solve the full Manhattan assignment; skip above ``max_points``."""
    ideal, sites = _validate(ideal_positions, legal_sites)
    if len(ideal) > max_points:
        print(f"exact_hungarian skipped: {len(ideal)} points exceeds cap {max_points}")
        return None
    cost = np.abs(ideal[:, None, :] - sites[None, :, :]).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    assignment = np.full(len(ideal), -1, dtype=int)
    assignment[rows] = cols
    return _finish(ideal, sites, assignment)


def _bucket_cell_size(sites: np.ndarray, requested: float | None) -> float:
    if requested and requested > 0:
        return float(requested)
    diffs = []
    for axis in (0, 1):
        values = np.unique(sites[:, axis])
        diffs.extend(np.diff(values)[np.diff(values) > 0].tolist())
    return float(np.median(diffs)) if diffs else 1.0


def _nearest_free(ideal: np.ndarray, sites: np.ndarray, free: set[int], cell: float,
                  buckets: dict[tuple[int, int], list[int]]) -> int:
    bx, by = np.floor(ideal / cell).astype(int)
    max_radius = max(max(abs(int(np.floor(sites[:, 0].min() / cell)) - bx),
                      abs(int(np.floor(sites[:, 0].max() / cell)) - bx)),
                     max(abs(int(np.floor(sites[:, 1].min() / cell)) - by),
                      abs(int(np.floor(sites[:, 1].max() / cell)) - by))) + 1
    best_idx, best_dist = None, float("inf")
    for radius in range(max_radius + 1):
        for ix in range(bx - radius, bx + radius + 1):
            for iy in range(by - radius, by + radius + 1):
                if max(abs(ix - bx), abs(iy - by)) != radius:
                    continue
                for idx in buckets.get((ix, iy), ()):
                    if idx not in free:
                        continue
                    dist = float(np.abs(ideal - sites[idx]).sum())
                    if dist < best_dist or (dist == best_dist and idx < (best_idx or idx + 1)):
                        best_idx, best_dist = idx, dist
        if best_idx is not None and best_dist <= max(0.0, (radius - 1) * cell):
            break
    if best_idx is None:
        raise RuntimeError("spatial hash could not find a free legal site")
    return best_idx


def greedy_nearest(ideal_positions: np.ndarray, legal_sites: np.ndarray,
                   cell_size: float | None = None) -> Assignment:
    """Greedy nearest-site assignment using spatial-hash buckets."""
    ideal, sites = _validate(ideal_positions, legal_sites)
    cell = _bucket_cell_size(sites, cell_size)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, point in enumerate(sites):
        buckets[tuple(np.floor(point / cell).astype(int))].append(idx)
    free = set(range(len(sites)))
    assignment = np.empty(len(ideal), dtype=int)
    order = np.argsort(ideal[:, 0] + ideal[:, 1], kind="stable")
    for row in order:
        idx = _nearest_free(ideal[row], sites, free, cell, buckets)
        assignment[row] = idx
        free.remove(idx)
    return _finish(ideal, sites, assignment)


def sliding_window(ideal_positions: np.ndarray, legal_sites: np.ndarray,
                   window_size_um: float | None = None, stride_um: float | None = None,
                   pitch_um: float | None = None,
                   dbu_per_um: int | None = None,
                   stats: dict[str, int] | None = None) -> Assignment:
    """ViaLegal-style overlapping tiled bipartite matching in microns.

    With ``pitch_um`` supplied, the published ViaLegal defaults are used:
    tile width/height = ``30 * pitch``, window = ``3 x 3`` tiles, and stride
    = ``2`` tiles. Explicit ``window_size_um``/``stride_um`` values are useful
    for experiments and are interpreted directly in microns.

    The paper says to update via locations after each window and to allow
    reassignment in later overlapping windows, but does not fully specify the
    traversal origin, boundary clipping, or convergence criterion. This
    implementation makes the following explicit choices: scan windows in
    row-major order from the lower-left legal-site coordinate; include points
    on both window boundaries; exclude sites owned by HBTs outside the current
    window; and perform one complete sweep, which is the paper's Algorithm 1
    procedure rather than an unbounded convergence loop.

    Costs are pure geometric Manhattan distances from each HBT's original
    ideal position to a candidate site. The timing-weighted variant described
    by ViaLegal is intentionally omitted because this benchmark has no timing
    data.
    """
    ideal, sites = _validate(ideal_positions, legal_sites)
    if pitch_um is not None and pitch_um <= 0:
        raise ValueError("pitch_um must be positive")
    if window_size_um is None:
        if pitch_um is None:
            raise ValueError("provide pitch_um or window_size_um")
        window_size_um = 3.0 * 30.0 * pitch_um
    if stride_um is None:
        if pitch_um is None:
            raise ValueError("provide pitch_um or stride_um")
        stride_um = 2.0 * 30.0 * pitch_um
    if window_size_um <= 0 or stride_um <= 0:
        raise ValueError("window_size_um and stride_um must be positive")
    if dbu_per_um not in (None, 1):
        raise ValueError("sliding_window expects micron coordinates; do not pass a DBU scale")
    window = float(window_size_um)
    stride = float(stride_um)

    # Coordinates are real microns at this function boundary. The legal grid
    # starts the tiled canvas; using sites rather than ideal points keeps the
    # window geometry tied to the manufacturing grid.
    origin = sites.min(axis=0).astype(float)
    max_site = sites.max(axis=0)
    x_origins = np.arange(origin[0], max_site[0] + stride * 0.5, stride)
    y_origins = np.arange(origin[1], max_site[1] + stride * 0.5, stride)

    # assignment[v] is the currently owned legal-site index, or -1 before the
    # first window reaches v. current[v] is the location queried by later
    # overlapping windows and is updated after every local LAP.
    assignment = np.full(len(ideal), -1, dtype=int)
    current = ideal.copy()
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
            rows = np.flatnonzero(np.all((current >= lower) & (current <= upper), axis=1))
            if len(rows) == 0:
                continue
            if stats is not None:
                stats["windows_with_rows"] += 1

            old_assignments = assignment[rows].copy()

            # A site currently held by a via outside this window is not
            # available. Sites held by vias inside the window are released and
            # can therefore participate in reassignment.
            inside = np.zeros(len(ideal), dtype=bool)
            inside[rows] = True
            occupied_outside = np.zeros(len(sites), dtype=bool)
            occupied = owner >= 0
            occupied_outside[occupied] = ~inside[owner[occupied]]
            candidates = np.flatnonzero(
                np.all((sites >= lower) & (sites <= upper), axis=1)
                & ~occupied_outside
            )
            if len(candidates) == 0:
                continue

            # Normal benchmark windows have more sites than HBTs. If a
            # pathological boundary window does not, retain all HBTs and let
            # the next overlap repair the unassigned ones rather than silently
            # losing an assignment.
            cost = np.abs(ideal[rows, None, :] - sites[candidates][None, :, :]).sum(axis=2)
            matched_rows, matched_sites = linear_sum_assignment(cost)
            for row in rows:
                old = assignment[row]
                if old >= 0 and owner[old] == row:
                    owner[old] = -1
                assignment[row] = -1
                current[row] = ideal[row]
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
                current[row] = sites[site]

    # A legal grid should make this unnecessary, but finish points that never
    # entered a clipped window using the same injective spatial-hash fallback.
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
