#!/usr/bin/env python3
"""A bounded BTAssign-style hierarchical legal-grid assignment baseline.

The public paper description does not specify the quadtree stopping rule,
boundary ownership, or how assignments are merged across levels.  This
implementation makes those choices explicit: quadtree nodes own their lower
and left boundaries and exclude their upper and right boundaries except at
the root; leaves/mergeable nodes solve a Manhattan Hungarian problem; and a
node rematches all rows in its subtree after its children have been solved
when the bounded candidate problem fits.  Large nodes recurse into children.

The implementation is deliberately a baseline, not a DGHL change.  It uses
bounded candidate-site sets at large nodes and a deterministic nearest-free
fallback only when a node cannot be split further.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    from .baselines import Assignment, _validate
except ImportError:  # pragma: no cover
    from baselines import Assignment, _validate


def _finish(ideal: np.ndarray, sites: np.ndarray, assignment: np.ndarray) -> Assignment:
    if np.any(assignment < 0) or len(np.unique(assignment)) != len(assignment):
        raise RuntimeError("BTAssign produced a non-injective assignment")
    distance = np.abs(ideal - sites[assignment]).sum(axis=1)
    return assignment, float(distance.sum())


def _candidate_sites(rows: np.ndarray, site_ids: np.ndarray, ideal: np.ndarray,
                     sites: np.ndarray, tree: cKDTree, limit: int) -> np.ndarray:
    """Keep a deterministic bounded union of nearest sites for a node."""
    if len(site_ids) <= limit:
        return site_ids
    k = min(len(site_ids), max(32, int(np.ceil(limit / max(len(rows), 1))) * 2))
    _, local = tree.query(ideal[rows], k=k, workers=1)
    local = np.unique(np.asarray(local, dtype=int).reshape(-1))
    chosen = local[np.isin(local, site_ids)]
    if len(chosen) < len(rows):
        # This is only a candidate-pool guard; the caller will use the full
        # node or nearest-free fallback if the pool cannot be injective.
        return site_ids
    if len(chosen) > limit:
        center = ideal[rows].mean(axis=0)
        order = np.argsort(np.abs(sites[chosen] - center).sum(axis=1), kind="stable")
        chosen = chosen[order[:limit]]
    return chosen


def _nearest_free(rows: np.ndarray, ideal: np.ndarray, sites: np.ndarray,
                  tree: cKDTree, owner: np.ndarray, assignment: np.ndarray) -> None:
    """Deterministic injective fallback for an unsplittable node."""
    order = np.argsort(ideal[rows, 0] + ideal[rows, 1], kind="stable")
    width = min(len(sites), 4096)
    for row in rows[order]:
        _, neighbours = tree.query(ideal[row], k=width, workers=1)
        chosen = next((int(idx) for idx in np.atleast_1d(neighbours)
                       if owner[int(idx)] < 0), None)
        if chosen is None:
            raise RuntimeError(f"BTAssign could not find a free site for row {int(row)}")
        assignment[row] = chosen
        owner[chosen] = int(row)


def _overlap_reassign(ideal: np.ndarray, sites: np.ndarray, tree: cKDTree,
                      assignment: np.ndarray, owner: np.ndarray,
                      lower: np.ndarray, upper: np.ndarray,
                      max_rows: int, max_sites: int,
                      levels: int = 2, passes: int = 2) -> tuple[int, int]:
    """Improve assignments in overlapping quadtree neighborhoods.

    Each level visits two-cell windows with one-cell stride. Rows near child
    boundaries therefore appear in both neighboring windows and can exchange
    sites. Only strict total-cost improvements are committed.
    """
    changes = 0
    sweeps = 0
    extent = upper - lower
    for _ in range(passes):
        pass_changes = 0
        for level in range(1, levels + 1):
            cell = extent / (2 ** level)
            if np.any(cell <= 0):
                continue
            keys = np.floor((ideal - lower) / cell).astype(int)
            occupied_keys = set(map(tuple, keys.tolist()))
            windows = set()
            for ix, iy in occupied_keys:
                for dx in (-1, 0):
                    for dy in (-1, 0):
                        windows.add((ix + dx, iy + dy))
            for ix, iy in sorted(windows):
                lo = lower + np.array([ix, iy], dtype=float) * cell
                hi = np.minimum(upper, lo + 2.0 * cell)
                lo = np.maximum(lower, lo)
                if np.any(hi <= lo):
                    continue
                current = sites[np.maximum(assignment, 0)]
                in_window = (np.all((ideal >= lo) & (ideal <= hi), axis=1) |
                             np.all((current >= lo) & (current <= hi), axis=1))
                ideal_frac = (ideal - lower) / cell
                boundary_distance = np.min(np.abs(ideal_frac - np.round(ideal_frac)), axis=1) * float(np.min(cell))
                rows = np.flatnonzero(in_window & (boundary_distance <= 0.25 * np.min(cell)))
                if len(rows) == 0:
                    continue
                candidate_ids = np.flatnonzero(np.all((sites >= lo) & (sites <= hi), axis=1))
                if len(candidate_ids) == 0:
                    continue
                edge_distance = np.min(np.column_stack((
                    np.abs(ideal[rows, 0] - lo[0]),
                    np.abs(ideal[rows, 0] - hi[0]),
                    np.abs(ideal[rows, 1] - lo[1]),
                    np.abs(ideal[rows, 1] - hi[1]))), axis=1)
                order = np.argsort(edge_distance, kind="stable")
                for start in range(0, len(rows), max_rows):
                    batch = rows[order[start:start + max_rows]]
                    candidates = _candidate_sites(batch, candidate_ids, ideal,
                                                  sites, tree, max_sites)
                    current_sites = assignment[batch].copy()
                    candidates = np.unique(np.concatenate((candidates, current_sites)))
                    if len(candidates) > max_sites:
                        continue
                    active = candidates[(owner[candidates] < 0) |
                                         np.isin(owner[candidates], batch)]
                    if len(active) < len(batch):
                        continue
                    old_cost = np.abs(ideal[batch] - sites[current_sites]).sum()
                    cost = np.abs(ideal[batch, None, :] - sites[active][None, :, :]).sum(axis=2)
                    matched_rows, matched_sites = linear_sum_assignment(cost)
                    proposed = current_sites.copy()
                    proposed[matched_rows] = active[matched_sites]
                    new_cost = np.abs(ideal[batch] - sites[proposed]).sum()
                    if new_cost >= old_cost - 1e-9:
                        continue
                    for row in batch:
                        old = int(assignment[row])
                        if old >= 0 and owner[old] == row:
                            owner[old] = -1
                    for row, site in zip(batch, proposed):
                        assignment[row] = int(site)
                        owner[int(site)] = int(row)
                    pass_changes += int(np.count_nonzero(proposed != current_sites))
            sweeps += 1
        changes += pass_changes
        if pass_changes == 0:
            break
    return changes, sweeps

def btassign(ideal_positions: np.ndarray, legal_sites: np.ndarray, pitch_um: float,
             max_cell_rows: int = 512, max_cell_sites: int = 10000,
             stats: dict[str, Any] | None = None) -> Assignment:
    """Assign HBTs using bounded quadtree Hungarian matching.

    ``pitch_um`` is part of the common benchmark API; the quadtree itself is
    geometric and therefore does not require it beyond validation.
    """
    ideal, sites = _validate(ideal_positions, legal_sites)
    if pitch_um <= 0 or max_cell_rows <= 0 or max_cell_sites <= 0:
        raise ValueError("invalid BTAssign parameters")
    tree = cKDTree(sites)
    assignment = np.full(len(ideal), -1, dtype=int)
    owner = np.full(len(sites), -1, dtype=int)
    nodes = 0
    hungarian_nodes = 0
    fallback_rows = 0

    lower = np.minimum(sites.min(axis=0), ideal.min(axis=0))
    upper = np.maximum(sites.max(axis=0), ideal.max(axis=0))

    def solve(rows: np.ndarray, site_ids: np.ndarray,
              lo: np.ndarray, hi: np.ndarray, depth: int = 0) -> None:
        nonlocal nodes, hungarian_nodes, fallback_rows
        if len(rows) == 0:
            return
        nodes += 1
        if len(rows) <= max_cell_rows:
            candidates = _candidate_sites(rows, site_ids, ideal, sites, tree, max_cell_sites)
            if len(candidates) <= max_cell_sites and len(candidates) >= len(rows):
                # Child assignments are provisional; rematching here is the
                # bottom-up merge step of this implementation.
                for row in rows:
                    old = int(assignment[row])
                    if old >= 0 and owner[old] == row:
                        owner[old] = -1
                    assignment[row] = -1
                active = candidates[owner[candidates] < 0]
                if len(active) >= len(rows):
                    cost = np.abs(ideal[rows, None, :] - sites[active][None, :, :]).sum(axis=2)
                    matched_rows, matched_sites = linear_sum_assignment(cost)
                    for local_row, local_site in zip(matched_rows, matched_sites):
                        row = int(rows[local_row])
                        site = int(active[local_site])
                        assignment[row] = site
                        owner[site] = row
                    if np.all(assignment[rows] >= 0):
                        hungarian_nodes += 1
                        return
        if depth >= 24 or np.allclose(lo, hi):
            unassigned = rows[assignment[rows] < 0]
            fallback_rows += len(unassigned)
            _nearest_free(unassigned, ideal, sites, tree, owner, assignment)
            return
        mid = (lo + hi) / 2.0
        quadrants = (
            (np.array([lo[0], lo[1]]), np.array([mid[0], mid[1]]),
             (ideal[rows, 0] <= mid[0]) & (ideal[rows, 1] <= mid[1]),
             (sites[site_ids, 0] <= mid[0]) & (sites[site_ids, 1] <= mid[1])),
            (np.array([mid[0], lo[1]]), np.array([hi[0], mid[1]]),
             (ideal[rows, 0] > mid[0]) & (ideal[rows, 1] <= mid[1]),
             (sites[site_ids, 0] > mid[0]) & (sites[site_ids, 1] <= mid[1])),
            (np.array([lo[0], mid[1]]), np.array([mid[0], hi[1]]),
             (ideal[rows, 0] <= mid[0]) & (ideal[rows, 1] > mid[1]),
             (sites[site_ids, 0] <= mid[0]) & (sites[site_ids, 1] > mid[1])),
            (np.array([mid[0], mid[1]]), np.array([hi[0], hi[1]]),
             (ideal[rows, 0] > mid[0]) & (ideal[rows, 1] > mid[1]),
             (sites[site_ids, 0] > mid[0]) & (sites[site_ids, 1] > mid[1])),
        )
        before = assignment[rows].copy()
        for child_lo, child_hi, row_mask, site_mask in quadrants:
            child_rows = rows[row_mask]
            child_sites = site_ids[site_mask]
            if len(child_rows):
                solve(child_rows, child_sites, child_lo, child_hi, depth + 1)
        # If child boundary choices left any rows unassigned, repair locally.
        missing = rows[assignment[rows] < 0]
        if len(missing):
            fallback_rows += len(missing)
            _nearest_free(missing, ideal, sites, tree, owner, assignment)

    root_rows = np.arange(len(ideal), dtype=int)
    root_sites = np.arange(len(sites), dtype=int)
    solve(root_rows, root_sites, lower, upper)
    overlap_changes, overlap_sweeps = _overlap_reassign(
        ideal, sites, tree, assignment, owner, lower, upper,
        max_cell_rows, max_cell_sites)
    if stats is not None:
        stats.clear()
        stats.update({"nodes": nodes, "hungarian_nodes": hungarian_nodes,
                      "fallback_rows": fallback_rows,
                      "overlap_changes": overlap_changes,
                      "overlap_sweeps": overlap_sweeps})
    return _finish(ideal, sites, assignment)

