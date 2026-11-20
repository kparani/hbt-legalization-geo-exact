#!/usr/bin/env python3
"""Density-Guided Hierarchical Legalization (DGHL).

DGHL is a deterministic, training-free HBT legalizer.  It builds a sparse
coarse contention map, schedules high-pressure regions first, solves bounded
local assignments, and performs improving refinement passes.  All coordinates
at this module boundary are real microns.

The implementation deliberately never constructs an ``N_HBT x N_sites`` cost
matrix.  ``cKDTree`` supplies bounded candidate neighborhoods and Hungarian is
used only when both local dimensions are small enough.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    from .baselines import Assignment, _validate
except ImportError:  # Support direct execution through run_benchmark.py.
    from baselines import Assignment, _validate


def _keys(points: np.ndarray, origin: np.ndarray, cell_um: float) -> np.ndarray:
    """Return integer coarse-cell keys for micron coordinates."""
    return np.floor((points - origin) / cell_um).astype(np.int64)


def _counts(keys: np.ndarray) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = defaultdict(int)
    for x, y in keys:
        result[(int(x), int(y))] += 1
    return result


def _local_candidates(tree: cKDTree, sites: np.ndarray, lower: np.ndarray,
                      upper: np.ndarray, max_candidates: int) -> np.ndarray:
    """Query a bounded rectangular micron window without scanning all sites."""
    center = (lower + upper) / 2.0
    radius = float(np.linalg.norm(upper - lower) / 2.0)
    candidates = np.asarray(tree.query_ball_point(center, radius), dtype=int)
    if len(candidates) == 0:
        return candidates
    keep = np.all((sites[candidates] >= lower) & (sites[candidates] <= upper), axis=1)
    candidates = candidates[keep]
    if len(candidates) > max_candidates:
        # Keep the sites closest to the window center. This preserves a local
        # near-exact problem while imposing a hard memory bound.
        distance = np.abs(sites[candidates] - center).sum(axis=1)
        candidates = candidates[np.argsort(distance, kind="stable")[:max_candidates]]
    return candidates


def _apply_matching(rows: np.ndarray, candidates: np.ndarray, ideal: np.ndarray,
                    sites: np.ndarray, owner: np.ndarray, assignment: np.ndarray,
                    allow_improvement_only: bool) -> int:
    """Apply an injective local LAP and return the number of site changes."""
    if len(rows) == 0 or len(candidates) < len(rows):
        return 0
    active_sites = candidates[(owner[candidates] < 0) | np.isin(owner[candidates], rows)]
    if len(active_sites) < len(rows):
        return 0
    cost = np.abs(ideal[rows, None, :] - sites[active_sites][None, :, :]).sum(axis=2)
    matched_rows, matched_sites = linear_sum_assignment(cost)
    if len(matched_rows) != len(rows):
        return 0
    proposed = assignment[rows].copy()
    proposed[matched_rows] = active_sites[matched_sites]
    old_cost = np.abs(ideal[rows] - sites[assignment[rows]]).sum() if np.all(assignment[rows] >= 0) else np.inf
    new_cost = np.abs(ideal[rows] - sites[proposed]).sum()
    if allow_improvement_only and new_cost >= old_cost - 1e-9:
        return 0

    changed = int(np.count_nonzero((assignment[rows] >= 0) & (assignment[rows] != proposed)))
    for row in rows:
        old = int(assignment[row])
        if old >= 0 and owner[old] == row:
            owner[old] = -1
    for row, site in zip(rows, proposed):
        assignment[row] = int(site)
        owner[int(site)] = int(row)
    return changed


def _nearest_candidate_assignment(rows: np.ndarray, ideal: np.ndarray, sites: np.ndarray,
                                  tree: cKDTree, owner: np.ndarray, assignment: np.ndarray,
                                  k: int = 64) -> int:
    """Assign rows using bounded KD-tree neighborhoods, expanding if needed."""
    if len(rows) == 0:
        return 0
    changed = 0
    remaining = rows.copy()
    for width in (k, min(k * 4, len(sites)), min(k * 16, len(sites))):
        if len(remaining) == 0:
            break
        _, neighbors = tree.query(ideal[remaining], k=width, workers=1)
        neighbors = np.atleast_2d(neighbors)
        next_remaining = []
        for local, row in enumerate(remaining):
            chosen = None
            for site in np.atleast_1d(neighbors[local]):
                site = int(site)
                if owner[site] < 0:
                    chosen = site
                    break
            if chosen is None:
                next_remaining.append(int(row))
                continue
            old = int(assignment[row])
            if old >= 0 and owner[old] == row:
                owner[old] = -1
            assignment[row] = chosen
            owner[chosen] = int(row)
            if old >= 0 and old != chosen:
                changed += 1
        remaining = np.asarray(next_remaining, dtype=int)
    if len(remaining):
        # Dense contention can exhaust the bounded neighborhood for a small
        # number of stragglers. Query a larger neighborhood one point at a
        # time in that case; this remains bounded by the number of stragglers
        # and never creates an HBT-by-all-sites matrix.
        if len(remaining) > 2048:
            raise RuntimeError(f"DGHL could not find free sites for {len(remaining)} HBTs")
        width = min(len(sites), max(k * 256, 16384))
        for row in remaining:
            _, neighbors = tree.query(ideal[row], k=width, workers=1)
            chosen = next((int(site) for site in np.atleast_1d(neighbors)
                           if owner[int(site)] < 0), None)
            if chosen is None:
                raise RuntimeError(f"DGHL could not find a free site for HBT {int(row)}")
            assignment[row] = chosen
            owner[chosen] = int(row)
    return changed


def dghl(ideal_positions: np.ndarray, legal_sites: np.ndarray, pitch_um: float,
         contention_threshold: float = 1.0,
         min_window_size: float = 1.0,
         initial_window_multiplier: float = 3.0,
         coarse_cell_multiplier: float = 30.0,
         shrink_factor: float = 0.5,
         max_passes: int = 2,
         max_local_hungarian_hbts: int = 256,
         max_local_sites: int = 10000,
         stats: dict[str, Any] | None = None) -> Assignment:
    """Legalize micron coordinates using density-guided hierarchical refinement.

    ``coarse_cell_multiplier`` defines the potential-map cell as a multiple of
    pitch. ``initial_window_multiplier`` defines the initial window as that
    many coarse cells per dimension. Windows shrink by ``shrink_factor`` until
    ``min_window_size`` or ``max_passes`` is reached. The contention threshold
    is the HBT/site ratio below which a region is considered non-contentious.
    These named arguments are intended to be tunable by future experiments.
    """
    ideal, sites = _validate(ideal_positions, legal_sites)
    if pitch_um <= 0 or min_window_size <= 0 or coarse_cell_multiplier <= 0:
        raise ValueError("pitch_um, min_window_size, and coarse_cell_multiplier must be positive")
    if not 0 < shrink_factor <= 1 or initial_window_multiplier <= 0 or max_passes <= 0:
        raise ValueError("invalid DGHL refinement parameters")
    tree = cKDTree(sites)
    origin = sites.min(axis=0).astype(float)
    cell = float(coarse_cell_multiplier * pitch_um)
    ideal_keys = _keys(ideal, origin, cell)
    site_keys = _keys(sites, origin, cell)
    ideal_counts = _counts(ideal_keys)
    site_counts = _counts(site_keys)
    ratios = {key: ideal_counts[key] / max(site_counts.get(key, 0), 1)
              for key in ideal_counts}

    assignment = np.full(len(ideal), -1, dtype=int)
    owner = np.full(len(sites), -1, dtype=int)
    current = ideal.copy()
    local_matches = 0
    reassignment_changes = 0
    regions_processed = 0

    for pass_index in range(max_passes):
        current_keys = _keys(current, origin, cell)
        current_counts = _counts(current_keys)
        free_counts = _counts(site_keys[owner < 0])
        region_keys = sorted(current_counts,
                             key=lambda key: (-(current_counts[key] / max(free_counts.get(key, 0), 1)),
                                               -ratios.get(key, 0.0), key))
        window = max(float(min_window_size),
                     float(initial_window_multiplier * cell) * (shrink_factor ** pass_index))
        for key in region_keys:
            center = origin + (np.asarray(key, dtype=float) + 0.5) * cell
            lower = center - window / 2.0
            upper = center + window / 2.0
            rows = np.flatnonzero(np.all((current >= lower) & (current <= upper), axis=1))
            if len(rows) == 0:
                continue
            if pass_index == 0:
                rows = rows[assignment[rows] < 0]
            candidates = _local_candidates(tree, sites, lower, upper, max_local_sites)
            if len(candidates) == 0:
                continue
            if len(rows) <= max_local_hungarian_hbts and len(candidates) <= max_local_sites:
                changed = _apply_matching(rows, candidates, ideal, sites, owner, assignment,
                                          allow_improvement_only=pass_index > 0)
                if changed or pass_index == 0:
                    local_matches += len(rows)
                    reassignment_changes += changed
            else:
                if pass_index == 0:
                    changed = _nearest_candidate_assignment(rows, ideal, sites, tree, owner, assignment)
                    reassignment_changes += changed
            regions_processed += 1
            # A high-ratio region is allowed to receive the full initial
            # window. Sparse regions are handled by the bounded nearest path.
            if ratios.get(key, 0.0) < contention_threshold and pass_index > 0:
                continue

        current = np.where(assignment[:, None] >= 0, sites[np.maximum(assignment, 0)], ideal)

    leftovers = np.flatnonzero(assignment < 0)
    if len(leftovers):
        reassignment_changes += _nearest_candidate_assignment(leftovers, ideal, sites, tree, owner, assignment)
    if np.any(assignment < 0) or len(np.unique(assignment)) != len(assignment):
        raise RuntimeError("DGHL produced a non-injective assignment")
    if stats is not None:
        stats.clear()
        stats.update({"regions_processed": regions_processed,
                      "local_matches": local_matches,
                      "reassignment_changes": reassignment_changes,
                      "contentious_regions": sum(r >= contention_threshold for r in ratios.values()),
                      "coarse_cell_um": cell,
                      "initial_window_um": initial_window_multiplier * cell})
    distances = np.abs(ideal - sites[assignment]).sum(axis=1)
    return assignment, float(distances.sum())
