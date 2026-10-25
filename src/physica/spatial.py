"""Deterministic 2D broad-phase AABB collision *candidates*, not contacts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class AABB:
    """Immutable 2D bounds with finite built-in int/float coordinates."""

    id: str
    lower: tuple[float, float]
    upper: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise ValueError("AABB id must be a nonempty string")
        # Strip a string subclass's overridden ordering/hash behavior without
        # rejecting valid StrEnum or other string-compatible input values.
        canonical_id = str.__str__(self.id)
        if not canonical_id:
            raise ValueError("AABB id must be a nonempty string")
        object.__setattr__(self, "id", canonical_id)
        try:
            if len(self.lower) != 2 or len(self.upper) != 2:
                raise ValueError("AABB must have 2D lower and upper coordinates")
            # A frozen dataclass does not freeze an aliased caller-owned list.
            # Read exactly two coordinates and retain immutable tuples instead.
            lower = (self.lower[0], self.lower[1])
            upper = (self.upper[0], self.upper[1])
        except (TypeError, IndexError, KeyError) as exc:
            raise ValueError("AABB must have 2D lower and upper coordinates") from exc
        if any(type(x) not in (int, float) for x in lower + upper):
            raise ValueError("AABB coordinates must be finite built-in int/float numbers")
        try:
            finite = all(isfinite(x) for x in lower + upper)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("AABB coordinates must be finite numbers") from exc
        if not finite:
            raise ValueError("AABB coordinates must be finite")
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise ValueError("AABB bounds cannot be reversed")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


def _validate(bounds: Sequence[AABB]) -> tuple[AABB, ...]:
    result = tuple(bounds)
    if any(not isinstance(box, AABB) for box in result):
        raise ValueError("all bounds must be AABB instances")
    if len({box.id for box in result}) != len(result):
        raise ValueError("AABB ids must be unique")
    return result


def broad_phase_pairs(bounds: Sequence[AABB]) -> tuple[tuple[str, str], ...]:
    """Find 2D *candidate* collisions using a sorted x-axis sweep.

    Touching AABBs count as candidates. This routine does not determine actual
    shape collision, penetration, contact force, or time of impact. Results are
    sorted by identifier and deterministic across input orders.
    """
    items = sorted(_validate(bounds), key=lambda b: (b.lower[0], b.id))
    active: list[AABB] = []
    candidates: set[tuple[str, str]] = set()
    for current in items:
        active = [prior for prior in active if prior.upper[0] >= current.lower[0]]
        for prior in active:
            if (prior.lower[1] <= current.upper[1]
                    and current.lower[1] <= prior.upper[1]):
                candidates.add(tuple(sorted((prior.id, current.id))))
        active.append(current)
    return tuple(sorted(candidates))


def brute_force_pairs(bounds: Sequence[AABB]) -> tuple[tuple[str, str], ...]:
    """Quadratic exhaustive AABB-overlap reference for broad-phase testing."""
    items = _validate(bounds)
    pairs: list[tuple[str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            left, right = items[i], items[j]
            if (left.lower[0] <= right.upper[0]
                and right.lower[0] <= left.upper[0]
                and left.lower[1] <= right.upper[1]
                and right.lower[1] <= left.upper[1]):
                pairs.append(tuple(sorted((left.id, right.id))))
    return tuple(sorted(pairs))


def swept_broad_phase_pairs(
    bounds: Sequence[AABB],
    displacements: Mapping[str, tuple[float, float]],
) -> tuple[tuple[str, str], ...]:
    """Conservative moving 2D AABB *candidates*, not contacts or impact times.

    Each AABB translates linearly by its displacement over one abstract step.
    The union of starting/ending axis bounds encloses every intermediate AABB.
    Applying broad_phase_pairs to these envelopes cannot miss a simultaneous
    overlap, but it may include pairs that never overlap at the same instant.
    No movement, collision resolution or state mutation is performed.
    """
    items = _validate(bounds)
    ids = {box.id for box in items}
    if not isinstance(displacements, Mapping) or set(displacements) != ids:
        raise ValueError("displacements must exactly match AABB IDs")
    # Keep the swept bounds exact. Python silently rounds a large int when it
    # is added to a float displacement (e.g. 2**53+1 + 1.0), which can shrink
    # an envelope and make a supposedly conservative broad phase miss a pair.
    # Tuples are (id, x_min, y_min, x_max, y_max) in exact rational values.
    envelopes: list[tuple[str, Fraction, Fraction, Fraction, Fraction]] = []
    for box in items:
        motion = displacements[box.id]
        if type(motion) is not tuple or len(motion) != 2:
            raise ValueError("displacement must be a finite 2D tuple")
        try:
            finite_motion = all(type(v) in (int, float) and isfinite(v)
                                for v in motion)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("displacement must be a finite 2D tuple") from exc
        if not finite_motion:
            raise ValueError("displacement must be a finite 2D tuple")

        delta = tuple(Fraction(v) for v in motion)
        lower = tuple(Fraction(v) for v in box.lower)
        upper = tuple(Fraction(v) for v in box.upper)
        end_lower = tuple(lower[axis] + delta[axis] for axis in (0, 1))
        end_upper = tuple(upper[axis] + delta[axis] for axis in (0, 1))
        try:
            finite_endpoints = all(isfinite(float(v))
                                   for v in end_lower + end_upper)
        except (ValueError, OverflowError) as exc:
            raise ValueError("translated AABB bounds must remain finite") from exc
        if not finite_endpoints:
            raise ValueError("translated AABB bounds must remain finite")
        envelopes.append((
            box.id,
            min(lower[0], end_lower[0]), min(lower[1], end_lower[1]),
            max(upper[0], end_upper[0]), max(upper[1], end_upper[1]),
        ))
    active: list[tuple[str, Fraction, Fraction, Fraction, Fraction]] = []
    candidates: set[tuple[str, str]] = set()
    for current in sorted(envelopes, key=lambda row: (row[1], row[0])):
        active = [prior for prior in active if prior[3] >= current[1]]
        for prior in active:
            if prior[2] <= current[4] and current[2] <= prior[4]:
                candidates.add(tuple(sorted((prior[0], current[0]))))
        active.append(current)
    return tuple(sorted(candidates))


@dataclass(frozen=True)
class SweptAABBInterval:
    """Closed normalized time interval when one candidate pair overlaps."""

    pair: tuple[str, str]
    enter_time: float
    exit_time: float


def swept_aabb_intervals(
    bounds: Sequence[AABB],
    displacements: Mapping[str, tuple[float, float]],
) -> tuple[SweptAABBInterval, ...]:
    """Analytic simultaneous-overlap intervals for linearly translating AABBs.

    Returns at most one closed interval per pair within [0, 1], including
    exact edge/corner touches. Contact predicates use exact-real arithmetic
    over input numbers (binary64 floats treated exactly, ints as integers), avoiding
    false touching caused by separately rounded floating-point translations.
    Returned float times are rounded representations of rational boundaries.

    These intervals are AABB overlaps, not shape contacts, forces, collision
    responses, or calibrated time-of-impact predictions.
    """
    items = _validate(bounds)
    ids = {box.id for box in items}
    if not isinstance(displacements, Mapping) or set(displacements) != ids:
        raise ValueError("displacements must exactly match AABB IDs")
    # A Mapping may compute values dynamically. Capture each immutable motion
    # tuple once so validation, broad-phase pruning and exact pair geometry
    # refer to the same observed trajectory for the entire calculation.
    motions = {box.id: displacements[box.id] for box in items}
    # Reuse the conservative envelope implementation for finite motion,
    # tuple shape and translated-endpoint validation before doing pair math.
    candidates = swept_broad_phase_pairs(items, motions)
    by_id = {box.id: box for box in items}
    result: list[SweptAABBInterval] = []

    for left_id, right_id in candidates:
        left, right = by_id[left_id], by_id[right_id]
        left_delta, right_delta = motions[left_id], motions[right_id]
        enter, exit = Fraction(0), Fraction(1)
        possible = True

        for axis in (0, 1):
            relative = Fraction(left_delta[axis]) - Fraction(right_delta[axis])
            near = Fraction(right.lower[axis]) - Fraction(left.upper[axis])
            far = Fraction(right.upper[axis]) - Fraction(left.lower[axis])
            if relative == 0:
                if near > 0 or far < 0:
                    possible = False
                    break
                continue

            axis_start, axis_end = sorted((near / relative, far / relative))
            enter = max(enter, axis_start)
            exit = min(exit, axis_end)
            if enter > exit:
                possible = False
                break

        if possible:
            result.append(SweptAABBInterval((left_id, right_id), float(enter), float(exit)))

    return tuple(result)
