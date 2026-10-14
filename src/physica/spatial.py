"""Deterministic 2D broad-phase AABB collision *candidates*, not contacts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class AABB:
    id: str
    lower: tuple[float, float]
    upper: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("AABB id must be a nonempty string")
        if len(self.lower) != 2 or len(self.upper) != 2:
            raise ValueError("AABB must have 2D lower and upper coordinates")
        if any(not isfinite(x) for x in self.lower + self.upper):
            raise ValueError("AABB coordinates must be finite")
        if any(lo > hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("AABB bounds cannot be reversed")


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
    envelopes: list[AABB] = []
    for box in items:
        motion = displacements[box.id]
        if (type(motion) is not tuple or len(motion) != 2
                or any(type(v) not in (int, float) or not isfinite(v)
                       for v in motion)):
            raise ValueError("displacement must be a finite 2D tuple")
        dx, dy = motion
        end_lower = (box.lower[0] + dx, box.lower[1] + dy)
        end_upper = (box.upper[0] + dx, box.upper[1] + dy)
        if any(not isfinite(v) for v in end_lower + end_upper):
            raise ValueError("translated AABB bounds must remain finite")
        envelopes.append(AABB(
            box.id,
            (min(box.lower[0], end_lower[0]),
             min(box.lower[1], end_lower[1])),
            (max(box.upper[0], end_upper[0]),
             max(box.upper[1], end_upper[1])),
        ))
    return broad_phase_pairs(envelopes)
