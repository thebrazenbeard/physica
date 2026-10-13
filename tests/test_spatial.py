"""Broad-phase candidate generation, with an independent brute-force oracle."""
from itertools import combinations
from random import Random
import pytest

from physica.spatial import AABB, broad_phase_pairs, brute_force_pairs


def test_overlapping_boxes_and_edge_touch_are_candidates():
    bounds = (
        AABB("a", (0., 0.), (1., 1.)),
        AABB("b", (1., .5), (2., 1.5)),  # edge touch
        AABB("c", (4., 4.), (5., 5.)),
        AABB("d", (.2, .2), (.5, .5)),
    )
    expected = {("a", "b"), ("a", "d")}
    assert set(brute_force_pairs(bounds)) == expected
    assert set(broad_phase_pairs(bounds)) == expected


def test_2d_sweep_candidates_agree_with_independent_quadratic_oracle():
    rng = Random(11)
    for n in (2, 4, 30, 120):
        boxes = []
        for i in range(n):
            x, y = rng.uniform(-5, 5), rng.uniform(-5, 5)
            w, h = rng.uniform(0, 2), rng.uniform(0, 2)
            boxes.append(AABB(str(i), (x, y), (x + w, y + h)))
        assert broad_phase_pairs(boxes) == brute_force_pairs(boxes)
        assert broad_phase_pairs(list(reversed(boxes))) == brute_force_pairs(boxes)


def test_output_order_is_stable_and_excludes_self_pairs():
    items = [
        AABB("b", (0., 0.), (1., 1.)),
        AABB("a", (0., 0.), (1., 1.)),
        AABB("c", (0., 0.), (1., 1.)),
    ]
    assert broad_phase_pairs(items) == (("a", "b"), ("a", "c"), ("b", "c"))


@pytest.mark.parametrize("items", [
    [AABB("a", (0., 0.), (1., 1.)), AABB("a", (2., 2.), (3., 3.))],
    [AABB("a", (0., 0.), (1., 1.)), "bad"],
])
def test_reject_duplicate_ids_or_inconsistent_objects(items):
    with pytest.raises(ValueError):
        broad_phase_pairs(items)


@pytest.mark.parametrize("kwargs", [
    {"id": "a", "lower": (1., 0.), "upper": (0., 1.)},
    {"id": "a", "lower": (0., 0.), "upper": (float("inf"), 1.)},
    {"id": "a", "lower": (0.,), "upper": (1.,)},
    {"id": "", "lower": (0., 0.), "upper": (1., 1.)},
])
def test_invalid_aabb_rejected(kwargs):
    with pytest.raises(ValueError):
        AABB(**kwargs)


def test_only_broadphase_disclaimer():
    # AABBs can overlap even when real rotated/nonrectangular shapes do not.
    assert broad_phase_pairs.__doc__ is not None
    assert "candidate" in broad_phase_pairs.__doc__.lower()
