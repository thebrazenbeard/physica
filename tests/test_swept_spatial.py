"""Swept translation candidates: independent sampled-time adversarial checks."""
from itertools import combinations
from fractions import Fraction
from random import Random

import pytest

from physica.spatial import AABB, broad_phase_pairs, swept_broad_phase_pairs


def overlap(a, b):
    """Independent strict-axis corner overlap, with edge contacts included."""
    return all(
        a.lower[axis] <= b.upper[axis] and b.lower[axis] <= a.upper[axis]
        for axis in (0, 1)
    )


def translated(box, delta, t):
    return AABB(box.id,
                (box.lower[0] + t * delta[0], box.lower[1] + t * delta[1]),
                (box.upper[0] + t * delta[0], box.upper[1] + t * delta[1]))


def test_swept_pair_crosses_mid_step_without_endpoint_overlap():
    a, b = AABB("a", (0., 0.), (1., 1.)), AABB("b", (2., 0.), (3., 1.))
    assert broad_phase_pairs((a, b)) == ()
    assert broad_phase_pairs((translated(a, (4., 0.), 1), b)) == ()
    assert swept_broad_phase_pairs((a, b), {"a": (4., 0.), "b": (0., 0.)}) == (("a", "b"),)


def test_two_swapping_bodies_are_not_missed():
    a = AABB("a", (0., 0.), (1., 1.))
    b = AABB("b", (3., 0.), (4., 1.))
    result = swept_broad_phase_pairs((a, b), {"a": (3., 0.), "b": (-3., 0.)})
    assert result == (("a", "b"),)


def test_stationary_identity_equals_static_broad_phase():
    boxes = (AABB("b", (0., 0.), (1., 1.)),
             AABB("a", (1., 0.), (2., 1.)),
             AABB("c", (5., 0.), (6., 1.)))
    deltas = {box.id: (0., 0.) for box in boxes}
    assert swept_broad_phase_pairs(boxes, deltas) == broad_phase_pairs(boxes)
    assert swept_broad_phase_pairs(tuple(reversed(boxes)), deltas) == broad_phase_pairs(boxes)


def test_conservative_false_positive_is_not_a_contact_claim():
    a = AABB("a", (0., 0.), (1., 1.))
    b = AABB("b", (3., 0.), (4., 1.))
    motion = {"a": (4., 0.), "b": (4., 0.)}
    assert swept_broad_phase_pairs((a, b), motion) == (("a", "b"),)
    assert all(not overlap(translated(a, motion["a"], i / 10),
                           translated(b, motion["b"], i / 10))
               for i in range(11))


def test_randomized_independent_sampled_overlap_is_subset_of_swept_pairs():
    rng = Random(72043)
    for count in (2, 5, 24, 64):
        for _ in range(12):
            bounds = []
            motions = {}
            for i in range(count):
                x, y = rng.uniform(-30, 30), rng.uniform(-30, 30)
                w, h = rng.uniform(.01, 5), rng.uniform(.01, 5)
                name = str(i)
                bounds.append(AABB(name, (x, y), (x + w, y + h)))
                motions[name] = (rng.uniform(-14, 14), rng.uniform(-14, 14))
            candidates = set(swept_broad_phase_pairs(bounds, motions))
            assert swept_broad_phase_pairs(list(reversed(bounds)), motions) == tuple(sorted(candidates))
            for tick in range(21):
                t = tick / 20
                now = [translated(b, motions[b.id], t) for b in bounds]
                observed = {tuple(sorted((a.id, b.id)))
                            for a, b in combinations(now, 2) if overlap(a, b)}
                assert observed <= candidates


@pytest.mark.parametrize("motion", [
    {}, {"a": (0., 0.), "b": (0., 0.)},
    {"a": [1., 0.]}, {"a": (True, 0.)},
    {"a": (float("nan"), 0.)}, {"a": (float("inf"), 0.)},
    {"a": (1.,)}, {"a": "no"}, {"a": (0., 0., 0.)},
    {"a": (1e308, 0.)},
])
def test_reject_bad_displacements_or_membership(motion):
    with pytest.raises(ValueError):
        swept_broad_phase_pairs((AABB("a", (1e308, 0.), (1e308, 1.)),), motion)


def test_input_objects_and_motion_map_not_modified():
    boxes = (AABB("a", (0., 0.), (1., 1.)), AABB("b", (2., 2.), (3., 3.)))
    motion = {"a": (1., 0.), "b": (-1., 0.)}
    original = dict(motion)
    swept_broad_phase_pairs(boxes, motion)
    assert motion == original
    assert boxes[0].lower == (0., 0.)


def test_empty_world_has_empty_candidate_set():
    assert swept_broad_phase_pairs((), {}) == ()


def test_large_exact_integer_endpoint_touch_is_not_pruned_by_float_addition():
    q = 2**53
    a = AABB("a", (q + 1, 0), (q + 1, 1))
    b = AABB("b", (q + 2, 0), (q + 2, 1))
    # Mathematically the one-unit displacement touches b at t=1.
    # float(q+1) rounds *down*, so naive int+float misses the endpoint.
    assert swept_broad_phase_pairs((a, b), {"a": (1., 0.), "b": (0., 0.)}) == (("a", "b"),)


def test_large_integer_positive_width_boxes_touch_at_end_of_sweep():
    q = 2**53
    a = AABB("a", (q, 0), (q + 1, 1))
    b = AABB("b", (q + 2, 0), (q + 3, 1))
    assert swept_broad_phase_pairs((a, b), {"a": (1., 0.), "b": (0., 0.)}) == (("a", "b"),)


def test_large_integer_sweep_conservatively_contains_exact_sampled_contacts():
    q = 2**53
    for displacement in (-3., -2., -1., 0., 1., 2., 3.):
        for offset in (-3, -2, -1, 0, 1, 2, 3):
            a = AABB("a", (q + 1, 0), (q + 1, 1))
            b = AABB("b", (q + 1 + offset, 0), (q + 1 + offset, 1))
            candidates = swept_broad_phase_pairs(
                (a, b), {"a": (displacement, 0.), "b": (0., 0.)})
            for time in (Fraction(0), Fraction(1, 2), Fraction(1)):
                x = Fraction(a.lower[0]) + Fraction(displacement) * time
                if x == Fraction(b.lower[0]):
                    assert ("a", "b") in candidates


def test_extreme_integer_displacement_has_stable_validation_error():
    a = AABB("a", (0, 0), (1, 1))
    with pytest.raises(ValueError, match="finite"):
        swept_broad_phase_pairs((a,), {"a": (10**1000, 0.)})
