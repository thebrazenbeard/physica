"""Independent geometric and endpoint references for swept AABB overlap times."""

from itertools import combinations
from fractions import Fraction
from math import isfinite
from random import Random

from physica.spatial import AABB, SweptAABBInterval, swept_aabb_intervals


def test_horizontal_crossing_exact_times():
    boxes = (AABB("a", (0., 0.), (1., 1.)),
             AABB("b", (3., 0.), (4., 1.)))
    # A's right edge reaches x=3 at t=1/2; its left edge reaches x=4 at t=1.
    assert swept_aabb_intervals(boxes, {"a": (4., 0.), "b": (0., 0.)}) == (
        SweptAABBInterval(("a", "b"), .5, 1.),
    )


def test_overlapping_motions_must_coincide_in_both_axes():
    boxes = (AABB("a", (0., 0.), (1., 1.)),
             AABB("b", (2., 4.), (3., 5.)))
    # X overlap only during [0.25, 0.75], Y overlap only during [0.75, 1].
    assert swept_aabb_intervals(boxes, {"a": (4., 0.), "b": (0., -4.)}) == (
        SweptAABBInterval(("a", "b"), .75, .75),
    )


def test_no_simultaneous_overlap_despite_envelope_candidate():
    boxes = (AABB("a", (0., 0.), (1., 1.)),
             AABB("b", (3., 0.), (4., 1.)))
    # Equal motion preserves separation; broad-phase translation envelopes
    # still overlap because they discard the common time coordinate.
    assert swept_aabb_intervals(boxes, {"a": (4., 0.), "b": (4., 0.)}) == ()


def test_stationary_overlap_and_touch_are_closed_full_intervals():
    boxes = (AABB("c", (1., 0.), (2., 1.)),
             AABB("b", (0., 0.), (1., 1.)),
             AABB("a", (8., 0.), (9., 1.)))
    motions = {box.id: (0., 0.) for box in boxes}
    assert swept_aabb_intervals(boxes, motions) == (
        SweptAABBInterval(("b", "c"), 0., 1.),
    )
    assert swept_aabb_intervals(tuple(reversed(boxes)), motions) == (
        SweptAABBInterval(("b", "c"), 0., 1.),
    )


def test_initial_touch_followed_by_separation_is_zero_duration():
    boxes = (AABB("a", (0., 0.), (1., 1.)),
             AABB("b", (1., 0.), (2., 1.)))
    assert swept_aabb_intervals(boxes, {"a": (-1., 0.), "b": (1., 0.)}) == (
        SweptAABBInterval(("a", "b"), 0., 0.),
    )


def test_opposite_axis_movement_and_end_corner_touch():
    boxes = (AABB("a", (0., 0.), (1., 1.)),
             AABB("b", (3., 3.), (4., 4.)))
    assert swept_aabb_intervals(boxes, {"a": (1., 1.), "b": (-1., -1.)}) == (
        SweptAABBInterval(("a", "b"), 1., 1.),
    )


def test_sampled_exact_real_overlaps_fit_analytic_intervals():
    """Fraction geometry avoids cancellation in a float-sampled oracle."""
    rng = Random(20261025)
    for _ in range(30):
        boxes = []
        motions = {}
        for index in range(10):
            x, y = rng.uniform(-3., 3.), rng.uniform(-3., 3.)
            w, h = rng.uniform(.1, 2.), rng.uniform(.1, 2.)
            box = AABB(str(index), (x, y), (x + w, y + h))
            boxes.append(box)
            motions[box.id] = (rng.uniform(-4., 4.), rng.uniform(-4., 4.))
        records = swept_aabb_intervals(boxes, motions)
        assert all(isfinite(record.enter_time) and isfinite(record.exit_time)
                   and 0. <= record.enter_time <= record.exit_time <= 1.
                   for record in records)
        assert records == tuple(sorted(records,
                                       key=lambda record: (record.pair, record.enter_time)))
        for tick in range(21):
            t = tick / 20
            exact_t = Fraction(t)
            for left, right in combinations(boxes, 2):
                left_delta, right_delta = motions[left.id], motions[right.id]
                if all(
                    Fraction(left.lower[axis]) + exact_t * Fraction(left_delta[axis])
                    <= Fraction(right.upper[axis]) + exact_t * Fraction(right_delta[axis])
                    and Fraction(right.lower[axis]) + exact_t * Fraction(right_delta[axis])
                    <= Fraction(left.upper[axis]) + exact_t * Fraction(left_delta[axis])
                    for axis in (0, 1)
                ):
                    pair = tuple(sorted((left.id, right.id)))
                    assert any(record.pair == pair
                               and record.enter_time <= t <= record.exit_time
                               for record in records)


def test_binary64_computed_end_touch_not_an_exact_analytic_contact():
    a = AABB("a", (0., 0.), (1., 1.))
    b = AABB("b", (1.1, 0.), (2.1, 1.))
    # Float endpoint arithmetic rounds to a touch; exact real arithmetic on
    # these *binary64 inputs* leaves a small gap throughout [0,1].
    assert a.upper[0] + .1 == b.lower[0]
    assert swept_aabb_intervals((a, b), {"a": (.1, 0.), "b": (0., 0.)}) == ()


def test_rounded_endpoint_contact_does_not_fill_a_gap():
    point_a = AABB("a", (1e20, 0.), (1e20, 0.))
    point_b = AABB("b", (1e20, 0.), (1e20, 0.))
    # Exact-real trajectories separate immediately after t=0, although
    # the computed t=1 float positions both round back to 1e20.
    assert 1e20 - 1. == 1e20 + 1. == 1e20
    assert swept_aabb_intervals((point_a, point_b), {
        "a": (-1., 0.), "b": (1., 0.)
    }) == (
        SweptAABBInterval(("a", "b"), 0., 0.),
    )


def test_interior_rounded_touch_is_not_exact_analytic_touch():
    a = AABB("a", (0., 0.), (1., 1.))
    b = AABB("b", (1.1, 0.), (2.1, 1.))
    # A sampled float at t=.5 rounds to touching, but exact-real arithmetic
    # on the binary64 inputs does not.
    assert 1. + .5 * .2 == 1.1
    interval, = swept_aabb_intervals((a, b), {"a": (.2, 0.), "b": (0., 0.)})
    assert interval.enter_time > .5


def test_exact_representable_end_touch_remains_included():
    a = AABB("a", (0., 0.), (1., 1.))
    b = AABB("b", (1.125, 0.), (2.125, 1.))
    assert swept_aabb_intervals((a, b), {"a": (.125, 0.), "b": (0., 0.)}) == (
        SweptAABBInterval(("a", "b"), 1., 1.),
    )


def test_endpoint_overflow_and_motion_shape_rejected():
    box = AABB("a", (1e308, 0.), (1e308, 1.))
    for motions in ({"a": (1e308, 0.)}, {"a": (True, 0.)},
                    {"a": (0., float("nan"))}, {}):
        try:
            swept_aabb_intervals((box,), motions)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid motion rejection for {motions}")


def test_extreme_finite_relative_motion_does_not_overflow_interval_math():
    boxes = (AABB("a", (-1e308, 0.), (-1e308, 1.)),
             AABB("b", (1e308, 0.), (1e308, 1.)))
    motions = {"a": (1e308, 0.), "b": (-1e308, 0.)}
    # Direct float subtraction of the motions would overflow; rational
    # differences give the mathematically exact t=1 corner/edge touch.
    assert swept_aabb_intervals(boxes, motions) == (
        SweptAABBInterval(("a", "b"), 1., 1.),
    )
