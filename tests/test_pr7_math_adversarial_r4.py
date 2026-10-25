"""Independent exact-rational oracle for swept AABB overlap claims.

This test module is applied only to the safe R3 composition; the unsafe
Draft PR #7 checkout and workspace tasks are never executed.
"""

from fractions import Fraction
from itertools import combinations
from random import Random

from physica.spatial import AABB, swept_aabb_intervals, swept_broad_phase_pairs


def _fraction_overlap_at(left, right, moves, time):
    """Directly intersect four exact translated edge inequalities."""
    t = Fraction(time)
    for axis in (0, 1):
        ll = Fraction(left.lower[axis]) + t * Fraction(moves[left.id][axis])
        lu = Fraction(left.upper[axis]) + t * Fraction(moves[left.id][axis])
        rl = Fraction(right.lower[axis]) + t * Fraction(moves[right.id][axis])
        ru = Fraction(right.upper[axis]) + t * Fraction(moves[right.id][axis])
        if lu < rl or ru < ll:
            return False
    return True


def test_opposite_extreme_finite_motion_exact_contact():
    boxes = (
        AABB("a", (-1e308, 0.), (-1e308, 1.)),
        AABB("b", (1e308, 0.), (1e308, 1.)),
    )
    moves = {"a": (1e308, 0.), "b": (-1e308, 0.)}
    assert moves["a"][0] - moves["b"][0] == float("inf")
    interval, = swept_aabb_intervals(boxes, moves)
    assert interval.pair == ("a", "b")
    assert (interval.enter_time, interval.exit_time) == (1., 1.)


def test_mixed_large_int_bounds_do_not_lose_touch():
    q = 2**53
    boxes = (AABB("a", (q, 0), (q + 1, 1)),
             AABB("b", (q + 2, 0), (q + 3, 1)))
    moves = {"a": (1., 0.), "b": (0., 0.)}
    assert _fraction_overlap_at(*boxes, moves, 1)
    assert swept_broad_phase_pairs(boxes, moves) == (("a", "b"),)
    interval, = swept_aabb_intervals(boxes, moves)
    assert (interval.enter_time, interval.exit_time) == (1., 1.)


def test_exact_oracle_randomized_discrete_times_no_false_negatives():
    rng = Random(20261025_07)
    motions = (-3., -2., -1., -.5, 0., .5, 1., 2., 3.)
    offset = (-2**53, -100., 0., 100., 2**53)
    times = tuple(Fraction(i, 8) for i in range(9))
    for scene in range(500):
        base = rng.choice(offset)
        boxes = []
        moves = {}
        for index in range(4):
            x = base + rng.randint(-3, 3)
            y = rng.randint(-3, 3)
            width = rng.randint(0, 3)
            height = rng.randint(0, 3)
            box = AABB(str(index), (x, y), (x + width, y + height))
            boxes.append(box)
            moves[box.id] = (rng.choice(motions), rng.choice(motions))
        records = {record.pair: record for record in swept_aabb_intervals(boxes, moves)}
        candidates = set(swept_broad_phase_pairs(boxes, moves))
        assert records.keys() <= candidates
        for left, right in combinations(boxes, 2):
            pair = tuple(sorted((left.id, right.id)))
            for time in times:
                if _fraction_overlap_at(left, right, moves, time):
                    assert pair in records, (scene, pair, time)
                    assert records[pair].enter_time <= float(time) <= records[pair].exit_time
