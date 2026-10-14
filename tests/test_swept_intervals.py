"""Continuous overlap intervals for translating AABBs, with independent samples."""
from itertools import permutations
from math import isclose
from random import Random

import pytest
from physica.spatial import AABB, brute_force_pairs, swept_aabb_intervals


def test_crossing_between_nonoverlapping_endpoint_frames():
    bounds = (
        AABB("moving", (0., 0.), (1., 1.)),
        AABB("stationary", (2., 0.), (3., 1.)),
    )
    hits = swept_aabb_intervals(bounds, {"moving": (4.,0.), "stationary": (0.,0.)})
    assert len(hits)==1
    assert hits[0].pair == ("moving","stationary")
    assert isclose(hits[0].enter_time, .25, abs_tol=1e-12)
    assert isclose(hits[0].exit_time, .75, abs_tol=1e-12)


def test_disjoint_motion_times_not_a_false_positive_contact():
    boxes = (
        AABB("rising", (0.,0.), (1.,1.)),
        AABB("falling", (4.,0.), (5.,1.)),
    )
    assert swept_aabb_intervals(boxes, {
        "rising": (4.,4.), "falling": (-4.,-4.)
    }) == ()


def test_initial_overlap_and_static_edge_touch_both_reported():
    boxes=(AABB("a",(0.,0.),(1.,1.)),AABB("b",(1.,0.),(2.,1.)))
    for ordered in permutations(boxes):
        hits=swept_aabb_intervals(ordered,{"a":(0.,0.),"b":(0.,0.)})
        assert len(hits)==1
        assert hits[0].pair==("a","b")
        assert (hits[0].enter_time,hits[0].exit_time)==(0.,1.)


def test_touching_only_at_final_time_is_included():
    boxes=[AABB("a",(0.,0.),(1.,1.)),AABB("b",(3.,0.),(4.,1.))]
    hits=swept_aabb_intervals(boxes,{"a":(2.,0.),"b":(0.,0.)})
    assert hits[0].enter_time == 1.0
    assert hits[0].exit_time == 1.0


def test_randomized_time_samples_are_never_missed():
    rng=Random(20261014)
    for _ in range(45):
        bounds=[]
        motion={}
        for i in range(7):
            x,y=rng.uniform(-3,3),rng.uniform(-3,3)
            w,h=rng.uniform(0.1,1.8),rng.uniform(0.1,1.8)
            key=f"box{i}"
            bounds.append(AABB(key,(x,y),(x+w,y+h)))
            motion[key]=(rng.uniform(-4,4),rng.uniform(-4,4))
        hits={h.pair:h for h in swept_aabb_intervals(bounds,motion)}
        for n in range(21):
            t=n/20
            sample=[]
            for b in bounds:
                dx,dy=motion[b.id]
                sample.append(AABB(
                    b.id,
                    (b.lower[0]+dx*t,b.lower[1]+dy*t),
                    (b.upper[0]+dx*t,b.upper[1]+dy*t)))
            for pair in brute_force_pairs(sample):
                assert pair in hits, (pair,t)
                assert hits[pair].enter_time-1e-10 <= t <= hits[pair].exit_time+1e-10


@pytest.mark.parametrize("motion",[
    {"a":(0.,0.)},
    {"a":(0.,0.),"b":(0.,0.),"ghost":(0.,0.)},
    {"a":(float("inf"),0.),"b":(0.,0.)},
    {"a":(float("nan"),0.),"b":(0.,0.)},
    {"a":(True,0.),"b":(0.,0.)},
    {"a":"left","b":(0.,0.)},
    {"a":(0.,),"b":(0.,0.)},
])
def test_fail_closed_for_nonfinite_missing_or_malformed_motion(motion):
    boxes=[AABB("a",(0.,0.),(1.,1.)),AABB("b",(2.,2.),(3.,3.))]
    with pytest.raises(ValueError):
        swept_aabb_intervals(boxes,motion)


def test_overflowing_relative_motion_fails_instead_of_inventing_contact():
    boxes=[AABB("a",(0.,0.),(1.,1.)),AABB("b",(2.,2.),(3.,3.))]
    with pytest.raises(ValueError):
        swept_aabb_intervals(boxes,{"a":(1e308,0.),"b":(-1e308,0.)})


def test_api_explicitly_limited_to_axis_aligned_translation():
    assert "not" in swept_aabb_intervals.__doc__.lower()
    assert "rigid-body" in swept_aabb_intervals.__doc__.lower()


def test_analytic_pairs_are_subset_of_conservative_swept_envelopes():
    from physica.spatial import swept_broad_phase_pairs
    boxes = [
        AABB("a", (0., 0.), (1., 1.)),
        AABB("b", (3., 0.), (4., 1.)),
        AABB("c", (10., 10.), (11., 11.)),
    ]
    moves = {"a": (4., 0.), "b": (0., 0.), "c": (-4., -1.)}
    analytic = {interval.pair for interval in swept_aabb_intervals(boxes, moves)}
    conservative = set(swept_broad_phase_pairs(boxes, moves))
    assert analytic <= conservative
