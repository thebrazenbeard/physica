import math

import pytest

from physica.ark_bridge import MotionCalibration, predict_ark_move
from physica.spatial import AABB


def intention(**overrides):
    data = dict(
        schema_version="VERA_ARK_ACTION_INTENTION_V1",
        intention_id="id", perception_id="p", session_id="s", step=0,
        action_type="MOVE", x=1.0, y=0.0, duration_ms=500,
        reason="test", planner="test", created_at="test", dry_run=True,
    )
    data.update(overrides)
    return data


def test_predicted_swept_candidate_not_contact():
    result = predict_ark_move(
        intention(), position_m=(0, 0),
        calibration=MotionCalibration((2, 0), (0, 2)), radius_m=0.1,
        obstacles=(AABB("wall", (0.9, -0.2), (1.1, 0.2)),
                   AABB("far", (5, 5), (6, 6))),
    )
    assert result.end_m == (1.0, 0.0)
    assert result.potential_obstacle_ids == ("wall",)
    assert "BROAD_PHASE_ONLY" in result.qualifier


@pytest.mark.parametrize("change", [
    {"dry_run": False}, {"action_type": "LOOK"}, {"x": float("nan")},
    {"x": 1.1}, {"duration_ms": 1001}, {"duration_ms": True},
    {"schema_version": "other"}, {"step": True},
])
def test_rejects_invalid_or_live(change):
    with pytest.raises(ValueError):
        predict_ark_move(intention(**change), position_m=(0, 0),
                         calibration=MotionCalibration((1, 0), (0, 1)))


def test_requires_calibration_and_bounds():
    with pytest.raises(ValueError):
        MotionCalibration((math.inf, 0), (0, 1))
    with pytest.raises(ValueError):
        predict_ark_move(intention(), position_m=(0, 0), calibration=None)
    with pytest.raises(ValueError):
        predict_ark_move(intention(), position_m=(0, 0),
                         calibration=MotionCalibration((1, 0), (0, 1)),
                         obstacles=(AABB("__ark_motion__", (0, 0), (1, 1)),))
