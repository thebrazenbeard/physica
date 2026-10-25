import math
from collections.abc import Mapping

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


def test_huge_integer_calibration_is_rejected_as_invalid_input():
    # Python math.isfinite() raises OverflowError for ints beyond binary64.
    with pytest.raises(ValueError, match="calibration"):
        MotionCalibration((10**1000, 0), (0, 1))


@pytest.mark.parametrize("input_case", ["stick", "position", "radius"])
def test_huge_integer_move_inputs_fail_closed(input_case):
    move = intention()
    position = (0, 0)
    radius = 0
    if input_case == "stick":
        move["x"] = 10**1000
    elif input_case == "position":
        position = (10**1000, 0)
    else:
        radius = 10**1000
    with pytest.raises(ValueError):
        predict_ark_move(move, position_m=position,
                         calibration=MotionCalibration((1, 0), (0, 1)),
                         radius_m=radius)


def test_intention_mapping_cannot_change_stick_after_validation():
    class ChangingX(Mapping):
        def __init__(self):
            self.data = intention(x=.5, duration_ms=1000)
            self.x_reads = 0

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, key):
            if key == "x":
                self.x_reads += 1
                return .5 if self.x_reads == 1 else 100.
            return self.data[key]

    action = ChangingX()
    result = predict_ark_move(action, position_m=(0., 0.),
                              calibration=MotionCalibration((1., 0.), (0., 1.)))
    assert result.end_m == (.5, 0.)
    assert action.x_reads == 1


def test_calibration_snapshots_mutable_axis_lists():
    x_axis = [1., 0.]
    y_axis = [0., 1.]
    calibration = MotionCalibration(x_axis, y_axis)
    x_axis[0] = 100.
    y_axis[1] = 100.
    assert calibration.x_axis == (1., 0.)
    assert calibration.y_axis == (0., 1.)
    result = predict_ark_move(intention(duration_ms=1000), position_m=(0., 0.),
                              calibration=calibration)
    assert result.end_m == (1., 0.)
