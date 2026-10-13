"""Optional, simulation-only interpretation of VERA ARK MOVE intentions.

Controller axes are NOT physical coordinates. A caller must provide an explicit,
locally justified stick-to-world velocity calibration. This module cannot emit
controller reports, contact claims, or actuator commands.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Any, Sequence

from .spatial import AABB, broad_phase_pairs


@dataclass(frozen=True)
class MotionCalibration:
    """World velocity in metres/second per normalized stick axis.

    Columns of the 2x2 matrix correspond to the x and y stick axes. This is
    an externally supplied assumption, not a measured/calibrated model.
    """
    x_axis: tuple[float, float]
    y_axis: tuple[float, float]

    def __post_init__(self) -> None:
        if (len(self.x_axis) != 2 or len(self.y_axis) != 2
                or any(type(v) not in (int, float) or not isfinite(v)
                       for v in self.x_axis + self.y_axis)):
            raise ValueError("calibration must contain four finite real numbers")


@dataclass(frozen=True)
class MotionCandidate:
    start_m: tuple[float, float]
    end_m: tuple[float, float]
    duration_s: float
    potential_obstacle_ids: tuple[str, ...]
    qualifier: str = "UNCALIBRATED_WORLD_MODEL_AND_BROAD_PHASE_ONLY"


def predict_ark_move(
    action: Mapping[str, Any],
    *,
    position_m: tuple[float, float],
    calibration: MotionCalibration,
    radius_m: float = 0.0,
    obstacles: Sequence[AABB] = (),
) -> MotionCandidate:
    """Bounded kinematic prediction; rejects arbitrary or live actions.

    Requires the ARK V1 full ActionIntention JSON shape, not raw input.
    A swept bounding box is conservative and can produce false positives.
    No evidence here proves an actual collision, movement, or safe path.
    """
    fields = {
        "schema_version", "intention_id", "perception_id", "session_id",
        "step", "action_type", "x", "y", "duration_ms", "reason",
        "planner", "created_at", "dry_run",
    }
    if not isinstance(action, Mapping) or set(action) != fields:
        raise ValueError("expected exact VERA ARK V1 intention fields")
    if action["schema_version"] != "VERA_ARK_ACTION_INTENTION_V1":
        raise ValueError("unsupported action schema")
    if action["dry_run"] is not True or action["action_type"] != "MOVE":
        raise ValueError("only dry-run MOVE intentions can be predicted")
    for name in ("x", "y"):
        v = action[name]
        if type(v) not in (int, float) or not isfinite(v) or not -1 <= v <= 1:
            raise ValueError(f"{name} must be finite and within [-1,1]")
    duration = action["duration_ms"]
    if type(duration) is not int or not 0 <= duration <= 1000:
        raise ValueError("duration_ms must be an integer within [0,1000]")
    if type(action["step"]) is not int or not 0 <= action["step"] <= 1_000_000:
        raise ValueError("invalid step")
    if not isinstance(calibration, MotionCalibration):
        raise ValueError("explicit MotionCalibration required")
    if (len(position_m) != 2 or
            any(type(v) not in (int, float) or not isfinite(v) for v in position_m)):
        raise ValueError("position_m must be two finite numbers")
    if type(radius_m) not in (int, float) or not isfinite(radius_m) or radius_m < 0:
        raise ValueError("radius_m must be nonnegative and finite")
    dt = duration / 1000.0
    end = (
        position_m[0] + dt * (action["x"] * calibration.x_axis[0] +
                               action["y"] * calibration.y_axis[0]),
        position_m[1] + dt * (action["x"] * calibration.x_axis[1] +
                               action["y"] * calibration.y_axis[1]),
    )
    if any(not isfinite(v) for v in end):
        raise ValueError("predicted position overflow")
    swept = AABB("__ark_motion__", (
        min(position_m[0], end[0]) - radius_m,
        min(position_m[1], end[1]) - radius_m,
    ), (
        max(position_m[0], end[0]) + radius_m,
        max(position_m[1], end[1]) + radius_m,
    ))
    if any(box.id == swept.id for box in obstacles):
        raise ValueError("reserved obstacle identifier")
    pairs = broad_phase_pairs((swept, *obstacles))
    hits = tuple(sorted(b if a == swept.id else a
                        for a, b in pairs if swept.id in (a, b)))
    return MotionCandidate(
        (float(position_m[0]), float(position_m[1])),
        end, dt, hits,
    )
