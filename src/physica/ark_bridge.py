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


def _finite_real(value: object) -> bool:
    """Validate an input as a finite real representable by this float model.

    ``math.isfinite`` converts integers to float, raising ``OverflowError``
    for otherwise valid Python integers outside the binary64 range. Invalid
    externally supplied values must fail validation, not escape as an error.
    """
    if type(value) not in (int, float):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class MotionCalibration:
    """World velocity in metres/second per normalized stick axis.

    Columns of the 2x2 matrix correspond to the x and y stick axes. This is
    an externally supplied assumption, not a measured/calibrated model.
    """
    x_axis: tuple[float, float]
    y_axis: tuple[float, float]

    def __post_init__(self) -> None:
        # A frozen dataclass does not freeze mutable values supplied by callers.
        # Snapshot each axis so subsequent list mutation cannot alter a prediction.
        x_axis = tuple(self.x_axis)
        y_axis = tuple(self.y_axis)
        if (len(x_axis) != 2 or len(y_axis) != 2
                or not all(_finite_real(v) for v in x_axis + y_axis)):
            raise ValueError("calibration must contain four finite real numbers")
        object.__setattr__(self, "x_axis", x_axis)
        object.__setattr__(self, "y_axis", y_axis)


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
    if not isinstance(action, Mapping):
        raise ValueError("expected exact VERA ARK V1 intention fields")
    # A custom Mapping can return different values on successive reads. Take one
    # snapshot, then validate and compute solely from the validated values.
    values = dict(action)
    if set(values) != fields:
        raise ValueError("expected exact VERA ARK V1 intention fields")
    if values["schema_version"] != "VERA_ARK_ACTION_INTENTION_V1":
        raise ValueError("unsupported action schema")
    if values["dry_run"] is not True or values["action_type"] != "MOVE":
        raise ValueError("only dry-run MOVE intentions can be predicted")
    for name in ("x", "y"):
        v = values[name]
        if not _finite_real(v) or not -1 <= v <= 1:
            raise ValueError(f"{name} must be finite and within [-1,1]")
    duration = values["duration_ms"]
    if type(duration) is not int or not 0 <= duration <= 1000:
        raise ValueError("duration_ms must be an integer within [0,1000]")
    if type(values["step"]) is not int or not 0 <= values["step"] <= 1_000_000:
        raise ValueError("invalid step")
    if not isinstance(calibration, MotionCalibration):
        raise ValueError("explicit MotionCalibration required")
    position = tuple(position_m)
    if (len(position) != 2 or
            not all(_finite_real(v) for v in position)):
        raise ValueError("position_m must be two finite numbers")
    if not _finite_real(radius_m) or radius_m < 0:
        raise ValueError("radius_m must be nonnegative and finite")
    dt = duration / 1000.0
    end = (
        position[0] + dt * (values["x"] * calibration.x_axis[0] +
                            values["y"] * calibration.y_axis[0]),
        position[1] + dt * (values["x"] * calibration.x_axis[1] +
                            values["y"] * calibration.y_axis[1]),
    )
    if any(not isfinite(v) for v in end):
        raise ValueError("predicted position overflow")
    swept = AABB("__ark_motion__", (
        min(position[0], end[0]) - radius_m,
        min(position[1], end[1]) - radius_m,
    ), (
        max(position[0], end[0]) + radius_m,
        max(position[1], end[1]) + radius_m,
    ))
    if any(box.id == swept.id for box in obstacles):
        raise ValueError("reserved obstacle identifier")
    pairs = broad_phase_pairs((swept, *obstacles))
    hits = tuple(sorted(b if a == swept.id else a
                        for a, b in pairs if swept.id in (a, b)))
    return MotionCandidate(
        (float(position[0]), float(position[1])),
        end, dt, hits,
    )
