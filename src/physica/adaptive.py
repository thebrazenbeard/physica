"""Auditable explicit Dormand–Prince 5(4) adaptive IVP solver.

This is a deterministic, nonstiff reference solver, not a stiff solver, a
dense-output integrator, or an implicit differential-algebraic solver.

The 5th-order estimate advances the state; a 4th-order embedded estimate
controls local error. Accepted samples do not form a uniform time grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, sqrt
from typing import Sequence

from .numerics import RHS, State, Trajectory, _evaluate, _state

# Explicit seven-stage Dormand–Prince tableau, 5th order + 4th order embedded.
_C = (0., 1/5, 3/10, 4/5, 8/9, 1., 1.)
_A = (
    (),
    (1/5,),
    (3/40, 9/40),
    (44/45, -56/15, 32/9),
    (19372/6561, -25360/2187, 64448/6561, -212/729),
    (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
    (35/384, 0., 500/1113, 125/192, -2187/6784, 11/84),
)
_B5 = (35/384, 0., 500/1113, 125/192, -2187/6784, 11/84, 0.)
_B4 = (5179/57600, 0., 7571/16695, 393/640,
       -92097/339200, 187/2100, 1/40)


@dataclass(frozen=True)
class AdaptiveResult:
    trajectory: Trajectory
    accepted_steps: int
    rejected_steps: int
    evaluations: int
    rtol: float
    atol: float
    initial_step: float
    min_step: float
    max_step: float

    @property
    def attempted_steps(self) -> int:
        return self.accepted_steps + self.rejected_steps


def solve_adaptive(
    rhs: RHS,
    initial: Sequence[float],
    *,
    t0: float = 0.,
    t1: float = 10.,
    dt: float = 0.1,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    min_step: float = 1e-12,
    max_step: float = inf,
    max_attempts: int = 100_000,
) -> AdaptiveResult:
    """Solve a nonstiff IVP with local weighted-RMS error control.

    `dt` is the initial attempted step, not output cadence. The loop is
    bounded by `max_attempts`, counting both accepted and rejected steps.
    Overflow, invalid RHS dimensions and impossible tolerances fail closed.
    """
    if not all(isfinite(v) for v in (t0, t1, dt, rtol, atol, min_step)):
        raise ValueError("integration inputs must be finite")
    if not t1 > t0 or dt <= 0:
        raise ValueError("requires t1 > t0 and dt > 0")
    if not (0 < rtol < 1) or atol <= 0:
        raise ValueError("requires 0 < rtol < 1 and atol > 0")
    if min_step <= 0 or not (isfinite(max_step) or max_step == inf):
        raise ValueError("invalid step bounds")
    if max_step <= 0 or min_step > max_step:
        raise ValueError("requires 0 < min_step <= max_step")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    y = _state(initial)
    t = float(t0)
    times = [t]
    states = [y]
    h = min(dt, max_step, t1 - t0)
    accepted = rejected = evaluations = 0

    while t < t1:
        if accepted + rejected >= max_attempts:
            raise RuntimeError("adaptive solver exceeded max_attempts")
        remaining = t1 - t
        h = min(h, remaining, max_step)
        if t + h <= t:
            raise RuntimeError("adaptive solver cannot advance floating-point time")
        if h < min_step and remaining > min_step:
            raise RuntimeError("adaptive solver step below min_step")

        stages: list[State] = []
        for c, row in zip(_C, _A):
            state = _state((
                yi + h * sum(row[j] * stages[j][i] for j in range(len(row)))
                for i, yi in enumerate(y)
            ), len(y))
            stages.append(_evaluate(rhs, t + c * h, state))
            evaluations += 1

        high = _state((
            yi + h * sum(w * k[i] for w, k in zip(_B5, stages))
            for i, yi in enumerate(y)
        ), len(y))
        low = _state((
            yi + h * sum(w * k[i] for w, k in zip(_B4, stages))
            for i, yi in enumerate(y)
        ), len(y))
        scaled_error = [
            (a - b) / (atol + rtol * max(abs(yi), abs(a)))
            for a, b, yi in zip(high, low, y)
        ]
        error = sqrt(sum(v*v for v in scaled_error) / len(y))
        if not isfinite(error):
            raise RuntimeError("nonfinite adaptive solver error")

        if error <= 1.:
            t = float(t1) if h >= remaining else float(t + h)
            if t <= times[-1]:
                raise RuntimeError("adaptive solver did not advance time")
            y = high
            times.append(t)
            states.append(y)
            accepted += 1
        else:
            rejected += 1

        factor = 5. if error == 0 else max(0.2, min(5., 0.9 * error ** (-0.2)))
        if error > 1.:
            factor = min(1., factor)
        next_h = h * factor
        if error > 1. and (next_h < min_step or t + next_h <= t):
            raise RuntimeError("adaptive solver cannot satisfy local tolerance")
        h = next_h

    trajectory = Trajectory(tuple(times), tuple(states), "rk45")
    return AdaptiveResult(trajectory, accepted, rejected, evaluations,
                          rtol, atol, dt, min_step, max_step)
