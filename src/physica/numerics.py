"""Transparent reference integrators, not production-grade stiff/DAE solvers.

All quantities are SI. Inputs are checked rather than silently clipped or cast.
Explicit RK4 supports general first-order systems. Velocity Verlet is for
velocity-INDEPENDENT acceleration; damping/drag must not use it.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Callable, Sequence

State = tuple[float, ...]
RHS = Callable[[float, State], Sequence[float]]
Acceleration = Callable[[float, State], Sequence[float]]


@dataclass(frozen=True)
class Trajectory:
    times: tuple[float, ...]
    states: tuple[State, ...]
    method: str

    def __post_init__(self) -> None:
        if not self.times or len(self.times) != len(self.states):
            raise ValueError("time and state arrays must be nonempty and aligned")
        n = len(self.states[0])
        if n < 1 or any(len(s) != n for s in self.states):
            raise ValueError("all states must have the same positive dimension")
        if any(not isfinite(x) for t in self.times for x in (t,)):
            raise ValueError("time must be finite")
        if any(not isfinite(x) for s in self.states for x in s):
            raise ValueError("trajectory contains a nonfinite state")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("trajectory times must be strictly increasing")

    @property
    def final(self) -> State:
        return self.states[-1]


def _state(items: Sequence[float], n: int | None = None) -> State:
    result = tuple(float(v) for v in items)
    if not result or (n is not None and len(result) != n):
        raise ValueError("invalid state vector dimension")
    if not all(isfinite(v) for v in result):
        raise ValueError("state contains NaN or infinity")
    return result


def _steps(t0: float, t1: float, dt: float, max_steps: int):
    if not all(isfinite(x) for x in (t0, t1, dt)):
        raise ValueError("time bounds and step must be finite")
    if dt <= 0 or t1 <= t0:
        raise ValueError("requires dt > 0 and t1 > t0")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be positive")
    count = ceil((t1 - t0) / dt)
    if count > max_steps:
        raise ValueError("integration exceeds max_steps")
    # Uniform partition avoids a tiny terminal step caused by roundoff.
    step = (t1 - t0) / count
    return count, step


def _evaluate(rhs: RHS, t: float, state: State) -> State:
    return _state(rhs(t, state), len(state))


def integrate(rhs: RHS, initial: Sequence[float], *, t0: float = 0.,
              t1: float = 10., dt: float = .01, method: str = "rk4",
              max_steps: int = 1_000_000) -> Trajectory:
    """Integrate using Euler, RK4, or adaptive Dormand–Prince RK45.

    For RK45, `dt` is an initial step, not a fixed output cadence. Use
    `solve_adaptive` to configure tolerances and inspect solver diagnostics.
    """
    if method == "rk45":
        from .adaptive import solve_adaptive
        return solve_adaptive(rhs, initial, t0=t0, t1=t1,
                              dt=dt, max_attempts=max_steps).trajectory
    count, h = _steps(t0, t1, dt, max_steps)
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler', 'rk4', or 'rk45'")
    state = _state(initial)
    times = [float(t0)]
    states = [state]
    for i in range(count):
        t = float(t0 + h * i)
        k1 = _evaluate(rhs, t, state)
        if method == "euler":
            state = _state([x + h * v for x, v in zip(state, k1)])
        else:
            k2 = _evaluate(rhs, t + h / 2, _state([x + h * k / 2 for x, k in zip(state, k1)]))
            k3 = _evaluate(rhs, t + h / 2, _state([x + h * k / 2 for x, k in zip(state, k2)]))
            k4 = _evaluate(rhs, t + h, _state([x + h * k for x, k in zip(state, k3)]))
            state = _state([
                x + h * (a + 2 * b + 2 * c + d) / 6
                for x, a, b, c, d in zip(state, k1, k2, k3, k4)
            ])
        times.append(float(t1) if i == count - 1 else float(t0 + (i + 1) * h))
        states.append(state)
    return Trajectory(tuple(times), tuple(states), method)


def velocity_verlet(acceleration: Acceleration, position: Sequence[float],
                    velocity: Sequence[float], *, t0: float = 0.,
                    t1: float = 10., dt: float = .01,
                    max_steps: int = 1_000_000) -> Trajectory:
    """Velocity-Verlet for a(t, q), *never* a(t, q, v).

    The returned state is (all positions, all velocities).
    """
    count, h = _steps(t0, t1, dt, max_steps)
    q, v = _state(position), _state(velocity)
    if len(q) != len(v):
        raise ValueError("position and velocity dimensions differ")
    n = len(q)
    states = [q + v]
    times = [float(t0)]
    for i in range(count):
        t = float(t0 + i * h)
        a = _state(acceleration(t, q), n)
        q_new = _state([qi + h * vi + .5 * h * h * ai for qi, vi, ai in zip(q, v, a)])
        a_new = _state(acceleration(t + h, q_new), n)
        v = _state([vi + .5 * h * (ai + ani) for vi, ai, ani in zip(v, a, a_new)])
        q = q_new
        times.append(float(t1) if i == count - 1 else float(t0 + (i + 1) * h))
        states.append(q + v)
    return Trajectory(tuple(times), tuple(states), "velocity_verlet")
