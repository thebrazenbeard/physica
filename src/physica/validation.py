"""Claim-scoped numerical diagnostics; passing them does not validate a physical theory."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log2
from typing import Callable

from .numerics import Trajectory, integrate, State


@dataclass(frozen=True)
class EnergyReport:
    initial: float
    final: float
    max_absolute_drift: float
    max_relative_drift: float | None
    tolerance: float
    criterion: str
    passed: bool


def energy_drift(trajectory: Trajectory, energy: Callable[[State], float],
                 *, tolerance: float = 1e-3) -> EnergyReport:
    """Check drift relative to any nonzero starting energy, or absolute at zero.

    Relative drift is dimensionless and invariant under energy-unit scaling,
    but not under an additive shift of the energy zero. Choose and document
    the model's physically meaningful energy reference before interpreting it.
    An absolute check is necessary only when starting energy is exactly zero;
    in that case `tolerance` has the same units as the supplied energy model.
    An absolute or relative drift exceeding the finite binary64 range raises
    ValueError instead of emitting infinity into strict JSON receipts.
    Both criteria measure numerical behavior, not physical model validity.
    """
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative and finite")
    levels = [float(energy(s)) for s in trajectory.states]
    if not all(isfinite(e) for e in levels):
        raise ValueError("energy produced a nonfinite value")
    baseline = levels[0]
    drift = max(abs(e - baseline) for e in levels)
    if not isfinite(drift):
        raise ValueError("absolute energy drift exceeds representable range")
    # No arbitrary near-zero cutoff: a tiny but nonzero initial energy still
    # has a meaningful dimensionless relative drift. Absolute tolerance has
    # physical energy units and only applies at exactly zero baseline.
    relative = drift / abs(baseline) if baseline != 0. else None
    if relative is not None and not isfinite(relative):
        raise ValueError("relative energy drift exceeds representable range")
    criterion = "relative" if relative is not None else "absolute"
    passed = (relative if relative is not None else drift) <= tolerance
    return EnergyReport(baseline, levels[-1], drift, relative,
                        tolerance, criterion, passed)


@dataclass(frozen=True)
class ConvergenceReport:
    coarse_error: float
    fine_error: float
    observed_order: float | None
    required_order: float
    passed: bool


def convergence_report(rhs, initial, exact_final: State, *, t0: float = 0.,
                       t1: float = 1., dt: float = .2, method: str = "rk4",
                       min_order: float = 3.5) -> ConvergenceReport:
    """Compare exact-solution errors at h and h/2; exact solution is caller-supplied."""
    if not isfinite(min_order) or min_order < 0:
        raise ValueError("min_order must be finite and nonnegative")
    coarse = integrate(rhs, initial, t0=t0, t1=t1, dt=dt, method=method).final
    fine = integrate(rhs, initial, t0=t0, t1=t1, dt=dt / 2, method=method).final
    exact = tuple(float(x) for x in exact_final)
    if len(exact) != len(coarse) or not all(isfinite(x) for x in exact):
        raise ValueError("exact_final must be finite and match state dimension")
    a = max(abs(x - y) for x, y in zip(coarse, exact))
    b = max(abs(x - y) for x, y in zip(fine, exact))
    order = log2(a / b) if a > 0 and b > 0 else None
    passed = b <= a and (order is not None and order >= min_order or a == b == 0)
    return ConvergenceReport(a, b, order, min_order, passed)
