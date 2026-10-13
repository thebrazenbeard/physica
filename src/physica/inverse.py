"""Bounded, provenance-labelled oscillator inverse problems.

Inverse results provide numerical *local* sensitivity diagnostics, never a
certificate of structural identifiability or physical instrument calibration.
SciPy and NumPy are optional dependencies, imported only for fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, cosh, exp, hypot, isfinite, sin, sinh, sqrt
from random import Random
from typing import Sequence

from .adaptive import solve_adaptive
from .models import HarmonicOscillator


@dataclass(frozen=True)
class Observation:
    """A time, displacement (m), and assumed 1-sigma scale (m)."""

    time: float
    position: float
    sigma: float

    def __post_init__(self) -> None:
        if not all(isfinite(v) for v in (self.time, self.position, self.sigma)):
            raise ValueError("observations must be finite")
        if self.time < 0 or self.sigma <= 0:
            raise ValueError("requires nonnegative time and positive sigma")


@dataclass(frozen=True)
class InverseReport:
    stiffness: float
    damping: float
    optimizer_converged: bool
    optimizer_message: str
    iterations: int
    jacobian_rank: int
    condition_number: float | None
    identifiability: str
    training_count: int
    holdout_count: int
    training_rmse: float
    holdout_rmse: float
    weighted_training_cost: float
    source_label: str
    evidence_ceiling: str
    model: str = "linear oscillator: known mass/initial state, zero external force"
    time_unit: str = "s"
    displacement_unit: str = "m"
    stiffness_unit: str = "N/m"
    damping_unit: str = "N*s/m"
    qualification: str = "NUMERICAL_INVERSE_ONLY"


def _initial(initial: Sequence[float]) -> tuple[float, float]:
    if len(initial) != 2 or not all(isfinite(x) for x in initial):
        raise ValueError("initial must be a finite (position,velocity) pair")
    return (float(initial[0]), float(initial[1]))


def _analytic_displacement(
    time: float, *, mass: float, stiffness: float, damping: float,
    initial: tuple[float, float],
) -> float:
    """Closed-form zero-force oscillator, independent of numerical solver."""
    x0, v0 = initial
    decay = damping / (2. * mass)
    disc = stiffness / mass - decay * decay
    core = v0 + decay * x0
    if disc > 0:
        w = sqrt(disc)
        return exp(-decay * time) * (x0 * cos(w * time) + core * sin(w * time) / w)
    if disc < 0:
        w = sqrt(-disc)
        return exp(-decay * time) * (x0 * cosh(w * time) + core * sinh(w * time) / w)
    return exp(-decay * time) * (x0 + core * time)


def synthetic_oscillator_observations(
    *, times: Sequence[float], mass: float, stiffness: float, damping: float,
    initial: Sequence[float], sigma: float = .01, seed: int = 0,
    noise: bool = True,
) -> tuple[Observation, ...]:
    """Seeded demonstration data from an *analytic*, idealized law.

    `sigma` sets an assumed data-error scale. Even noisy synthetic data
    do not provide empirical model or device validation.
    """
    HarmonicOscillator(mass=mass, stiffness=stiffness, damping=damping)
    y0 = _initial(initial)
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if len(times) < 4 or any(not isfinite(t) or t < 0 for t in times):
        raise ValueError("requires at least four finite nonnegative times")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("times must strictly increase")
    rng = Random(seed)
    return tuple(
        Observation(
            float(t),
            _analytic_displacement(
                float(t), mass=mass, stiffness=stiffness,
                damping=damping, initial=y0,
            ) + (rng.gauss(0., sigma) if noise else 0.),
            float(sigma),
        )
        for t in times
    )


def _predict(
    observations: Sequence[Observation], *, mass: float, stiffness: float,
    damping: float, initial: tuple[float, float],
) -> list[float]:
    """Use Physica's own RK45, advancing to *each* observation timestamp."""
    system = HarmonicOscillator(mass=mass, stiffness=stiffness, damping=damping)
    t = 0.
    y = initial
    output: list[float] = []
    for item in observations:
        if item.time > t:
            duration = item.time - t
            path = solve_adaptive(
                system.rhs, y, t0=t, t1=item.time,
                dt=min(.1, duration), rtol=1e-9, atol=1e-11,
                max_attempts=10_000,
            )
            y = path.trajectory.final
            t = item.time
        output.append(y[0])
    return output


def fit_oscillator(
    observations: Sequence[Observation], *, initial: Sequence[float],
    mass: float, initial_guess: tuple[float, float],
    holdout_indices: Sequence[int], source_label: str,
    parameter_scales: tuple[float, float] = (1., 1.),
    max_nfev: int = 200,
    max_condition: float = 1e8,
) -> InverseReport:
    """Fit stiffness+damping from position-only samples of known-mass system.

    The holdout observations do not appear in the optimizer residual/Jacobian.
    A full-rank **local linearization** is not a structural/global proof.
    """
    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("inverse fitting requires: pip install physica[inverse]") from exc

    obs = tuple(observations)
    if len(obs) < 5 or any(not isinstance(o, Observation) for o in obs):
        raise ValueError("requires at least five Observation records")
    if any(b.time <= a.time for a, b in zip(obs, obs[1:])):
        raise ValueError("observation times must strictly increase")
    y0 = _initial(initial)
    if not isinstance(source_label, str) or not source_label.strip():
        raise ValueError("source_label must be a nonempty, caller-asserted provenance label")
    if not all(isfinite(v) and v > 0 for v in parameter_scales):
        raise ValueError("parameter scales must be positive and finite")
    if not isfinite(max_condition) or max_condition <= 1:
        raise ValueError("max_condition must be finite and greater than one")
    if type(max_nfev) is not int or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    HarmonicOscillator(mass=mass, stiffness=initial_guess[0], damping=initial_guess[1])

    hold = tuple(holdout_indices)
    if (
        not hold or len(set(hold)) != len(hold)
        or any(type(i) is not int or i < 0 or i >= len(obs) for i in hold)
        or len(obs) - len(hold) < 3
    ):
        raise ValueError("requires distinct in-range holdout indices and >=3 training points")
    hold_set = set(hold)
    train_indices = tuple(i for i in range(len(obs)) if i not in hold_set)
    train = tuple(obs[i] for i in train_indices)
    test = tuple(obs[i] for i in sorted(hold_set))

    def residual(params):
        predictions = _predict(
            train, mass=mass, stiffness=float(params[0]),
            damping=float(params[1]), initial=y0,
        )
        return np.asarray([
            (prediction - item.position) / item.sigma
            for prediction, item in zip(predictions, train)
        ])

    fit = least_squares(
        residual, x0=np.asarray(initial_guess, dtype=float),
        bounds=([1e-9, 0.], [1e6, 1e5]),
        method="trf", max_nfev=max_nfev, x_scale=parameter_scales,
    )
    stiffness, damping = (float(fit.x[0]), float(fit.x[1]))
    J = fit.jac * np.asarray(parameter_scales)[None, :]
    values = np.linalg.svd(J, compute_uv=False)
    if values.size and values[0] > 0:
        threshold = max(1e-10 * values[0], np.finfo(float).eps * max(J.shape) * values[0])
        rank = int(np.sum(values > threshold))
    else:
        rank = 0
    condition = (
        float(values[0] / values[-1])
        if rank == 2 and values[-1] > 0 else None
    )
    if not fit.success:
        diagnostic = "OPTIMIZER_NOT_CONVERGED"
    elif rank < 2:
        diagnostic = "RANK_DEFICIENT"
    elif condition is not None and condition > max_condition:
        diagnostic = "ILL_CONDITIONED"
    else:
        diagnostic = "LOCAL_FULL_RANK"

    train_predictions = _predict(train, mass=mass, stiffness=stiffness,
                                 damping=damping, initial=y0)
    test_predictions = _predict(test, mass=mass, stiffness=stiffness,
                                damping=damping, initial=y0)
    def rmse(predictions, samples):
        return sqrt(sum((a - b.position) ** 2 for a, b in zip(predictions, samples))
                    / len(samples))

    return InverseReport(
        stiffness=stiffness, damping=damping,
        optimizer_converged=bool(fit.success),
        optimizer_message=str(fit.message), iterations=int(fit.nfev),
        jacobian_rank=rank, condition_number=condition,
        identifiability=diagnostic, training_count=len(train),
        holdout_count=len(test), training_rmse=rmse(train_predictions, train),
        holdout_rmse=rmse(test_predictions, test),
        weighted_training_cost=float(fit.cost), source_label=source_label,
        # An arbitrary source label is not evidence that a dataset is synthetic.
        evidence_ceiling="UNVERIFIED_MODEL_FIT_ONLY",
    )
