"""Reproducible simulation records with explicit solver and qualification scope."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from json import dumps
from math import isinf
from typing import Any

from . import __version__
from .adaptive import solve_adaptive
from .models import HarmonicOscillator, Pendulum, CartPole
from .numerics import integrate, velocity_verlet
from .validation import energy_drift


def _canonical(obj: Any) -> bytes:
    return dumps(obj, ensure_ascii=False, allow_nan=False,
                 sort_keys=True, separators=(",", ":")).encode("utf-8")


def simulate(model: str, *, initial: tuple[float, ...], t1: float = 10.,
             dt: float = .01, method: str = "rk4",
             parameters: dict[str, float] | None = None,
             energy_tolerance: float = 1e-3,
             include_trajectory: bool = True,
             rtol: float | None = None, atol: float | None = None,
             max_attempts: int | None = None) -> dict:
    """Return a content-addressed numerical run record, not a validation claim.

    For `rk45`, `dt` is an initial guess and `effective_dt` is null;
    accepted timestamps must be read from the trajectory.
    """
    parameters = dict(parameters or {})
    classes = {"oscillator": HarmonicOscillator, "pendulum": Pendulum,
               "cartpole": CartPole}
    if model not in classes:
        raise ValueError(f"unknown model {model!r}")
    system = classes[model](**parameters)
    needed = 4 if model == "cartpole" else 2
    if len(initial) != needed:
        raise ValueError(f"{model} expects {needed} state values")
    adaptive = None
    if method == "rk45":
        options = {}
        if rtol is not None:
            options["rtol"] = rtol
        if atol is not None:
            options["atol"] = atol
        if max_attempts is not None:
            options["max_attempts"] = max_attempts
        adaptive = solve_adaptive(system.rhs, initial, t1=t1, dt=dt, **options)
        traj = adaptive.trajectory
    else:
        if rtol is not None or atol is not None or max_attempts is not None:
            raise ValueError("adaptive solver options require method='rk45'")
        if method == "velocity_verlet":
            if model != "oscillator" or not system.conservative:
                raise ValueError("velocity_verlet requires an undamped/unforced oscillator")
            traj = velocity_verlet(system.acceleration, initial[:1],
                                   initial[1:], t1=t1, dt=dt)
        else:
            traj = integrate(system.rhs, initial, t1=t1, dt=dt, method=method)

    result: dict = {
        "schema": "physica.simulation.v2" if adaptive else "physica.simulation.v1",
        "version": __version__,
        "model": model,
        "parameters": asdict(system),
        "initial": list(initial),
        "t0": 0.,
        "t1": t1,
        "requested_dt": dt,
        "effective_dt": None if adaptive else (t1 / (len(traj.times) - 1)),
        "method": method,
        "steps": len(traj.times) - 1,
        "final_state": list(traj.final),
        "qualification": "NUMERICAL_OUTPUT_ONLY",
        "assumptions": [
            "SI units",
            "idealized equations as defined in physica.models",
            ("adaptive-step explicit Dormand-Prince 5(4) local-error control"
             if adaptive else "fixed-step explicit numerical integration"),
        ],
    }
    if adaptive:
        result["adaptive"] = {
            "accepted_steps": adaptive.accepted_steps,
            "rejected_steps": adaptive.rejected_steps,
            "attempted_steps": adaptive.attempted_steps,
            "evaluations": adaptive.evaluations,
            "rtol": adaptive.rtol,
            "atol": adaptive.atol,
            "initial_step": adaptive.initial_step,
            "min_step": adaptive.min_step,
            "max_step": None if isinf(adaptive.max_step) else adaptive.max_step,
        }
    if model != "cartpole":
        if system.conservative:
            result["energy_check"] = asdict(
                energy_drift(traj, system.energy, tolerance=energy_tolerance)
            )
        else:
            result["energy_check"] = None
            result["energy_check_explanation"] = (
                "External work/dissipation: constant energy is not expected."
            )
    if include_trajectory:
        result["trajectory"] = [
            {"time": t, "state": list(s)} for t, s in zip(traj.times, traj.states)
        ]
    result["sha256"] = sha256(_canonical(result)).hexdigest()
    return result


def verify_receipt(receipt: dict) -> bool:
    if not isinstance(receipt, dict):
        return False
    digest = receipt.get("sha256")
    body = {k: v for k, v in receipt.items() if k != "sha256"}
    try:
        return isinstance(digest, str) and sha256(_canonical(body)).hexdigest() == digest
    except (TypeError, ValueError):
        return False
