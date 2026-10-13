"""Deterministic experiment receipts with explicit scope, not scientific certification."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from json import dumps
from typing import Any

from . import __version__
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
             include_trajectory: bool = True) -> dict:
    """Return a content-addressed, reproducible numerical run record.

    No output makes claims about real-world calibration, validation, or safety.
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
    if method == "velocity_verlet":
        if model != "oscillator" or not system.conservative:
            raise ValueError("velocity_verlet requires an undamped/unforced oscillator")
        traj = velocity_verlet(system.acceleration, initial[:1],
                               initial[1:], t1=t1, dt=dt)
    else:
        traj = integrate(system.rhs, initial, t1=t1, dt=dt, method=method)
    result: dict = {
        "schema": "physica.simulation.v1",
        "version": __version__,
        "model": model,
        "parameters": asdict(system),
        "initial": list(initial),
        "t0": 0.,
        "t1": t1,
        "requested_dt": dt,
        "effective_dt": (t1 / (len(traj.times) - 1)),
        "method": method,
        "steps": len(traj.times) - 1,
        "final_state": list(traj.final),
        "qualification": "NUMERICAL_OUTPUT_ONLY",
        "assumptions": [
            "SI units",
            "idealized equations as defined in physica.models",
            "fixed-step explicit numerical integration",
        ],
    }
    if model != "cartpole":
        if system.conservative:
            report = energy_drift(traj, system.energy, tolerance=energy_tolerance)
            result["energy_check"] = asdict(report)
        else:
            result["energy_check"] = None
            result["energy_check_explanation"] = "External work/dissipation: constant energy is not expected."
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
