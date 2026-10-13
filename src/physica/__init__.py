"""Physica: evidence-bound computational physics and scientific ML."""

from .numerics import Trajectory, integrate, velocity_verlet
from .models import HarmonicOscillator, Pendulum, CartPole
from .validation import energy_drift, convergence_report

__version__ = "0.1.0"
__all__ = [
    "Trajectory", "integrate", "velocity_verlet",
    "HarmonicOscillator", "Pendulum", "CartPole",
    "energy_drift", "convergence_report",
]
