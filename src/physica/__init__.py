"""Physica: evidence-bound computational physics and scientific ML."""

from .numerics import Trajectory, integrate, velocity_verlet
from .adaptive import AdaptiveResult, solve_adaptive
from .models import HarmonicOscillator, Pendulum, CartPole
from .validation import energy_drift, convergence_report

__version__ = "0.2.0"
__all__ = [
    "Trajectory", "integrate", "velocity_verlet", "AdaptiveResult", "solve_adaptive",
    "HarmonicOscillator", "Pendulum", "CartPole",
    "energy_drift", "convergence_report",
]
