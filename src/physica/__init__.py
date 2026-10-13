"""Physica: evidence-bound computational physics and scientific ML."""

from .numerics import Trajectory, integrate, velocity_verlet
from .adaptive import AdaptiveResult, solve_adaptive
from .models import HarmonicOscillator, Pendulum, CartPole
from .validation import energy_drift, convergence_report
from .spatial import AABB, broad_phase_pairs, brute_force_pairs
from .inverse import Observation, InverseReport, fit_oscillator, synthetic_oscillator_observations
from .rl import (CartPoleEnv, Transition, ReplayBuffer, TabularQAgent,
                 TrainingReport, auxiliary_targets, train_tabular_cartpole)

__version__ = "0.3.0"
__all__ = [
    "Trajectory", "integrate", "velocity_verlet", "AdaptiveResult", "solve_adaptive",
    "HarmonicOscillator", "Pendulum", "CartPole",
    "energy_drift", "convergence_report",
    "AABB", "broad_phase_pairs", "brute_force_pairs",
    "Observation", "InverseReport", "fit_oscillator", "synthetic_oscillator_observations",
    "CartPoleEnv", "Transition", "ReplayBuffer", "TabularQAgent",
    "TrainingReport", "auxiliary_targets", "train_tabular_cartpole",
]
