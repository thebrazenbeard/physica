"""Analytic mechanical laws in SI units with explicit modeling assumptions."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Callable

from .numerics import State


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")


@dataclass(frozen=True)
class HarmonicOscillator:
    mass: float = 1.
    stiffness: float = 1.
    damping: float = 0.
    force: float = 0.

    def __post_init__(self):
        _positive("mass", self.mass)
        _positive("stiffness", self.stiffness)
        _nonnegative("damping", self.damping)
        if not isfinite(self.force):
            raise ValueError("force must be finite")

    def rhs(self, t: float, state: State) -> State:
        x, v = state
        return (v, (self.force - self.stiffness * x - self.damping * v) / self.mass)

    def acceleration(self, t: float, positions: State) -> State:
        if self.damping or self.force:
            raise ValueError("velocity-Verlet requires conservative oscillator")
        return (-self.stiffness * positions[0] / self.mass,)

    def energy(self, state: State) -> float:
        x, v = state
        return .5 * self.mass * v * v + .5 * self.stiffness * x * x

    @property
    def conservative(self) -> bool:
        return self.damping == 0 and self.force == 0


@dataclass(frozen=True)
class Pendulum:
    mass: float = 1.
    length: float = 1.
    gravity: float = 9.80665
    damping: float = 0.

    def __post_init__(self):
        for name in ("mass", "length", "gravity"):
            _positive(name, getattr(self, name))
        _nonnegative("damping", self.damping)

    def rhs(self, t: float, state: State) -> State:
        theta, omega = state
        return (omega, -(self.gravity / self.length) * sin(theta) - self.damping * omega)

    def energy(self, state: State) -> float:
        theta, omega = state
        return .5 * self.mass * (self.length * omega)**2 + self.mass * self.gravity * self.length * (1 - cos(theta))

    @property
    def conservative(self) -> bool:
        return self.damping == 0


@dataclass(frozen=True)
class CartPole:
    """Frictionless cart-pole, pole length is half-length, angle zero upright.

    Force input is an external *command*, not a closed-loop controller.
    """
    cart_mass: float = 1.
    pole_mass: float = .1
    half_length: float = .5
    gravity: float = 9.80665

    def __post_init__(self):
        for name in ("cart_mass", "pole_mass", "half_length", "gravity"):
            _positive(name, getattr(self, name))

    def rhs(self, t: float, state: State, force: float = 0.) -> State:
        if not isfinite(force):
            raise ValueError("force must be finite")
        x, dx, theta, dtheta = state
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.half_length
        s, c = sin(theta), cos(theta)
        temp = (force + pole_mass_length * dtheta * dtheta * s) / total_mass
        d2theta = (self.gravity * s - c * temp) / (
            self.half_length * (4. / 3. - self.pole_mass * c * c / total_mass)
        )
        d2x = temp - pole_mass_length * d2theta * c / total_mass
        return (dx, d2x, dtheta, d2theta)


def controlled_cartpole(model: CartPole, controller: Callable[[float, State], float]):
    """Explicit actuator/controller boundary; no hidden action authorization."""
    def rhs(t: float, state: State) -> State:
        return model.rhs(t, state, float(controller(t, state)))
    return rhs
