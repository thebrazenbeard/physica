"""Analytic mechanical laws in SI units with explicit modeling assumptions."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, frexp, inf, isfinite, ldexp, sin
from typing import Callable

from .numerics import State


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _nonnegative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")


def _scaled_nonnegative_product(*factors: float) -> float:
    """Multiply nonnegative floats without overflowing compensating factors.

    Keep each factor's binary exponent separate until the final rounding. This
    also preserves subnormal results when tiny factors are offset by large ones.
    """
    if any(value == 0. for value in factors):
        return 0.
    mantissa, exponent = 1., 0
    for value in factors:
        fraction, power = frexp(value)
        mantissa *= fraction
        exponent += power
    try:
        return ldexp(mantissa, exponent)
    except OverflowError:
        return inf


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
        # 1-cos(theta) loses all significance near theta=0; the half-angle
        # identity preserves tiny but physically nonzero potential energies.
        # Delay multiplication until the end: even finite physical energies
        # can otherwise have overflowing intermediate mass/length products.
        kinetic = _scaled_nonnegative_product(.5, self.mass, self.length,
                                              self.length, abs(omega), abs(omega))
        if abs(theta) < 1e-3:
            # Halving a subnormal theta can itself round to zero. Retain raw
            # theta factors; only the well-conditioned Taylor correction uses
            # theta**2, whose underflow here is harmless.
            theta_sq = theta * theta
            correction = .5 - theta_sq / 24. + theta_sq * theta_sq / 720.
            potential = _scaled_nonnegative_product(
                self.mass, self.gravity, self.length,
                abs(theta), abs(theta), correction)
        else:
            half_sine = abs(sin(theta / 2.))
            potential = _scaled_nonnegative_product(2., self.mass, self.gravity,
                                                    self.length, half_sine, half_sine)
        return kinetic + potential

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
