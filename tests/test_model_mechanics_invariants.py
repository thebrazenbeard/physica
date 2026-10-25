"""Analytic balance checks for Physica's idealized frictionless cart-pole.

The independent reference is the planar mechanics of a uniform rod of length
2*l pivoted on a translating cart. These checks do not calibrate that model
against any physical apparatus.
"""

import math

import pytest

from physica.models import CartPole
from physica.numerics import integrate


def _mechanical_energy(model: CartPole, state: tuple[float, ...]) -> float:
    """Uniform-rod kinetic plus gravitational potential energy (SI units)."""
    _, velocity, angle, angular_velocity = state
    cart_and_pole = model.cart_mass + model.pole_mass
    coupling = model.pole_mass * model.half_length
    # A uniform rod of total length 2*l has I_about_pivot = 4*m*l**2/3.
    return (
        .5 * cart_and_pole * velocity**2
        + coupling * velocity * angular_velocity * math.cos(angle)
        + (2. / 3.) * model.pole_mass * model.half_length**2 * angular_velocity**2
        + coupling * model.gravity * math.cos(angle)
    )


def _horizontal_momentum(model: CartPole, state: tuple[float, ...]) -> float:
    _, velocity, angle, angular_velocity = state
    return (
        (model.cart_mass + model.pole_mass) * velocity
        + model.pole_mass * model.half_length * angular_velocity * math.cos(angle)
    )


@pytest.mark.parametrize("configuration,state,force", [
    ({}, (0., 0., 0., 0.), 0.),
    ({}, (.1, .4, -.4, .5), 0.),
    ({}, (-.3, 1.7, 2.8, -.9), 3.5),
    ({"cart_mass": 3., "pole_mass": .25, "half_length": 1.2, "gravity": 9.5},
     (2., -1.4, -2.1, 2.4), -7.),
    ({"cart_mass": .3, "pole_mass": 1.7, "half_length": .4, "gravity": 1.62},
     (0., .01, math.pi / 2., -.2), 1.25),
])
def test_cartpole_lagrangian_force_and_power_identities(configuration, state, force):
    """The RHS must satisfy d(p_x)/dt=F and d(E)/dt=F*x_dot."""
    model = CartPole(**configuration)
    _, velocity, theta, omega = state
    _, cart_acceleration, _, angular_acceleration = model.rhs(0., state, force)
    total_mass = model.cart_mass + model.pole_mass
    ml = model.pole_mass * model.half_length
    c, s = math.cos(theta), math.sin(theta)

    momentum_rate = (
        total_mass * cart_acceleration
        + ml * (angular_acceleration * c - omega**2 * s)
    )
    energy_rate = (
        (total_mass * velocity + ml * omega * c) * cart_acceleration
        + (ml * velocity * c + 4. / 3. * ml * model.half_length * omega)
        * angular_acceleration
        - ml * velocity * omega**2 * s
        - ml * model.gravity * omega * s
    )
    assert momentum_rate == pytest.approx(force, rel=1e-12, abs=1e-12)
    assert energy_rate == pytest.approx(force * velocity, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("force", [0., 1.5])
def test_cartpole_integrated_energy_work_and_horizontal_impulse(force):
    """Finite-window RK4 trajectories should converge to the balance laws."""
    model = CartPole(cart_mass=1.3, pole_mass=.3, half_length=.65)
    # Near the stable hanging position; no reset or balance controller.
    initial = (0., .13, math.pi + .18, -.08)
    trajectory = integrate(
        lambda t, state: model.rhs(t, state, force),
        initial, t0=0., t1=2., dt=.005, method="rk4",
    )
    final = trajectory.final
    expected_work = force * (final[0] - initial[0])
    expected_impulse = force * 2.
    assert _mechanical_energy(model, final) - _mechanical_energy(model, initial) == (
        pytest.approx(expected_work, rel=5e-9, abs=5e-10)
    )
    assert _horizontal_momentum(model, final) - _horizontal_momentum(model, initial) == (
        pytest.approx(expected_impulse, rel=5e-9, abs=5e-10)
    )
