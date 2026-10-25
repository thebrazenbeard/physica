"""Independent small-angle limit checks for pendulum energy diagnostics.

For a static pendulum, gravitational potential energy is
    m*g*l*(1-cos(theta)) ~ m*g*l*theta**2/2
as theta -> 0. The series limit is independent of the production formula.
"""

from decimal import Decimal, localcontext
import math

import pytest

from physica.models import Pendulum
from physica.numerics import Trajectory
from physica.validation import energy_drift


@pytest.mark.parametrize("angle", [1e-8, -1e-9, 1e-12, -1e-14, 1e-150])
def test_pendulum_small_angle_potential_matches_independent_taylor_limit(angle):
    pendulum = Pendulum(mass=1.5, length=2., gravity=9.80665)
    expected = 0.5 * pendulum.mass * pendulum.gravity * pendulum.length * angle**2
    actual = pendulum.energy((angle, 0.))
    assert actual > 0.
    assert actual == pytest.approx(expected, rel=1e-14, abs=0.)


def test_energy_drift_exposes_changes_between_two_tiny_nonzero_angles():
    pendulum = Pendulum()
    trajectory = Trajectory((0., 1.), ((1e-9, 0.), (2e-9, 0.)), "reference")
    report = energy_drift(trajectory, pendulum.energy, tolerance=0.01)
    assert report.criterion == "relative"
    assert report.max_relative_drift == pytest.approx(3., rel=1e-14)
    assert not report.passed


def test_zero_angle_potential_and_kinetic_energy_contract():
    pendulum = Pendulum(mass=2., length=3., gravity=9.8)
    assert pendulum.energy((0., 0.)) == 0.
    assert pendulum.energy((0., 2.)) == 36.


def test_pendulum_potential_avoids_premature_multiplication_overflow():
    # Original small-angle fix multiplied 2*mass first, overflowing although
    # gravity compensates the scale and the physical potential is finite.
    pendulum = Pendulum(mass=1e308, gravity=1e-308)
    expected = 1. - math.cos(1.)
    assert pendulum.energy((1., 0.)) == pytest.approx(expected, rel=2e-15)


def test_pendulum_potential_preserves_scaled_tiny_angle_and_subnormal_value():
    # Binary64-safe independent small-angle reference using exact input floats
    # represented in high-precision decimal arithmetic.
    with localcontext() as ctx:
        ctx.prec = 80
        huge = Pendulum(mass=1e308, gravity=1e308, length=1e-308)
        tiny_angle = 1e-154
        expected = float(
            Decimal.from_float(huge.mass)
            * Decimal.from_float(huge.gravity)
            * Decimal.from_float(huge.length)
            * Decimal.from_float(tiny_angle)**2 / 2
        )
    assert huge.energy((tiny_angle, 0.)) == pytest.approx(expected, rel=3e-15)

    tiny = Pendulum(mass=1e-308, gravity=1e-308, length=1e308)
    expected_subnormal = (1. - math.cos(1.)) * 1e-308
    assert tiny.energy((1., 0.)) == pytest.approx(expected_subnormal, rel=1e-12, abs=0.)


def test_pendulum_kinetic_energy_avoids_intermediate_square_overflow():
    # (length*omega)**2 exceeds binary64, but small mass makes the final
    # kinetic energy finite: 0.5*1e-308*(1e308)**2 = 5e307.
    pendulum = Pendulum(mass=1e-308, length=1e308)
    assert pendulum.energy((0., 1.)) == pytest.approx(5e307, rel=2e-15)


def test_subnormal_angles_are_not_halved_before_compensating_multiplication():
    # theta/2 can round to zero or a different integer multiple of minsubnormal
    # although the combination m*g*l*theta**2/2 is fully representable.
    pendulum = Pendulum(mass=1e308, gravity=1e308, length=1e308)
    min_subnormal = 5e-324
    with localcontext() as ctx:
        ctx.prec = 100
        prefactor = (Decimal.from_float(pendulum.mass)
                     * Decimal.from_float(pendulum.gravity)
                     * Decimal.from_float(pendulum.length))
        for k in range(1, 200):
            angle = k * min_subnormal
            expected = float(prefactor * Decimal.from_float(angle)**2 / 2)
            actual = pendulum.energy((angle, 0.))
            assert actual == pytest.approx(expected, rel=3e-15), (k, actual, expected)


@pytest.mark.parametrize("angle", [-1.001e-3, -.999e-3, .999e-3, 1.001e-3])
def test_small_angle_branch_boundary_matches_higher_order_series(angle):
    # Independent degree-eight Taylor reference straddles the small-angle
    # branch, including both signs of displacement.
    with localcontext() as ctx:
        ctx.prec = 80
        a = Decimal.from_float(angle)
        potential = a**2 / 2 - a**4 / 24 + a**6 / 720 - a**8 / 40320
        expected = float(potential)
    assert Pendulum(mass=1., gravity=1., length=1.).energy((angle, 0.)) == (
        pytest.approx(expected, rel=2e-15))
