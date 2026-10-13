import math

import pytest

from physica.numerics import Trajectory, integrate, velocity_verlet
from physica.models import HarmonicOscillator, Pendulum, CartPole, controlled_cartpole
from physica.validation import energy_drift, convergence_report


def test_rk4_oscillator_matches_analytic_solution():
    model = HarmonicOscillator()
    path = integrate(model.rhs, (1., 0.), t1=2 * math.pi, dt=.02)
    assert path.times[-1] == 2 * math.pi
    assert path.final[0] == pytest.approx(1., abs=2e-8)
    assert path.final[1] == pytest.approx(0., abs=2e-8)


def test_rk4_has_expected_convergence_order():
    model = HarmonicOscillator()
    target = (math.cos(1.), -math.sin(1.))
    report = convergence_report(model.rhs, (1., 0.), target, dt=.2)
    assert report.passed
    assert report.observed_order > 3.5


def test_euler_is_less_accurate_than_rk4():
    m = HarmonicOscillator()
    reference = math.cos(2.)
    euler = integrate(m.rhs, (1., 0.), t1=2., dt=.1, method="euler")
    rk4 = integrate(m.rhs, (1., 0.), t1=2., dt=.1, method="rk4")
    assert abs(rk4.final[0] - reference) < abs(euler.final[0] - reference)


def test_verlet_conserves_oscillator_energy_long_run():
    m = HarmonicOscillator(mass=2., stiffness=3.)
    tr = velocity_verlet(m.acceleration, (1.,), (0.,), t1=200., dt=.03)
    report = energy_drift(tr, m.energy, tolerance=.001)
    assert report.passed
    assert report.max_relative_drift < .001


def test_pendulum_energy_conservation_and_bounds():
    m = Pendulum()
    tr = integrate(m.rhs, (.2, 0.), dt=.01, t1=20.)
    assert energy_drift(tr, m.energy, tolerance=1e-5).passed


def test_zero_initial_energy_uses_absolute_criterion():
    m = HarmonicOscillator()
    tr = integrate(m.rhs, (0., 0.), t1=1.)
    report = energy_drift(tr, m.energy)
    assert report.criterion == "absolute"
    assert report.max_relative_drift is None
    assert report.passed


def test_damped_oscillator_dissipates_energy():
    m = HarmonicOscillator(damping=.3)
    tr = integrate(m.rhs, (1., 0.), dt=.05, t1=8.)
    values = [m.energy(s) for s in tr.states]
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))
    assert values[-1] < values[0]


def test_trajectory_final_uneven_requested_step():
    tr = integrate(lambda t, y: (1.,), (0.,), t0=2., t1=3., dt=.3)
    assert tr.times[0] == 2.
    assert tr.times[-1] == 3.
    assert tr.final[0] == pytest.approx(1.)


@pytest.mark.parametrize("opts", [
    {"dt": 0.}, {"dt": -1.}, {"dt": math.nan},
    {"t1": 0.}, {"t1": math.inf}, {"dt": 1e-9, "max_steps": 50}
])
def test_time_fail_closed(opts):
    with pytest.raises(ValueError):
        integrate(lambda t, x: (1.,), (1.,), **opts)


@pytest.mark.parametrize("state", [(math.nan,), (math.inf,), ()])
def test_invalid_initial_state_rejected(state):
    with pytest.raises(ValueError):
        integrate(lambda t, y: (1.,), state, t1=1.)


def test_rhs_dimension_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        integrate(lambda t, y: (1., 2.), (0.,), t1=1.)
    with pytest.raises(ValueError):
        integrate(lambda t, y: (math.nan,), (0.,), t1=1.)


def test_wrong_verlet_dimensions_rejected():
    with pytest.raises(ValueError):
        velocity_verlet(lambda t, q: q, (1., 2.), (0.,))


def test_verlet_refuses_nonconservative_oscillator():
    with pytest.raises(ValueError, match="conservative"):
        velocity_verlet(HarmonicOscillator(damping=.1).acceleration,
                        (1.,), (0.,), t1=1.)


def test_cartpole_equilibrium_and_control_boundary():
    m = CartPole()
    assert m.rhs(0., (0., 0., 0., 0.)) == (0., 0., 0., 0.)
    rhs = controlled_cartpole(m, lambda t, state: 1.)
    assert rhs(0., (0., 0., 0., 0.))[1] > 0
    assert rhs(0., (0., 0., 0., 0.))[3] < 0


def test_bad_model_parameters_rejected():
    with pytest.raises(ValueError):
        HarmonicOscillator(mass=0.)
    with pytest.raises(ValueError):
        Pendulum(length=-1.)
    with pytest.raises(ValueError):
        CartPole(pole_mass=math.inf)


def test_invalid_trajectory_rejected():
    with pytest.raises(ValueError):
        Trajectory((0., 0.), ((1.,), (2.,)), "invalid")
    with pytest.raises(ValueError):
        Trajectory((0., 1.), ((1.,), (float("nan"),)), "invalid")
