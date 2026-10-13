"""Behavioral tests for explicit Dormand–Prince and scientific evidence boundaries."""
import json
import math
import subprocess
import sys

import pytest

from physica import HarmonicOscillator, Pendulum, CartPole
from physica.adaptive import solve_adaptive
from physica.numerics import integrate
from physica.experiment import simulate, verify_receipt
from physica.validation import energy_drift


def test_rk45_matches_exact_oscillator():
    oscillator = HarmonicOscillator()
    path = integrate(oscillator.rhs, (1., 0.), t1=10., dt=1., method="rk45")
    assert path.method == "rk45"
    assert path.times[-1] == 10.
    assert abs(path.final[0] - math.cos(10.)) < 1e-4
    assert abs(path.final[1] + math.sin(10.)) < 1e-4


def test_tolerance_measurably_changes_solver_effort_and_accuracy():
    rhs = lambda t, y: (y[0],)
    coarse = solve_adaptive(rhs, (1.,), t1=5., dt=2., rtol=1e-3, atol=1e-6)
    fine = solve_adaptive(rhs, (1.,), t1=5., dt=2., rtol=1e-10, atol=1e-12)
    true = math.exp(5.)
    assert abs(fine.trajectory.final[0] - true) < abs(coarse.trajectory.final[0] - true)
    assert fine.accepted_steps > coarse.accepted_steps
    assert fine.trajectory.times[-1] == 5.
    assert fine.evaluations == 7 * fine.attempted_steps
    assert fine.rejected_steps > 0


def test_nonuniform_time_grid_and_no_hidden_fixed_step_claim():
    s = solve_adaptive(Pendulum().rhs, (.5, 0.), t1=7., dt=1., rtol=1e-8)
    gaps = [b-a for a,b in zip(s.trajectory.times, s.trajectory.times[1:])]
    assert len(gaps) == s.accepted_steps
    assert len({round(x, 6) for x in gaps}) > 2
    assert all(x > 0 for x in gaps)
    assert s.trajectory.times[-1] == 7.
    assert energy_drift(s.trajectory, Pendulum().energy, tolerance=1e-5).passed


def test_vector_system_versus_independent_scipy_reference():
    scipy = pytest.importorskip("scipy.integrate")
    model = CartPole(cart_mass=1.2, pole_mass=.2, half_length=.4)
    y0 = (0., .1, .5, -.2)
    result = solve_adaptive(model.rhs, y0, t1=3.3, dt=.4, rtol=2e-9, atol=1e-11)
    reference = scipy.solve_ivp(
        lambda t, y: model.rhs(t, tuple(y)),
        (0., 3.3), y0, method="DOP853", rtol=2e-12, atol=1e-13,
    )
    assert reference.success
    assert all(abs(a-b) < 1e-6 for a,b in zip(result.trajectory.final, reference.y[:, -1]))


@pytest.mark.parametrize("bad", [
    {"dt": 0.}, {"dt": float("nan")}, {"t1": 0.}, {"t0": 2., "t1": 1.},
    {"rtol": 0.}, {"rtol": 1.}, {"atol": 0.}, {"atol": float("inf")},
    {"min_step": 0.}, {"max_step": 0.}, {"min_step": 2., "max_step": 1.},
    {"max_attempts": 0}, {"max_attempts": True},
])
def test_invalid_solver_settings_fail_closed(bad):
    options = {"t1": 1.}
    options.update(bad)
    with pytest.raises(ValueError):
        solve_adaptive(lambda t, s: (s[0],), (1.,), **options)


def test_nonfinite_rhs_and_wrong_dimension_fail_closed():
    with pytest.raises(ValueError):
        solve_adaptive(lambda t, s: (float("nan"),), (1.,), t1=1.)
    with pytest.raises(ValueError):
        solve_adaptive(lambda t, s: (1., 2.), (1.,), t1=1.)


def test_iteration_budget_is_hard_limit():
    with pytest.raises(RuntimeError, match="max_attempts"):
        solve_adaptive(lambda t, y: (y[0],), (1.,), t1=50., dt=10.,
                       rtol=1e-10, atol=1e-13, max_attempts=1)


def test_minimum_step_fails_on_impossible_tolerance():
    with pytest.raises(RuntimeError, match="local tolerance"):
        solve_adaptive(lambda t, y: (y[0],), (1.,), t1=10., dt=1.,
                       rtol=1e-12, atol=1e-14, min_step=.9)


def test_adaptive_experiment_receipt_is_explicit_and_reproducible():
    a = simulate("oscillator", initial=(1., 0.), t1=10., dt=1.,
                 method="rk45", include_trajectory=False)
    b = simulate("oscillator", initial=(1., 0.), t1=10., dt=1.,
                 method="rk45", include_trajectory=False)
    assert a == b and verify_receipt(a)
    assert a["schema"] == "physica.simulation.v2"
    assert a["effective_dt"] is None
    assert a["adaptive"]["accepted_steps"] == a["steps"]
    assert a["adaptive"]["evaluations"] >= 7*a["steps"]
    assert "adaptive" in " ".join(a["assumptions"])
    assert a["qualification"] == "NUMERICAL_OUTPUT_ONLY"


def test_cli_accepts_rk45_and_tolerances():
    cp = subprocess.run([sys.executable, "-m", "physica", "simulate", "pendulum",
                         "--initial", ".1", "0", "--t1", "2", "--dt", ".5",
                         "--method", "rk45", "--rtol", "1e-8",
                         "--atol", "1e-10", "--summary-only"],
                        text=True, capture_output=True,
                        env=__import__("os").environ.copy())
    assert cp.returncode == 0, cp.stderr
    rec = json.loads(cp.stdout)
    assert rec["adaptive"]["rtol"] == 1e-8
    assert rec["adaptive"]["atol"] == 1e-10
    assert verify_receipt(rec)
