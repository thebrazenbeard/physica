import math

import pytest

torch = pytest.importorskip("torch")

from physica.learning import train_poisson, torch_rk4, neural_oscillator


def test_torch_rk4_differentiates_through_parameter():
    coefficient = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    y0 = torch.tensor([1.], dtype=torch.float64)
    tr = torch_rk4(lambda t, y: -coefficient * y, y0, t1=1., steps=40)
    assert tr.shape == (41, 1)
    assert float(tr[-1].detach()) == pytest.approx(math.exp(-2.), abs=1e-6)
    tr[-1].sum().backward()
    assert coefficient.grad is not None
    assert float(coefficient.grad) == pytest.approx(-math.exp(-2.), abs=1e-5)


def test_neural_oscillator_initially_matches_physics():
    system = neural_oscillator(mass=2., stiffness=4.)
    y = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float32)
    f = system(0., y)
    assert torch.allclose(f, torch.tensor([[0., -2.], [1., 0.]]))
    tr = torch_rk4(system, y[0], t1=.5, steps=10)
    loss = tr[-1].sum()
    loss.backward()
    assert any(p.grad is not None for p in system.parameters())


def test_poisson_hard_boundary_and_training_progress():
    _, report = train_poisson(steps=45, hidden=12, collocation=16, seed=11, lr=2e-3)
    assert report.final_residual_mse < report.first_residual_mse
    assert report.boundary_max_error == 0.
    assert math.isfinite(report.holdout_relative_l2)


def test_poisson_reproducible_cpu():
    _, a = train_poisson(steps=5, hidden=8, collocation=8, seed=3)
    _, b = train_poisson(steps=5, hidden=8, collocation=8, seed=3)
    assert a == b


def test_torch_shape_guard():
    with pytest.raises(ValueError):
        torch_rk4(lambda t, y: y[:1], torch.zeros(2), t1=1., steps=1)
