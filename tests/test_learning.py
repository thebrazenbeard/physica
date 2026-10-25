import math

import pytest

torch = pytest.importorskip("torch")

from physica.learning import (train_poisson, torch_rk4, neural_oscillator,
                              _poisson_holdout_grid, _default_device_is_cpu)


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


def test_default_device_guard_on_cpu_factory():
    assert _default_device_is_cpu(torch)


def test_seeded_neural_oscillator_reproducible_and_preserves_caller_rng():
    torch.manual_seed(889)
    before = torch.random.get_rng_state().clone()
    cuda_before = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else [])
    model_a = neural_oscillator(seed=41)
    model_b = neural_oscillator(seed=41)
    model_c = neural_oscillator(seed=42)
    assert torch.equal(torch.random.get_rng_state(), before)
    if cuda_before:
        assert all(torch.equal(before_state, after_state)
                   for before_state, after_state in zip(
                       cuda_before, torch.cuda.get_rng_state_all()))

    weights_a = [p.detach().clone() for p in model_a.parameters()]
    weights_b = [p.detach().clone() for p in model_b.parameters()]
    weights_c = [p.detach().clone() for p in model_c.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(weights_a, weights_b))
    assert any(not torch.equal(a, c) for a, c in zip(weights_a, weights_c))
    # All models initially produce the same physical baseline despite their
    # different internal random weights because the correction head is zeroed.
    probe = torch.tensor([[1., .25], [-.3, .7]])
    assert torch.equal(model_a(0., probe), model_c(0., probe))


def test_seeded_neural_oscillator_rejects_nonnumeric_seed():
    with pytest.raises(ValueError, match="seed"):
        neural_oscillator(seed=True)


def test_seeded_neural_oscillator_rejects_non_cpu_default_device():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA accelerator")
    original = torch.empty(0).device
    torch.set_default_device("cuda")
    try:
        with pytest.raises(ValueError, match="CPU default"):
            neural_oscillator(seed=9)
    finally:
        torch.set_default_device(original)


def test_poisson_hard_boundary_and_training_progress():
    _, report = train_poisson(steps=45, hidden=12, collocation=16, seed=11, lr=2e-3)
    assert report.final_residual_mse < report.first_residual_mse
    assert report.boundary_max_error == 0.
    assert math.isfinite(report.holdout_relative_l2)


def test_poisson_reproducible_cpu():
    _, a = train_poisson(steps=5, hidden=8, collocation=8, seed=3)
    _, b = train_poisson(steps=5, hidden=8, collocation=8, seed=3)
    assert a == b


def test_poisson_seed_does_not_modify_caller_cpu_rng():
    torch.manual_seed(113)
    before = torch.random.get_rng_state().clone()
    train_poisson(steps=1, hidden=4, collocation=2, seed=13)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_poisson_rejects_non_cpu_default_without_changing_cuda_rng():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA accelerator")
    original = torch.empty(0).device
    torch.set_default_device("cuda")
    try:
        before = torch.cuda.get_rng_state_all()
        with pytest.raises(ValueError, match="CPU default"):
            train_poisson(steps=1, hidden=4, collocation=2, seed=13)
        assert all(torch.equal(x, y) for x, y in
                   zip(before, torch.cuda.get_rng_state_all()))
    finally:
        torch.set_default_device(original)


@pytest.mark.parametrize("collocation", [2, 39, 199, 399])
def test_poisson_actual_binary64_holdout_disjoint_from_collocation(collocation):
    train = torch.linspace(0., 1., collocation + 2,
                           dtype=torch.float64)[1:-1].unsqueeze(1)
    holdout = _poisson_holdout_grid(torch, train, collocation)
    assert holdout.shape == (199, 1)
    assert bool(torch.all(holdout[1:, 0] > holdout[:-1, 0]))
    assert not bool(torch.any(holdout == train.T))


def test_torch_shape_guard():
    with pytest.raises(ValueError):
        torch_rk4(lambda t, y: y[:1], torch.zeros(2), t1=1., steps=1)
