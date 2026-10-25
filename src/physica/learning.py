"""Optional PyTorch scientific ML: PINN and differentiable neural dynamics.

Neural fits are experimental approximations. They do not inherit the validity
of the equations imposed as training losses.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import pi, isfinite


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install Physica with the 'learn' extra to enable PyTorch") from exc
    return torch


def _default_device_is_cpu(torch) -> bool:
    """Use a no-RNG tensor factory; torch.get_default_device needs Torch 2.3+."""
    return torch.empty(0).device.type == "cpu"


@dataclass(frozen=True)
class PINNReport:
    steps: int
    collocation: int
    seed: int
    first_residual_mse: float
    final_residual_mse: float
    holdout_relative_l2: float
    boundary_max_error: float


def _poisson_holdout_rationals(collocation: int) -> tuple[Fraction, ...]:
    """199 interior samples disjoint from i/(collocation+1) in exact arithmetic.

    If one equalled a training coordinate, multiplication by 400*(n+1)
    would give 2*i*(n+1)+1 = 400*j: odd equals even, impossible.
    """
    return tuple(Fraction(i, 200) + Fraction(1, 400 * (collocation + 1))
                 for i in range(1, 200))


def _poisson_holdout_grid(torch, training, collocation: int):
    """Also eliminate any equality introduced by binary64 rounding."""
    samples = torch.tensor(
        [float(value) for value in _poisson_holdout_rationals(collocation)],
        dtype=training.dtype, device=training.device,
    )
    collocation_points = training.detach().reshape(-1)
    while True:
        indices = torch.searchsorted(collocation_points, samples)
        in_range = indices < collocation_points.numel()
        collision = in_range & (collocation_points[
            indices.clamp(max=collocation_points.numel() - 1)
        ] == samples)
        if not bool(collision.any()):
            break
        samples[collision] = torch.nextafter(
            samples[collision], torch.ones_like(samples[collision]))
        if not bool((samples < 1.).all()):
            raise FloatingPointError("no interior representable holdout sample")
    if not bool((samples[1:] > samples[:-1]).all()):
        raise FloatingPointError("binary64 holdout samples not strictly increasing")
    return samples.unsqueeze(1)


def train_poisson(*, steps: int = 250, hidden: int = 24,
                  collocation: int = 32, seed: int = 0, lr: float = 1e-3):
    """PINN for -u''(x) = pi² sin(pi*x), x in (0,1), u(0)=u(1)=0.

    Analytic spatial holdout reference: u(x)=sin(pi*x), sampled away from
    the training collocation coordinates. Hard boundary ansatz
    u=x(1-x)N(x) so endpoints are enforced regardless of optimizer.
    Requires CPU as PyTorch's default device for isolated seeded training;
    it does not support accelerator-default training. Returns (trained
    torch.nn.Module, PINNReport); not a validated PDE solver.
    """
    torch = _torch()
    if steps <= 0 or hidden < 2 or collocation < 2 or lr <= 0:
        raise ValueError("invalid training configuration")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not _default_device_is_cpu(torch):
        raise ValueError("train_poisson requires a CPU default device")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1)
        ).double()
    class HardBoundary(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            return x * (1. - x) * self.backbone(x)

    model = HardBoundary(net)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.linspace(0., 1., collocation + 2, dtype=torch.float64)[1:-1].unsqueeze(1)
    x.requires_grad_(True)
    first = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        u = model(x)
        ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
        residual = -uxx - (pi**2) * torch.sin(pi * x)
        loss = torch.mean(residual * residual)
        if first is None:
            first = float(loss.detach())
        if not torch.isfinite(loss):
            raise FloatingPointError("PINN residual loss diverged")
        loss.backward()
        optimizer.step()
    # Re-evaluate after final optimizer step (not the previous step's loss).
    u = model(x)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
    final = float(torch.mean((-uxx - pi**2 * torch.sin(pi * x))**2).detach())
    with torch.no_grad():
        holdout = _poisson_holdout_grid(torch, x, collocation)
        reference = torch.sin(pi * holdout)
        prediction = model(holdout)
        relative = float(torch.linalg.vector_norm(prediction - reference) /
                         torch.linalg.vector_norm(reference))
        boundaries = torch.tensor([[0.], [1.]], dtype=torch.float64)
        boundary = float(torch.max(torch.abs(model(boundaries))))
    return model, PINNReport(steps, collocation, seed, first, final, relative, boundary)


def torch_rk4(rhs, initial, *, t0: float = 0., t1: float = 1., steps: int = 20):
    """Differentiable unrolled classical RK4; torch tensor shape [..., state_dim].

    Intended for small, nonstiff tasks. Does not claim adaptive stiff solver support.
    """
    torch = _torch()
    if steps < 1 or not t1 > t0:
        raise ValueError("requires steps >= 1 and t1 > t0")
    if not isinstance(initial, torch.Tensor) or not initial.is_floating_point():
        raise TypeError("initial must be a floating-point torch tensor")
    h = (t1 - t0) / steps
    y = initial
    states = [y]
    for i in range(steps):
        t = t0 + h * i
        k1 = rhs(t, y)
        k2 = rhs(t + .5*h, y + .5*h*k1)
        k3 = rhs(t + .5*h, y + .5*h*k2)
        k4 = rhs(t + h, y + h*k3)
        if any(k.shape != y.shape for k in (k1, k2, k3, k4)):
            raise ValueError("ODE rhs returned wrong state shape")
        y = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        if not bool(torch.isfinite(y).all()):
            raise FloatingPointError("nonfinite learned dynamics")
        states.append(y)
    return torch.stack(states, dim=0)


def neural_oscillator(*, mass: float = 1., stiffness: float = 1.,
                      damping: float = 0., hidden: int = 16,
                      seed: int | None = None):
    """Construct physical baseline + learnable acceleration residual network.

    The residual can violate energy conservation. Use held-out validation
    before treating its predictions as scientific evidence. An explicit seed
    reproducibly initializes the residual network on a CPU default device
    without changing the caller's PyTorch random state. Move the returned
    model to another device after initialization if needed. Without a seed,
    standard PyTorch initialization and random-state behavior are preserved.
    """
    torch = _torch()
    if not all(isfinite(v) for v in (mass, stiffness, damping)) or mass <= 0 or stiffness <= 0 or damping < 0 or hidden < 2:
        raise ValueError("invalid oscillator configuration")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer or None")
    if seed is not None and not _default_device_is_cpu(torch):
        raise ValueError("seeded initialization requires a CPU default device")

    class NeuralOscillator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.correction = torch.nn.Sequential(
                torch.nn.Linear(2, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, 1)
            )
            # Zero correction at initialization preserves the physical baseline.
            torch.nn.init.zeros_(self.correction[-1].weight)
            torch.nn.init.zeros_(self.correction[-1].bias)

        def forward(self, t, state):
            x, v = state[..., 0], state[..., 1]
            accel = (-stiffness * x - damping * v) / mass
            accel = accel + self.correction(state).squeeze(-1)
            return torch.stack((v, accel), dim=-1)

    if seed is None:
        return NeuralOscillator()
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed would also reseed CUDA/MPS/XPU generators, which
        # are deliberately not included in this CPU-only fork.
        torch.default_generator.manual_seed(seed)
        return NeuralOscillator()
