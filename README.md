# Physica

**Physics-aware computation, with numerical evidence ahead of neural claims.**

Physica is a Python research foundation for physics-informed machine learning,
differentiable differential equations, and simulated embodied dynamics. This
repository began as a one-paragraph concept on October 13, 2026. The first
implementation provides transparent mechanical reference models, fixed-step
numerical solvers, a reproducible Poisson PINN benchmark, and differentiable
PyTorch dynamics.

**Status: executable V0 research foundation. Not a scientifically validated
general-purpose physics engine, autonomous physical controller, or safety-rated
simulation package.**

## Install

Python 3.11+:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

For neural differential equations and PINN training:

```bash
python -m pip install -e ".[all]"
python -m pytest -q
```

The non-ML simulations require only Python's standard library. PyTorch is an
optional dependency; the corresponding tests skip when unavailable.

## Run

```bash
# SI units; state = position in meters, velocity in meters/second
python -m physica simulate oscillator --initial 1 0 --t1 6.2831853 --dt 0.02 --summary-only

# Energy-preserving velocity-Verlet integrator: undamped, unforced oscillator only
python -m physica simulate oscillator --initial 1 0 --t1 100 --dt 0.03 --method velocity_verlet --summary-only

# Nonconservative system: conserved-energy check explicitly inapplicable
python -m physica simulate oscillator --initial 1 0 --param damping=0.2 --t1 10 --summary-only

# Idealized pendulum (theta in radians, angular speed radians/s)
python -m physica simulate pendulum --initial 0.2 0 --t1 10 --summary-only

# Cart-pole (x, x velocity, upright-angle radians, angular velocity)
python -m physica simulate cartpole --initial 0 0 0.1 0 --t1 2 --summary-only

# PyTorch PINN: -u'' = pi² sin(pi*x), u(0)=u(1)=0
python -m physica pinn-poisson --steps 250 --seed 0

# Differentiable neural residual force fitted on synthetic data
python examples/neural_oscillator.py
```

Leave off `--summary-only` to include every simulated timestep. Add
`--out results.json` to save JSON. Every simulation has a SHA-256 content
fingerprint, exact configuration, method, initial/final state, declared
assumptions and a qualification ceiling. **This digest proves neither
authenticity nor physical truth.**

## Library

```python
from math import cos, sin
from physica import HarmonicOscillator, integrate, convergence_report, energy_drift

system = HarmonicOscillator(mass=1.0, stiffness=1.0)
path = integrate(system.rhs, (1.0, 0.0), t1=1.0, dt=0.01, method="rk4")
check = energy_drift(path, system.energy, tolerance=1e-4)
order = convergence_report(system.rhs, (1.0, 0.0),
                           (cos(1), -sin(1)), t1=1.0)
assert check.passed and order.passed
```

### Components

| Module | Included |
| --- | --- |
| `physica.numerics` | Explicit Euler, classical RK4, velocity Verlet for velocity-independent acceleration, strict input guards |
| `physica.models` | Harmonic oscillator, nonlinear pendulum, idealized cart-pole with explicit external force |
| `physica.validation` | Energy drift and exact-reference step-halving convergence diagnostics |
| `physica.experiment` | Reproducible JSON simulation and canonical content fingerprint |
| `physica.learning` | PyTorch differentiable RK4, neural acceleration residual, hard-Dirichlet Poisson PINN |
| `physica.cli` | Local simulation and PINN commands |

## Research and provenance

All **93 repositories** in the connected owner portfolio were structurally
audited, including **23 private repositories**. The private audit is kept
outside this public repository. Relevant *public* donor ideas are traced in
[Research and provenance](docs/RESEARCH_AND_PROVENANCE.md), with source commit
bindings; no proprietary portfolio code was copied. External frameworks
including DeepXDE, PhysicsNeMo, NeuralPDE, PyTorch, and torchdiffeq were
researched for architectures and failure modes.

Read [Architecture](docs/ARCHITECTURE.md) for equations, interfaces,
assumptions, evidence boundaries and intended next research frontier.

## Verification ceiling

A green test suite establishes **software behavior for the tested subject**.
It does *not* establish calibrated real-world predictions. PINN training
residuals, held-out solution error, numerical convergence, physical-model
adequacy, and deployment safety are separate questions. This V0 foundation has
no stiff solver, finite-element solver, full PDE suite, hardware actuator
interface, or physical uncertainty model.

Current source code is independently implemented. No license grant has yet
been selected for this repository.
