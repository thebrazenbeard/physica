# Physica

**Physics-aware computation, with numerical evidence ahead of neural claims.**

Physica is a Python research foundation for physics-informed machine learning,
differentiable differential equations, and simulated embodied dynamics. This
repository began as a one-paragraph concept on October 13, 2026. The first
implementation provides mechanical reference models, fixed-step and adaptive
nonstiff ODE solvers, synthetic inverse-parameter estimation, 2D collision
candidate detection, a bounded simulated reinforcement-learning environment,
a reproducible Poisson PINN benchmark, and differentiable PyTorch dynamics.

**Status: executable V0.3 research foundation. Not a scientifically validated
general-purpose physics engine, autonomous physical controller, or safety-rated
simulation package.**

## Install

Python 3.11+:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

For independent SciPy DOP853 and oscillator inverse-problem checks, install
`.[test,reference]` (or `.[test,inverse]`). The CI pipeline installs this
extra separately to ensure that those tests actually run.

For neural differential equations and PINN training:

```bash
python -m pip install -e ".[all]"
python -m pytest -q
```

The non-ML simulations require only Python's standard library. PyTorch is an
optional dependency; the corresponding tests skip when unavailable.

CI checks ordinary editable installs across Python 3.11–3.13, explicitly
imports SciPy and PyTorch in their respective optional-dependency jobs so
that missing reference/learning coverage fails instead of silently skipping,
and installs the built wheel in a fresh environment outside the source tree.
The wheel smoke test checks distribution import and a verifiable oscillator
receipt; it does not constitute physical-model validation.

## Run

```bash
# SI units; state = position in meters, velocity in meters/second
python -m physica simulate oscillator --initial 1 0 --t1 6.2831853 --dt 0.02 --summary-only

# Adaptive Dormand-Prince 5(4): dt is an initial guess, not a sampling interval
python -m physica simulate oscillator --initial 1 0 --t1 40 --dt 1 --method rk45 --rtol 1e-9 --atol 1e-12 --summary-only

# Symplectic velocity-Verlet (bounded drift, not exact energy preservation)
# Applies to an undamped, unforced oscillator only
python -m physica simulate oscillator --initial 1 0 --t1 100 --dt 0.03 --method velocity_verlet --summary-only

# Nonconservative system: conserved-energy check explicitly inapplicable
python -m physica simulate oscillator --initial 1 0 --param damping=0.2 --t1 10 --summary-only

# Idealized pendulum (theta in radians, angular speed radians/s)
python -m physica simulate pendulum --initial 0.2 0 --t1 10 --summary-only

# Cart-pole (x, x velocity, upright-angle radians, angular velocity)
python -m physica simulate cartpole --initial 0 0 0.1 0 --t1 2 --summary-only

# Demonstration of tabular Q-learning, with frozen evaluation on separate seeds
python -m physica rl-cartpole --episodes 50 --evaluate 10 --seed 3 --max-steps 200

# Fit known-mass oscillator stiffness and damping from JSON observations
# Input JSON records: [{"time": 0, "position": 1, "sigma": 0.01}, ...]
# Generate the explicitly synthetic noisy fixture before fitting
python examples/synthetic_inverse_data.py > observations.json
python -m physica inverse-oscillator --data observations.json --mass 1 --initial 1 0 --guess 2 0.1 --holdout-every 5 --source-label "synthetic example"

# PyTorch PINN: -u'' = pi² sin(pi*x), u(0)=u(1)=0
python -m physica pinn-poisson --steps 250 --seed 0

# Differentiable neural residual force fitted on synthetic data
python examples/neural_oscillator.py
```

Leave off `--summary-only` to include every simulated timestep. Add
`--out results.json` to save JSON. Every simulation has a SHA-256 content
fingerprint, exact configuration, method, initial/final state, declared
assumptions and a qualification ceiling. Adaptive runs record accepted and
rejected steps, RHS evaluation count and error tolerances under the V2 receipt
schema; they return nonuniform timestamps and do not claim a fixed output dt. **This digest proves neither
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
| `physica.adaptive` | Dormand–Prince RK45 with weighted-RMS error control, accepted/rejected step counts, minimum step and attempt guards |
| `physica.models` | Harmonic oscillator, nonlinear pendulum, idealized cart-pole with explicit external force |
| `physica.validation` | Energy drift and exact-reference step-halving convergence diagnostics |
| `physica.experiment` | Reproducible JSON simulation and canonical content fingerprint |
| `physica.learning` | PyTorch differentiable RK4, neural acceleration residual, hard-Dirichlet Poisson PINN |
| `physica.inverse` | Known-mass oscillator stiffness/damping fit, synthetic observations, disjoint holdout and local SVD sensitivity diagnostics (SciPy optional) |
| `physica.spatial` | Independently verified 2D AABB broad-phase candidate sweep; **not** contact or rigid-body physics |
| `physica.rl` | Simulated discrete-force cart-pole, bounded replay, auxiliary reward/state-delta targets, tabular Q-learning, frozen-seed evaluation |
| `physica.cli` | Local simulation, inverse-fit, bounded RL, and PINN commands |

## Research and provenance

All **93 repositories** in the connected owner portfolio were structurally
audited, including **23 private repositories**. The private audit is kept
outside this public repository. Relevant *public* donor ideas are traced in
[Research and provenance](docs/RESEARCH_AND_PROVENANCE.md), with source commit
bindings; no proprietary portfolio code was copied. External frameworks
including DeepXDE, PhysicsNeMo, NeuralPDE, PyTorch, and torchdiffeq were
researched for architectures and failure modes. SciPy DOP853 provides an
independent numerical reference in the optional `reference` test extra.

Read [Architecture](docs/ARCHITECTURE.md), the
[Adaptive solver contract](docs/ADAPTIVE_SOLVER.md), and the
[eleven-repository engine/RL research review](docs/ENGINE_AND_RL_RESEARCH.md)
for bounded interfaces, numerical evidence and future integration options.
The linked `miyosuda/unreal` project concerns DeepMind **UNREAL auxiliary-task
reinforcement learning**, not Unreal Engine.

## Verification ceiling

A green test suite establishes **software behavior for the tested subject**.
It does *not* establish calibrated real-world predictions. PINN training
residuals, held-out solution error, numerical convergence, physical-model
adequacy, and deployment safety are separate questions. The inverse fit does not establish
global identifiability; its observation source label is caller-asserted.
The RL baseline and reward-balanced replay sampler report no demonstrated
training improvement or physical transfer. Spatial AABB overlaps are only
potential candidates, not contacts.
Physica still has no stiff/implicit solver, dense event solver, finite-element
solver, full PDE suite, hardware actuator interface, or calibrated physical
uncertainty model.

Current source code is independently implemented. No license grant has yet
been selected for this repository.

## Swept broad-phase candidates

`AABB` identifiers are normalized to nonempty immutable built-in `str` values.
String subclasses (including `StrEnum`) are accepted, but their overridden
comparison/hash behavior is discarded to preserve canonical pair order.
`AABB` stores immutable two-coordinate tuples. It accepts finite built-in
`int`/`float` coordinates and snapshots two-item input lists; NumPy scalars,
arrays, booleans and mutable custom numeric objects are not supported as
coordinates. Convert such values deliberately before constructing an AABB.

`physica.spatial.swept_broad_phase_pairs(bounds, displacements)` accepts the
same immutable 2D `AABB` values as `broad_phase_pairs`, plus **exactly one**
finite displacement tuple per box identifier. Linear translation envelopes
are swept through one abstract step. The algorithm returns sorted candidate
ID pairs, conservatively enclosing possible overlap at any intermediate time.
It is deterministic and side-effect free.

The swept envelope boundaries are evaluated using exact rational arithmetic
over the supplied finite integers and binary64 floats. In particular,
large integer coordinates must not be rounded inward by mixed int/float
endpoint addition before broad-phase pruning. Translated endpoints that
overflow the finite binary64 range remain invalid inputs.

This is **not** time-of-impact, real contacts, impulse dynamics, or a collision
response solver. Its envelope can report pairs that never coincide in time.
No physical-world effect or ARK runtime binding is authorized.

The randomized oracle samples 21 instants in multiple moving-body scenes
and checks that every independent sampled overlap is included in candidates.
The sampling is evidence for the tested configurations, not a formal proof of
all continuous trajectories.

`physica.spatial.swept_aabb_intervals(bounds, displacements)` additionally
returns sorted `SweptAABBInterval(pair, enter_time, exit_time)` records for
**simultaneous** closed overlap intervals of moving AABB pairs, with time
normalized to `[0, 1]` and at most one interval per pair. Analytic per-axis
intervals are intersected, so
envelope candidates whose boxes never coincide in time are excluded.
Edge and corner touching count as zero-duration overlap. Inputs use the same
strict finite-motion and translated-endpoint checks as the conservative sweep.
Each displacement is read once from the supplied mapping and held fixed for
the calculation; this is not an atomic snapshot of a concurrently mutated map.

Floating-point policy: contact predicates use exact-real arithmetic on the
finite binary64 input values via Python rational arithmetic. Separately
rounded float translations can disagree: `1.0 + 0.1 == 1.1` in Python, but
those particular binary64 values do **not** touch in exact arithmetic at
`t=1`; the interval routine correctly excludes that pair. An exactly
representable touch (such as `1.0 + 0.125 == 1.125`) remains included.
Returned float-valued interval endpoints are rounded approximations of exact
rational times; they are **not** exact membership predicates at the final bit
near a boundary. No broad numerical epsilon is used. This is a bounded
small-scene research routine, not an optimized large-world spatial index.
Intervals remain geometric AABB overlaps, not impulse/contact solutions or
validated physical time-of-impact predictions.
