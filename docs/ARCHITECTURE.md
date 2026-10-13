# Physica V0.3 architecture

```text
CLI -> experiment (model + solver + receipt)
            |          |
          models -> numerics (Euler, RK4, velocity Verlet)
            |           adaptive (explicit RK45, nonuniform time grid)
            |          |
            +------> validation (energy, convergence, analytic comparands)
            |
          learning (optional PyTorch)
            - Poisson PINN with hard zero-Dirichlet boundaries
            - differentiable torch RK4
            - neural residual oscillator

inverse -> scipy/numpy optional: known-mass oscillator parameter estimation
          analytic synthetic generator vs numerical RK45 prediction
          frozen timestamp holdout + local sensitivity rank diagnostic

spatial -> 2D AABB x-axis sweep + independent quadratic oracle
           candidate overlaps ONLY; no narrow-phase or contact solver

rl -> cart-pole model + RK4 + synthetic discrete-action environment
      tabular Q-learning with disjoint evaluation seeds
      seeded FIFO replay + reward/state-delta auxiliary targets
      no hardware interface; not the DeepMind UNREAL training algorithm
```

## Boundaries

- Numerical results reflect only governing equations, initial values and
  integration choices. A physically plausible output is not a calibrated result.
- Synthetic inverse observations declare SI units, time, assumed uncertainty
  scales and frozen train/holdout split. Caller-supplied provenance labels are
  not independently verified. Real sensor data requires separate calibration,
  timing and measurement chain evidence.
- The training/evaluation boundary is explicit; analytic Poisson holdout uses
  points different from the collocation grid.
- The cart-pole controller produces a candidate force in a **simulation**.
  Nothing here actuates physical machinery.
- Numerical error and physical-model error are separate; a solver can be
  accurate for an incorrect model.
- Physica has no dependency on the rest of the portfolio at runtime.

## Example equations

Oscillator, state `(x, v)`, `m x'' + c x' + k x = F`.
Conservative mechanical energy `E = (m*v*v+k*x*x)/2` is only constant when
`c=0` and `F=0`.

Pendulum, `theta'' = -(g/L) sin(theta) - damping * theta'`.
Point-mass bob model, with rotational friction term measured in inverse seconds.

Poisson benchmark on `x in (0,1)`:
`-u''(x) = pi^2 sin(pi*x)`, `u(0)=u(1)=0`.
The analytic solution is `u(x)=sin(pi*x)`. The network uses
`u_theta(x)=x(1-x) N_theta(x)`, satisfying endpoint values exactly.

Cart-pole follows the standard idealized frictionless equations with nonzero
pole inertia (uniform rod), angle zero upright, and pole length parameter
as **half** the pole's physical length.

## Reproduction and qualification

Install the package and run `python -m pytest -q`. Fixed-step comparisons
are in `tests/test_numerics.py`, adaptive-error-control and SciPy DOP853
comparison in `tests/test_adaptive.py`. Torch tests skip without `torch`;
the SciPy cross-reference test skips locally without `scipy`, but CI installs
both dependencies in dedicated jobs. Inverse tests are SciPy-dependent and
must run explicitly in CI. Spatial and RL tests are pure Python.
Read the [adaptive solver contract](ADAPTIVE_SOLVER.md) and
[eleven-repository engine / RL source study](ENGINE_AND_RL_RESEARCH.md) for
step-rejection, candidate-collision semantics, and learning limits.

Receipt digest is over canonical JSON excluding the digest itself. The schema
version scopes interpretation. It is not a cryptographically signed record.

Source-present, tests-pass, training-fits, model-validated and deployment-safe
are distinct statuses.
