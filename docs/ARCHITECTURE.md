# Physica V0 architecture

```text
CLI -> experiment (model + solver + receipt)
            |          |
          models -> numerics (pure Python reference integration)
            |          |
            +------> validation (energy, convergence, analytic comparands)
            |
          learning (optional PyTorch)
            - Poisson PINN with hard zero-Dirichlet boundaries
            - differentiable torch RK4
            - neural residual oscillator
```

## Boundaries

- Numerical results reflect only governing equations, initial values and
  integration choices. A physically plausible output is not a calibrated result.
- Observations/data, when added, must carry units, instrument provenance,
  measured time, uncertainty and split membership.
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

Install the package and run `python -m pytest -q`. Baseline scientific
comparisons are exercised by `tests/test_numerics.py`. Optional Torch
neural tests are skipped when `torch` is not installed; CI exercises both
baseline and Torch jobs.

Receipt digest is over canonical JSON excluding the digest itself. The schema
version scopes interpretation. It is not a cryptographically signed record.

Source-present, tests-pass, training-fits, model-validated and deployment-safe
are distinct statuses.
