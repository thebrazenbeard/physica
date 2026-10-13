# Adaptive Dormand–Prince V0.2: numerical contract

Status: implemented reference solver with deterministic validation tests.
This is **not** a stiff solver, root/event locator, dense-output interpolator,
or scientifically validated model of measured reality.

## Scheme

`physica.adaptive.solve_adaptive` implements the explicit 7-stage
Dormand–Prince 5(4) tableau. Fifth-order local extrapolation advances the
state; the difference from the embedded fourth-order formula estimates
per-step error. Error is measured as a weighted root-mean-square of

`(y5[i] - y4[i]) / (atol + rtol * max(abs(y_old[i]), abs(y5[i])))`.

A trial is accepted when RMS <= 1. Timestep adjustment uses a
0.9 safety factor and order-five power scaling, capped between 0.2 and 5.0.
On rejection, the proposed factor cannot increase the step. All seven
derivative evaluations are counted in the diagnostic receipt. FSAL reuse is
not implemented. No polynomial dense output is implemented.

## Failure containment

- Input t0/t1, dt and tolerances must be finite; dt, atol, min_step > 0,
  0 < rtol < 1, and min_step <= max_step.
- RHS dimensionality and all state components must remain finite.
- A trial cannot advance if floating-point resolution makes `t+h == t`.
- A rejected step that cannot reach the declared minimum step fails rather
  than silently accepting an unconverged state.
- `max_attempts` counts both accepted and rejected steps, bounding compute.
- The exact final boundary is included; the final remainder can be shorter
  than min_step solely to land at t1.
- The accepted timestamps are **nonuniform**. `dt` is the initial proposed
  step, not a uniform sampling interval. V2 receipts mark effective_dt null.

## Numerical evidence

`tests/test_adaptive.py` checks against:
- analytic exponentials and harmonic oscillators;
- tolerance tightening, rejected-step accounting, minimum step and budget;
- conservative pendulum energy drift;
- independent SciPy `solve_ivp(method="DOP853")` final-state reference on a
  nontrivial cart-pole IVP;
- deterministic, tamper-detectable but **not authenticated** receipts;
- CLI tolerance forwarding and schema integrity.

The repository's SciPy CI job intentionally installs the independent solver
rather than allowing `pytest.importorskip` to hide that comparison. Existing
standard-library and Torch jobs remain separate evidence classes.

## Hostile internal review, not independent review

> **Challenge:** An adaptive local error controller can pass its own estimates
> while producing inaccurate long-time trajectories on chaotic dynamics.

Response: true. Holdout reference comparisons and invariant monitoring are
separate checks. RK45 local estimates do not bound global error or certify
the right governing equations.

> **Challenge:** Explicit step-size adaptation may attempt many tiny steps
> on a stiff system, giving misleading confidence or exhausting resources.

Response: explicit solver only, with finite step/attempt budgets and exceptions.
Stiffness detection and implicit methods remain an open requirement.

> **Challenge:** A passing SciPy comparison is not an independent scientific
> validation if both solvers implement an idealized cart-pole equation.

Response: correct. It is **numerical cross-implementation evidence for one
initial-value problem**, not an empirical model validation.

> **Challenge:** Reproducible SHA-256 simulation fingerprints could be
> mistaken for a trustworthy scientific observation record.

Response: a hash detects altered content relative to a trusted digest.
No signer, witness, empirical observation provenance, or authorization is
asserted. No physical machine actuation is exposed by Physica.

## Research links

- [SciPy RK45 reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html)
  (Dormand–Prince 5(4), error tolerance semantics and quartic dense output).
- [2026 survey of PINN methods and benchmarking](https://www.sciencedirect.com/science/article/pii/S0925231226007149):
  distinguishes training success from external accuracy and reproducibility.
- [2026 review of formulation, training and evaluation](https://www.mdpi.com/2079-3197/14/10/242):
  residual error alone cannot establish conservation or robustness.

Physica is an independently written small implementation of the classical
formula; no SciPy code was copied into the repository.
