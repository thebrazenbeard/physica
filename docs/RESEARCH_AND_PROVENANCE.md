# Physica research and design record

Date: 2026-10-13. Status: **implementation foundation**, not a scientifically
validated all-purpose physics/engineering system.

## Problem formulation

Physica targets physics-informed machine learning, neural differential equations,
and embodied dynamics. A coherent initial build needs:
1. A transparent deterministic differential-equation baseline, with analytic checks.
2. Explicit physical assumptions, units, conserved quantities and forcing terms.
3. A differentiable integration path capable of backpropagating through dynamics.
4. A reproducible PINN benchmark with independently evaluated solution error.
5. An embodied state/action example whose control input does not masquerade as
   real actuator authority.

A PINN training residual is *not* evidence that the underlying physical law is
true or that the fitted model generalizes. Models can overfit collocation grids,
lose sharp scales, satisfy weighted losses unevenly, and omit crucial terms.
Numerical baseline and physical/model-validity qualification must stay separate.

## Research: relevant external implementations and limitations

| Reference | Mechanism studied | Physica V0 decision |
| --- | --- | --- |
| [DeepXDE forward demos](https://deepxde.readthedocs.io/en/stable/demos/pinn_forward.html) | ODE/PDE problems, diverse boundary conditions, reference solvers | Start with one exact-solution Poisson case, not a large PDE DSL. |
| [NVIDIA PhysicsNeMo v2.0 PINNs](https://docs.nvidia.com/physicsnemo/26.05/user-guide/pinns-tutorials/index.html) | Explicit PyTorch network, differential operator, residual, optimizer | Keep independent residual, boundary, held-out error metrics. No heavy framework dependency yet. |
| [NVIDIA migration guide](https://github.com/NVIDIA/physicsnemo/blob/main/v2.0-MIGRATION-GUIDE.md) | Sym functionality moved into PhysicsNeMo, explicit PyTorch workflow | Avoid committing to legacy PhysicsNeMo-Sym APIs. |
| [SciML NeuralPDE](https://docs.sciml.ai/NeuralPDE/stable/) | Adaptive loss, symbolic operators and quadrature strategies | Candidate for future benchmark comparison, not silently imported. |
| [PyTorch func APIs](https://docs.pytorch.org/docs/stable/func.api.html) | Differentiable Jacobian and vectorization transforms | Start with double-precision autograd for Poisson derivatives. Later profile jacrev/vmap for multidimensional PDEs. |
| [torchdiffeq](https://github.com/rtqichen/torchdiffeq) | Differentiable integration and adjoints | Reference for eventual adaptive solvers; this build uses a clearly labeled unrolled RK4. |
| [PINN failure study (2025)](https://www.sciencedirect.com/science/article/pii/S089360802400892X) | PDE vs boundary loss imbalance | Use hard zero-Dirichlet construction; compare held-out *solution* to analytic truth. |
| [PINN engineering limits (2025)](https://www.sciencedirect.com/science/article/pii/S0360835225008502) | Plausible but wrong predictions from idealized laws | Refuse to promote training metrics into real-world validation. |

These are research references, not dependencies, endorsements, or source-code donors.

## Portfolio-wide audit

A read-only tree inventory was run against **all 93 owner repositories** with the
authenticated GitHub connection (93/93 successful, 23 private) on 2026-10-13.
The detailed report was written outside the Physica repository so that private
names, paths, and source details are not published. The reproducible audit script
is `tools/portfolio_audit.py`.

Deep inspection below is of selected **public repositories** whose source or
documentation is materially relevant. These are separately owned systems; no
external portfolio code was copied into Physica, and no existing portfolio service
is a mandatory Physica runtime dependency.

| Source at inspected exact head | Useful mechanism | Physica application |
| --- | --- | --- |
| [world-zero@837fa825](https://github.com/thebrazenbeard/world-zero/tree/837fa825a2e8972f74a91089337513b55f732aa9) | System-dynamics falsification, timestep dependence, invariant records, benchmarks | Conservative energy check, analytic error and solver convergence tests. Not a claim of model validation. |
| [noema@157fd886](https://github.com/thebrazenbeard/noema/tree/157fd886ab34443968d030f2109cb5125903c1ce) | Evidence/source fingerprinting, deterministic experiments | Content-hashed simulation receipts with explicit qualification ceiling. Hash is **not** a digital signature or authenticity proof. |
| [spm@894bc57f](https://github.com/thebrazenbeard/spm/tree/894bc57f45e8329bbe89235b7b6886cdefbdfda4) | Distinct training, evaluation, and generalization evidence | PINN residual, analytic holdout, and boundary error remain separate. |
| [hc-brain@efe72ec2](https://github.com/thebrazenbeard/hc-brain/tree/efe72ec24f2ff4baf825bc37efc8295adbe6489c) | Simulation/counterfactual scope and effect/authority separation | Cart-pole is a simulated state/action contract only. No authorization of real machinery. |
| [semanticatlas@62d1aa82](https://github.com/thebrazenbeard/semanticatlas/tree/62d1aa82631af482408073e6586a34385be7f970) | Claims and provenance classification | Separate analytic solution, numerical approximation, learning result, and observation. |
| [temporal@e4fe336d](https://github.com/thebrazenbeard/temporal/tree/e4fe336d41ede471671fe4e83652055994e67e2c) | Time ordering without meaning promotion | Monotonic time grid, exact requested end time, no inferred chronology beyond simulation. |
| [discovery@bd476c7f](https://github.com/thebrazenbeard/discovery/tree/bd476c7f074bb6af3297d9d79b803583a6b103fa) | Earn reuse through concrete consumers, not elegance | Physica remains standalone; promote shared libraries only after independent consumers and comparative evidence. |
| [vera-mesh@3e03bfa7](https://github.com/thebrazenbeard/vera-mesh/tree/3e03bfa7415295330c145c9990155ec13ed3742e) | Transport not authority | No network-actuator or deployment authority implied by a simulation. |
| [vera-mono@a14fccaa](https://github.com/thebrazenbeard/vera-mono/tree/a14fccaad3470a2b5283eecbcd7f0a138069ebf9) | Self-contained implementation and modular integration boundaries | No runtime dependency on Vera internals. |
| [project-runner@bf65b31c](https://github.com/thebrazenbeard/project-runner/tree/bf65b31c2be030946cfb590eedeba5574c480ae4) | Model/version tracking and execution promotion separation | Git source version, simulation receipt, and scientific qualification are different statuses. |

The remaining inventory entries were **structurally scanned**, not all deeply read
or deemed physics donors. Reuse of every repository as a code dependency would be
unjustified, error-prone, and in some cases cross private/public boundaries.

## Physica V0 evidence classes

- **Analytic exact case:** harmonic oscillator, selected constant-coefficient
  cases, 1D Poisson equation. Exact only under the modeled assumptions.
- **Numerical implementation:** Euler, classical RK4, velocity Verlet
  (position-only acceleration), plus explicit Dormand–Prince 5(4) adaptive
  integration with local error control. All methods are nonstiff.
- **Training result:** reproducible PyTorch CPU Poisson residual optimization;
  raw residual, boundary error and held-out solution error.
- **Embodied model:** idealized cart-pole differential equations with explicit
  external force input. Not a physically qualified controller.
- **Receipt:** SHA-256 content fingerprint of simulation output; no timestamp
  freshness, trusted signer, authenticated source or prediction claim.

## Known limitations and falsification plan

1. No stiff/DAE/implicit methods, adaptive dense output, event roots,
   collision/contact, finite elements, electromagnetics, thermal or fluid PDE
   solver yet. Adaptive RK45 is explicit, nonstiff and locally controlled.
2. Explicit integration may become unstable. Compare solver families, local
   tolerances, time resolution, energy drift and analytic trajectories before
   trusting output.
3. Conservation checks are **inapplicable** when there is damping or external
   forcing. Track external work/dissipation before asserting an energy ledger.
4. PINNs can converge on training residual but remain wrong on unobserved
   domains, regimes or physical processes. V0 has only a 1D controlled benchmark.
5. Differentiability alone does not establish physical correctness.
6. The neural residual accelerator is unconstrained by conservation;
   never automatically deploy or trust it on hardware.
7. No data assimilation, physical parameter identifiability, sensor calibration,
   uncertainty propagation, or independent benchmark suite is yet qualified.
8. Content hashes detect changes relative to a trusted recorded hash; attackers
   can recompute both payload and digest. They prove neither origin nor truth.

The V0.2 [adaptive solver review](ADAPTIVE_SOLVER.md) documents SciPy
DOP853 cross-implementation checks. These establish numerical agreement on
specified initial-value problems, not validated physical models.

Suggested next frontier: inverse problems with frozen synthetic train/holdout
partitions, stiff/implicit benchmark, parameter identifiability measurement,
units validation and checkpointed experiment manifests.

## Licensing boundary

Several portfolio repositories are explicitly **source-visible, proprietary**
rather than open source. Physica's code was written independently against standard
mathematical definitions. No repository was copied or made a mandatory import.
Physica itself currently has no license grant; do not assume permission for
commercial distribution by third parties.
