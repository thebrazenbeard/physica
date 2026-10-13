# Physica V0.3 external-engine and reinforcement-learning research

Research date: October 13, 2026. Source heads are immutable Git commits read
through connected GitHub, not claims about ongoing upstream compatibility.
These projects were inspected as **candidate mechanisms and constraints**.
No external code was copied and none of these eleven engines, tools or
frameworks was installed as a mandatory Physica dependency.

## Source-by-source disposition

| Source, inspected Git head | Actual scope and relevant mechanism | Decision |
| --- | --- | --- |
| [PhysicsJS @ b9eca16](https://github.com/wellcaffeinated/PhysicsJS/tree/b9eca1634b6db222571e7e820a09404513fe2e46) | Modular JavaScript rigid-body engine, now archived; maintainer says no longer maintained | Architecture reference only, not a runtime dependency |
| [Jolt Physics @ a4b24c3](https://github.com/jrouwe/JoltPhysics/tree/a4b24c3fadd7eade0435f27e9d6a07564273eb5f) | C++ multithreaded rigid bodies, asynchronous mutation and collision queries | Candidate for later isolated native 3D adapter, not an interchangeably drop-in Python solver |
| [PhysicsNeMo @ f9b2721](https://github.com/NVIDIA/physicsnemo/tree/f9b2721dfbc0299a39bd6322a039b0dd268a35cb) | NVIDIA PyTorch physics ML, models, training recipes | Benchmark/reference for future scalable scientific ML; no GPU or CUDA expectation in core |
| [TheAlgorithms C++ @ fdcc29b](https://github.com/TheAlgorithms/C-Plus-Plus/tree/fdcc29b0ac2848e9d1a8d926e171cc5b39f97759) | Educational algorithm implementations, not a calibrated physics engine | Independent algorithm-study/reference only |
| [Isaac Lab @ bbcbab6](https://github.com/isaac-sim/IsaacLab/tree/bbcbab6b64d5538c1e6800646d57b462389c1410) | GPU robot learning atop Isaac Sim, tasks, sensors and training environments; evolving develop branch | Borrow explicit environment, action, observation and training/evaluation interface concepts |
| [MuJoCo @ 4110357](https://github.com/google-deepmind/mujoco/tree/4110357126a3c6f35d828500044d67d7ba660533) | Articulated contact dynamics and robotics with C and Python APIs | Candidate future **opt-in** cross-engine cartpole / contact benchmark |
| [Box2D @ ac7c751](https://github.com/erincatto/box2d/tree/ac7c751eaeddbabdc1c4d41ae4f3a25d78627790) | 2D shapes, broad-phase AABB tree, narrow-phase, time of impact, rigid bodies | Implement independent 2D sweep-and-prune AABB *candidate* detection plus quadratic oracle; no contacts or forces |
| [UnrealEnginePython @ 4b5da5b](https://github.com/20tab/UnrealEnginePython/tree/4b5da5bf4ca598e8d0f67eb938749bc381a67d6c) | Legacy Unreal Engine 4 Python embedding; README says development on hold | Future optional adapter consideration only, avoid legacy dependency |
| [miyosuda/unreal @ 31d4886](https://github.com/miyosuda/unreal/tree/31d4886149412fa248f6efa490ab65bd2f425cde) | **UNREAL = Unsupervised REinforcement and Auxiliary Learning**, replication of DeepMind paper on DeepMind Lab / old TensorFlow; **not related to Unreal Engine** | Use deterministic replay, reward and next-state auxiliary *target interfaces* and a bounded RL training baseline; this is **not** a reproduction of UNREAL |
| [UnrealCLR @ 8a517e2](https://github.com/nxrighthere/UnrealCLR/tree/8a517e23b3b76399d77d50a3b8ad0cb8887c2a8e) | Unreal Engine .NET/C# interoperability and host isolation | Future optional interop research, outside Physica Python scientific core |
| [Dumper-7 @ dd8fe34](https://github.com/Encryqed/Dumper-7/tree/dd8fe34df8283378a1055386eee195005a100924) | Reverse-engineered UE4/UE5 SDK generator using injection into games | No scientific-physics application; **not installed, used, or integrated** |

Git heads above are source review boundaries, **not** technical qualification of
the full repositories or permission to copy their code. Licenses, native ABI,
compatibility, and distribution requirements would need separate decisions
before any actual dependency integration.

## Bounded pieces built in V0.3

### Inverse model, with separate holdout

`physica.inverse` fits stiffness and viscous damping of the **known-mass,
zero-external-force** harmonic oscillator from position observations. Data
records have times, displacement and caller-supplied uncertainty scale (sigma).
The synthetic fixture is generated from an **analytic formula**, while the fit
uses **Physica's numerical RK45 solver**. It therefore does not simply fit its
own data-generation routine. NumPy/SciPy bounded least-squares optimize only on
training observations; withheld timestamps never appear in optimizer residuals.

A rank/condition assessment uses singular values of the local weighted Jacobian.
`LOCAL_FULL_RANK` describes **only** the local numerical linearization at the
reported optimum. `RANK_DEFICIENT`, `ILL_CONDITIONED` and
`OPTIMIZER_NOT_CONVERGED` prevent promoting a low residual into a unique
parameter claim. Source labels are **caller asserted**, not authenticated.
No structural/global identifiability, calibrated sensors, Bayesian uncertainty,
noise-distribution certification or scientific validation is claimed.

### Collision candidate, no solver

`physica.spatial` supplies validated 2D axis-aligned boxes, x-axis sweep
candidate pairs and a separate O(n²) overlap reference. Touching faces count
as candidates. Randomized synthetic sets compare both algorithms.
Overlap of bounding boxes is **not** narrow-phase intersection of rotated or
non-rectangular shapes; the module applies **no contact impulses, force,
friction or time-of-impact calculation**.

### RL training baseline and auxiliary *targets*

`physica.rl.CartPoleEnv` is a simulated, bounded, discrete-action environment.
Its transition calls existing RK4 with Physica's idealized `CartPole`. Every
step produces observation, discrete command (-1/0/+1), scalar reward, next
observation, terminal/truncation flags and dt. It has no device bridge.

`TabularQAgent` is an explicitly lossy discretization with a basic temporal
difference update. Training uses a deterministic seed range; frozen-policy
evaluation uses a separate seed range, no updates and no exploratory action.
This reports descriptive returns **without claiming improvement** over any
baseline or real-world transfer. `ReplayBuffer` is bounded FIFO with seeded
samples. `auxiliary_targets` exposes immediate reward and state-delta labels.
Unlike the original UNREAL paper, V0.3 **does not yet train auxiliary neural
heads**, implement pixel control, or compare on Atari/DeepMind Lab.

## Research anchors

- [DeepMind UNREAL paper, Jaderberg et al., 2016](https://arxiv.org/abs/1611.05397):
  shared representation and auxiliary reward/task signals. This is **not**
  Unreal Engine technology.
- [Box2D collision docs](https://box2d.org/documentation/md_collision.html):
  broad-phase AABBs are only potential overlaps; independent shape overlap and
  contact solving are separate stages.
- [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html):
  residual minimization and numerical Jacobian; convergence does not prove
  parameter identifiability.
- [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/):
  robotics environment/reference scope, not a Physica dependency.

## Next qualification and integration gates

1. For RL, add auxiliary-head training and compare frozen evaluation across
   **matched random seeds** against an equal-compute no-auxiliary ablation.
2. For inverse problems, add a structural/analytic identifiability treatment,
   interval uncertainty and realistic calibrated sensor observations.
3. For 3D physics, benchmark optional MuJoCo and Jolt native integrations,
   with explicit ABI/licensing/resource budgets.
4. For spatial, implement narrow-phase geometry and independently reviewed
   collision/contact invariants before calling Physica a rigid-body engine.

Sources reviewed and code implemented are different evidence classes.

## Source-file inspection beyond repository README

The full recursive GitHub trees of **all eleven** supplied repositories were
read (none returned truncated), followed by direct inspection of the source
files and technical guides below. The distinction is important: this is a
**tree-wide inventory plus selected implementation inspection**, not a
line-by-line audit of thousands of source files or a compatibility test.

| Repository | Specific source/code read | Actual implementation insight |
| --- | --- | --- |
| PhysicsJS | [world implementation](https://github.com/wellcaffeinated/PhysicsJS/blob/b9eca1634b6db222571e7e820a09404513fe2e46/src/core/world.js) | The world defaults to Verlet, a fixed simulation timestep, bounded iterations per frame and sleep thresholds; timing/sleep are runtime engine concerns |
| Jolt Physics | [Architecture](https://github.com/jrouwe/JoltPhysics/blob/a4b24c3fadd7eade0435f27e9d6a07564273eb5f/Docs/Architecture.md) | Bodies have static/dynamic/kinematic modes, shapes and motion properties; native multicore collision queries are a separate architecture from a scalar ODE engine |
| PhysicsNeMo | [Darcy FNO validator](https://github.com/NVIDIA/physicsnemo/blob/f9b2721dfbc0299a39bd6322a039b0dd268a35cb/examples/cfd/darcy_fno/validator.py) | Inverts normalization for comparison of prediction and target, in addition to a loss; metrics need physically interpretable units and separate validation |
| TheAlgorithms/C++ | [segment intersection implementation](https://github.com/TheAlgorithms/C-Plus-Plus/blob/fdcc29b0ac2848e9d1a8d926e171cc5b39f97759/geometry/line_segment_intersection.cpp) | Orientation/on-segment predicates are *geometric* intersection tests, not time-resolved contact simulation; educational implementations are not drop-in authoritative physics |
| Isaac Lab | [DirectRLEnv code](https://github.com/isaac-sim/IsaacLab/blob/bbcbab6b64d5538c1e6800646d57b462389c1410/source/isaaclab/isaaclab/envs/direct_rl_env.py) | Gymnasium-compatible RL interface and support for parallel environments; our local environment follows a bounded observation/action/reset/step contract without claiming GPU vectorization |
| MuJoCo | [modeling documentation](https://github.com/google-deepmind/mujoco/blob/4110357126a3c6f35d828500044d67d7ba660533/doc/modeling.rst) | MJCF XML compiles articulated physical models with defaults and pipeline options; any future cross-engine benchmark requires explicit scene/units mapping |
| Box2D | [broad-phase C source](https://github.com/erincatto/box2d/blob/ac7c751eaeddbabdc1c4d41ae4f3a25d78627790/src/broad_phase.c) | Uses dynamic trees and proxy/pair management by body class; Physica's independent sorted sweep only enumerates **potential** 2D AABB pairs, not a Box2D port |
| UnrealEnginePython | [Physics API guide](https://github.com/20tab/UnrealEnginePython/blob/4b5da5bf4ca598e8d0f67eb938749bc381a67d6c/docs/Physics_API.md) | Enables component physics with collision profiles and receives overlap/hit callbacks; an engine adapter would need explicit and potentially effectful actor authority |
| miyosuda/unreal | [experience replay](https://github.com/miyosuda/unreal/blob/31d4886149412fa248f6efa490ab65bd2f425cde/train/experience.py) | Experience contains action, terminal status, clipped reward, pixel-change and prior action/reward; tracks zero vs nonzero reward events for auxiliary learning |
| UnrealCLR | [managed interop manual](https://github.com/nxrighthere/UnrealCLR/blob/8a517e23b3b76399d77d50a3b8ad0cb8887c2a8e/MANUAL.md) | .NET managed assemblies load into Unreal, with engine data-passing/lifecycle boundaries; interoperability is not a physics solver and requires supported engine/toolchain |
| Dumper-7 | [SDK usage instructions](https://github.com/Encryqed/Dumper-7/blob/dd8fe34df8283378a1055386eee195005a100924/UsingTheSDK.md) | Generates UE SDK artifacts for injected DLL projects and game-specific layouts; not a scientific interface nor an appropriate Physica integration |

**Decision after deeper review:** Keep Physica's dependency-free physical
simulation and spatial/RL baseline. Prefer optional, separately tested future
bindings for MuJoCo/Jolt/Box2D or scalable scientific-learning frameworks.
Do **not** treat the two Unreal Engine plugins, the unrelated UNREAL algorithm,
or the SDK dumper as interchangeable products. In particular, Dumper-7's
game-process injection has no role in this research foundation.

**UNREAL-specific increment:** `ReplayBuffer.sample_balanced_rewards` uses
an approximately even split of zero and nonzero reward experiences where
available, with bounded no-replacement sampling, seeded determinism, and a
fallback for a missing/rare class. This borrows the *idea* of reward-aware
sampling from the implementation while keeping a small, independent API.
The new function does **not** train a reward predictor or add the full UNREAL
architecture.

Additional scientific sources:
- [DeepMind's UNREAL overview](https://deepmind.google/blog/reinforcement-learning-with-unsupervised-auxiliary-tasks/) explains pixel control and reward prediction as **trained tasks**; target construction alone is not auxiliary learning.
- [Lam et al. practical identifiability review](https://www.sciencedirect.com/science/article/pii/S0378475422001227) distinguishes local/practical parameter recoverability from structural identifiability.
