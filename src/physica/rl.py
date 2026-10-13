"""Bounded synthetic cart-pole reinforcement learning with no actuator authority.

Inspired by training/evaluation and auxiliary-task ideas in UNREAL and Isaac
Lab; the implementation is independent and not a reproduction of UNREAL.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Sequence

from .models import CartPole
from .numerics import integrate

State4 = tuple[float, float, float, float]


def _state4(value: Sequence[float]) -> State4:
    data = tuple(float(v) for v in value)
    if len(data) != 4 or any(not isfinite(v) for v in data):
        raise ValueError("state must contain four finite components")
    return data  # type: ignore[return-value]


@dataclass(frozen=True)
class Transition:
    observation: State4
    action: int
    reward: float
    next_observation: State4
    terminated: bool
    truncated: bool
    dt: float


@dataclass(frozen=True)
class TrainingReport:
    training_episodes: int
    evaluation_episodes: int
    total_training_steps: int
    mean_training_return: float
    mean_evaluation_return: float
    seed: int
    qualifier: str = "SYNTHETIC_SIMULATION_ONLY"
    evidence_limit: str = (
        "NOT a real-world policy validation, not a UNREAL reproduction, "
        "not a safety-rated controller, and not evidence of policy improvement."
    )


class CartPoleEnv:
    """Local-only simulated discrete-force cart-pole with bounded episodes."""

    def __init__(self, *, dt: float = .02, force_magnitude: float = 10.,
                 position_limit: float = 2.4, angle_limit: float = .20944,
                 max_episode_steps: int = 200, model: CartPole | None = None):
        if not all(isfinite(v) and v > 0 for v in
                   (dt, force_magnitude, position_limit, angle_limit)):
            raise ValueError("environment scales must be positive and finite")
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        self.dt = dt
        self.force_magnitude = force_magnitude
        self.position_limit = position_limit
        self.angle_limit = angle_limit
        self.max_episode_steps = max_episode_steps
        if model is not None and not isinstance(model, CartPole):
            raise ValueError("model must be a CartPole")
        self.model = model or CartPole()
        self._state: State4 | None = None
        self._step_count = 0
        self._finished = False

    def reset(self, *, seed: int = 0, initial: Sequence[float] | None = None) -> State4:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        rng = Random(seed)
        self._state = _state4(
            initial if initial is not None
            else tuple(rng.uniform(-.05, .05) for _ in range(4))
        )
        self._step_count = 0
        self._finished = False
        return self._state

    def step(self, action: int) -> Transition:
        if type(action) is not int or action not in (-1, 0, 1):
            raise ValueError("action must be one of -1, 0, or 1")
        if self._state is None or self._finished:
            raise RuntimeError("episode must be reset before step")
        before = self._state
        force = self.force_magnitude * action
        result = integrate(
            lambda t, y: self.model.rhs(t, y, force),
            before, t0=0., t1=self.dt, dt=self.dt, method="rk4",
            max_steps=1,
        )
        after = _state4(result.final)
        self._step_count += 1
        terminated = (
            abs(after[0]) > self.position_limit
            or abs(after[2]) > self.angle_limit
        )
        truncated = self._step_count >= self.max_episode_steps and not terminated
        self._finished = terminated or truncated
        self._state = after
        return Transition(before, action, 0. if terminated else 1., after,
                          terminated, truncated, self.dt)


class ReplayBuffer:
    """FIFO replay with reproducible uniform sampling; no optimizer coupling."""

    def __init__(self, *, capacity: int, seed: int = 0):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.capacity = capacity
        self._rng = Random(seed)
        self._data: list[Transition] = []

    @classmethod
    def from_transitions(cls, transitions: Sequence[Transition], *,
                         capacity: int, seed: int = 0) -> "ReplayBuffer":
        result = cls(capacity=capacity, seed=seed)
        for transition in transitions:
            result.append(transition)
        return result

    def append(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise ValueError("replay contains only Transition objects")
        if len(self._data) >= self.capacity:
            self._data.pop(0)
        self._data.append(transition)

    def sample(self, size: int) -> tuple[Transition, ...]:
        if type(size) is not int or size < 1 or size > len(self._data):
            raise ValueError("sample size must be positive and <= replay size")
        return tuple(self._rng.sample(self._data, size))

    def sample_balanced_rewards(self, size: int) -> tuple[Transition, ...]:
        """Sample a roughly even mix of zero and nonzero rewards.

        Uses available examples if one class is rare or absent; no synthetic
        positive samples are invented. This is a replay *sampler*, not UNREAL's
        trained reward-prediction auxiliary head.
        """
        if type(size) is not int or size < 1 or size > len(self._data):
            raise ValueError("sample size must be positive and <= replay size")
        zero = [i for i, t in enumerate(self._data) if t.reward == 0.]
        nonzero = [i for i, t in enumerate(self._data) if t.reward != 0.]
        first = self._rng.sample(zero, min(size // 2, len(zero)))
        second = self._rng.sample(nonzero, min(size // 2, len(nonzero)))
        chosen = first + second
        selected = set(chosen)
        remaining = [i for i in range(len(self._data)) if i not in selected]
        chosen.extend(self._rng.sample(remaining, size - len(chosen)))
        self._rng.shuffle(chosen)
        return tuple(self._data[i] for i in chosen)

    def __len__(self) -> int:
        return len(self._data)


def auxiliary_targets(transition: Transition) -> tuple[float, State4]:
    """Reward-prediction and next-state-delta *targets*, not trained networks."""
    if not isinstance(transition, Transition):
        raise ValueError("expected Transition")
    return (transition.reward,
            _state4(tuple(b - a for a, b in
                          zip(transition.observation, transition.next_observation))))


class TabularQAgent:
    """Discrete Q-learning educational reference, not a deep RL agent.

    State bucketing is deliberately lossy and not calibrated to general tasks.
    """

    def __init__(self, *, alpha: float = .15, gamma: float = .99,
                 epsilon: float = .1, seed: int = 0):
        if not all(isfinite(v) for v in (alpha, gamma, epsilon)):
            raise ValueError("agent parameters must be finite")
        if not (0 < alpha <= 1 and 0 <= gamma <= 1 and 0 <= epsilon <= 1):
            raise ValueError("alpha in (0,1], gamma/epsilon in [0,1]")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self._rng = Random(seed)
        self._q: dict[tuple[tuple[int, ...], int], float] = {}

    @staticmethod
    def _bucket(obs: Sequence[float]) -> tuple[int, ...]:
        o = _state4(obs)
        scales = (.4, .6, .035, .6)
        return tuple(max(-12, min(12, int(round(x / s))))
                     for x, s in zip(o, scales))

    def q_value(self, observation: Sequence[float], action: int) -> float:
        if type(action) is not int or action not in (-1, 0, 1):
            raise ValueError("invalid discrete action")
        return self._q.get((self._bucket(observation), action), 0.)

    def choose_action(self, observation: Sequence[float], *,
                      explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice((-1, 0, 1))
        # Preserve stable deterministic tie-breaking.
        return max((0, -1, 1), key=lambda a: self.q_value(observation, a))

    def learn(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise ValueError("expected Transition")
        old = self.q_value(transition.observation, transition.action)
        future = (0. if transition.terminated
                  else max(self.q_value(transition.next_observation, action)
                           for action in (-1, 0, 1)))
        target = transition.reward + self.gamma * future
        key = (self._bucket(transition.observation), transition.action)
        self._q[key] = old + self.alpha * (target - old)


def train_tabular_cartpole(
    *, episodes: int, evaluation_episodes: int = 10,
    seed: int = 0, max_episode_steps: int = 200,
) -> TrainingReport:
    """Train on one synthetic seed range; evaluate frozen policy on another.

    Evaluation runs do not call learn and do not use exploratory actions.
    Returns descriptive metrics, without any performance or transfer claims.
    """
    if (type(episodes) is not int or not 1 <= episodes <= 10000
            or type(evaluation_episodes) is not int
            or not 1 <= evaluation_episodes <= 10000
            or type(seed) is not int):
        raise ValueError("episodes/evaluation_episodes must be in [1,10000], seed integer")
    agent = TabularQAgent(seed=seed)
    env = CartPoleEnv(max_episode_steps=max_episode_steps)
    train_returns: list[float] = []
    steps = 0
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        score = 0.
        while True:
            step = env.step(agent.choose_action(obs))
            agent.learn(step)
            score += step.reward
            steps += 1
            obs = step.next_observation
            if step.terminated or step.truncated:
                break
        train_returns.append(score)
    eval_returns: list[float] = []
    for ep in range(evaluation_episodes):
        obs = env.reset(seed=seed + 1_000_000 + ep)
        score = 0.
        while True:
            step = env.step(agent.choose_action(obs, explore=False))
            score += step.reward
            obs = step.next_observation
            if step.terminated or step.truncated:
                break
        eval_returns.append(score)
    return TrainingReport(
        training_episodes=episodes, evaluation_episodes=evaluation_episodes,
        total_training_steps=steps,
        mean_training_return=sum(train_returns) / episodes,
        mean_evaluation_return=sum(eval_returns) / evaluation_episodes,
        seed=seed,
    )
