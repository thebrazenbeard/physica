"""A bounded RL seam, replay, and a reproducible tabular training baseline."""
from dataclasses import replace
import math

import pytest

from physica.rl import (
    CartPoleEnv, Transition, ReplayBuffer, TabularQAgent,
    auxiliary_targets, train_tabular_cartpole,
)


def test_seeded_reset_and_rollout_are_reproducible():
    a, b = CartPoleEnv(dt=.02), CartPoleEnv(dt=.02)
    assert a.reset(seed=7) == b.reset(seed=7)
    steps_a = [a.step(x) for x in (1, -1, 0, 1, 1)]
    steps_b = [b.step(x) for x in (1, -1, 0, 1, 1)]
    assert steps_a == steps_b
    assert all(t.next_observation != t.observation for t in steps_a)


def test_one_step_agrees_with_independent_numeric_integration():
    from physica.models import CartPole
    from physica.numerics import integrate
    model = CartPole()
    env = CartPoleEnv(dt=.03)
    start = (0., 0., .1, 0.)
    env.reset(initial=start)
    actual = env.step(1)
    expected = integrate(lambda t, y: model.rhs(t, y, env.force_magnitude),
                         start, t1=.03, dt=.03, method="rk4").final
    assert actual.next_observation == pytest.approx(expected, abs=1e-12)


def test_terminal_and_truncated_states_have_an_end_of_episode_boundary():
    env = CartPoleEnv(max_episode_steps=2)
    env.reset(initial=(0.,0.,0.,0.))
    assert env.step(0).truncated is False
    assert env.step(0).truncated is True
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(initial=(3.0, 0., .2, 0.))
    assert env.step(0).terminated is True


@pytest.mark.parametrize("action", [2, "1", 1.5, True, None])
def test_invalid_actions_are_rejected(action):
    env = CartPoleEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(action)


def test_tabular_q_learning_changes_value_from_reward():
    agent = TabularQAgent(alpha=.5, epsilon=0., gamma=.95, seed=1)
    s = (0., 0., 0., 0.)
    sample = Transition(s, 1, 1., s, True, False, .02)
    assert agent.q_value(s, 1) == 0.
    agent.learn(sample)
    assert agent.q_value(s, 1) == pytest.approx(.5)


def test_replay_buffer_fifo_and_auxiliary_targets():
    buffer = ReplayBuffer(capacity=3, seed=42)
    transitions = []
    for i in range(5):
        start = (float(i), 0., 0., 0.)
        target = (float(i+1), 0., 0., 0.)
        t = Transition(start, 1, float(i % 2), target, False, False, .02)
        transitions.append(t)
        buffer.append(t)
    assert len(buffer) == 3
    sampled = buffer.sample(3)
    assert all(x in transitions[2:] for x in sampled)
    assert sampled == ReplayBuffer.from_transitions(
        transitions[2:], capacity=3, seed=42).sample(3)
    reward, change = auxiliary_targets(transitions[-1])
    assert reward == 0.
    assert change == (1., 0., 0., 0.)


def test_reward_balanced_replay_is_bounded_and_reproducible():
    transitions = []
    for i in range(12):
        state = (float(i), 0., 0., 0.)
        reward = 1. if i < 3 else 0.
        transitions.append(Transition(state, 0, reward, state, False, False, .02))
    a = ReplayBuffer.from_transitions(transitions, capacity=12, seed=8)
    b = ReplayBuffer.from_transitions(transitions, capacity=12, seed=8)
    sample = a.sample_balanced_rewards(6)
    assert sample == b.sample_balanced_rewards(6)
    assert len(sample) == 6
    assert len(set(sample)) == 6
    assert sum(bool(x.reward) for x in sample) == 3
    assert sum(not x.reward for x in sample) == 3


def test_reward_balanced_sampling_falls_back_to_available_class():
    t = Transition((0.,0.,0.,0.), 0, 0., (0.,0.,0.,0.), False, False, .02)
    buffer = ReplayBuffer.from_transitions([t,t,t], capacity=3, seed=4)
    assert len(buffer.sample_balanced_rewards(2)) == 2
    with pytest.raises(ValueError):
        buffer.sample_balanced_rewards(4)


def test_train_and_frozen_evaluation_are_repeatable_and_bounded():
    r1 = train_tabular_cartpole(episodes=8, evaluation_episodes=3, seed=44,
                                max_episode_steps=30)
    r2 = train_tabular_cartpole(episodes=8, evaluation_episodes=3, seed=44,
                                max_episode_steps=30)
    assert r1 == r2
    assert r1.training_episodes == 8
    assert r1.evaluation_episodes == 3
    assert 0 <= r1.total_training_steps <= 240
    assert 0 <= r1.mean_evaluation_return <= 30
    assert r1.qualifier == "SYNTHETIC_SIMULATION_ONLY"


def test_training_does_not_claim_real_world_or_performance():
    r = train_tabular_cartpole(episodes=2, evaluation_episodes=2, seed=1,
                               max_episode_steps=10)
    assert "NOT" in r.evidence_limit.upper()
    assert r.mean_training_return >= 0


def test_training_cli_json_report():
    import json
    import subprocess
    import sys
    result = subprocess.run([
        sys.executable, "-m", "physica", "rl-cartpole",
        "--episodes", "6", "--evaluate", "3", "--seed", "9",
        "--max-steps", "20",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["training_episodes"] == 6
    assert record["evaluation_episodes"] == 3
    assert record["qualifier"] == "SYNTHETIC_SIMULATION_ONLY"


def test_invalid_configuration_fails_closed():
    for config in ({"dt": 0}, {"force_magnitude": math.inf},
                   {"max_episode_steps": 0}, {"angle_limit": -1}):
        with pytest.raises(ValueError):
            CartPoleEnv(**config)
    with pytest.raises(ValueError):
        train_tabular_cartpole(episodes=0)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)
