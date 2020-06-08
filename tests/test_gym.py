import gym
import pytest

import highway_env

envs = [
    "SingleGoal-v0",
    "SingleGymGoal-v0",
    "SingleGoalIDM-v0",
    "DoubleGoal-v0"
]


@pytest.mark.parametrize("env_spec", envs)
def test_env_step(env_spec):
    env = gym.make(env_spec)

    obs = env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
