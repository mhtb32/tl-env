import gym
# noinspection PyUnresolvedReferences
import tl_env


def test_env_step():
    env = gym.make("double-goal-v0")

    obs = env.reset()
    reward = 0.
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert reward <= env.unwrapped.MAX_REWARD
