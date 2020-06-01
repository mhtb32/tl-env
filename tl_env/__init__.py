from gym.envs.registration import register


register(
    id='SingleGoal-v0',
    entry_point='tl_env.envs:SingleGoalEnv',
    max_episode_steps=20
)


register(
    id='SingleGymGoal-v0',
    entry_point='tl_env.envs:SingleGymGoalEnv',
    max_episode_steps=20
)


register(
    id='DoubleGoal-v0',
    entry_point='tl_env.envs:DoubleGoalEnv',
    max_episode_steps=35
)
