from gym.envs.registration import register


register(
    id='SingleGoal-v0',
    entry_point='tl_env.envs:SingleGoalEnv'
)


register(
    id='SingleGymGoal-v0',
    entry_point='tl_env.envs:SingleGymGoalEnv'
)


register(
    id='SingleGoalIDM-v0',
    entry_point='tl_env.envs:SingleGoalIDMEnv'
)


register(
    id='DoubleGoal-v0',
    entry_point='tl_env.envs:DoubleGoalEnv'
)
