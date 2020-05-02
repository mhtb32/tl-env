from gym.envs.registration import register

register(
    id='double-goal-v0',
    entry_point='tl_env.envs:DoubleGoalEnv'
)
