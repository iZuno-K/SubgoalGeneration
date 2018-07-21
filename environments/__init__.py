from gym.envs.registration import register

register(
    id='ContinuousSpaceMaze-v0',
    entry_point='environments.continuous_space_maze:ContinuousSpaceMaze',
    max_episode_steps=1000
)

register(
    id='MountainCarContinuousColor-v0',
    entry_point='environments.MountainCarContinuousColor:Continuous_MountainCar_Color_Env',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 999},
    reward_threshold=90.0,
)

register(
    id='FrozenLakeDeterministic-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': True},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)
