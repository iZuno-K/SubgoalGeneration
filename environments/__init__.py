from gym.envs.registration import register, make, spec

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
    id='MountainCarContinuousOneTurn-v0',
    entry_point='environments.MountainCarContinuousModify:Continuous_MountainCarOneTurnEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 999},
)

register(
    id='FrozenLakeDeterministic-v0',
    # entry_point='gym.envs.toy_text:FrozenLakeEnv',
    entry_point='environments.descrete_maze:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

register(
    id='CliffMazeDeterministic-v0',
    entry_point='environments.cliff_maze:CliffMazeEnv',
    kwargs={'map_name' : '9x9', 'is_slippery': False},
    max_episode_steps=100,
)

register(
    id='AntNoForce-v0',
    entry_point='environments.ant_without_force:AntNoForceEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Walker2dWOFallReset-v0',
    max_episode_steps=1000,
    entry_point='environments.walker2d_without_falling_reset:Walker2dWithoutFallingResetEnv',
)
