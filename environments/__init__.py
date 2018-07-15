from gym.envs.registration import register

register(
    id='ContinuousSpaceMaze-v0',
    entry_point='environments.continuous_space_maze:ContinuousSpaceMaze',
    max_episode_steps=1000
)