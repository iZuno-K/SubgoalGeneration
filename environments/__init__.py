from gym.envs.registration import register

register(
    id='ContinuousSpaceMazeEnv-v0',
    entry_point='environments.contiuous_space_maze:ContinuousSpaceMaze',
    max_episode_steps=1000
)