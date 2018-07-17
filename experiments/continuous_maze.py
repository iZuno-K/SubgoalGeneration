import numpy as np
import gym
import environments
from environments.continuous_space_maze import ContinuousSpaceMaze
from environments.continuous_space_maze import Flat_dim
from sac.algos import SAC
from sac.policies import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.sampler import SimpleSampler
from sac.misc.instrument import run_sac_experiment
import tensorflow as tf
import misc.mylogger as mylogger
from sac.envs import (
    GymEnv,
    MultiDirectionSwimmerEnv,
    MultiDirectionAntEnv,
    MultiDirectionHumanoidEnv,
    CrossMazeAntEnv,
)
from rllab.envs.normalized_env import normalize
from datetime import datetime
from pytz import timezone

def main():
    # env = ContinuousSpaceMaze()
    env_id = 'NormalizedContinuousSpaceMaze'
    env = normalize(ContinuousSpaceMaze(), normalize_obs=True)
    # env = normalize(GymEnv('HalfCheetah-v2'))


    print('environment set done')

    # define value function
    layer_size = 100

    qf = NNQFunction(env_spec=env.spec,
                     hidden_layer_sizes=(layer_size, layer_size))
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(layer_size, layer_size))

    # use GMM policy
    policy = GMMPolicy(
        env_spec=env.spec,
        K=4,
        hidden_layer_sizes=[layer_size, layer_size],
        qf=qf,
        reg=1e-3,
        squash=True
    )

    # TODO
    max_replay_buffer_size = int(1e6)
    sampler_params = {'max_path_length': 1000, 'min_pool_size': 1000, 'batch_size': 128}
    base_kwargs = dict(
        epoch_length=1000,
        n_epochs=2000,
        # scale_reward=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    pool = SimpleReplayBuffer(env_spec=env.spec, max_replay_buffer_size=max_replay_buffer_size)
    sampler = SimpleSampler(**sampler_params)
    base_kwargs = dict(base_kwargs, sampler=sampler)

    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=pool,
        qf=qf,
        vf=vf,
        lr=3e-4,
        scale_reward=1.,
        discount=0.99,
        tau=1e-2,
        target_update_interval=1,
        action_prior='uniform',
        save_full_state=False,
    )
    print("1")
    name = env_id + datetime.now().strftime("-%m%d-%Hh-%Mm-%ss")
    mylogger.make_log_dir(name)

    algorithm._sess.run(tf.global_variables_initializer())

    print("2")
    algorithm.train()


if __name__ == '__main__':
    main()
