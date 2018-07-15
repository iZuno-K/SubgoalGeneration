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


def main():
    env = ContinuousSpaceMaze()
    # env = gym.make('ContinuousSpaceMaze-v0')
    # env.spec.observation_space = Flat_dim(env.observation_space.shape)
    # env.spec.action_space = Flat_dim(env.action_space.shape)
    print('environment set done')

    # define value function
    layer_size = 30

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
    )

    # TODO
    max_replay_buffer_size = int(1e6)
    sampler_params = {'max_path_length': 1000, 'min_pool_size': 10, 'batch_size': 16}
    base_kwargs = dict(
        epoch_length=10,
        n_epochs=5,
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
    algorithm._sess.run(tf.global_variables_initializer())

    print("2")
    algorithm.train()


if __name__ == '__main__':
    main()
