from environments.continuous_space_maze import ContinuousSpaceMaze
from sac.algos import SAC
from sac.policies import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.sampler import SimpleSampler
import tensorflow as tf
import misc.mylogger as mylogger
from sac.envs import GymEnv

from datetime import datetime
from pytz import timezone
import argparse
import os

def main(root_dir, seed, entropy_coeff, n_epochs, dynamic_coeff, clip_norm, regularize):

    tf.set_random_seed(seed=seed)
    env = GymEnv('MountainCarContinuous-v0')
    env.min_action = env.action_space.low[0]
    env.max_action = env.action_space.high[0]

    env.env.seed(seed)
    max_replay_buffer_size = int(1e6)
    sampler_params = {'max_path_length': 1000, 'min_pool_size': 1000, 'batch_size': 128}
    sampler = SimpleSampler(**sampler_params)

    entropy_coeff = entropy_coeff
    dynamic_coeff = dynamic_coeff
    # env_id = 'ContinuousSpaceMaze{}_{}_RB{}_entropy_{}__Normalize'.format(goal[0], goal[1], max_replay_buffer_size, entropy_coeff)
    env_id = 'MountainCarContinuous_RB1e6_entropy{}_epoch{}__Normalize_uniform'.format(entropy_coeff, n_epochs)
    env_id = env_id + '_dynamicCoeff' if dynamic_coeff else env_id

    os.makedirs(root_dir, exist_ok=True)
    env_dir = os.path.join(root_dir, env_id)
    os.makedirs(env_dir, exist_ok=True)
    current_log_dir = os.path.join(env_dir, 'seed{}'.format(seed))
    mylogger.make_log_dir(current_log_dir)

    # env_id = 'Test'

    print(env_id)
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
    base_kwargs = dict(
        epoch_length=1000,
        n_epochs=n_epochs,
        # scale_reward=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=20,
        eval_deterministic=True,
    )

    pool = SimpleReplayBuffer(env_spec=env.spec, max_replay_buffer_size=max_replay_buffer_size)
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
        dynamic_coeff=dynamic_coeff,
        entropy_coeff=entropy_coeff,
        clip_norm=clip_norm,
    )

    # name = env_id + datetime.now().strftime("-%m%d-%Hh-%Mm-%ss")
    # mylogger.make_log_dir(name)

    algorithm._sess.run(tf.global_variables_initializer())
    algorithm.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--entropy-coeff', type=float, default=0.)
    parser.add_argument('--dynamic-coeff', type=bool, default=False)
    parser.add_argument('--n-epochs', type=int, default=2000)
    parser.add_argument('--clip-norm', type=float, default=None)

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    main(**args)
