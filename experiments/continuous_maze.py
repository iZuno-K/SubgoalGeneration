from environments.continuous_space_maze import ContinuousSpaceMaze
from sac.algos import SAC
from sac.policies import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.sampler import SimpleSampler, NormalizeSampler
# from sac.misc.instrument import run_sac_experiment
import tensorflow as tf
import misc.mylogger as mylogger
# from sac.envs import (
#     GymEnv,
#     MultiDirectionSwimmerEnv,
#     MultiDirectionAntEnv,
#     MultiDirectionHumanoidEnv,
#     CrossMazeAntEnv,
# )
# from rllab.envs.normalized_env import normalize
from datetime import datetime
from pytz import timezone
import argparse
import os
from datetime import datetime
import yaml


def main(env, seed, entropy_coeff, n_epochs, dynamic_coeff, clip_norm, normalize_obs, buffer_size,
         max_path_length, min_pool_size, batch_size):

    tf.set_random_seed(seed=seed)

    # define value function
    layer_size = 100
    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(layer_size, layer_size))
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

    max_replay_buffer_size = buffer_size
    pool = SimpleReplayBuffer(env_spec=env.spec, max_replay_buffer_size=max_replay_buffer_size)
    sampler_params = {'max_path_length': max_path_length, 'min_pool_size': min_pool_size, 'batch_size': batch_size}
    sampler = NormalizeSampler(**sampler_params) if normalize_obs else SimpleSampler(**sampler_params)

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
        clip_norm=clip_norm
    )


    algorithm._sess.run(tf.global_variables_initializer())
    algorithm.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n-epochs', type=int, default=2000)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--normalize-obs', type=int, default=1, help="whether normalize observation online")
    parser.add_argument('--buffer-size', type=int, default=1e6)
    # sampler
    parser.add_argument('--max-path-length', type=int, default=1000)
    parser.add_argument('--min-pool-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    # my experiment parameter
    parser.add_argument('--entropy-coeff', type=float, default=0.)
    parser.add_argument('--dynamic-coeff', type=bool, default=False)
    parser.add_argument('--path-mode', type=str, default="Double")
    parser.add_argument('--reward-mode', type=str, default="Dense", help="Dense or Sparse")
    parser.add_argument('--terminate-dist', type=int, default=0, help="whether terminate episode when goal-state-distance < 1")
    parser.add_argument('--opt-log-name', type=str, default=None)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    # set environment
    seed = args['seed']
    env = ContinuousSpaceMaze(seed=seed, path_mode=args.pop('path_mode'),
                              reward_mode=args.pop('reward_mode'), terminate_dist=args.pop('terminate_dist'))
    # set log directory
    env_id = env.spec.id
    print(env_id)
    root_dir = args.pop('root_dir')
    opt_log_name = args.pop('opt_log_name')
    os.makedirs(root_dir, exist_ok=True)
    date = datetime.strftime(datetime.now(), '%m%d')
    date = date + opt_log_name if opt_log_name is not None else date
    current_log_dir = os.path.join(root_dir, env_id, date, 'seed{}'.format(seed))
    mylogger.make_log_dir(current_log_dir)

    # save parts of hyperparameters
    with open(os.path.join(current_log_dir, "hyparam.yaml"), 'w') as f:
        yaml.dump(args, f, default_flow_style=False)

    args.update({'env': env})
    main(**args)
