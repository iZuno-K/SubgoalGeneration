from environments.continuous_space_maze import ContinuousSpaceMaze
from sac.algos import SAC
from sac.policies import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.sampler import SimpleSampler, NormalizeSampler
# from sac.misc.instrument import run_sac_experiment
import tensorflow as tf
import misc.mylogger as mylogger
from sac.envs import GymEnv
# from rllab.envs.normalized_env import normalize
from pytz import timezone
import argparse
import os
from datetime import datetime
import yaml
from algorithms.knack_based_policy import KnackBasedPolicy

import environments


def main(env, seed, entropy_coeff, n_epochs, dynamic_coeff, clip_norm, normalize_obs, buffer_size,
         max_path_length, min_pool_size, batch_size, policy_mode, eval_model):

    tf.set_random_seed(seed=seed)
    env.min_action = env.action_space.low[0]
    env.max_action = env.action_space.high[0]
    if hasattr(env, "seed"):
        env.seed(seed)
    else:
        env.env.seed(seed)

    # define value function
    layer_size = 100
    qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(layer_size, layer_size))
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(layer_size, layer_size))
    print("here")

    # use GMM policy
    if policy_mode == "GMMPolicy":
        # use GMM policy
        policy = GMMPolicy(
            env_spec=env.spec,
            K=4,
            hidden_layer_sizes=[layer_size, layer_size],
            qf=qf,
            reg=1e-3,
            squash=True
        )
    else:
        _, mode = str(policy_mode).split('-')
        if _ != "Knack":
            raise AssertionError("policy_mode should be GMMPolicy or Knack-p_control or Knack-exploitation or Knack-exploration")
        else:
            policy = KnackBasedPolicy(
                a_lim_lows=env.action_space.low,
                a_lim_highs=env.action_space.high,
                mode=mode,
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
    if eval_model is None:
        algorithm.train()
    else:
        eval_render(algorithm, eval_model)


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
    parser.add_argument('--env-id', type=str, default='HalfCheetah-v2')
    parser.add_argument('--entropy-coeff', type=float, default=0.)
    parser.add_argument('--dynamic-coeff', type=bool, default=False)
    parser.add_argument('--opt-log-name', type=str, default=None)
    parser.add_argument('--policy-mode', default="GMMPolicy", choices=["GMMPolicy", "Knack-p_control", "Knack-exploitation", "Knack-exploration"])
    parser.add_argument('--eval-model', type=str, default=None)

    return vars(parser.parse_args())

def eval_render(algorithm, eval_model):
    with algorithm._sess.as_default():
        algorithm._saver.restore(algorithm._sess, eval_model)
        env = algorithm._env
        import h5py
        import numpy as np

        movie_dir = os.path.join(os.path.dirname(eval_model), "movie")
        os.makedirs(movie_dir, exist_ok=True)
        movie = []

        if hasattr(algorithm._policy, "_is_deterministic"):
            algorithm._policy._is_deterministic = True

        if hasattr(env, "env"):
            env = env.env

        for i in range(1):
            obs = env.reset()
            done = False
            steps = 0
            while not done:
                steps += 1
                # img = env.render(mode='rgb_array')
                env.render()
                action, _ = algorithm.policy.get_action(obs.flatten())
                obs, rew, done, _ = env.step(action)
                # movie.append(img)
                if steps > 1000:
                    break

        # movie = np.array(movie)
        # print(movie.shape)
        # print(movie[0])
        # f = h5py.File(os.path.join(movie_dir, '{}.h5'.format("movie")), 'w')
        # f.create_dataset('imgs', data=np.asarray(movie), compression='lzf')
        # f.close()
        # print(movie_dir)


if __name__ == '__main__':
    import multiprocessing
    print(multiprocessing.cpu_count())

    args = parse_args()

    # set environment
    seed = args['seed']
    env_id = args.pop('env_id')
    env = GymEnv(env_id)

    # set log directory
    root_dir = args.pop('root_dir')
    opt_log_name = args.pop('opt_log_name')
    if args['eval_model'] is None:
        env_id = env.env_id
        # print(env_id)
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
