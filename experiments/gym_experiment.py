from environments.continuous_space_maze import ContinuousSpaceMaze
from sac.algos import SAC
from sac.policies import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.sampler import SimpleSampler, NormalizeSampler
# from sac.misc.instrument import run_sac_experiment
import tensorflow as tf
# import misc.mylogger as mylogger
import misc.log_scheduler as mylogger
import misc.baselines_logger as logger
from sac.envs import GymEnv
# from rllab.envs.normalized_env import normalize
import argparse
import os
from datetime import datetime
import yaml
from algorithms.knack_based_policy import KnackBasedPolicy, EExploitationPolicy

import environments
import numpy as np
import optuna
import multiprocessing

def wrap(trial, args):
    return_list = []
    args.update({"return_list": return_list})
    p = multiprocessing.Process(main(trial, **args))
    p.start()
    p.join()
    return return_list[0]


def main(trial, optuna, env, seed, entropy_coeff, n_epochs, dynamic_coeff, clip_norm, normalize_obs, buffer_size,
         max_path_length, min_pool_size, batch_size, policy_mode, eval_model, e, eval_n_episodes, eval_n_frequency,
         knack_thresh, return_list=None):
    if optuna:
        logger.configure(logger.get_dir(), log_suffix="_optune{}".format(trial.number), enable_std_out=False)
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
    elif policy_mode == "EExploitation":
        policy = EExploitationPolicy(env_spec=env.spec,
                                     K=4,
                                     hidden_layer_sizes=[layer_size, layer_size],
                                     qf=qf,
                                     reg=1e-3,
                                     squash=True,
                                     e=e
                                     )

    elif policy_mode == "Knack-exploration" or policy_mode == "kurtosis":
        policy = KnackBasedPolicy(
            a_lim_lows=env.action_space.low,
            a_lim_highs=env.action_space.high,
            env_spec=env.spec,
            K=4,
            hidden_layer_sizes=[layer_size, layer_size],
            qf=qf,
            vf=vf,
            reg=1e-3,
            squash=True,
            metric="kurtosis",
            knack_thresh=knack_thresh,
            optuna_trial=trial,
        )
    elif policy_mode == "signed_variance":
        policy = KnackBasedPolicy(
            a_lim_lows=env.action_space.low,
            a_lim_highs=env.action_space.high,
            env_spec=env.spec,
            K=4,
            hidden_layer_sizes=[layer_size, layer_size],
            qf=qf,
            vf=vf,
            reg=1e-3,
            squash=True,
            metric="signed_variance",
            knack_thresh=knack_thresh,
            optuna_trial=trial,
        )
    elif policy_mode == "negative_signed_variance":
        policy = KnackBasedPolicy(
            a_lim_lows=env.action_space.low,
            a_lim_highs=env.action_space.high,
            env_spec=env.spec,
            K=4,
            hidden_layer_sizes=[layer_size, layer_size],
            qf=qf,
            vf=vf,
            reg=1e-3,
            squash=True,
            metric="negative_signed_variance",
            knack_thresh=knack_thresh,
            optuna_trial=trial,
        )
    elif policy_mode == "small_variance":
        policy = KnackBasedPolicy(
            a_lim_lows=env.action_space.low,
            a_lim_highs=env.action_space.high,
            env_spec=env.spec,
            K=4,
            hidden_layer_sizes=[layer_size, layer_size],
            qf=qf,
            vf=vf,
            reg=1e-3,
            squash=True,
            metric="variance",
            knack_thresh=knack_thresh,
            optuna_trial=trial,
        )
    elif "kurtosis-" in policy_mode:
        policy = KnackBasedPolicy(
            a_lim_lows=env.action_space.low,
            a_lim_highs=env.action_space.high,
            env_spec=env.spec,
            K=4,
            hidden_layer_sizes=[layer_size, layer_size],
            qf=qf,
            vf=vf,
            reg=1e-3,
            squash=True,
            metric=policy_mode,
            knack_thresh=knack_thresh,
            optuna_trial=trial,
        )
    else:
        raise AssertionError("policy_mode should be GMMPolicy or Knack-exploration or Knack-exploration or signed_variance or variance")
        
    # TODO
    base_kwargs = dict(
        epoch_length=1000,
        n_epochs=n_epochs,
        # scale_reward=1,
        n_train_repeat=1,
        eval_render=False,
        eval_n_episodes=eval_n_episodes,
        eval_deterministic=True,
        eval_n_frequency=eval_n_frequency
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
        avg_return = algorithm.train()
        if return_list is not None:
            return_list.append(avg_return)
        tf.reset_default_graph()
        # algorithm._sess.close()
        # del algorithm
        return avg_return

    else:
        make_expert_data(algorithm, eval_model, seed, stochastic=False)
        # eval_render(algorithm, eval_model, seed)

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
    parser.add_argument('--policy-mode', default="Knack-exploration",
                        choices=["GMMPolicy", "Knack-exploration", "EExploitation", "signed_variance", "negative_signed_variance", "small_variance", "kurtosis-signed_variance", "kurtosis-negative_signed_variance", "kurtosis-small_variance", "kurtosis-negative_singed_variance_no_threshold"])
    parser.add_argument('--eval-model', type=str, default=None)
    parser.add_argument('--e', type=float, default=1.)
    
    parser.add_argument('--eval_n_episodes', type=int, default=20)  # the num of episode to calculate an averaged return when an evaluation
    parser.add_argument('--eval_n_frequency', type=int, default=1)  # an evaluation per eval_n_frequency epochs
    parser.add_argument('--knack_thresh', type=float, default=0.8)
    parser.add_argument('--save_array_flag', choices=[0, 1], type=int, default=1)
    parser.add_argument('--optuna', action='store_true')

    return vars(parser.parse_args())


def eval_render(algorithm, eval_model, seed):
    import imageio
    from moviepy.editor import ImageSequenceClip
    import numpy as np
    import cv2
    from glob import glob
    import re
    parser = re.compile(r'.*_epoch(\d+)\.npz')

    movie_second = 7

    with algorithm._sess.as_default():
        algorithm._saver.restore(algorithm._sess, eval_model)

        knack_files = glob(os.path.join(os.path.dirname(eval_model), "experienced/*.npz"))
        epoch_num = [int(parser.match(f_name).group(1)) for f_name in knack_files]
        print(max(epoch_num))
        max_id = np.argmax(np.asarray(epoch_num))
        final_knacks = np.load(knack_files[max_id])['knack_kurtosis']

        env = algorithm._env

        movie_dir = os.path.join(os.path.dirname(eval_model), "movie")

        os.makedirs(movie_dir, exist_ok=True)

        if hasattr(algorithm._policy, "_is_deterministic"):
            algorithm._policy._is_deterministic = True

        if hasattr(env, "env"):
            env = env.env

        # np.random.seed(seed)
        # env.seed(seed)
        fps = 1 / env.dt
        step_thresh = movie_second / env.dt
        imgs = []
        for i in range(10):
            obs = env.reset()
            done = False
            steps = 0
            _min = np.min(final_knacks)
            _max = np.max(final_knacks)
            print("start episode {}".format(i))
            while not done:
                steps += 1
                # env.render()
                img = env.render(mode='rgb_array', width=256, height=256)
                v, mean, var, kurtosis = algorithm._policy.calc_and_update_knack([obs])
                knack_value = kurtosis[0]
                # _min = min(knack_value, _min)
                # _max = max(knack_value, _max)
                print(knack_value)
                knack_value = (knack_value - _min) / (_max - _min)
                if knack_value > 0.8:  ## TODO hyper param
                    print("knack {}".format(knack_value))
                    # algorithm._policy._is_deterministic = False
                    action, _ = algorithm.policy.get_action(obs.flatten())
                    # action = max(env.action_space.high) * np.random.normal(0, max(env.action_space.high)**2)
                    # action = env.action_space.sample()
                    # algorithm._policy._is_deterministic = True
                    # additional append img
                    [imgs.append(img) for i in range(10 - 1)]
                    pass
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = np.tile(gray[:, :, np.newaxis], (1, 1, 3))
                    action, _ = algorithm.policy.get_action(obs.flatten())
                imgs.append(img)
                obs, rew, done, _ = env.step(action)

                if steps > step_thresh:
                    break
        imgs = np.asarray(imgs)
        save_path = os.path.join(movie_dir, 'movie.mp4')
        imageio.mimwrite(save_path, imgs, fps=fps)
        print(save_path)


def make_expert_data(algorithm, eval_model, seed, stochastic=False):

    with algorithm._sess.as_default():
        algorithm._saver.restore(algorithm._sess, eval_model)

        if stochastic:
            knack_file = os.path.join(os.path.dirname(eval_model), "array/epoch0_2001.npz")
            final_knacks = np.load(knack_file)['knack_kurtosis'][-1]

        env = algorithm._env

        if hasattr(env, "env"):
            env = env.env

        # np.random.seed(seed)
        # env.seed(seed)
        num_data = 1500
        steps_thresh = 1000
        data = {'acs': [], 'ep_rets': [], 'obs': [], 'rews': []}
        for i in range(num_data):
            obs = env.reset()
            done = False
            steps = 0
            ret = 0
            tmp_data = {'acs': [], 'obs': [], 'rews': []}
            if stochastic:
                _min = np.min(final_knacks)
                _max = np.max(final_knacks)
            print("start episode {}".format(i))
            while not done:
                steps += 1
                # env.render()
                if stochastic:
                    if hasattr(algorithm.pi, "knack_thresh"):
                        v, mean, var, kurtosis = algorithm._policy.calc_and_update_knack([obs])
                        knack_value = kurtosis[0]
                        # _min = min(knack_value, _min)
                        # _max = max(knack_value, _max)
                        knack_value = (knack_value - _min) / (_max - _min)
                        if knack_value > 0.8:  ## TODO hyper param
                            print("knack {}".format(knack_value))
                            was = algorithm._policy._is_deterministic
                            algorithm._policy._is_deterministic = True
                            action, _ = algorithm.policy.get_action(obs.flatten())
                            algorithm._policy._is_deterministic = was
                        else:
                            action, _ = algorithm.policy.get_action(obs.flatten())
                    else:
                        algorithm._policy._is_deterministic = False
                        action, _ = algorithm.policy.get_action(obs.flatten())
                else:
                    if hasattr(algorithm._policy, "_is_deterministic"):
                        algorithm._policy._is_deterministic = True
                    action, _ = algorithm.policy.get_action(obs.flatten())

                obs_next, rew, done, _ = env.step(action)
                tmp_data['obs'].append(obs)
                tmp_data['acs'].append(action)
                tmp_data['rews'].append(rew)
                ret += rew

                obs = obs_next
                if steps >= steps_thresh:
                    done = True


            data['ep_rets'].append(ret)
            for k, v in tmp_data.items():
                data[k].append(v)

    np.savez_compressed("a.npz", **data)
    print("return mean: {}".format(np.mean(data['ep_rets'])))


if __name__ == '__main__':
    args = parse_args()

    # set environment
    seed = args['seed']
    env_id = args.pop('env_id')
    env = GymEnv(env_id)

    # set log directory
    root_dir = args.pop('root_dir')
    opt_log_name = args.pop('opt_log_name')
    logger2 = mylogger.get_logger()
    if args['eval_model'] is None:
        env_id = env.env_id
        # print(env_id)
        # os.makedirs(root_dir, exist_ok=True)
        # current_log_dir = root_dir
        # current_log_dir = os.path.join(root_dir, env_id, 'seed{}'.format(seed))
        current_log_dir = root_dir
        # mylogger.make_log_dir(current_log_dir)
        logger2.set_log_dir(current_log_dir, exist_ok=True)
        logger2.set_save_array_flag(args.pop("save_array_flag"))
        logger.configure(dir=current_log_dir, enable_std_out=False)

        # save parts of hyperparameters
        with open(os.path.join(current_log_dir, "hyparam.yaml"), 'w') as f:
            yaml.dump(args, f, default_flow_style=False)

    args.update({'env': env})
    if args["optuna"]:
        study = optuna.create_study(study_name='knack_threshold_small_variance_{}'.format(env_id), storage='mysql://root@192.168.2.76/optuna',
        # study = optuna.create_study(study_name='test', storage='mysql://root@192.168.2.76/optuna',
                                    pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=args["n_epochs"] / 3),
                                    direction="maximize", load_if_exists=True)
        # study.optimize(lambda trial: main(trial, **args), timeout=24 * 60 * 60)
        study.optimize(lambda trial: main(trial, **args), n_trials=2)
    else:
        args.update({'trial': None})
        main(**args)
    logger2.force_write()
