from sac.algos import SAC
from sac.policies import GMMPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.misc.sampler import SimpleSampler, NormalizeSampler
import tensorflow as tf
from sac.envs import GymEnv
import os
from algorithms.knack_based_policy import KnackBasedPolicy, EExploitationPolicy
import numpy as np
from experiments.gym_experiment import parse_args
import multiprocessing as mp
from copy import deepcopy
from analysis.q_function_analysis_histogram import draw_histogram


def sign_variance_knack(q_values, q_var_max, q_var_min):
    """
    :param q_values:  shape:(data_num, action_sample_num)
    :param q_var_max:
    :param q_var_min:
    :return:
    """
    # calc variance
    q_variance = np.var(q_values, axis=1)  # shape:(data_num,)
    # q_variance = q_variance / (q_var_max - q_var_min)  # shape:(data_num,)

    # calc sign
    q_mean = np.mean(q_values, axis=1)
    diff = q_values - np.expand_dims(q_mean, axis=1)  # shape:(data_num, action_sample_num)
    nums_greater_than_mean = np.sum(diff >= 0, axis=1)  # shape:(data_num,)
    nums_smaller_than_mean = np.sum(diff < 0, axis=1)  # shape:(data_num,)
    signs = np.ones(len(q_values))  # shape:(data_num,)
    signs[nums_smaller_than_mean < nums_greater_than_mean] = -1.

    return signs * q_variance, diff  # shape:(data_num,)


def main(env_id, seed, entropy_coeff, n_epochs, dynamic_coeff, clip_norm, normalize_obs, buffer_size,
         max_path_length, min_pool_size, batch_size, policy_mode, eval_model, e,
         eval_n_episodes, eval_n_frequency, stochastic=True):
    tf.set_random_seed(seed=seed)

    env = GymEnv(env_id)
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
    elif policy_mode == "EExploitationPolicy":
        policy = EExploitationPolicy(env_spec=env.spec,
                                     K=4,
                                     hidden_layer_sizes=[layer_size, layer_size],
                                     qf=qf,
                                     reg=1e-3,
                                     squash=True,
                                     e=e
                                     )

    else:
        _, mode = str(policy_mode).split('-')
        if _ != "Knack":
            raise AssertionError(
                "policy_mode should be GMMPolicy or Knack-p_control or Knack-exploitation or Knack-exploration")
        else:
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
                metric="kurtosis"
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
    # -------------- setting done ------------------------

    # -------------- main process ------------------------
    with algorithm._sess.as_default():
        algorithm._saver.restore(algorithm._sess, eval_model)

        if stochastic:
            knack_file = os.path.join(os.path.dirname(eval_model), "array/epoch0_2001.npz")
            final_knacks = np.load(knack_file)['knack_kurtosis'][-1]
            final_variance = np.load(knack_file)['knack'][-1]
            final_variance_max = np.max(final_variance)
            final_variance_min = np.min(final_variance)
            print("variance max: {}, min: {}".format(final_variance_max, final_variance_min))

            knack_thresh = 0.8  ## TODO hyper param

        env = algorithm._env

        if hasattr(env, "env"):
            env = env.env

        # np.random.seed(seed)
        # env.seed(seed)
        num_data = 25  # num_data * nprocess == 1500
        steps_thresh = 1000
        data = {'obs': [],}
        for i in range(num_data):
            obs = env.reset()
            done = False
            steps = 0
            ret = 0
            if stochastic:
                _min = np.min(final_knacks)
                _max = np.max(final_knacks)
            print("start episode {}".format(i))
            while not done:
                steps += 1
                # env.render()
                if stochastic:
                    if hasattr(algorithm.policy, "knack_thresh"):
                        v, mean, var, kurtosis, signed_variance = algorithm._policy.calc_and_update_knack([obs])
                        knack_value = kurtosis[0]
                        # _min = min(knack_value, _min)
                        # _max = max(knack_value, _max)
                        knack_value = (knack_value - _min) / (_max - _min)
                        if knack_value > knack_thresh:  ## TODO hyper param
                            # print("knack {}".format(knack_value))
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
                data['obs'].append(obs)
                ret += rew

                obs = obs_next
                if steps >= steps_thresh:
                    done = True


    # np.savez_compressed("a.npz", **data)
    # print("return mean: {}".format(np.mean(data['ep_rets'])))
    v, q_for_knack, knack, knack_kurtosis = [], [], [], []
    for i in range(int(len(data["obs"]) / 100)):
        _v, _q_for_knack, _knack, _knack_kurtosis, _signed_variance = algorithm.policy.calc_and_update_knack(observations=np.array(data["obs"][i * 100: (i+1) * 100]))
        v.append(_v), q_for_knack.append(_q_for_knack), knack.append(_knack), knack_kurtosis.append(_knack_kurtosis)

    v, q_for_knack, knack, knack_kurtosis = np.concatenate(v, axis=0), np.concatenate(q_for_knack, axis=0), np.concatenate(knack, axis=0), np.concatenate(knack_kurtosis, axis=0)
    new_knack, diff_to_mean = sign_variance_knack(q_for_knack, final_variance_max, final_variance_min)
    knack_kurtosis = (knack_kurtosis - _min) / (_max - _min)
    knack_or_not = knack_kurtosis > knack_thresh

    return {"knack_kurtosis": knack_kurtosis, "knack_or_not": knack_or_not, "sign_variance_knack": new_knack, "variance": knack, "diff_to_meam": diff_to_mean}

if __name__ == '__main__':
    args = parse_args()
    print(os.path.abspath(__file__))


    # set environment
    seed = args['seed']
    args['stochastic'] = True
    # set log directory
    root_dir = args.pop('root_dir')
    opt_log_name = args.pop('opt_log_name')
    save_dir = os.path.dirname(args["eval_model"])

    def wrap(_args):
        return main(**_args)

    nprocess = 2
    args_s = [deepcopy(args) for i in range(nprocess)]
    for i in range(nprocess):
        seed = np.random.randint(low=0, high=1000)
        # print(seed)
        args_s[i]["seed"] = seed

    # data = wrap(args)
    with mp.Pool(nprocess) as p:
        data = p.map(wrap, args_s)

    print("data reshaping ...")
    n = len(data)
    keys = list(data[0].keys())
    out = {}
    print(n)
    for k in keys:
        out[k] = np.concatenate([d[k] for d in data], axis=0)
    print("data reshaping done")

    draw_histogram(out, save_path=os.path.join(save_dir, "knack_hist.pdf"))
    # print("saving ...")
    # np.savez_compressed("b.npz", **out)
