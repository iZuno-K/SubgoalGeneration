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


def main(env_id, seed, entropy_coeff, n_epochs, dynamic_coeff, clip_norm, normalize_obs, buffer_size,
         max_path_length, min_pool_size, batch_size, policy_mode, eval_model, e, stochastic):
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
                mode=mode,
                env_spec=env.spec,
                K=4,
                hidden_layer_sizes=[layer_size, layer_size],
                qf=qf,
                vf=vf,
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
    # -------------- setting done ------------------------

    # -------------- main process ------------------------
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
        num_data = 50  # num_data * nprocess == 1500
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

    # np.savez_compressed("a.npz", **data)
    # print("return mean: {}".format(np.mean(data['ep_rets'])))
    return data

if __name__ == '__main__':
    args = parse_args()

    # set environment
    seed = args['seed']
    args['stochastic'] = False
    # set log directory
    root_dir = args.pop('root_dir')
    opt_log_name = args.pop('opt_log_name')


    def wrap(_args):
        return main(**_args)

    nprocess = 30
    args_s = [deepcopy(args) for i in range(nprocess)]
    for i in range(nprocess):
        seed = np.random.randint(low=0, high=1000)
        # print(seed)
        args_s[i]["seed"] = seed

    with mp.Pool(nprocess) as p:
        data = p.map(wrap, args_s)

    print("data reshaping ...")
    n = len(data)
    out = data[0]
    for d in data[1:]:
        for k, v in d.items():
            # print(k, v)
            out[k].extend(v)
    print("data reshaping done")

    print("saving ...")
    np.savez_compressed("a.npz", **out)
    print("averaged return is: ", np.mean(out["ep_rets"]))
