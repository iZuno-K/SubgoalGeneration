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
import numpy as np
from time import sleep

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

def main(root_dir):
    # tf.set_random_seed(seed=seed)
    # env = GymEnv('MountainCarContinuous-v0')
    env = GymEnv('MountainCarContinuousColor-v0')

    max_replay_buffer_size = int(1e6)
    sampler_params = {'max_path_length': 1000, 'min_pool_size': 1000, 'batch_size': 128}

    # TODO Normalize or not
    sampler = SimpleSampler(**sampler_params)

    entropy_coeff = 0.
    dynamic_coeff = True

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
        n_epochs=10,
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
        entropy_coeff=entropy_coeff
    )


    algorithm._sess.run(tf.global_variables_initializer())


    # TODO Normalize or not
    # Currently only MountainCar is available
    with algorithm._sess.as_default():
        model_file = os.path.join(root_dir, 'model')
        algorithm._saver.restore(algorithm._sess, model_file)

        for i in range(1):
            obs = env.reset()
            env.env.render()
            sleep(4.0)
            traj = [obs]
            done = False

            while not done:
                env.env.render()
                action = algorithm.policy.get_action(obs.flatten())
                obs, rew, done, _ = env.step(action)
                traj.append(obs.flatten())

            knack, knack_kurtosis = sub_goal_detect(algorithm, traj)
            idxs = np.argsort(knack_kurtosis)
            # idxs = np.argsort(knack)
            print(idxs[::-1])

            COL = MplColorHelper('Blues', np.min(knack_kurtosis), np.max(knack_kurtosis))
            for j, s in enumerate(traj):
                env.env.state = np.array(traj[j])
                rgba = COL.get_rgb(knack_kurtosis[j])
                env.env.render(car_rgba=rgba)
            sleep(1.0)

            for idx in idxs[::-1]:
                obs = env.reset()
                env.env.state = np.array(traj[0])
                rgba = COL.get_rgb(knack_kurtosis[0])
                env.env.render(car_rgba=rgba)
                for j in range(idx+1):
                    env.env.state = np.array(traj[j])
                    rgba = COL.get_rgb(knack_kurtosis[j])

                    # env.env.viewer.geoms[1].set_color(*(0.0, 0.0, 1.0))
                    env.env.render(car_rgba=rgba)
                sleep(0.5)

def sub_goal_detect(alg, traj):
    n_sample = 1000
    trajs = traj
    for i in range(n_sample - 1):
        trajs = np.concatenate((trajs, traj))

    q_values = alg._sess.run(alg.q_for_state_importance_ops, feed_dict={
        alg._observations_ph: trajs})  # debug OK: correctly sample different action per same state
    q_values = q_values.reshape(n_sample, len(traj))  # debug OK: Correct order by reshape
    q_1_moment = np.mean(q_values, axis=0)
    diffs_pow2 = np.square(q_values - q_1_moment)
    q_2_moment = np.mean(diffs_pow2, axis=0)
    q_4_moment = np.mean(np.square(diffs_pow2), axis=0)
    knack = q_2_moment
    knack_kurtosis = q_4_moment / np.square(q_2_moment)
    return knack, knack_kurtosis

class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    main(**args)
