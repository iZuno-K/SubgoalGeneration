import abc
import gtimer as gt

import numpy as np

# from rllab.misc import logger
from rllab.algos.base import Algorithm

from sac.core.serializable import deep_clone
from sac.misc import tf_utils
from sac.misc.sampler import rollouts

# my
import misc.mylogger as mylogger
import misc.log_scheduler as mylogger2
import misc.baselines_logger as logger
import os
import tensorflow as tf

from misc import debug
import time
class RLAlgorithm(Algorithm):
    """Abstract RLAlgorithm.

    Implements the _train and _evaluate methods to be used
    by classes inheriting from RLAlgorithm.
    """

    def __init__(
            self,
            sampler,
            n_epochs=1000,
            n_train_repeat=1,
            epoch_length=2000,
            eval_n_episodes=10,
            eval_n_frequency=1,
            eval_deterministic=True,
            eval_render=False,
            control_interval=1,
    ):
        """
        Args:
            n_epochs (`int`): Number of epochs to run the training for.
            n_train_repeat (`int`): Number of times to repeat the training
                for single time step.
            epoch_length (`int`): Epoch length.
            eval_n_episodes (`int`): Number of rollouts to evaluate.
            eval_deterministic (`int`): Whether or not to run the policy in
                deterministic mode when evaluating policy.
            eval_render (`int`): Whether or not to render the evaluation
                environment.
        """
        self.sampler = sampler

        self._n_epochs = n_epochs
        self._n_train_repeat = n_train_repeat
        self._epoch_length = epoch_length
        self._control_interval = control_interval

        self._eval_n_episodes = eval_n_episodes
        self._eval_n_frequency = eval_n_frequency
        self._eval_deterministic = eval_deterministic
        self._eval_render = eval_render

        self._sess = tf_utils.get_default_session()

        self._env = None
        self._policy = None
        self._pool = None

        self.log_writer = None

    def _train(self, env, policy, pool, qf=None, vf=None, saver=None, _ec=None, dynamic_ec=False):
        """Perform RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            pool (`PoolBase`): Sample pool to add samples to
        """

        self._init_training(env, policy, pool)
        self.sampler.initialize(env, policy, pool)
        # my
        save_episodes = 2
        save_knack_episodes = 50
        if dynamic_ec:
            dicrese_rate = _ec / self._n_epochs
        positive_visit_count = np.zeros([50, 50])
        logger2 = mylogger2.get_logger()
        os.makedirs(os.path.join(logger2.log_dir, 'model'), exist_ok=logger2.exist_ok)

        with self._sess.as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)
            episode_states = []
            # for epoch in gt.timed_for(range(self._n_epochs + 1), save_itrs=True):
            for epoch in gt.timed_for(range(self._n_epochs + 1), save_itrs=True):
                # logger.push_prefix('Epoch #%d | ' % epoch)
                epoch_states = []
                train_terminal_states = []
                for t in range(self._epoch_length):
                    # TODO.codeconsolidation: Add control interval to sampler
                    done, _n_episodes, next_obs, info = self.sampler.sample()
                    epoch_states.append(next_obs)
                    episode_states.append(next_obs)
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        self._do_training(
                            iteration=t + epoch * self._epoch_length,
                            batch=self.sampler.random_batch())
                    gt.stamp('train')

                    if done:
                        if hasattr(env, 'id'):
                            if "Maze" in env.id:
                                train_terminal_states.append(next_obs.tolist())

                        if info:  #["reached_goal"]
                            experienced_states = np.array(episode_states, dtype=np.int32).T  # (states_dim, steps)
                            positive_visit_count_hist, xedges, yedges = \
                                np.histogram2d(x=experienced_states[0], y=experienced_states[1], bins=50, range=[[0, 50], [0, 50]])
                            positive_visit_count += positive_visit_count_hist
                            episode_states = []

                    # if done and (_n_episodes % 2 == 0):
                    # if _n_episodes == save_episodes:
                    #     mylogger.write()
                    #     save_episodes += 2
                    # if _n_episodes == save_knack_episodes:
                    #     v_map, knack_map, knack_map_kurtosis = self._value_and_knack_map()
                    #     save_path = os.path.join(mylogger._my_map_log_dir, 'episode'+str(save_knack_episodes)+'.npz')
                    #     np.savez(save_path, v_map=v_map, knack_map=knack_map, knack_map_kurtosis=knack_map_kurtosis)
                    #     save_knack_episodes += 50

                if epoch % self._eval_n_frequency == 0:
                    eval_average_return = self._evaluate(epoch)
                    logger.record_tabular('eval_average_return', eval_average_return)
                    if hasattr(self.policy, "optuna_trial"):
                        if self.policy.optuna_trial is not None:
                            self.policy.optuna_trial.report(eval_average_return, epoch)  # report intermediate_value
                else:
                    logger.record_tabular('eval_average_return', np.nan)

                gt.stamp('eval')

                # params = self.get_snapshot(epoch)
                # logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('time-train', times_itrs['train'][-1])
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-sample', times_itrs['sample'][-1])
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('total_step', self.sampler._total_samples)
                logger.record_tabular('total_episode', self.sampler._n_episodes)
                # mylogger.data_update(key="epoch", val=epoch)
                # logger2.add_csv_data({"epoch": epoch})

                # self.sampler.log_diagnostics()

                # mylogger.write()

                # os.makedirs(os.path.join(mylogger._my_log_parent_dir, 'experienced'), exist_ok=True)
                os.makedirs(os.path.join(logger2.log_dir, 'experienced'), exist_ok=True)
                if logger2.save_array_flag:
                    if hasattr(self.policy, "knack_thresh"):
                        v, q_for_knack, knack, knack_kurtosis, signed_variance_t = self.policy.calc_and_update_knack(epoch_states)
                        # v = self.calc_value_and_knack_map(option_states=epoch_states, v_only=True)
                        kwargs1 = {'description': os.path.join('epoch{}'.format(epoch)),
                                   'states': np.array(epoch_states), 'knack': knack, 'knack_kurtosis': knack_kurtosis,
                                   'v': v, 'signed_variance': signed_variance_t}
                        # 'q_for_knack': q_for_knack,
                        logger2.add_array_data(kwargs1)
                        gt.stamp("calc knacks")
                    else:
                        v, knack, knack_kurtosis, q_for_knack = self.calc_value_and_knack_map(option_states=epoch_states)
                        # kwargs1 = {'file': os.path.join(logger2.log_qdir, 'experienced', '_epoch{}.npz'.format(epoch)),
                        kwargs1 = {'description': os.path.join('epoch{}'.format(epoch)),
                                   'states': np.array(epoch_states), 'knack': knack, 'knack_kurtosis': knack_kurtosis,
                                   'v': v}
                        # 'q_for_knack': q_for_knack,
                        logger2.add_array_data(kwargs1)

                    if epoch % 2 == 0:
                        if self.env.observation_space.flat_dim <= 2:
                            os.makedirs(os.path.join(logger2.log_dir, "map"), exist_ok=True)
                            map_save_path = os.path.join(logger2.log_dir, "map", 'epoch' + str(epoch) + '.npz')
                            v_map, knack_map, knack_map_kurtosis, q_1_moment_map = self.calc_value_and_knack_map()
                            kwargs = {'file': map_save_path, 'knack_map': knack_map, 'knack_map_kurtosis': knack_map_kurtosis,
                                      'q_1_moment': q_1_moment_map, 'train_terminal_states': np.asarray(train_terminal_states),
                                      'v_map': v_map, 'visit_count': positive_visit_count}
                            # save_thread2 = Thread(group=None, target=np.savez_compressed, kwargs=kwargs)
                            # save_thread2.start()
                            np.savez_compressed(**kwargs)

                    if epoch % 10 == 0:
                      #  TODO save only parameters
                        saver.save(self._sess, os.path.join(logger2.log_dir, 'model'))
                        gt.stamp("tf save")

                if dynamic_ec:
                    self._sess.run(tf.assign(_ec, _ec - dicrese_rate))

                logger.dump_tabular()
                # logger.dump_tabular(with_prefix=False)
                # logger.pop_prefix()
                # del logger._tabular[:]
                logger2.write()
                # gt.stamp('eval')
                # print(gt.report())

            if logger2.save_array_flag:
                saver.save(self._sess, os.path.join(logger2.log_dir, 'model'))

            self.sampler.terminate()
            return eval_average_return

    def _evaluate(self, epoch):
        """Perform evaluation for the current policy.

        :param epoch: The epoch number.
        :return: None
        """

        if self._eval_n_episodes < 1:
            return

        with self._policy.deterministic(self._eval_deterministic):
            paths = rollouts(self._eval_env, self._policy,
                             self.sampler._max_path_length, self._eval_n_episodes,
                            )

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]

        eval_average_return = np.mean(total_returns)
        # logger.record_tabular('return-min', np.min(total_returns))
        # logger.record_tabular('return-max', np.max(total_returns))
        # logger.record_tabular('return-std', np.std(total_returns))
        # logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        # logger.record_tabular('episode-length-min', np.min(episode_lengths))
        # logger.record_tabular('episode-length-max', np.max(episode_lengths))
        # logger.record_tabular('episode-length-std', np.std(episode_lengths))

        # mylogger.data_update(key='eval_average_return', val=np.mean(total_returns))
        logger2 = mylogger2.get_logger()
        # logger2.add_csv_data({'eval_average_return': np.mean(total_returns)})

        if hasattr(self._eval_env, 'id'):
            if "Maze" in self._eval_env:
                terminal_states = [path['next_observations'][-1].tolist() for path in paths]
                # mylogger.data_update(key='eval_terminal_states', val=terminal_states)
                logger2.add_array_data({'eval_terminal_states': terminal_states})

        self._eval_env.log_diagnostics(paths)
        if self._eval_render:
            self._eval_env.render(paths)

        # iteration = epoch*self._epoch_length
        # batch = self.sampler.random_batch()
        # self.log_diagnostics(iteration, batch)

        return eval_average_return

    @abc.abstractmethod
    def log_diagnostics(self, iteration, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError

    @abc.abstractmethod
    def _do_training(self, iteration, batch):
        raise NotImplementedError

    #my
    @abc.abstractmethod
    def calc_value_and_knack_map(self, option_states=None, v_only=False):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_training(self, env, policy, pool):
        """Method to be called at the start of training.

        :param env: Environment instance.
        :param policy: Policy instance.
        :return: None
        """

        self._env = env
        if self._eval_n_episodes > 0:
            # TODO: This is horrible. Don't do this. Get rid of this.
            import tensorflow as tf
            with tf.variable_scope("low_level_policy", reuse=True):
                self._eval_env = deep_clone(env)
        self._policy = policy
        self._pool = pool

    @property
    def policy(self):
        return self._policy

    @property
    def env(self):
        return self._env

    @property
    def pool(self):
        return self._pool
