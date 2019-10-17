import abc
import gtimer as gt

import numpy as np

# from rllab.misc import logger
from rllab.algos.base import Algorithm

from sac.core.serializable import deep_clone
from sac.misc import tf_utils
from sac.misc.sampler import rollouts

# my
import misc.log_scheduler as mylogger2
import misc.baselines_logger as logger
import os
import tensorflow as tf

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
        if dynamic_ec:
            dicrese_rate = _ec / self._n_epochs

        logger2 = mylogger2.get_logger()
        os.makedirs(os.path.join(logger2.log_dir, 'model'), exist_ok=logger2.exist_ok)

        with self._sess.as_default():
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)
            for epoch in gt.timed_for(range(self._n_epochs), save_itrs=True):
                # logger.push_prefix('Epoch #%d | ' % epoch)
                epoch_states = []
                kurtosis = []
                signed_variance = []
                for t in range(self._epoch_length):
                    # TODO.codeconsolidation: Add control interval to sampler
                    done, _n_episodes, obs, next_obs, info = self.sampler.sample()
                    epoch_states.append(obs)

                    state_importances = self.policy.calc_knack([obs])
                    kurtosis.append(state_importances["kurtosis"])
                    signed_variance.append(state_importances["signed_variance"])  # be careful of batch_ready < epoch_length
                    if not self.sampler.batch_ready():
                        continue
                    gt.stamp('sample')

                    for i in range(self._n_train_repeat):
                        self._do_training(
                            iteration=t + epoch * self._epoch_length,
                            batch=self.sampler.random_batch())
                    gt.stamp('train')

                # evaluation
                if epoch % self._eval_n_frequency == 0:
                    eval_average_return = self._evaluate(epoch)
                    logger.record_tabular('eval_average_return', eval_average_return)
                    if hasattr(self.policy, "optuna_trial"):
                        if self.policy.optuna_trial is not None:
                            self.policy.optuna_trial.report(eval_average_return, epoch)  # report intermediate_value
                else:
                    logger.record_tabular('eval_average_return', np.nan)
                gt.stamp('eval')

                # logging about time and step
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

                # logging about array
                if hasattr(self.policy, "current_knack_thresh"):
                    current_knack_thresh = self.policy.current_knack_thresh
                    _ = self.policy.calc_and_update_knack(epoch_states)
                if logger2.save_array_flag:
                    kwargs1 = {'epoch': epoch, 'states': np.array(epoch_states),
                               'knack_kurtosis': np.array(kurtosis), 'signed_variance': np.array(signed_variance)}
                    if hasattr(self.policy, "current_knack_thresh"):
                        kwargs1.update({'current_knack_thresh':  current_knack_thresh})
                        kwargs1.update(self.policy.get_q_params())
                    logger2.add_array_data(kwargs1)

                    if epoch % 10 == 0:
                        #  TODO save only parameters
                        saver.save(self._sess, os.path.join(logger2.log_dir, 'model'))
                        gt.stamp("tf save")
                gt.stamp("calc knacks")

                if dynamic_ec:
                    self._sess.run(tf.assign(_ec, _ec - dicrese_rate))

                logger.dump_tabular()
                logger2.write()
                # print(gt.report())

            # finalize processing
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

        with self.policy.deterministic(self._eval_deterministic):
            paths = rollouts(self._eval_env, self.policy,
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
