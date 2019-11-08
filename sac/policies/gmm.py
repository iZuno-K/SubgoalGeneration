""" Gaussian mixture policy. """

from contextlib import contextmanager
import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.misc import logger

from sac.distributions import GMM
from sac.policies import NNPolicy
from sac.misc import tf_utils
from tensorflow.contrib.distributions import Uniform

EPS = 1e-6


class GMMPolicy(NNPolicy, Serializable):
    """Gaussian Mixture Model policy"""
    def __init__(self, env_spec, K=2, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, qf=None, name='gmm_policy'):
        """
        Args:
            env_spec (`rllab.EnvSpec`): Specification of the environment
                to create the policy for.
            K (`int`): Number of mixture components.
            hidden_layer_sizes (`list` of `int`): Sizes for the Multilayer
                perceptron hidden layers.
            reg (`float`): Regularization coeffiecient for the GMM parameters.
            squash (`bool`): If True, squash the GMM the gmm action samples
               between -1 and 1 with tanh.
            qf (`ValueFunction`): Q-function approximator.
        """
        Serializable.quick_init(self, locals())

        self._hidden_layers = hidden_layer_sizes
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim
        self._K = K
        self._is_deterministic = False
        self._fixed_h = None
        self._squash = squash
        self._qf = qf
        self._reg = reg

        self.name = name
        self.build()

        self._n_approx = 1000  # the numbers of samples to estimate variance of Q(s,.)
        self.a_lim_lows = env_spec.action_space.low
        self.a_lim_highs = env_spec.action_space.high
        self.q_kurtosis_t, self.signed_variance_t = self.build_calc_knack_t()

        self._scope_name = (
            tf.get_variable_scope().name + "/" + name
        ).lstrip("/")

        # TODO.code_consolidation: This should probably call
        # `super(GMMPolicy, self).__init__`
        super(NNPolicy, self).__init__(env_spec)

    def actions_for(self, observations, latents=None,
                    name=None, reuse=tf.AUTO_REUSE,
                    with_log_pis=False, regularize=False):
        name = name or self.name

        with tf.variable_scope(name, reuse=reuse):
            distribution = GMM(
                K=self._K,
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                cond_t_lst=(observations,),
                reg=self._reg
            )

        raw_actions = tf.stop_gradient(distribution.x_t)
        actions = tf.tanh(raw_actions) if self._squash else raw_actions

        # TODO: should always return same shape out
        # Figure out how to make the interface for `log_pis` cleaner
        if with_log_pis:
            # TODO.code_consolidation: should come from log_pis_for
            log_pis = distribution.log_p_t
            if self._squash:
                log_pis -= self._squash_correction(raw_actions)
            return actions, log_pis

        return actions

    def build(self):
        self._observations_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Ds),
            name='observations',
        )

        self._latents_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='latents',
        )

        self.sample_z = tf.random_uniform([], 0, self._K, dtype=tf.int32)

        # TODO.code_consolidation:
        # self.distribution is used very differently compared to the
        # `LatentSpacePolicy`s distribution.
        # This does not use `self.actions_for` because we need to manually
        # access e.g. `self.distribution.mus_t`
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.distribution = GMM(
                K=self._K,
                hidden_layers_sizes=self._hidden_layers,
                Dx=self._Da,
                cond_t_lst=(self._observations_ph,),
                reg=self._reg,
            )

        raw_actions = tf.stop_gradient(self.distribution.x_t)
        self._actions = tf.tanh(raw_actions) if self._squash else raw_actions
        # TODO.code_consolidation:
        # This should be standardized with LatentSpacePolicy/NNPolicy
        # self._determistic_actions = self.actions_for(self._observations_ph,
        #                                              self._latents_ph)

    @overrides
    def get_actions(self, observations):
        """Sample actions based on the observations.

        If `self._is_deterministic` is True, returns a greedily sampled action
        for the observations. If False, return stochastically sampled action.

        TODO.code_consolidation: This should be somewhat similar with
        `LatentSpacePolicy.get_actions`.
        """
        if self._is_deterministic: # Handle the deterministic case separately.
            if self._qf is None: raise AttributeError

            feed_dict = {self._observations_ph: observations}

            # TODO.code_consolidation: these shapes should be double checked
            # for case where `observations.shape[0] > 1`
            mus = tf.get_default_session().run(
                self.distribution.mus_t, feed_dict)[0]  # K x Da

            squashed_mus = np.tanh(mus) if self._squash else mus
            qs = self._qf.eval(observations, squashed_mus)

            # if self._fixed_h is not None:
            #     h = self._fixed_h # TODO.code_consolidation: this needs to be tiled
            # else:
            h = np.argmax(qs) # TODO.code_consolidation: check the axis

            actions = squashed_mus[h, :][None]
            return actions

        return super(GMMPolicy, self).get_actions(observations)

    def _squash_correction(self, actions):
        if not self._squash: return 0
        return tf.reduce_sum(tf.log(1 - tf.tanh(actions) ** 2 + EPS), axis=1)

    @contextmanager
    def deterministic(self, set_deterministic=True, latent=None):
        """Context manager for changing the determinism of the policy.

        See `self.get_action` for further information about the effect of
        self._is_deterministic.

        Args:
            set_deterministic (`bool`): Value to set the self._is_deterministic
                to during the context. The value will be reset back to the
                previous value when the context exits.
            latent (`Number`): Value to set the latent variable to over the
                deterministic context.
        """
        was_deterministic = self._is_deterministic
        old_fixed_h = self._fixed_h

        self._is_deterministic = set_deterministic
        if set_deterministic:
            if latent is None: latent = self.sample_z.eval()
            self._fixed_h = latent

        yield

        self._is_deterministic = was_deterministic
        self._fixed_h = old_fixed_h

    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records the mean, min, max, and standard deviation of the GMM
        means, component weights, and covariances.
        """

        feeds = {self._observations_ph: batch['observations']}
        sess = tf_utils.get_default_session()
        mus, log_sigs, log_ws = sess.run(
            (
                self.distribution.mus_t,
                self.distribution.log_sigs_t,
                self.distribution.log_ws_t,
            ),
            feeds
        )

        logger.record_tabular('gmm-mus-mean', np.mean(mus))
        logger.record_tabular('gmm-mus-min', np.min(mus))
        logger.record_tabular('gmm-mus-max', np.max(mus))
        logger.record_tabular('gmm-mus-std', np.std(mus))
        logger.record_tabular('gmm-log-w-mean', np.mean(log_ws))
        logger.record_tabular('gmm-log-w-min', np.min(log_ws))
        logger.record_tabular('gmm-log-w-max', np.max(log_ws))
        logger.record_tabular('gmm-log-w-std', np.std(log_ws))
        logger.record_tabular('gmm-log-sigs-mean', np.mean(log_sigs))
        logger.record_tabular('gmm-log-sigs-min', np.min(log_sigs))
        logger.record_tabular('gmm-log-sigs-max', np.max(log_sigs))
        logger.record_tabular('gmm-log-sigs-std', np.std(log_sigs))

    def build_calc_knack_t(self):
        # copy observation
        duplicated_obs = tf.tile(self._observations_ph, [1, self._n_approx])  # (batch, self._n_approx * self.Ds)
        duplicated_obs = tf.reshape(duplicated_obs, (-1, self._Ds))  # (batch * self._n_approx, self.Ds)

        action_sample_t = Uniform(low=self.a_lim_lows, high=self.a_lim_highs)
        action_shape = (tf.shape(self._observations_ph)[0], self._n_approx)
        action_sample = action_sample_t.sample(action_shape)  # (batch, self._n_approx, self._Da)
        action_sample = tf.cast(action_sample, tf.float32)
        action_sample = tf.reshape(action_sample, (-1, self._Da)) # (batch * self._n_approx, self._Da)

        qf_t = self._qf.get_output_for(duplicated_obs, action_sample, reuse=True)  # (batch * self._n_approx)
        qf_t = tf.reshape(qf_t, (-1, self._n_approx))  # (batch, self._n_approx)
        q_mean_t = tf.reduce_mean(qf_t, axis=1)  # (batch,)
        diff = qf_t - tf.expand_dims(q_mean_t, axis=1)  # (batch, self._n_approx)
        q_var_t = tf.reduce_mean(tf.square(diff), axis=1)  # (batch,)
        q_kurtosis_t = tf.reduce_mean(tf.pow(diff, 4), axis=1) / q_var_t / q_var_t

        nums_greater_than_mean = tf.reduce_sum(tf.cast(diff >= 0, tf.float32), axis=1)  # shape:(batch,)
        nums_smaller_than_mean = tf.reduce_sum(tf.cast(diff < 0, tf.float32), axis=1)  # shape:(batch,)
        signs = tf.sign(nums_smaller_than_mean - nums_greater_than_mean + 0.1)  # shape:(batch,)  avoid signs == 0 by adding tiny value 0.1 (< 1)
        signed_variance = signs * q_var_t

        return q_kurtosis_t, signed_variance

    def calc_knack(self, observations):
        """
        :param observations: (batch, self._Ds)
        :return:
        """
        sess = tf.get_default_session()
        feed = {self._observations_ph: observations}
        kurtosis, signed_variance = sess.run([self.q_kurtosis_t, self.signed_variance_t], feed)
        return {'kurtosis': kurtosis, 'signed_variance': signed_variance}  # each :(batch,)

    @staticmethod
    def get_q_params():
        tmp = tf.trainable_variables()
        _vars = []
        for var in tmp:
            if 'qf' in var.name:
                _vars.append(var)
        _vars = sorted(_vars, key=lambda x: x.name)
        vars_ndarray = tf.get_default_session().run(_vars)  # tf.Variable --> ndarray
        return {var.name: array for var, array in zip(_vars, vars_ndarray)}

    def build_assign_q_graph(self):
        _vars = [v for v in tf.trainable_variables() if 'qf' in v.name]
        self.assign_q_phs = {v.name: tf.placeholder(tf.float32, v.shape) for v in _vars}
        self.assign_q_ops = [tf.assign(v, self.assign_q_phs[v.name]) for v in _vars]

    def assign_q_params(self, param_dict):
        if not hasattr(self, "assign_q_ops"):
            self.build_assign_q_graph()
        feed_dict = {self.assign_q_phs[k]: v for k, v in param_dict.items()}
        tf.get_default_session().run(self.assign_q_ops, feed_dict)
