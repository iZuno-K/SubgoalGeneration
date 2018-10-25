from sac.policies.gmm import GMMPolicy
from rllab.misc.overrides import overrides
import tensorflow as tf
from tensorflow.contrib.distributions import Uniform
import numpy as np


class KnackBasedPolicy(GMMPolicy):
    def __init__(self, a_lim_lows, a_lim_highs, env_spec, qf, K=2, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, name='gmm_policy'):
        """
        Args:
            a_lim_lows (numpy.ndarray): the lower limits of action
            a_lim_highs (numpy.ndarray): the higher limits of action
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
        super(KnackBasedPolicy, self).__init__(env_spec, K, hidden_layer_sizes, reg, squash, qf, name)

        self.target_knack = 0
        self._actions_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='latents',
        )
        self._n_approx = 1000  # the numbers of samples to estimate variance of Q(s,.)
        self.a_lim_lows = a_lim_lows
        self.a_lim_highs = a_lim_highs
        self.q_mean_t, self.q_var_t, self.q_kurtosis_t = self.build_calc_knack_t()

        self.metric = 2  # if you measure knack by variance, set 1 if by kurtosis, set 2
        self.knack_thresh = 0.8
        self.normalize_params = {'min': 0, 'max': 1}


    def build_calc_knack_t(self):
        # copy observation
        duplicated_obs = tf.tile(self.observation_space, [1, self._n_approx])  # (batch, self._n_approx * self.Ds)
        duplicated_obs = tf.reshape(duplicated_obs, (-1, self._Ds))  # (batch * self._n_approx, self.Ds)

        action_sample_t = Uniform(low=self.a_lim_lows, high=self.a_lim_highs)
        action_shape = (tf.shape(self._observations_ph[0]), self._n_approx)
        action_sample = action_sample_t.sample(action_shape)  # (batch, self._n_approx, self._Da)
        action_sample = tf.reshape(action_sample, (-1, self._Da)) # (batch * self._n_approx, self._Da)

        qf_t = self._qf.get_output_for(duplicated_obs, action_sample, reuse=True)  # (batch * self._n_approx)
        qf_t = tf.reshape(qf_t, (-1, self._n_approx))  # (batch, self._n_approx)
        q_mean_t = tf.reduce_mean(qf_t, axis=1)  # (batch,)
        q_var_t = tf.reduce_mean(tf.square(qf_t - q_mean_t), axis=1)  # (batch,)
        q_kurtosis_t = tf.reduce_mean(tf.pow(qf_t - q_mean_t, 4), axis=1) / q_var_t

        return q_mean_t, q_var_t, q_kurtosis_t

    def calc_knack(self, observations):
        """
        :param observation: (batch, self._Ds)
        :return:
        """
        sess = tf.get_default_session()
        feed = {self._observations_ph: observations}
        mean, var, kurtosis = sess.run([self.q_mean_t, self.q_var_t, self.q_kurtosis_t], feed)
        return mean, var, kurtosis # each :(batch,)

    def update_normalize_params(self, _min, _max):
        self.normalize_params['min'] = _min
        self.normalize_params['max'] = _max

    @overrides
    def get_actions(self, observations):
        knacks = self.calc_knack(observations)
        knack = knacks[self.metric]
        knack = (knack - self.normalize_params['mean']) / self.normalize_params['max']
        if knack > self.knack_thresh:
            was = self._is_deterministic
            self._is_deterministic = True
            actions = super(KnackBasedPolicy, self).get_actions(observations)
            self._is_deterministic = was
            return actions
        else:
            # TODO action when not knack
            # P
            return

