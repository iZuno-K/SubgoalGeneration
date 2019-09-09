from sac.policies.gmm import GMMPolicy
from rllab.misc.overrides import overrides
import tensorflow as tf
from tensorflow.contrib.distributions import Uniform
import numpy as np
from rllab.core.serializable import Serializable


class KnackBasedPolicy(GMMPolicy, Serializable):
    def __init__(self, a_lim_lows, a_lim_highs, mode, env_spec, qf, vf, K=2, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, name='gmm_policy'):
        """
        Args:
            a_lim_lows (numpy.ndarray): the lower limits of action
            a_lim_highs (numpy.ndarray): the higher limits of action
            mode (str): mode before knack state
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
        super(KnackBasedPolicy, self).__init__(env_spec, K, hidden_layer_sizes, reg, squash, qf, name)


        self._actions_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='latents',
        )
        self._n_approx = 1000  # the numbers of samples to estimate variance of Q(s,.)
        self.a_lim_lows = a_lim_lows
        self.a_lim_highs = a_lim_highs

        self.metric = 2  # if you measure knack by variance, set 1 if by kurtosis, set 2.
        self.knack_thresh = 0.8
        self.normalize_params = {'min': 0, 'max': 1}  # TODO
        self.target_knack = None
        self.target_knack_value = -1e6
        self.p = 0.1
        self.before_knack = True
        self.mode = mode  # p_control or exploitation or exploration

        self._vf_t = vf.get_output_for(self._observations_ph, reuse=True)
        self.q_mean_t, self.q_var_t, self.q_kurtosis_t, self.knack_min_t, self.knack_max_t, self.knack_target_t  = self.build_calc_knack_t()

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
        q_var_t = tf.reduce_mean(tf.square(qf_t - q_mean_t), axis=1)  # (batch,)
        q_kurtosis_t = tf.reduce_mean(tf.pow(qf_t - q_mean_t, 4), axis=1) / q_var_t / q_var_t
        knacks = [q_mean_t, q_var_t, q_kurtosis_t]
        _min = tf.reduce_min(knacks[self.metric])
        _max = tf.reduce_max(knacks[self.metric])
        _argmax = tf.argmax(knacks[self.metric])
        knack_state = self._observations_ph[_argmax]

        return q_mean_t, q_var_t, q_kurtosis_t, _min, _max, knack_state

    def calc_knack(self, observations):
        """
        :param observations: (batch, self._Ds)
        :return:
        """
        sess = tf.get_default_session()
        feed = {self._observations_ph: observations}
        mean, var, kurtosis = sess.run([self.q_mean_t, self.q_var_t, self.q_kurtosis_t], feed)
        return mean, var, kurtosis # each :(batch,)

    def update_normalize_params(self, _min, _max):
        self.normalize_params['min'] = _min
        self.normalize_params['max'] = _max if _min == _max else 1 + _min


    def update_target_knack(self, observation):
        self.target_knack = observation

    def p_control(self, observations):
        """
        we assume we know the relation of delta_s and action to suffice it.
        This can be replaced by MPC with transition model.
        :param observations: (batch, self.Ds)
        :return:
        """
        delta_s = self.p * (self.target_knack - observations)
        actions = np.clip(delta_s, self.a_lim_lows, self.a_lim_highs)
        return actions

    def act_before_knack(self, observations):
        """
        action before encountering knack in an episode
        :param observations:
        :return:
        """
        if self.mode == "p_control":
            actions = self.p_control(observations)
        elif self.mode == "exploitation":
            was = self._is_deterministic
            self._is_deterministic = True
            actions = super(KnackBasedPolicy, self).get_actions(observations)
            self._is_deterministic = was
        elif self.mode == "exploration":
            actions = super(KnackBasedPolicy, self).get_actions(observations)
        else:
            raise AssertionError("self.mode should be p_control or exploitation or exploration")
        return actions

    def calc_and_update_knack(self, observations):
        """
        使う状況は、最適化が終了した直後。
        直前のエポックのstateを全て入れる。最大最小ターゲットが更新される。全てのコツ度が出力される
        :param observations: (batch, self.Ds)
        :return:
        """
        sess = tf.get_default_session()
        feed = {self._observations_ph: observations}
        vf, mean, var, kurtosis, _min, _max, target_state = sess.run([self._vf_t, self.q_mean_t, self.q_var_t, self.q_kurtosis_t, self.knack_min_t, self.knack_max_t, self.knack_target_t], feed)
        self.update_normalize_params(_min, _max)
        self.update_target_knack(target_state)
        return vf, mean, var, kurtosis  # each :(batch,), min max are scalar

    @overrides
    def get_actions(self, observations):
        """
        :param observations: (batch, self.Ds)
        :return:
        """
        # TODO we assume single observation. Batch > 1 is not available
        if len(observations) > 1:
            raise AssertionError("get_actions should receive single observation: batch size=1, but received batch size={}".format(len(observations)))
        knacks_value = self.calc_knack(observations)
        knack_value = knacks_value[self.metric][0]
        _min = self.normalize_params['min']
        _max = self.normalize_params['max']
        # if knack_value > _max:
        #     self.update_normalize_params(_min, _max)  # TODO unexpected error ? max is not be updated
        #     self.update_target_knack(observations)
        knack_value = (knack_value - _min) / (_max - _min)
        # if knack_value > self.knack_thresh and self.target_knack is not None:  # on knack
        #     self.before_knack = False
        if knack_value > self.knack_thresh:
            # TODO: act deterministic optimal action with probability 95%
            if np.random.rand() < 0.95:
                was = self._is_deterministic
                self._is_deterministic = True
                actions = super(KnackBasedPolicy, self).get_actions(observations)
                self._is_deterministic = was
                return actions
            else:
                actions = super(KnackBasedPolicy, self).get_actions(observations)
                return actions
        else:
            if self.before_knack and self.target_knack is not None:  # before knack
                actions = self.act_before_knack(observations)
                return actions
            else:  # after knack or first epoch with no knack info
                actions = super(KnackBasedPolicy, self).get_actions(observations)
                return actions

    @overrides
    def reset(self, dones=None):
        """
        正規化項をいつ更新するか
        目標状態をいつ更新するか
        1,
        :param dones:
        :return:
        """
        self.before_knack = True


class EExploitationPolicy(GMMPolicy, Serializable):
    """
    Act exploitation(deterministic) with probability e
    """

    def __init__(self, env_spec, qf, K=2, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, name='EExploitation_policy', e=0):
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
        super(EExploitationPolicy, self).__init__(env_spec, K, hidden_layer_sizes, reg, squash, qf, name)
        self.e = e

        self._actions_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='latents',
        )

    @overrides
    def get_actions(self, observations):
        """
        :param observations: (batch, self.Ds)
        :return:
        """
        # TODO we assume single observation. Batch > 1 is not available
        if len(observations) > 1:
            raise AssertionError("get_actions should receive single observation: batch size=1, but received batch size={}".format(len(observations)))

        if np.random.rand() < self.e:
            was = self._is_deterministic
            self._is_deterministic = True
            actions = super(EExploitationPolicy, self).get_actions(observations)
            self._is_deterministic = was
        else:
            actions = super(EExploitationPolicy, self).get_actions(observations)

        return actions
