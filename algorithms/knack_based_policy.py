from sac.policies.gmm import GMMPolicy
from rllab.misc.overrides import overrides
import tensorflow as tf
import numpy as np
from rllab.core.serializable import Serializable


class KnackBasedPolicy(GMMPolicy, Serializable):
    def __init__(self, env_spec, qf, vf, K=2, hidden_layer_sizes=(100, 100), reg=1e-3,
                 squash=True, name='gmm_policy', metric="kurtosis", exploitation_ratio=0.8, optuna_trial=False):
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
            metric: kurtosis or signed_variance or variance
        """
        Serializable.quick_init(self, locals())
        super(KnackBasedPolicy, self).__init__(env_spec, K, hidden_layer_sizes, reg, squash, qf, name)

        self._actions_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self._Da),
            name='latents',
        )

        self.metric = metric  # if you measure knack by variance, set 1 if by kurtosis, set 2, set 3 if you use signed variance

        self.optuna_trial = optuna_trial
        if self.optuna_trial is not None:
            self.exploitation_ratio = self.optuna_trial.suggest_uniform('knack_thresh', 0.2, 0.95)
        else:
            self.exploitation_ratio = exploitation_ratio
        self.current_knack_thresh = 1e8  # enough large value

    def calc_knack_value_by_metric(self, knacks):
        """
        :param knacks:  assume the output of self.calc_knack (dictionary of array)
        :return:
        """
        if self.metric == "kurtosis":
            knack = knacks["kurtosis"]
        elif self.metric == "signed_variance":
            knack = knacks["signed_variance"]
        elif self.metric == "negative_signed_variance":
            knack = - knacks["signed_variance"]  # negatively larger value regard as knack
        elif self.metric == "small_variance":
            knack = - np.abs(knacks["signed_variance"])  # absolutely smaller value regard as knack
        else:
            raise NotImplementedError
        return knack

    def calc_and_update_knack(self, observations):
        """
        使う状況は、最適化が終了した直後。
        直前のエポックのstateを全て入れる。最大最小ターゲットが更新される。全てのコツ度が出力される
        :param observations: (batch, self.Ds)
        :return:
        """
        knacks = self.calc_knack(observations)

        # ここで、割合を考慮してthresholdを持ってくる
        _idx = len(observations) * (1 - self.exploitation_ratio)
        idx1, idx2 = int(_idx) - 1, int(_idx + 0.5) - 1  # subtract -1 since idx starts with 0
        knack = self.calc_knack_value_by_metric(knacks)
        knack = np.sort(knack)
        self.current_knack_thresh = knack[idx1] * 0.5 + knack[idx2] * 0.5

        return knacks["kurtosis"], knacks["signed_variance"]  # each :(batch,), min max are scalar

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
        knack_value = self.calc_knack_value_by_metric(knacks_value)[0]

        if knack_value > self.current_knack_thresh:
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
            actions = super(KnackBasedPolicy, self).get_actions(observations)
            return actions


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
