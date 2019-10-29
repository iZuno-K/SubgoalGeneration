import numpy as np
from math import isnan
import random


class Qlearning(object):
    """
    Discrete state-action space
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.3, epsilon=0.1, decay_rate=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = np.zeros((state_dim, action_dim))
        self.v_table = np.zeros(state_dim)

        self.gamma = gamma
        self.initial_alpha = alpha
        self.initial_epsilon = epsilon
        self.alpha = alpha
        self.epsilon = epsilon
        self.a_dim = action_dim

        self.decay_rate = decay_rate

    def update(self, trajectory):
        # about terminal state
        s0 = trajectory[-1][0]
        a = trajectory[-1][1]
        r = trajectory[-1][2]
        s1 = trajectory[-1][3]
        if isnan(self.q_table[s1, a]):
            self.q_table[s1, a] = 0.

        self.q_table[s0, a] = (1. - self.alpha) * self.q_table[s0, a] + self.alpha * r
        self.v_table[s0] = (1. - self.alpha) * self.v_table[s0] + self.alpha * r

        for traj in trajectory[:0:-1]:  # reverse order (exclude terminal state)
            s0 = traj[0]
            a = traj[1]
            r = traj[2]
            s1 = traj[3]

            self.q_table[s0, a] = (1. - self.alpha) * self.q_table[s0, a] + self.alpha * (
                    r + self.gamma * np.nanmax(self.q_table[s1]))
            self.v_table[s0] = (1. - self.alpha) * self.v_table[s0] + self.alpha * (
                    r + self.gamma * self.v_table[s1])

    def decay(self):
        if self.decay_rate is not None:
            # self.alpha = self.initial_alpha - self.alpha / self.decay_rate
            self.epsilon -= self.decay_rate

    def optimal_action(self, state):
        max_q = None
        for i, q in enumerate(self.q_table[state]):
            if max_q is None:
                max_q = q
            else:
                if max_q < q:
                    max_q = q
        # consider multiple maximums
        candidate = np.where(self.q_table[state] == max_q)[0]
        l = len(candidate)
        idx = random.randint(0, l - 1)
        return candidate[idx]

    def act(self, state, exploration=True):
        """
        確率εで探索
        :param state:
        :param exploration:
        :return:
        """
        if exploration:
            if np.random.uniform(0, 1) <= self.epsilon:
                return random.randint(0, self.a_dim - 1)
            else:
                return self.optimal_action(state)
        else:
            a = self.optimal_action(state)
            return a

    def save_table(self, save_path):
        np.save(save_path, self.q_table)

    def load_table(self, load_path):
        self.q_table = np.load(load_path)

    @staticmethod
    def calc_decay_rate(self, initial_eps, final_eps, decay_num):
        decay_rate = (initial_eps - final_eps) / decay_num
        return decay_rate

class KnackBasedQlearnig(Qlearning):
    """
    Discrete state-action space
    """

    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.3, epsilon=0.001, exploitation_ratio=0.2, metric="large_variance",
                 decay_rate=None):
        super(KnackBasedQlearnig, self).__init__(state_dim, action_dim, gamma, alpha, epsilon, decay_rate)

        self.state_importance = np.zeros(state_dim)
        self.exploitation_ratio = exploitation_ratio
        self.current_knack_thresh = 1e8
        self.bottleneck_exploitation_ratio = 0.95
        self.metric = metric
        self.epsilon = self.calc_epsilon(self.bottleneck_exploitation_ratio, self.exploitation_ratio, epsilon)
        self.reference_epsilon = epsilon

    def update(self, trajectory):
        super(KnackBasedQlearnig, self).update(trajectory)
        for traj in trajectory:
            s0 = traj[0]
            self.state_importance[s0] = self.calc_subgoal(s0)
        self.current_knack_thresh = self.calc_threshold(self.state_importance, self.exploitation_ratio)

    def calc_subgoal(self, s):
        """
        Assumption
        deterministic policy
        TODO: variance order
        :param s:
        :return:
        """
        kurtosis = 0
        q_mean = np.mean(self.q_table[s])
        variance = np.var(self.q_table[s])

        for a in range(self.action_dim):
            kurtosis += (self.q_table[s, a] - q_mean) ** 4
        kurtosis = kurtosis / (variance ** 2) / self.action_dim

        # signed_variance
        diff = self.q_table[s] - q_mean  # shape=(action_num,)
        greater_than_mean = np.count_nonzero(diff >= 0)
        smaller_than_mean = np.count_nonzero(diff < 0)  # TODO どっちも同じ数だったら，varianceが大きくても0になっちゃうよね？
        sign = np.sign(smaller_than_mean - greater_than_mean)
        if sign == 0:
            sign = 1
        signed_variance = sign * variance

        # state_importance = kurtosis
        # state_importance = variance
        state_importance = self.calc_knack_value_by_metric({'kurtosis': kurtosis, 'signed_variance': signed_variance}, metric=self.metric)
        # reward scale invariance
        # diff = self.q_table.max() - self.q_table.min()
        # if diff != 0:
        #     state_importance /= diff
        return state_importance

    @staticmethod
    def calc_threshold(values, exploitation_ratio):
        # calc threshold of state importance
        _idx = len(values) * (1 - exploitation_ratio)
        idx1, idx2 = int(_idx), int(_idx + 0.5)
        knack = np.sort(values)
        current_knack_thresh = knack[idx1] * 0.5 + knack[idx2] * 0.5
        return current_knack_thresh

    def act(self, state, exploration=True):
        """
        コツ度の計算
        閾値との比較
        if コツ度 > 閾値:
         確率0.95で活用
         確率0.05でランダム探索
        else:
         確率εで探索
        :param state:
        :param exploration:
        :return:
        """
        # calc knack
        if exploration:
            state_importance = self.calc_subgoal(state)
            if state_importance > self.current_knack_thresh:
                if np.random.uniform(0, 1) < self.bottleneck_exploitation_ratio:
                    return self.optimal_action(state)
                else:
                    return random.randint(0, self.a_dim - 1)
            else:
                return super(KnackBasedQlearnig, self).act(state, exploration)
        else:
            a = self.optimal_action(state)
            return a

    def save_table(self, save_path):
        np.save(save_path, self.q_table)

    def load_table(self, load_path):
        self.q_table = np.load(load_path)

    @staticmethod
    def calc_epsilon(bottleneck_exploitation_ratio, exploitation_ratio, target_eps):
        """
        calc epsilon which corresponds to standard epsilon-greedy
        :return:
        """
        k = bottleneck_exploitation_ratio
        p = exploitation_ratio
        # target_eps = (1 - k) * p + (1 - p) * eps_dash
        eps_dash = (target_eps - (1 - k) * p) / (1 - p)
        return eps_dash

    @staticmethod
    def calc_knack_value_by_metric(knacks, metric):
        """
        :param knacks:  assume the output of self.calc_knack (dictionary of array)
        :return:
        """
        if metric == "kurtosis":
            knack = knacks["kurtosis"]
        elif metric == "signed_variance":
            knack = knacks["signed_variance"]
        elif metric == "negative_signed_variance":
            knack = - knacks["signed_variance"]  # negatively larger value regard as knack
        elif metric == "small_variance":
            knack = - np.abs(knacks["signed_variance"])  # absolutely smaller value regard as knack
        elif metric == "large_variance":
            knack = np.abs(knacks["signed_variance"])
        else:
            raise NotImplementedError
        return knack

    def decay(self):
        if self.decay_rate is not None:
            # self.alpha = self.initial_alpha - self.alpha / self.decay_rate
            self.reference_epsilon -= self.decay_rate
            self.epsilon = self.calc_epsilon(self.bottleneck_exploitation_ratio, self.exploitation_ratio, self.reference_epsilon)
