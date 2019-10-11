import numpy as np
from math import isnan
import random


class SubgoalGeneration(object):
    """
    Discrete state-action space
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.3, epsilon=0.001, total_timesteps_for_decay=None):
        # self.q_table = np.full((state_dim, action_dim), np.nan, np.float64)
        # self.v_table = np.full(state_dim, np.nan, np.float64)
        self.q_table = np.zeros((state_dim, action_dim))
        self.v_table = np.zeros(state_dim)

        self.subgoals = []
        self.state_importance = np.zeros(state_dim)

        self.gamma = gamma
        self.initial_alpha = alpha
        self.initial_epsilon = epsilon
        self.alpha = alpha
        self.epsilon = epsilon
        self.a_dim = action_dim
        
        self.total_timesteps_for_decay = total_timesteps_for_decay

    def update(self, trajectory):
        # about terminal state
        s0 = trajectory[-1][0]
        a = trajectory[-1][1]
        r = trajectory[-1][2]
        s1 = trajectory[-1][3]
        if isnan(self.q_table[s1, a]):
            self.q_table[s1, a] = 0.
        if isnan(self.v_table[s1]):
            self.v_table[s1] = 0.

        self.q_table[s0, a] = (1. - self.alpha) * self.q_table[s0, a] + self.alpha * r
        self.v_table[s0] = (1. - self.alpha) * self.v_table[s0] + self.alpha * r

        for traj in trajectory[:0:-1]:  # reverse order (exclude terminal state)
            s0 = traj[0]
            a = traj[1]
            r = traj[2]
            s1 = traj[3]

            if isnan(self.q_table[s0, a]):
                self.q_table[s0, a] = 0.
            if isnan(self.v_table[s0]):
                self.v_table[s0] = 0.

            self.q_table[s0, a] = (1. - self.alpha) * self.q_table[s0, a] + self.alpha * (
                        r + self.gamma * np.nanmax(self.q_table[s1]))
            self.v_table[s0] = (1. - self.alpha) * self.v_table[s0] + self.alpha * (
                        r + self.gamma * self.v_table[s1])
            self.calc_subgoal(s0)

        if self.total_timesteps_for_decay is not None:
            self.alpha = self.initial_alpha - self.alpha/self.total_timesteps_for_decay
            self.epsilon = self.initial_epsilon - self.epsilon/self.total_timesteps_for_decay
    
    def calc_subgoal(self, s):
        """
        Assumption
        reward >= 0 (to normalize variance by dividing q_mean)
        deterministic policy
        TODO: variance order
        :param s:
        :return:
        """
        a_idx = []
        q_mean = 0
        variance = 0
        kurtosis = 0
        for i, q in enumerate(self.q_table[s]):
            if not isnan(q):
                a_idx.append(i)
                q_mean += q
        q_mean /= len(a_idx)
        for a in a_idx:
            variance += (self.q_table[s, a] - q_mean) ** 2
        variance /= len(a_idx)

        for a in a_idx:
            kurtosis += (self.q_table[s, a] - q_mean) ** 4
        kurtosis = kurtosis / (variance ** 2) / len(a_idx)

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
        state_importance = signed_variance
        self.state_importance[s] = state_importance

        update_idx = None
        insert_idx = len(self.subgoals)
        if len(self.subgoals) != 0:
            for i, sv in enumerate(self.subgoals):
                if s == sv[0]:
                    update_idx = i
                if state_importance >= sv[1]:
                    insert_idx = i
                    break

        if update_idx is None:
            for i, sv in enumerate(self.subgoals[insert_idx:]):
                if s == sv[0]:
                    update_idx = i + insert_idx

        if update_idx is None:
            self.subgoals.insert(insert_idx, [s, state_importance])
        else:
            self.subgoals.insert(insert_idx, [s, state_importance])
            if update_idx >= insert_idx:
                del self.subgoals[update_idx + 1]
            else:
                del self.subgoals[update_idx]

    def optimal_action(self, state):
        max_q = None
        for i, q in enumerate(self.q_table[state]):
            if isnan(q):
                pass
            else:
                if max_q is None:
                    max_q = q
                else:
                    if max_q < q:
                        max_q = q

        if max_q is None:
            return random.randint(0, self.a_dim - 1)
        else:
            # consider multiple maximums
            candidate = np.where(self.q_table[state] == max_q)[0]
            l = len(candidate)
            idx = random.randint(0, l-1)
            return candidate[idx]

    def act(self, state, exploration=True):
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
