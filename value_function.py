import numpy as np
import random
from math import isnan


class Q_learning(object):
    def __init__(self, state_dim, action_dim, gamma=0.99, alpha=0.3, epsilon=0.001):
        self.q_table = np.full((state_dim, action_dim), np.nan, np.float64)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.a_dim = action_dim

    def update(self, trajectory):
        # reverse order
        for traj in trajectory[::-1]:
            s0 = traj[0]
            a = traj[1]
            r = traj[2]
            s1 = traj[3]
            if isnan(self.q_table[s0, a]):
                self.q_table[s0, a] = 0.
            if isnan(self.q_table[s1, a]):
                self.q_table[s1, a] = 0.
            self.q_table[s0, a] = (1. - self.alpha) * self.q_table[s0, a] + self.alpha * (
                        r + self.gamma * np.nanmax(self.q_table[s1]))

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


class Value_function(object):
    def __init__(self, state_dim, gamma=0.99, alpha=0.3):
        self.v_table = np.full(state_dim, np.nan, np.float64)
        self.gamma = gamma
        self.alpha = alpha

    def update(self, trajectory):
        # reverse order
        for traj in trajectory[::-1]:
            s0 = traj[0]
            # a = traj[1]
            r = traj[2]
            s1 = traj[3]

            if isnan(self.v_table[s0]):
                self.v_table[s0] = 0.
            if isnan(self.v_table[s1]):
                self.v_table[s1] = 0.
            self.v_table[s0] = (1. - self.alpha) * self.v_table[s0] + self.alpha * (
                        r + self.gamma * self.v_table[s1])

    def save_table(self, save_path):
        np.save(save_path, self.v_table)

    def load_table(self, load_path):
        self.v_table = np.load(load_path)
