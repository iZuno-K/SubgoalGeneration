import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class Hole(object):
    def __init__(self, center, radius):
        self.c = np.array(center)
        self.r = np.array(radius)


class ContinuousSpaceMaze(gym.Env):
    """50x50 2D continuous space maze"""
    def __init__(self):
        self.h1 = Hole(center=[23, 22], radius=14)
        self.h2 = Hole(center=[8, 42], radius=8)
        self.goal = np.array([30, 40])
        self.done = False
        self.state = np.array([0, 0]) + np.random.rand(1, 2)
        self.state_dim = (1, 2)
        self.action_dim = (1, 2)

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = 0.
        self.max_position = 50.

        self.low_state = np.array([self.min_position, self.min_position])
        self.high_state = np.array([self.max_position, self.max_position])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=self.action_dim)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.seed()
        self.reset()

    def reward(self, state):
        dist = np.linalg.norm(self.goal - state)
        if dist != 0:
            rew = 1.0 / dist
        else:
            rew = 1.0 / 1e-6
        return rew

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """deterministic transition"""
        if not self.done:

            # clip by maze border
            next_state = np.clip(self.state + action, self.min_position, self.max_position)
            r = self.reward(next_state)
            done = self.done_detection(state=next_state)

            self.state = next_state
            # observation, reward, done, info
            return self.state, r, done, {}
        else:
            raise Exception("reset required when an episode done")

    def done_detection(self, state):
        # if an agent is in a hole
        if np.linalg.norm(state - self.h1.c) <= self.h1.r:
            self.done = True
        if np.linalg.norm(state - self.h2.c) <= self.h2.r:
            self.done = True
        return self.done

    def reset(self):
        self.done = False
        self.state = np.array([0, 0]) + np.random.rand(1, 2)
        return self.state

def think_maze_layout():
    # h1 = Hole(center=[25, 20], radius=14)
    # h2 = Hole(center=[10, 40], radius=8)
    h1 = Hole(center=[23, 22], radius=14)
    h2 = Hole(center=[8, 42], radius=8)

    print(np.linalg.norm(h1.c - h2.c) - h1.r - h2.r)

    a = np.random.rand(50*50).reshape(50, 50)
    s = np.arange(50*50).reshape(50, 50)

    fig, ax = plt.subplots()

    im = ax.imshow(a, cmap='Reds')
    # Loop over data dimensions and create text annotations.
    # for i in range(a.shape[0]):
    #     for j in range(a.shape[1]):
    #         text = ax.text(j, i, s[i][j],
    #                         ha="center", va="center", color="w")

    c1 = patches.Circle(xy=h1.c, radius=h1.r, fc='k', ec='k')
    c2 = patches.Circle(xy=h2.c, radius=h2.r, fc='k', ec='k')
    ax.add_patch(c1)
    ax.add_patch(c2)

    plt.show()


if __name__ == '__main__':
    think_maze_layout()