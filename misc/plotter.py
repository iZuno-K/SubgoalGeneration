import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import json
from environments.continuous_space_maze import ContinuousSpaceMaze
import argparse
from matplotlib import patches, animation

def maze_plot(map, v_table, variances):
    """
    map and values must be the same size
    :param map:
    :param values:
    :return:
    """
    # sphinx_gallery_thumbnail_number = 2

    fig, (ax1, ax2) = plt.subplots(1, 2)

    im = ax1.imshow(v_table, cmap='Reds')
    # Loop over data dimensions and create text annotations.
    for i in range(v_table.shape[0]):
        for j in range(v_table.shape[1]):
            text = ax1.text(j, i, map[i][j],
                           ha="center", va="center", color="black")

    im = ax2.imshow(variances, cmap='Reds')
    # Loop over data dimensions and create text annotations.
    for i in range(variances.shape[0]):
        for j in range(variances.shape[1]):
            text = ax2.text(j, i, map[i][j],
                           ha="center", va="center", color="black")

    ax1.set_title("V(s)")
    ax2.set_title('state-importance')

    fig.tight_layout()
    plt.show()

def continuous_maze_plot(root_dir):
    save_path = os.path.join(root_dir, 'graphs')
    os.makedirs(save_path, exist_ok=True)
    # log_file = os.path.join(root_dir, 'log.json')
    # plot_log(log_file, save_path=save_path)

    map_files = glob(os.path.join(root_dir, 'maps/*.npz'))
    # plot_map(map_files=map_files, is_mask=False)
    # plot_map(map_files=map_files, is_mask=True)
    ani = map_animation_maker(root_dir=root_dir)
    ani.animate(save_path=save_path)


def log_reader(log_file):
    """decode my log format"""
    data = dict(
        total_step=[],
        mean_return=[],
        q_loss=[],
        v_loss=[],
        policy_loss=[]
    )
    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        l = line.replace('\n', '')
        dic = json.loads(l)
        for key in data.keys():
            data[key].append(dic[key])

    return data

def plot_log(log_file, save_path=None):
    data = log_reader(log_file)
    total_steps = data.pop('total_step')
    ylabels = {'mean_return': 'mean return', 'q_loss': 'loss', 'v_loss': 'loss', 'policy_loss': 'loss', }
    plt.style.use('mystyle2')
    fig, axes = plt.subplots(2, 2, sharex='col')
    for i, key in enumerate(data.keys()):
        axes[int(i/2), i % 2].set_title(key)
        axes[int(i/2), i % 2].set_ylabel(ylabels[key])
        if int(i/2) == 1:
            axes[int(i/2), i % 2].set_xlabel('total steps')
        axes[int(i/2), i % 2].plot(total_steps, data[key])

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'log.pdf'))

    plt.show()

def plot_map(map_files, is_mask=False, save_path=None):
    env = ContinuousSpaceMaze()
    plt.style.use('mystyle3')
    fig, axes = plt.subplots(2, 2)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

    # plot maze
    a = np.zeros([50, 50])
    im = axes[0, 0].imshow(a, cmap='binary')
    c1 = patches.Circle(xy=env.h1.c, radius=env.h1.r, fc='k', ec='k')
    c2 = patches.Circle(xy=env.h2.c, radius=env.h2.r, fc='k', ec='k')
    axes[0, 0].add_patch(c1)
    axes[0, 0].add_patch(c2)

    # data load
    data = np.load(map_files[-1])
    v_map = data['v_map'].reshape(25, 25)
    knack_map = data['knack_map'].reshape(25, 25)
    knack_map_kurtosis = data['knack_map_kurtosis'].reshape(25, 25)

    v_map = map_reshaper(v_map)
    knack_map = map_reshaper(knack_map)
    knack_map_kurtosis = map_reshaper(knack_map_kurtosis)
    if is_mask:
        mask = np.ones([50, 50])
        for i in range(50):
            for j in range(50):
                if (np.linalg.norm(np.asarray([j, i]) - env.h1.c) <= env.h1.r) or (np.linalg.norm(np.asarray([i, j]) - env.h2.c) <= env.h2.r):
                    mask[j, i] = 0.

        v_map[mask == 0] = np.min(v_map)
        knack_map[mask == 0] = np.min(knack_map)
        knack_map_kurtosis[mask == 0] = np.min(knack_map_kurtosis)
        # we have already set color-bar-min as 0.
        # v_map[mask == 0] = 0.
        # knack_map[mask == 0] = 0.
        # knack_map_kurtosis[mask == 0] = 0.

    axes[0, 1].imshow(v_map, cmap='Reds')
    axes[0, 1].set_title('V(s)')
    axes[1, 0].imshow(knack_map, cmap='Reds')
    axes[1, 0].set_title('knack map')
    axes[1, 1].imshow(knack_map_kurtosis, cmap='Reds')
    axes[1, 1].set_title('knack map kurtosis')
    c1 = patches.Circle(xy=env.h1.c, radius=env.h1.r, fc='k', ec='k', alpha=0.1)
    c2 = patches.Circle(xy=env.h2.c, radius=env.h2.r, fc='k', ec='k', alpha=0.1)
    axes[1, 1].add_patch(c1)
    axes[1, 1].add_patch(c2)
    plt.show()

def map_reshaper(map):
    """reshape 25x25 to 50x50"""
    a = [[map[int(i/2), int(j/2)] for j in range(50)] for i in range(50)]
    return np.array(a)

class map_animation_maker(object):
    """
    see misc/test.py for simpler program
    """
    def __init__(self, root_dir):
        self.map_files = glob(os.path.join(root_dir, 'maps/*.npz'))

        # figure configuration
        plt.style.use('mystyle3')
        # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.fig, self.axes = plt.subplots(2, 2, sharex='col', sharey='row')
        for axs in self.axes:
            for ax in axs:
                ax.set_yticklabels([])
                ax.set_xticklabels([])

        self.axes[0, 1].set_title('map')
        self.axes[0, 1].set_title('relative V(s)')
        self.axes[1, 0].set_title('relative knack map')
        self.axes[1, 1].set_title('relative knack map kurtosis')

        # prepare to draw updatable map
        # !!!!!!!! set vmin and vmax is important!!!!!!!!!
        # we normalize array in (0., 1.) to visualize
        tmp = np.zeros([50, 50])
        self.im00 = self.axes[0, 0].imshow(tmp, cmap='binary')
        self.im01 = self.axes[0, 1].imshow(tmp, cmap='Reds', animated=True, vmin=0., vmax=1.)
        self.im10 = self.axes[1, 0].imshow(tmp, cmap='Reds', animated=True, vmin=0., vmax=1.)
        self.im11 = self.axes[1, 1].imshow(tmp, cmap='Reds', animated=True, vmin=0., vmax=1.)

        # prepare to draw hole
        self.env = ContinuousSpaceMaze()
        hole1 = patches.Circle(xy=self.env.h1.c, radius=self.env.h1.r, fc='k', ec='k')
        hole2 = patches.Circle(xy=self.env.h2.c, radius=self.env.h2.r, fc='k', ec='k')
        self.axes[0, 0].add_patch(hole1)
        self.axes[0, 0].add_patch(hole2)
        self.axes[0, 0].text(0.5, 0.5, 'S', horizontalalignment='center', verticalalignment='center', fontsize=5)
        self.axes[0, 0].text(self.env.goal[0], self.env.goal[1], 'G', horizontalalignment='center', verticalalignment='center', fontsize=5)
        # self.axes[0, 0].text(20, 45, 'G', horizontalalignment='center',
        #                      verticalalignment='center', fontsize=5)

        # to avoid re-use artist, re-define
        hole1 = patches.Circle(xy=self.env.h1.c, radius=self.env.h1.r, fc='k', ec='k', alpha=0.2)
        hole2 = patches.Circle(xy=self.env.h2.c, radius=self.env.h2.r, fc='k', ec='k', alpha=0.2)
        self.circles = [self.axes[0, 1].add_patch(hole1), self.axes[0, 1].add_patch(hole2)]
        hole1 = patches.Circle(xy=self.env.h1.c, radius=self.env.h1.r, fc='k', ec='k', alpha=0.2)
        hole2 = patches.Circle(xy=self.env.h2.c, radius=self.env.h2.r, fc='k', ec='k', alpha=0.2)
        self.circles.extend([self.axes[1, 0].add_patch(hole1), self.axes[1, 0].add_patch(hole2)])
        hole1 = patches.Circle(xy=self.env.h1.c, radius=self.env.h1.r, fc='k', ec='k', alpha=0.2)
        hole2 = patches.Circle(xy=self.env.h2.c, radius=self.env.h2.r, fc='k', ec='k', alpha=0.2)
        self.circles.extend([self.axes[1, 1].add_patch(hole1), self.axes[1, 1].add_patch(hole2)])

    def updateifig(self, i):
        # print(i)
        title = self.fig.suptitle("epoch{}".format(i), fontsize=5)
        data = self.load_map_data(self.map_files[i*2], is_mask=True)
        self.im01.set_array(data['v_map'])
        self.im10.set_array(data['knack_map'])
        self.im11.set_array(data['knack_map_kurtosis'])
        parts = [self.im01, self.im10, self.im11]
        parts.extend(self.circles)
        parts.append(title)
        return parts

    def animate(self, save_path=None):
        ani = animation.FuncAnimation(self.fig, self.updateifig, frames=int(len(self.map_files)/2), interval=100, blit=True)
        if save_path is not None:
            # ani.save(os.path.join(save_path, 'anim.gif'), writer='imagemagick')
            ani.save(os.path.join(save_path, 'anim.mp4'), writer='ffmpeg')
        else:
            plt.show()

    def load_map_data(self, map_file, is_mask=True):
        data = np.load(map_file)
        v_map = data['v_map'].reshape(25, 25)
        knack_map = data['knack_map'].reshape(25, 25)
        knack_map_kurtosis = data['knack_map_kurtosis'].reshape(25, 25)

        # normalize array into (0., 1.) to visualize
        v_map = normalize(v_map)
        knack_map = normalize(knack_map)
        knack_map_kurtosis = normalize(knack_map_kurtosis)

        v_map = map_reshaper(v_map)
        knack_map = map_reshaper(knack_map)
        knack_map_kurtosis = map_reshaper(knack_map_kurtosis)
        if is_mask:
            mask = np.ones([50, 50])
            for i in range(50):
                for j in range(50):
                    if (np.linalg.norm(np.asarray([j, i]) - self.env.h1.c) <= self.env.h1.r) or (np.linalg.norm(np.asarray([i, j]) - self.env.h2.c) <= self.env.h2.r):
                        mask[j, i] = 0.

            # v_map[mask == 0] = np.min(v_map)
            # knack_map[mask == 0] = np.min(knack_map)
            # knack_map_kurtosis[mask == 0] = np.min(knack_map_kurtosis)
            v_map[mask == 0] = 0.
            knack_map[mask == 0] = 0.
            knack_map_kurtosis[mask == 0] = 0.

        map_data = {'v_map': v_map, 'knack_map': knack_map, 'knack_map_kurtosis': knack_map_kurtosis}
        return map_data


def normalize(arr):
    m = np.min(arr)
    arr = arr - m
    M = np.max(arr)
    arr = arr / M
    return arr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir',
                        type=str,
                        default=None)
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    continuous_maze_plot(args['root_dir'])
