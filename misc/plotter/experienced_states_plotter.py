import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
from environments.continuous_space_maze import ContinuousSpaceMaze
import argparse
from matplotlib import patches, animation


def normalize(arr):
    """
    normalize arr in [0, 1]
    :param numpy.ndarray arr:
    :return numpy.ndarray arr:
    """
    m = np.min(arr)
    arr = arr - m
    M = np.max(arr)
    arr = arr / M
    return arr


def map_reshaper(map):
    """
    reshape 25x25 to 50x50 (copy the nearest value)
    :param numpy.ndarray map: arr with shape (25, 25)
    :return numpy.ndarray map: arr with shape (50, 50):
    """
    a = [[map[int(i / 2), int(j / 2)] for j in range(50)] for i in range(50)]
    return np.array(a)


"""
1. 経験状態、コツ度のデータパス取得 done
2. ４つグラフを下準備（地図、V、knack, kurtosis）
3. 経験状態、コツ度のデータを読み込む、scatter
"""


class ExperiencedKnackAnimationMaker(object):
    """
    二次元状態空間上のコツ値をヒートマップで表示する。経験した状態にのみ打点する。
    ４つのグラフを同時に描画（マップ、V(s)、分散コツ、尖度コツ）
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.experienced_states_kancks_paths = glob(os.path.join(root_dir, 'experienced/*.npz'))
        self.frame_skip = 1  # skip data to reduce computation
        self.init_figure(root_dir)
        self.counter = 0

    def init_figure(self, root_dir):
        # figure configuration
        plt.style.use('mystyle3')
        # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.fig, self.axes = plt.subplots(2, 2, sharex='col', sharey='row')
        for ax in self.axes.flatten():
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        title = ['map', 'relative V(s)', 'relative knack map', 'relative knack map kurtosis']
        for t, ax in zip(title, self.axes.flatten()):
            ax.set_title(t)
            ax.set_xlim(0, 50)
            ax.set_ylim(50, 0)

        # prepare to draw hole
        modes = ['DoubleRevised', 'Double', 'Single', 'OneHole', 'EasierDouble']
        for mode in modes:
            if mode in root_dir:
                path_mode = mode
                break
        self.env = ContinuousSpaceMaze(goal=(20, 45), path_mode=path_mode)

        self.circles = []
        for ax in self.axes.flatten(): # to avoid re-use artist, re-define Circles
            hole1 = patches.Circle(xy=self.env.h1.c, radius=self.env.h1.r, fc='k', ec='k', alpha=0.2)
            hole2 = patches.Circle(xy=self.env.h2.c, radius=self.env.h2.r, fc='k', ec='k', alpha=0.2)
            self.circles.extend([ax.add_patch(hole1), ax.add_patch(hole2)])
        self.axes[0, 0].text(0., 0., 'S', horizontalalignment='center', verticalalignment='center', fontsize=5)
        self.axes[0, 0].text(self.env.goal[0], self.env.goal[1], 'G', horizontalalignment='center',
                             verticalalignment='center', fontsize=5)

        # prepare to draw states
        self.experienced_points = [ax.scatter(x=0, y=0, c=1., cmap='Blues', vmin=0., vmax=1., s=10, animated=True) for
                                   ax in self.axes.flatten()]
        self.experienced_points_scat = np.asarray(self.experienced_points)

    def updateifig(self, i):
        if (i % 20) == 0:
            self.data = self.load_data(self.experienced_states_kancks_paths[self.counter * self.frame_skip])
            self.counter += 1
        # data = self.load_map_data(self.experienced_states_kancks_paths[i * self.frame_skip])
        keys = ['states', 'v', 'knack', 'knack_kurtosis']
        data = {k: v[(i % 20) * 100: ((i % 20) + 1) * 100] for k, v in
                self.data.items()}  # 1 epoch consists 20 optimizations per 100 steps

        for j, scat in enumerate(self.experienced_points_scat.flatten()):
            if j == 0:  # upper left
                scat.set_offsets(data['states'])
            else:
                # set positions
                scat.set_offsets(data['states'])
                # set colors
                scat.set_array(data[keys[j]])

        parts = self.experienced_points_scat.flatten().tolist()
        parts.extend(self.circles)
        return parts

    def animate(self, save_path=None):
        """
        draw animation by calling updatefig func
        :param str save_path: path to save created animation
        :return:
        """
        interval = 100  # 1 frame per interval ms
        frames = int(20 * len(self.experienced_states_kancks_paths) / self.frame_skip)  # times to call updatefig
        blit = True  # acceralate computation
        ani = animation.FuncAnimation(self.fig, self.updateifig, frames=frames,
                                      interval=interval, blit=blit)
        if save_path is not None:
            ani.save(os.path.join(save_path, 'anim.mp4'), writer='ffmpeg')
        else:
            plt.show()

    def load_data(self, data_file):
        """
        load states and knack values from a file, and
        :param str data_file: file path .npz
        :return : dict of numpy.ndarray (same_dim, diff_dim)
        """
        data = np.load(data_file)
        states = data['states']
        knack = data['knack']
        knack_kurtosis = data['knack_kurtosis']
        v = data['q_1_moment']

        extracted_data = {'states': states, 'v': normalize(v), 'knack': normalize(knack),
                          'knack_kurtosis': normalize(knack_kurtosis)}
        return extracted_data


"""
経験状態を2epochずつ読みこむ。
コツドマップを読み込む
状態を50x50の区画に割り振る。
割降った区画でmaskをする。
表示。
頻度もカウントしよう。
"""

class TotalExperienceAnimationMaker(object):
    """
    ２次元経験状態空間上のコツ値のヒートマップを描画する。
    状態空間は離散化される。離散化された領域内に経験状態があればその離散化された状態は経験済みとする。
    （連続空間だと厳密に同じ状態を経験することはないので近しいものはまとめてしまうということ）
    """

    def __init__(self, root_dir):
        # initialize figure
        self.root_dir = root_dir
        self.experienced_states_kancks_paths = glob(os.path.join(root_dir, 'experienced/*.npz'))
        self.map_paths = glob(os.path.join(self.root_dir, 'maps/*.npz'))
        self.frame_skip = 1  # skip data to reduce computation
        self.counter = 0
        self.range = [[0, 50], [50, 0]]  # range of x, y coordinate. y range is reversed for visualization
        self.resolution = 50  # discritization num
        self.states_visit_counts = np.zeros([50, 50])
        self.save_name = 'knack_of_experienced_states.mp4'
        self.init_figure(root_dir)

    def init_figure(self, root_dir):
        # figure configuration
        plt.style.use('mystyle3')
        # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.fig, self.axes = plt.subplots(2, 2, sharex='col', sharey='row')
        for ax in self.axes.flatten():
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        title = ['map', 'relative V(s)', 'relative knack map', 'relative knack map kurtosis']
        for t, ax in zip(title, self.axes.flatten()):
            ax.set_title(t)
            ax.set_xlim(self.range[0])
            ax.set_ylim(self.range[1])

        # prepare to draw hole
        modes = ['DoubleRevised', 'Double', 'Single', 'OneHole', 'EasierDouble']
        for mode in modes:
            if mode in root_dir:
                path_mode = mode
                break

        self.env = ContinuousSpaceMaze(goal=(20, 45), path_mode=path_mode)
        self.circles = []
        for ax in self.axes.flatten():  # to avoid re-use artist, re-define Circles
            hole1 = patches.Circle(xy=self.env.h1.c, radius=self.env.h1.r, fc='k', ec='k', alpha=0.2)
            hole2 = patches.Circle(xy=self.env.h2.c, radius=self.env.h2.r, fc='k', ec='k', alpha=0.2)
            self.circles.extend([ax.add_patch(hole1), ax.add_patch(hole2)])
        self.axes[0, 0].text(0., 0., 'S', horizontalalignment='center', verticalalignment='center', fontsize=5)
        self.axes[0, 0].text(self.env.goal[0], self.env.goal[1], 'G', horizontalalignment='center',
                             verticalalignment='center', fontsize=5)

        # initialize heat-map (value must be in [0, 1])
        tmp = np.zeros([50, 50])
        self.im = np.array([ax.imshow(tmp, cmap='Blues', animated=True, vmin=0., vmax=1.) for ax in self.axes.flatten()]).reshape(2, 2)

    def load_map_data(self, path):
        """
        load data to visualize from path
        :param str path: path to data file
        :return dict of numpy.ndarray: keys() = v_map, knack_map, knack_map_kurtosis
        """
        data = np.load(path)
        v_map = data['q_1_moment'].reshape(25, 25)
        knack_map = data['knack_map'].reshape(25, 25)
        knack_map_kurtosis = data['knack_map_kurtosis'].reshape(25, 25)

        # normalize array into (0., 1.) to visualize
        v_map = normalize(v_map)
        knack_map = normalize(knack_map)
        knack_map_kurtosis = normalize(knack_map_kurtosis)

        v_map = map_reshaper(v_map)
        knack_map = map_reshaper(knack_map)
        knack_map_kurtosis = map_reshaper(knack_map_kurtosis)

        return {'v_map': v_map, 'knack_map': knack_map, 'knack_map_kurtosis': knack_map_kurtosis}

    def updateifig(self, i):
        """
        called by animate
        :param int i: iteration times
        :return:
        """
        map_data = self.load_map_data(self.map_paths[self.counter * self.frame_skip])
        # experienced states data is saved twice more than map
        experienced_states = []
        for i in range(min(0, self.counter - 1) * self.frame_skip * 2, self.counter * self.frame_skip * 2):
            experienced_states.extend(np.load(self.experienced_states_kancks_paths[i])['states'])  # (steps, states_dim)
        experienced_states = np.array(experienced_states, dtype=np.int32).T  # (states_dim, steps)
        visit_count_hist, xedges, yedges = np.histogram2d(x=experienced_states[0], y=experienced_states[1], bins=self.resolution,
                                                          range=[sorted(self.range[0]), sorted(self.range[1])])

        self.states_visit_counts += visit_count_hist

        # normalize among only experienced states
        mask = self.states_visit_counts > 0
        mask = mask.T  # mask[x][y] -> mask[y][x] for map_data[y][x]
        for k, v in map_data.items():
            _min = np.min(v[mask])
            _max = np.max(v[mask])
            v = (v - _min) / _max
            v = v * mask
            map_data[k] = v

        # update heat-map
        self.im[0, 0].set_array(mask)
        for im, key in zip(self.im.flatten()[1:], map_data.keys()):
            im.set_array(map_data[key])

        # make return list to matplot
        parts = self.im.flatten().tolist()
        parts.extend(self.circles)

        self.counter += 1

        return parts

    def animate(self, save_path=None):
        """
         draw animation by calling updatefig func
         :param str save_path: path to save created animation
         :return:
         """
        interval = 100  # 1 frame per interval ms
        frames = int(20 * len(self.experienced_states_kancks_paths) / self.frame_skip)  # times to call updatefig
        blit = True  # acceralate computation
        ani = animation.FuncAnimation(self.fig, self.updateifig, frames=frames,
                                      interval=interval, blit=blit)
        if save_path is not None:
            ani.save(os.path.join(save_path, self.save_name), writer='ffmpeg')
        else:
            plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    # animator = ExperiencedKnackAnimationMaker(root_dir=args['root_dir'])
    animator = TotalExperienceAnimationMaker(root_dir=args['root_dir'])
    save_path = os.path.join(args['root_dir'], 'graphs')
    os.makedirs(save_path, exist_ok=True)

    animator.animate(save_path=save_path)
