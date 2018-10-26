import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import json
from environments.continuous_space_maze import ContinuousSpaceMaze
import argparse
from matplotlib import patches, animation
import itertools
import csv
import misc.debug as debug

def maze_plot(map, v_table, variances):
    """
    map and values must be the same size
    :param map:
    :param values:
    :return:
    """
    # sphinx_gallery_thumbnail_number = 2
    plt.style.use('mystyle3')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for ax in [ax1, ax2]:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    im = ax1.imshow(v_table, cmap='Blues')
    # Loop over data dimensions and create text annotations.
    for i in range(v_table.shape[0]):
        for j in range(v_table.shape[1]):
            text = ax1.text(j, i, map[i][j],
                           ha="center", va="center", color="black")

    im = ax2.imshow(variances, cmap='Blues')
    # Loop over data dimensions and create text annotations.
    for i in range(variances.shape[0]):
        for j in range(variances.shape[1]):
            text = ax2.text(j, i, map[i][j],
                           ha="center", va="center", color="black")

    ax1.set_title("V(s)")
    ax2.set_title('state-importance')

    fig.tight_layout()
    plt.show()


def log_reader(log_file):
    """decode my log format"""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        l = line.replace('\n', '')
        dic = json.loads(l)
        if i == 0:
            data = dic
            for key in data.keys():
                data[key] = [data[key]]
        else:
            for key in data.keys():
                data[key].append(dic[key])

    return data

def csv_log_plotter(log_file, save_dir):
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        data = [a for a in reader]
        data = list(zip(*data))  # [[1., 'a', '1h'], [2., 'b', '2b']] -> [(1., 2.), ('a', 'b'), ('1h', '2h')]
        data_dict = {header[i]: list(data[i]) for i in range(len(header))}

    x_label = 'total/epochs'
    y_labels = ['rollout/return', 'eval/return', 'train/loss_critic', 'train/loss_actor']
    x_data = [int(i) for i in data_dict[x_label]]
    y_datas = [[float(i) for i in data_dict[ylabel]] for ylabel in y_labels]

    plt.style.use('mystyle2')
    fig, axes = plt.subplots(2, 2, sharex='col')

    for i, label in enumerate(y_labels):
        # axes[int(i/2), i % 2].set_title(label)
        axes[int(i/2), i % 2].set_ylabel(label)
        if int(i/2) == 1:
            axes[int(i/2), i % 2].set_xlabel(x_label)

        x, y = smooth_plot(x_data, y_datas[i], interval=10)
        axes[int(i / 2), i % 2].plot(x, y)

    fig.suptitle('DDPG Learning Curve')
    plt.savefig(os.path.join(save_dir, 'reward_curve.pdf'))

def plot_log(log_file, save_path=None, eval=False):
    data = log_reader(log_file)
    total_steps = data.pop('total_step')
    if eval:
        ylabels = {'eval_average_return': 'eval average return', 'q_loss': 'loss', 'v_loss': 'loss', 'policy_loss': 'loss', }
    else:
       ylabels = {'mean_return': 'mean return', 'q_loss': 'loss', 'v_loss': 'loss', 'policy_loss': 'loss', }

    plt.style.use('mystyle2')
    fig, axes = plt.subplots(2, 2, sharex='col')
    for i, key in enumerate(ylabels.keys()):
        axes[int(i/2), i % 2].set_title(key)
        axes[int(i/2), i % 2].set_ylabel(ylabels[key])
        if int(i/2) == 1:
            axes[int(i/2), i % 2].set_xlabel('total steps')

        x, y = smooth_plot(total_steps, data[key], interval=10)
        axes[int(i / 2), i % 2].plot(x, y)

        # axes[int(i/2), i % 2].plot(total_steps, data[key])

    if save_path is not None:
        _name = 'log_eval.pdf' if eval else 'log_train.pdf'
        plt.savefig(os.path.join(save_path, _name))
    else:
        plt.show()


def map_reshaper(map):
    """reshape 25x25 to 50x50"""
    a = [[map[int(i/2), int(j/2)] for j in range(50)] for i in range(50)]
    return np.array(a)


class MapAnimationMaker(object):
    def __init__(self, root_dir, is_mask=True):
        self.map_files = glob(os.path.join(root_dir, 'maps/*.npz'))
        if os.path.exists(os.path.join(root_dir, 'log.json')):
            data = log_reader(os.path.join(root_dir, 'log.json'))
            self.train_terminal_states = data['train_terminal_states']

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
            ax.set_ylim(50, 0)
            ax.set_xlim(0, 50)

        # prepare to draw updatable map
        # !!!!!!!! set vmin and vmax is important!!!!!!!!!
        # we normalize array in (0., 1.) to visualize
        # off set to draw on imshow coordinate (see misc.test.plot_test)
        # self.offset = np.array([0.5, 0.5])
        tmp = np.zeros([50, 50])
        self.im = np.array([ax.imshow(tmp, cmap='Blues', animated=True, vmin=0., vmax=1.) for ax in self.axes.flatten()]).reshape(2, 2)

        # prepare to draw hole
        modes = ['DoubleRevised', 'Double', 'Single', 'OneHole', 'EasierDouble']
        for mode in modes:
            if mode in root_dir:
                path_mode = mode
                break
        self.env = ContinuousSpaceMaze(goal=(20, 45), path_mode=path_mode)
        hole1 = patches.Circle(xy=self.env.h1.c, radius=self.env.h1.r, fc='k', ec='k')
        hole2 = patches.Circle(xy=self.env.h2.c, radius=self.env.h2.r, fc='k', ec='k')
        self.axes[0, 0].add_patch(hole1)
        self.axes[0, 0].add_patch(hole2)
        self.axes[0, 0].text(0.5, 0.5, 'S', horizontalalignment='center', verticalalignment='center', fontsize=5)
        self.axes[0, 0].text(self.env.goal[0], self.env.goal[1], 'G', horizontalalignment='center', verticalalignment='center', fontsize=5)
        # self.axes[0, 0].text(20, 45, 'G', horizontalalignment='center',
        #                      verticalalignment='center', fontsize=5)
        # to avoid re-use artist, re-define
        self.circles = []
        for ax in self.axes.flatten()[1:]:
            hole1 = patches.Circle(xy=self.env.h1.c, radius=self.env.h1.r, fc='k', ec='k', alpha=0.2)
            hole2 = patches.Circle(xy=self.env.h2.c, radius=self.env.h2.r, fc='k', ec='k', alpha=0.2)
            self.circles.extend([ax.add_patch(hole1), ax.add_patch(hole2)])
            # self.circles.extend([ax.add_patch(hole2)])

        # prepare to draw terminal state
        self.train_terminal_states_scat = [ax.scatter(x=0, y=0, c='y', s=10, animated=True) for ax in self.axes.flatten()]
        self.train_terminal_states_scat = np.asarray(self.train_terminal_states_scat)
        # whether to mask values of state in hole
        self.is_mask=is_mask
        self.frame_skip = 2

    def updateifig(self, i):
        # print(i)
        data = self.load_map_data(self.map_files[i * self.frame_skip], is_mask=self.is_mask)
        keys = ['v_map', 'knack_map', 'knack_map_kurtosis']
        for im, key in zip(self.im.flatten()[1:], keys):
            im.set_array(data[key])

        if hasattr(self, 'train_terminal_states'):
            for j, scat in enumerate(self.train_terminal_states_scat.flatten()):
                if j == 0:  # uupper left
                    arr = self.reshaper_to_scat(self.train_terminal_states[:i * self.frame_skip])
                    if arr != []:
                       scat.set_offsets(np.asarray(arr))
                else:
                    if i > 0:
                        arr = self.reshaper_to_scat(self.train_terminal_states[(i - 1) * self.frame_skip:i * self.frame_skip])
                        if arr != []:
                            scat.set_offsets(np.asarray(arr))

        parts = self.im.flatten()[1:].tolist()
        if hasattr(self, 'train_terminal_states'):
            parts.extend(self.train_terminal_states_scat.flatten().tolist())
        parts.extend(self.circles)
        return parts

    def animate(self, save_path=None):
        ani = animation.FuncAnimation(self.fig, self.updateifig, frames=int(len(self.map_files) / self.frame_skip), interval=100, blit=True)
        if save_path is not None:
            # ani.save(os.path.join(save_path, 'anim.gif'), writer='imagemagick')
            ani.save(os.path.join(save_path, 'anim.mp4'), writer='ffmpeg')
        else:
            plt.show()

    @staticmethod
    def reshaper_to_scat(array):
        reshaped = []
        for arr in array:
            if arr == []:
                pass
            else:
                reshaped.extend(arr)
        return reshaped

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
                    # arr[i, j] means (x, y) = (j, i)
                    if (np.linalg.norm(np.asarray([j, i]) - self.env.h1.c) <= self.env.h1.r) or (np.linalg.norm(np.asarray([j, i]) - self.env.h2.c) <= self.env.h2.r):
                        mask[i, j] = 0.

            # v_map[mask == 0] = np.min(v_map)
            # knack_map[mask == 0] = np.min(knack_map)
            # knack_map_kurtosis[mask == 0] = np.min(knack_map_kurtosis)
            v_map[mask == 0] = 0.
            knack_map[mask == 0] = 0.
            knack_map_kurtosis[mask == 0] = 0.

        map_data = {'v_map': v_map, 'knack_map': knack_map, 'knack_map_kurtosis': knack_map_kurtosis}
        return map_data

class MapAnimationMakerDDPG(MapAnimationMaker):
    def load_map_data(self, map_file, is_mask=True):
        data = np.load(map_file)
        v_map = data['q_1_moment'].reshape(25, 25)
        knack_map = data['knack_map'].reshape(25, 25)
        knack_map_kurtosis = data['knack_map_kurtosis'].reshape(25, 25)
        terminal_states = data['train_terminal_states']

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
                    # arr[i, j] means (x, y) = (j, i)
                    if (np.linalg.norm(np.asarray([j, i]) - self.env.h1.c) <= self.env.h1.r) or (np.linalg.norm(np.asarray([j, i]) - self.env.h2.c) <= self.env.h2.r):
                        mask[i, j] = 0.

            v_map[mask == 0] = 0.
            knack_map[mask == 0] = 0.
            knack_map_kurtosis[mask == 0] = 0.

        map_data = {'v_map': v_map, 'knack_map': knack_map, 'knack_map_kurtosis': knack_map_kurtosis, 'terminal_states':terminal_states}
        return map_data

    def updateifig(self, i):
        # print(i)
        data = self.load_map_data(self.map_files[i * self.frame_skip], is_mask=self.is_mask)
        keys = ['v_map', 'knack_map', 'knack_map_kurtosis']
        for im, key in zip(self.im.flatten()[1:], keys):
            im.set_array(data[key])

        for j, scat in enumerate(self.train_terminal_states_scat.flatten()):
            if j == 0:  # uupper left
                arr = data['terminal_states']
                if arr != []:
                   scat.set_offsets(np.asarray(arr))
            else:
                if i > 0:
                    arr = data['terminal_states']
                    if arr != []:
                        scat.set_offsets(np.asarray(arr))

        parts = self.im.flatten()[1:].tolist()
        if hasattr(self, 'train_terminal_states'):
            parts.extend(self.train_terminal_states_scat.flatten().tolist())
        parts.extend(self.circles)
        return parts


class MountainCarAnimationMaker(object):
    def __init__(self, root_dir):
        self.map_files = glob(os.path.join(root_dir, 'maps/*.npz'))
        data = log_reader(os.path.join(root_dir, 'log.json'))
        self.total_steps = data['total_step']
        self.average_return = data['eval_average_return']
        self.eval_terminal_states = data['eval_terminal_states']
        self.train_terminal_states = data['train_terminal_states']

        from sac.envs import GymEnv
        env = GymEnv('MountainCarContinuousColor-v0')
        low_state = env.env.low_state
        high_state = env.env.high_state
        extent = [low_state[0], high_state[0], high_state[1], low_state[1]]
        aspect = (high_state[0] - low_state[0]) / (high_state[1] - low_state[1])


        # figure configuration
        plt.style.use('mystyle3')
        self.fig, self.axes = plt.subplots(2, 2, sharex='col', sharey='row')
        title = ['average relative Q(s,a)', 'relative V(s)', 'relative knack map', 'relative knack map kurtosis']
        for t, ax in zip(title, self.axes.flatten()):
            ax.set_title(t)

        # prepare to draw updatable map
        # !!!!!!!! set vmin and vmax is important!!!!!!!!!
        # we normalize array in (0., 1.) to visualize
        # extent set tick label range
        tmp = np.zeros([25, 25])
        self.im = np.array([ax.imshow(tmp, cmap='Blues', animated=True, vmin=0., vmax=1., extent=extent) for ax in self.axes.flatten()]).reshape(2, 2)
        for ax in self.axes.flatten():
            ax.set_aspect(aspect)
            # ax.set_aspect('equal')

        self.frame_skip = 2

    def updateifig(self, i):
        # print(i)
        data = self.load_map_data(self.map_files[i * self.frame_skip])
        keys = ['q_mean_map', 'v_map', 'knack_map', 'knack_map_kurtosis']
        for im, key in zip(self.im.flatten(), keys):
            im.set_array(data[key])

        parts = self.im.flatten().tolist()
        return parts

    def animate(self, save_path=None):
        ani = animation.FuncAnimation(self.fig, self.updateifig, frames=int(len(self.map_files) / self.frame_skip), interval=100, blit=True)
        if save_path is not None:
            # ani.save(os.path.join(save_path, 'anim.gif'), writer='imagemagick')
            ani.save(os.path.join(save_path, 'anim.mp4'), writer='ffmpeg')
        else:
            plt.show()

    @staticmethod
    def load_map_data(map_file):
        data = np.load(map_file)
        q_mean_map = data['q_mean_map'].reshape(25, 25)
        v_map = data['v_map'].reshape(25, 25)
        knack_map = data['knack_map'].reshape(25, 25)
        knack_map_kurtosis = data['knack_map_kurtosis'].reshape(25, 25)

        # normalize array into (0., 1.) to visualize
        q_mean_map = normalize(q_mean_map)
        v_map = normalize(v_map)
        knack_map = normalize(knack_map)
        knack_map_kurtosis = normalize(knack_map_kurtosis)

        map_data = {'q_mean_map': q_mean_map,'v_map': v_map, 'knack_map': knack_map, 'knack_map_kurtosis': knack_map_kurtosis}
        return map_data

def MapMakerExperiencedState(root_dir):
    """
    visualize knack-value of only actually experienced states
    """
    # load data
    states_knacks_file = os.path.join(root_dir, 'experienced_knack_data.npz')
    data = np.load(states_knacks_file)
    states = data['states']  # (sample_num, state_dim=2)
    knack_kurtosis = data['knack_kurtosis']  # (sample_num, 1)
    knack_kurtosis = knack_kurtosis.squeeze()  # (sample_num,)



    plt.style.use('mystyle3')
    offset = 0.5
    x, y = zip(*states)

    # prepare environment
    modes = ['DoubleRevised', 'Double', 'Single', 'OneHole', 'EasierDouble']
    for mode in modes:
        if mode in root_dir:
            path_mode = mode
            break
    env = ContinuousSpaceMaze(goal=(20, 45), path_mode=path_mode)
    fig, axes = plt.subplots()

    debug.debug_histogram(debug=False)
    xedges = np.arange(0, 51) - offset
    yedges = np.arange(0, 51) - offset
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    axes.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]], cmap='Blues')


    # off set to draw on imshow coordinate (see misc.test.plot_test)
    hole1 = patches.Circle(xy=env.h1.c, radius=env.h1.r, fc='k', ec='k')
    hole2 = patches.Circle(xy=env.h2.c, radius=env.h2.r, fc='k', ec='k')
    axes.add_patch(hole1)
    axes.add_patch(hole2)
    axes.text(0., 0., 'S', horizontalalignment='center', verticalalignment='center', fontsize=5)
    axes.text(env.goal[0], env.goal[1], 'G', horizontalalignment='center', verticalalignment='center',
                         fontsize=5)
    # axes.scatter(x=x, y=y, c=knack_kurtosis, cmap='Blues', s=10)
    axes.scatter(x=x, y=y, c='y', s=10)
    axes.set_ylim(50, 0)
    axes.set_xlim(0, 50)


    # TODO max をみつけてそこだけ違う色で

    plt.savefig(os.path.join(root_dir, 'graphs', 'experienced_states.png'))
    # plt.show()




def normalize(arr):
    m = np.min(arr)
    arr = arr - m
    M = np.max(arr)
    arr = arr / M
    return arr

def smooth_plot(x_s, y_s, interval):
    """smooth plot by averaging"""
    sta = 0
    x = []
    y = []
    for i in range(int(len(x_s) / interval)):
        x.append(np.mean(x_s[sta: sta + interval]))
        y.append(np.mean(y_s[sta: sta + interval]))
        sta += interval
    return x, y

def continuous_maze_plot(root_dir, is_mask=True):
    save_path = os.path.join(root_dir, 'graphs')
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(os.path.join(root_dir, 'log.json')):
        log_file = os.path.join(root_dir, 'log.json')
        plot_log(log_file, save_path=save_path, eval=False)
        plot_log(log_file, save_path=save_path, eval=True)
    else:
        log_file = os.path.join(root_dir, 'progress.csv')
        csv_log_plotter(log_file=log_file, save_dir=save_path)


    # map_paths = glob(os.path.join(root_dir, 'maps/*.npz'))
    # plot_map(map_paths=map_paths, is_mask=False)
    # plot_map(map_paths=map_paths, is_mask=True)

    # ani = MapAnimationMaker(root_dir=root_dir, is_mask=is_mask)

    # ani = MapAnimationMakerDDPG(root_dir=root_dir, is_mask=is_mask)
    # ani.animate(save_path=save_path)

    #
    # ani = MountainCarAnimationMaker(root_dir=root_dir)
    # ani.animate(save_path=save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--bool-test', type=bool, default=False)
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()

    continuous_maze_plot(args['root_dir'], is_mask=True)
    # MapMakerExperiencedState(root_dir=args['root_dir'])  deplicated