import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from matplotlib.ticker import MaxNLocator
from misc.plotter.return_plotter import smooth_plot2
from scipy import stats
import argparse
import multiprocessing as mp

def wrap(states_i):
    return calc_cumulative_stdevs(states_i[0], states_i[1])


def calc_state_variance(states):
    # states: shape (epoch, epoch_steps, obs_shape)
    stdevs = np.std(states, axis=1)  # shape (epoch, obs_shape)
    stdevs = np.mean(stdevs, axis=1)  # shape (epoch,)

    # cumulative_stdevs = np.zeros_like(stdevs)
    pas = [(states, i) for i in np.arange(len(states))]
    with mp.Pool(int(mp.cpu_count() - 1)) as p:
        cumulative_stdevs = p.map(wrap, pas)
    # for i in range(len(cumulative_stdevs)):
    #     std = np.std(states[:i+1], axis=(0, 1))  # shape (obs_shape,)
    #     std = np.mean(std)
    #     cumulative_stdevs[i] = std

    return stdevs, cumulative_stdevs


def calc_cumulative_stdevs(states, idx):
    std = np.std(states[:idx + 1], axis=(0, 1))  # shape (obs_shape,)
    std = np.mean(std)
    return std


def plot_state_variance(stdevs, cumulative_stdevs, save_path=None):
    #  plot
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../misc/plotter/drawconfig.mplstyle"))
    fig, axis = plt.subplots(ncols=2, figsize=(6, 2))

    epochs = np.arange(len(stdevs))
    axis[0].plot(epochs, stdevs)
    axis[0].set_title("stdev in an epoch")
    axis[0].set_xlabel('epoch')
    axis[0].set_ylaebl("stdev")

    axis[1].plot(epochs, cumulative_stdevs)
    axis[1].set_title("stdev by the epoch")
    axis[1].set_xlabel('epoch')
    axis[1].set_ylaebl("stdev")

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def compare_plotter(root_dirs, labels, smooth=1, plot_mode="raw", save_path=None):
    """
    plot return curves to compare multiple learning-experiments
    :param root_dirs: list of parent directories of seed*
    :param labels: list of label of data to be compared
    :return:
    """
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../misc/plotter/drawconfig.mplstyle"))
    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    axis = axis.flatten()
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    log_file = "epoch*.npz"

    for root_dir, label, c in zip(root_dirs, labels, cycle):
        print('processing {}...'.format(root_dir))
        # data load
        print('data loading ...')
        array_logs = glob(os.path.join(root_dir, 'seed*/array', log_file))
        data = [calc_state_variance(np.load(file)['states']) for file in array_logs]
        stdevs, cumulative_stdevs = list(zip(*data))
        epochs = np.arange(len(stdevs[0]))

        # main plot
        print('plotting ...')
        if plot_mode == "raw":
            i = 0
            for _stdev, _cumulative_stdev in zip(stdevs, cumulative_stdevs):
                x, y = smooth_plot2(epochs, _stdev, interval=smooth)
                if i == 0:
                    axis[0].plot(x, y, color=c, lw=1.3, label=label)
                    axis[0+2].plot(x, y, color=c, lw=1.3, label=label)
                else:
                    axis[0].plot(x, y, color=c, lw=1.3)
                    axis[0+2].plot(x, y, color=c, lw=1.3)

                x, y = smooth_plot2(epochs, _cumulative_stdev, interval=smooth)
                if i == 0:
                    axis[1].plot(x, y, color=c, lw=1.3, label=label)
                    axis[1+2].plot(x, y, color=c, lw=1.3, label=label)
                else:
                    axis[1].plot(x, y, color=c, lw=1.3)
                    axis[1+2].plot(x, y, color=c, lw=1.3)
                i += 1
        elif plot_mode == "iqr":
            # interquartile
            i = 0
            for ys in zip(axis, stdevs, cumulative_stdevs):
                iqr1, median, iqr3 = stats.scoreatpercentile(ys, per=(25, 50, 75), axis=0)
                x, iqr1 = smooth_plot2(epochs, iqr1, interval=smooth)
                x, iqr3 = smooth_plot2(epochs, iqr3, interval=smooth)
                x, median = smooth_plot2(epochs, median, interval=smooth)
                axis[i].fill_between(x, iqr1, iqr3, color=c, alpha=0.2)
                axis[i+2].fill_between(x, iqr1, iqr3, color=c, alpha=0.2)
                axis[i].plot(x, median, color=c, label=label, lw=1.3)
                axis[i+2].plot(x, median, color=c, label=label, lw=1.3)
                i += 1
        elif plot_mode == "std":
            i = 0
            for ys in zip(axis, stdevs, cumulative_stdevs):
                std = np.std(ys, axis=0)
                mean = np.mean(ys, axis=0)
                x, std = smooth_plot2(epochs, std, interval=smooth)
                x, mean = smooth_plot2(epochs, mean, interval=smooth)
                axis[i].fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
                axis[i+2].fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
                axis[i].plot(x, mean, color=c, label=label, lw=1.3)
                axis[i+2].plot(x, mean, color=c, label=label, lw=1.3)
                i += 1
        else:
            raise NotImplementedError
        i += 1

    # set labels
    for i in range(1):
        axis[0 + i*2].set_title("stdev in an epoch")
        axis[0 + i*2].set_xlabel('epoch')
        axis[0 + i*2].set_ylabel("stdev")

        axis[1 + i*2].set_title("stdev by the epoch")
        axis[1 + i*2].set_xlabel('epoch')
        axis[1 + i*2].set_ylabel("stdev")

    axis[2].set_xlim(-1, 100)
    axis[3].set_xlim(-1, 100)

    if smooth > 1:
        fig.suptitle("std (prev {} epoch average)".format(smooth))
    else:
        fig.suptitle("std")
    for ax in axis:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
    fig.tight_layout()

    # save
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dirs', type=str, default=None)
    parser.add_argument('--labels', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=1)
    parser.add_argument('--plot_mode', type=str, default='raw')
    parser.add_argument('--save_path', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    # root_dirs =
    args = parse_args()
    start = "/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/"
    labels = "EExploitation^GMMPolicy^Knack-exploration^negative_signed_variance^small_variance^signed_variance"
    env = 'Walker2d-v2'

    args.root_dirs = [os.path.join(start, l, env) for l in labels.split('^')]
    args.labels = labels
    args.smooth = 50
    args.save_path = '/home/isi/karino/state_variance.pdf'

    # compare_plotter(args.root_dirs.split('^'), args.labels.split('^'), args.smooth, args.plot_mode, args.save_path)
    compare_plotter(args.root_dirs, args.labels.split('^'), args.smooth, args.plot_mode, args.save_path)

