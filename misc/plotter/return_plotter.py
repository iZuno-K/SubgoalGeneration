import matplotlib.pyplot as plt
import os
import argparse
import json
import numpy as np
import csv
from glob import glob
from scipy import stats

# _ls = [(0, (3, 5, 1, 5, 1, 5)), "dashdot", "dotted", "dashed", "solid"]

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


def smooth_plot2(x_s, y_s, interval):
    """smooth plot by averaging"""
    x = np.array(x_s)  # just copy
    y = []
    for i in range(len(y_s)):
        y.append(np.mean(y_s[max(0, i - interval):i]))
    return x, y

def load_from_my_format(log_file):
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

def csv_reader(log_file):
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダーを読み飛ばしたい時

        _data = [row for row in reader]

    data = {}
    for h, d in zip(header, data):
        data[h] = d
    return data


def log_reader(log_file):
    if ".json" in log_file:
        return load_from_my_format(log_file)
    elif ".csv" in log_file:
        return csv_reader(log_file)


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


def compare_reward_plotter(root_dirs, labels, mode="exploration", smooth=1):
    """
    plot return curves to compare multiple learning-experiments
    :param root_dirs: list of parent directories of seed*
    :param labels: list of label of data to be compared
    :return:
    """
    plt.style.use('mystyle2')
    # plt.style.use('powerpoint_style')
    fig, axis = plt.subplots()
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    log_file = "log.*"
    if mode == "exploration":
        y_key = "mean_return"
    elif mode == "exploitation":
        y_key = "eval_average_return"
    else:
        raise AssertionError("mode should be `exploration` or `exploitation` but received {}".format(mode))

    compare_two_returns = []
    xlabel = "total_step"
    # xlabel = "total_epochs"

    i = 0
    for root_dir, label, c in zip(root_dirs, labels, cycle):
        seeds_logs = glob(os.path.join(root_dir, '*', log_file))
        data = [log_reader(file) for file in seeds_logs]
        print(list(data[0].keys()))
        min_len = min([len(d[xlabel]) for d in data])
        print(min_len)
        _returns = np.array([d[y_key][:min_len] for d in data])
        _x = data[0][xlabel][: min_len]
        # _stats = np.mean(_returns, axis=0)
        _stats = np.median(_returns, axis=0)


        compare_two_returns.append(_returns)
        for _y in _returns:
            __x, __y = smooth_plot2(_x, _y, interval=smooth)
            axis.plot(__x, __y, color=c, alpha=0.2)
        __x, __y = smooth_plot2(_x, _stats, interval=smooth)
        axis.plot(__x, __y, color=c, label=label)
        i += 1

    axis.legend()
    axis.set_title("Compare learning-trajectory ({})".format(mode))
    axis.set_xlabel(xlabel)
    if smooth > 1:
        axis.set_ylabel("return (prev {} optimization average)".format(smooth))
    else:
        axis.set_ylabel("return")
    axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    fig.tight_layout()
    plt.show()

    l = min(len(compare_two_returns[0][0]), len(compare_two_returns[1][0]))

    # print(stats.ttest_rel(compare_two_returns[0][:, l-1], compare_two_returns[1][:, l-1]))
    # print(stats.ttest_ind(compare_two_returns[0][:, l-1], compare_two_returns[1][:, l-1], equal_var=False))
    # print(stats.mannwhitneyu(compare_two_returns[0][:, l-1], compare_two_returns[1][:, l-1], alternative='two-sided'))
    print(stats.mannwhitneyu(compare_two_returns[0][:, l - 1], compare_two_returns[1][:, l - 1]))
    np.savetxt("test1.txt", compare_two_returns[0][:, l-1], delimiter=',')
    np.savetxt("test2.txt", compare_two_returns[1][:, l - 1], delimiter=',')

def tmp():
    args = parse_args()
    root_dir = args['root_dir']
    save_path = os.path.join(root_dir, 'graphs')
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(os.path.join(root_dir, 'log.json')):
        log_file = os.path.join(root_dir, 'log.json')
        plot_log(log_file, save_path=save_path, eval=False)
        plot_log(log_file, save_path=save_path, eval=True)
    else:
        log_file = os.path.join(root_dir, 'progress.csv')
        csv_log_plotter(log_file=log_file, save_dir=save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--root-dirs', type=str, default=None, help="data root directories name separated by a `^`")
    parser.add_argument('--labels', type=str, default=None, help="label names separated by a `^`")
    parser.add_argument('--mode', type=str, default="exploration", help="exploration or exploitation")
    parser.add_argument('--smooth', type=int, default=1, help="smoothing interval")
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    root_dirs = args["root_dirs"].split('^')
    labels = args["labels"].split('^')
    compare_reward_plotter(root_dirs, labels, args['mode'], args["smooth"])

