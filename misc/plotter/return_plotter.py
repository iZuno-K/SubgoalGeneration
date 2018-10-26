import matplotlib.pyplot as plt
import os
import argparse
import json
import numpy as np
import csv
from glob import glob

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


def compare_reward_plotter(root_dirs, labels, mode="exploration"):
    """
    plot return curves to compare multiple learning-experiments
    :param root_dirs: list of parent directories of seed*
    :param labels: list of label of data to be compared
    :return:
    """
    plt.style.use('mystyle2')
    fig, axis = plt.subplots()
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    log_file = "log.json"
    if mode == "exploration":
        y_key = "mean_return"
    elif mode == "exploitation":
        y_key = "eval_average_return"
    else:
        raise AssertionError("mode should be `exploration` or `exploitation` but received {}".format(mode))

    for root_dir, label, c in zip(root_dirs, labels, cycle):
        seeds_logs = glob(os.path.join(root_dir, '*', log_file))
        data = [log_reader(file) for file in seeds_logs]
        min_len = min([len(d['total_step']) for d in data])
        _returns = np.array([d[y_key][:min_len] for d in data])
        _x = data[0]['total_step'][: min_len]
        _mean = np.mean(_returns, axis=0)
        for _y in _returns:
            __x, __y = smooth_plot(_x, _y, interval=10)
            axis.plot(__x, __y, color=c, alpha=0.2)
        __x, __y = smooth_plot(_x, _mean, interval=10)
        axis.plot(__x, __y, color=c, label=label)

    axis.legend()
    axis.set_title("Compare learning-trajectory ({})".format(mode))
    axis.set_xlabel("total steps")
    axis.set_ylabel("return")
    axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.show()


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
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    root_dirs = args["root_dirs"].split('^')
    labels = args["labels"].split('^')
    compare_reward_plotter(root_dirs, labels, args['mode'])

