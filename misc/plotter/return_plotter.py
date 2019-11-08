import matplotlib.pyplot as plt
import os
import argparse
import json
import numpy as np
import csv
from glob import glob
from scipy import stats
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d
import pandas as pd

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
        y.append(np.mean(y_s[max(0, i - interval):i+1]))
    return x, y


def smooth_plot_2dim(x_s, ys_s, interval):
    x = np.array(x_s)  # just copy
    y = []
    for i in range(len(ys_s[0])):
        y.append(np.mean(ys_s[:, max(0, i - interval):i+1], axis=1))
    return x, np.array(y).T


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
    # print(log_file)
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # ヘッダーを読み飛ばしたい時
            lh =  len(header)
            data = [row for row in reader if len(row) == lh]

        data = list(zip(*data))  # [[1., 'a', '1h'], [2., 'b', '2b']] -> [(1., 2.), ('a', 'b'), ('1h', '2h')]
        data_dict = {header[i]: np.array(data[i], dtype=np.float) for i in range(len(header))}
        return data_dict
    except:
        print("file: {} has some error".format(log_file))
        return {}

# def csv_reader(log_file):
#     df = pd.read_csv(log_file, sep=',')
#     if len(df) > 0:
#         data_dict = {k: df[k].values for k in df.keys()}
#     return data_dict

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


def compare_reward_plotter(root_dirs, legends, xlabel="total_step", ylabel="eval_average_return", smooth=1, plot_mode="raw", save_path=None):
    """
    plot return curves to compare multiple learning-experiments
    :param root_dirs: list of parent directories of seed*
    :param legends: list of label of data to be compared
    :return:
    """
    plt.style.use('mystyle2')
    # plt.style.use('powerpoint_style')
    fig, axis = plt.subplots()
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    log_file = "log.csv"
    compare_two_returns = []

    i = 0
    for root_dir, label, c in zip(root_dirs, legends, cycle):
        print(root_dir)
        seeds_logs = glob(os.path.join(root_dir, '*', log_file))
        data = [log_reader(file) for file in seeds_logs]
        print(list([len(d[xlabel]) for d in data]))
        print(list(data[0].keys()))
        min_len = min([len(d[xlabel]) for d in data])
        print(min_len)
        _returns = np.array([d[ylabel][:min_len] for d in data], dtype=np.float)
        # _x = np.array(data[0][xlabel][: min_len], dtype=np.float)
        _xs = np.array([d[xlabel][: min_len] for d in data], dtype=np.float)
        _x =  _xs[0]
        # _stats = np.mean(_returns, axis=0)
        _stats = np.median(_returns, axis=0)
        print("max return: {} its file is: {}".format(max(_returns[:, -1]), seeds_logs[np.argmax(_returns[:, -1])]))

        compare_two_returns.append(_returns)
        if plot_mode == "raw":
            for _x, _y in zip(_xs, _returns):
                __x, __y = smooth_plot2(_x, _y, interval=smooth)
                axis.plot(__x, __y, color=c, alpha=0.2, lw=1.3)

            __x, __y = smooth_plot2(_x, _stats, interval=smooth)
            axis.plot(__x, __y, color=c, label=label, lw=1.3)
        elif plot_mode == "iqr":
            # interquartile
            iqr1, _stats, iqr3 = stats.scoreatpercentile(_returns, per=(25, 50, 75), axis=0)
            __x, _iqr1 = smooth_plot2(_x, iqr1, interval=smooth)
            __x, _iqr3 = smooth_plot2(_x, iqr3, interval=smooth)
            axis.fill_between(__x, _iqr1, _iqr3, color=c, alpha=0.2)

            __x, __y = smooth_plot2(_x, _stats, interval=smooth)
            axis.plot(__x, __y, color=c, label=label, lw=1.3)
        else:
            raise NotImplementedError
        i += 1
        print(label)
        print(_returns[:, -1])

    axis.legend()
    axis.set_title("Compare learning-trajectory ({})".format(ylabel))
    axis.set_xlabel(xlabel)
    title = ylabel
    if smooth > 1:
        axis.set_ylabel("{} (prev {} optimization average)".format(title, smooth))
    else:
        axis.set_ylabel(title)
    axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    # plt.savefig("test.pdf")
    # plt.savefig("test.pdf", format="pdf", bbox_inches='tight')
    if save_path is not None:
        plt.savefig(save_path)
    else:
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


def plot_train_and_eval(log_file, smooth=1, save_path=None):
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawconfig.mplstyle"))
    fig, axis = plt.subplots(ncols=2, figsize=(6, 3))
    train_key = "mean_return"
    eval_key = "eval_average_return"
    xlabel = "total_step"

    data = log_reader(log_file)
    eval_return = np.array(data[eval_key], dtype=np.float)
    train_return = np.array(data[train_key], dtype=np.float)
    x = np.array(data[xlabel], dtype=np.float)

    # plot train
    _x, _y = smooth_plot2(x, train_return, interval=smooth)
    axis[0].plot(_x, _y, lw=1.3)
    axis[0].set_title("return (train)")

    # plot eval
    _x, _y = smooth_plot2(x, eval_return, interval=smooth)
    axis[1].plot(_x, _y, lw=1.3)
    axis[1].set_title("return (eval)")

    for ax in axis.flatten():
        ax.set_xlabel(xlabel)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if smooth > 1:
            ax.set_ylabel("return (prev {} optimization average)".format(smooth))
        else:
            ax.set_ylabel("return")

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def my_json2csv(file):
    data = load_from_my_format(log_file=file)
    csv_file = file[:-4] + "csv"
    # if os.path.exists(csv_file):
    #     pass
    # else:
    with open(csv_file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n', delimiter=',')
        writer.writerow(list(data.keys()))
        _data = data.values()
        _data = list(zip(*_data))
        l = len(_data)
        for i in range(l):
            writer.writerow(_data[i])


def my_json2csv_all(root_dirs):
    for root_dir in root_dirs:
        seeds_logs = glob(os.path.join(root_dir, '*', "log.json"))
        [my_json2csv(file) for file in seeds_logs]


def compare_return_plotter2(root_dirs, legends, xlabel="total_step", ylabel="eval_average_return", smooth=1, plot_mode="raw", save_path=None):
    """
    配列のサイズが違うくてもいい感じにやってくれる．
    x軸の値が違くてもいい感じに補間してやってくれる
    done ファイルの読み込み（からファイルかどうかの確認）

    :return:
    """
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./drawconfig.mplstyle"))

    fig, axis = plt.subplots()
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    log_file = "log.csv"

    statistical_test = []
    total_data = []

    for root_dir, legend, c in zip(root_dirs, legends, cycle):
        # load data
        seeds_logs = glob(os.path.join(root_dir, '*', log_file))
        data = [csv_reader(file) for file in seeds_logs]
        # remove broken file
        data = [d for d in data if len(d) > 0]
        # extract data
        ys = [d[ylabel] for d in data]  # the size may different among ys[0], ys[1], ...
        xs = [d[xlabel] for d in data]

        # max_len = max(map(len, xs))
        # ys = [y for y in ys if len(y) == max_len]
        # xs = [x for x in xs if len(x) == max_len]

        # plot raw
        if plot_mode == "raw":
            i = 0
            interp_flag = False
            y_news = []
            for x, y in zip(xs, ys):
                _x, _y = smooth_plot2(x, y, interval=smooth)
                if i == 0:
                    axis.plot(_x, _y, color=c, alpha=0.5, lw=1.3, label=legend)
                else:
                    axis.plot(_x, _y, color=c, alpha=0.5, lw=1.3)
                i += 1
                y_news.append(_y)

        elif plot_mode == "iqr" or plot_mode == 'std' or plot_mode == 'min_max':
            # interquartile
            try:
                tests = np.array(xs)
                if (tests == tests[0]).all():
                    print("No interpolation required")
                    x_new = tests[0]
                    y_news = np.array(ys)
                    interp_flag = False
                else:
                    print("Interpolation required")
                    x_new, y_news = interpolate(xs, ys)
                    interp_flag = True
            except:
                print("Interpolation required")
                x_new, y_news = interpolate(xs, ys)
                interp_flag = True

            # smooth
            x, y_news = smooth_plot_2dim(x_new, y_news, interval=smooth)
            if plot_mode == 'iqr':
                iqr1, median, iqr3 = stats.scoreatpercentile(y_news, per=(25, 50, 75), axis=0)
                axis.fill_between(x, iqr1, iqr3, color=c, alpha=0.2)
                axis.plot(x, median, color=c, label=legend, lw=1.3)

            elif plot_mode == 'std':
                _mean = np.mean(y_news, axis=0)
                _std = np.std(y_news, ddof=1, axis=0)
                axis.fill_between(x, _mean - _std, _mean + _std, color=c, alpha=0.2)

                x, median = smooth_plot2(x_new, _mean, interval=smooth)
                axis.plot(x, median, color=c, label=legend, lw=1.3)

            elif plot_mode == 'min_max':
                _mean = np.mean(y_news, axis=0)
                _min = np.min(y_news, axis=0)
                _max = np.max(y_news, axis=0)
                axis.fill_between(x, _min, _max, color=c, alpha=0.2)
                axis.plot(x, _mean, color=c, label=legend, lw=1.3)

        else:
            raise NotImplementedError

        # statistical test
        total_data.append(y_news)
        if len(root_dirs) == 2:
            # statistical_test.append([y[int(len(y)/2)] for y in y_news])
            statistical_test.append([y[-1] for y in y_news])

    # write title
    # axis.set_xlim(xmax=2e6)
    # axis.set_ylim(ymax=10)
    axis.legend()
    axis.set_title("Compare learning-trajectory ({})".format(ylabel))
    axis.set_xlabel(xlabel)
    ylabel = ylabel if not interp_flag else "{} (interpolated)".format(ylabel)
    if smooth > 1:
        axis.set_ylabel("{} (prev {} evaluation average)".format(ylabel, smooth))
    else:
        axis.set_ylabel(ylabel)
    axis.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    # plt.savefig("test.pdf")
    # plt.savefig("test.pdf", format="pdf", bbox_inches='tight')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    # statistical test
    if len(root_dirs) == 2:
        print("rank sum")
        print(stats.ranksums(*statistical_test))
        print([len(t) for t in statistical_test])

    save_txt_for_R_statistical_test(x, total_data, legends)

def interpolate(xs, ys):
    """
    :param xs: 横柚の値のリスト．サイズは違ってもいい．
    :param ys: 縦軸の値のリスト．サイズは違ってもいい．
    :return:  x_new. 新しい横軸．y_new：新しい縦軸の値．サイズは同じになる
    """
    # find confidential region
    x_min_max = max([x[0] for x in xs])
    x_max_min = min([x[-1] for x in xs])
    max_size = max([len(x) for x in xs])

    # interpolation
    ys_interp = [interp1d(x, y, kind='linear') for x, y in zip(xs, ys)]
    x_new = np.arange(x_min_max, x_max_min, (x_max_min - x_min_max) / max_size)
    y_news = np.array([y_interp(x_new) for y_interp in ys_interp])

    return x_new, y_news


def save_txt_for_R_statistical_test(x, data, column_titles):
    """
    data [methods, N(seeds), steps]
    :param data:
    :return:
    """
    data = np.array(data)
    # max_len = int(len(data[0][0]))
    # extract_steps = [int(max_len / 2), max_len]
    max_idx = len(data[0][0]) - 1
    half_idx = np.where(x == x[max_idx] / 2)[0][0]
    extract_steps = [half_idx, max_idx]
    # extracts = [data[:, :, s] for s in extract_steps]  # shape=(len(extract_steps), methods, N(seeds)
    _dict = {"step{}".format(int(x[s])): {l: list(data[i, :, s-1]) for i, l in enumerate(legends)} for s in extract_steps}
    with open('/tmp/dqn_test2.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Step', 'Method', 'y'])
        for s in extract_steps:  # factor A
            for i, l in enumerate(legends):  # factor B
                for y in data[i, :, s]:
                    writer.writerow(['step{}'.format(int(x[s])), l, y])
    # with open('/tmp/test.json', 'w') as f:
    #     json.dump(_dict, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--root-dirs', type=str, default=None, help="data root directories name separated by a `^`")
    parser.add_argument('--legends', type=str, default=None, help="legend names separated by a `^`")
    parser.add_argument('--ylabel', type=str, default=None, help="ylabel corresponds to column title of csv", nargs='*')
    parser.add_argument('--xlabel', type=str, default=None, help="xlabel corresponds to column title of csv")
    parser.add_argument('--smooth', type=int, default=1, help="smoothing interval")
    parser.add_argument('--plot_mode', type=str, choices=["raw", "iqr", "std", "min_max"], default="raw", help="plot all lines or iqr or std")
    parser.add_argument('--save_path', type=str, default=None, help="save file name (.pdf, .jpg, etc...)")

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    root_dirs = args["root_dirs"].split('^')
    legends = args["legends"].split('^')
    args["ylabel"] = ''.join([w + ' ' for w in args["ylabel"]])[:-1]

    # compare_reward_plotter(root_dirs, legends, args['xlabel'], args['ylabel'], args["smooth"], args["plot_mode"], args['save_path'])
    compare_return_plotter2(root_dirs, legends, args['xlabel'], args['ylabel'], args["smooth"], args["plot_mode"],
                            args['save_path'])
    # my_json2csv_all(root_dirs)
