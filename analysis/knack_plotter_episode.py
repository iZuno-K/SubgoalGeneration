import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import re
from analysis.common_params import CommonParams as CP
import argparse
from misc.plotter.return_plotter import smooth_plot

"""
1episode内で要所度がどう変わるかを、横軸step数でプロットする。
Plot the change of state-importance in an episode.

問題は何か？
1. 学習のどの段階のデータを用いてプロットするか
2. 複数プロットすると見えづらくなってしまう。
→ 学習最終盤epochのデータを使ってプロットを行う。７本ならプロットしても見える。
"""


def plot_importance_episode(data_root_dir, file_place_depth, target_epoch, save_dir, save_mode=".pdf", **kwargs):
    """
    :param data_root_dir:
    :param file_place_depth:
    :param save_dir:
    :param kwargs:後に追加された余分な条件分岐パラメータを引数にとっても動くようにするため。この関数では使用しない。
    :return:
    ファイル名取得
    データロード
    プロット
    コツの場所はそうわかるようにプロットする
    """
    # get file names
    extension = "*" + ".npz"
    depth = "*/" * file_place_depth
    file_reg = os.path.join(data_root_dir, depth) + extension
    file_names = glob(file_reg)

    # get target file_names
    target_file_names = []
    pre_target_file_names = []
    parser = re.compile(r'.*_epoch(\d+)\.npz')
    for file_name in file_names:
        if int(parser.match(file_name).group(1)) == target_epoch:
            target_file_names.append(file_name)
        if int(parser.match(file_name).group(1)) == target_epoch - 1:
            pre_target_file_names.append(file_name)
    # sort by seed
    parser2 = re.compile(r'.*seed(\d+).*')
    target_file_names = sorted(target_file_names, key=lambda x: int(parser2.match(x).group(1)))
    pre_target_file_names = sorted(pre_target_file_names, key=lambda x: int(parser2.match(x).group(1)))
    seeds = [int(parser2.match(x).group(1)) for x in target_file_names]

    # load data
    data = [np.load(tar_f_name) for tar_f_name in target_file_names]
    pre_data = [np.load(tar_f_name) for tar_f_name in pre_target_file_names]  # コツ計算のため

    # plot
    plt.style.use("mystyle2")
    cmap = plt.get_cmap("tab10")
    smooth_interval = 1

    # plot data
    i = 0
    for pre_d, d in zip(pre_data, data):
        # set canvas
        fig, axis = plt.subplots(2)
        # plot variance and kurtosis
        labels = ["knack", "knack_kurtosis"]
        # plot title
        cp = CP()
        for env_name in cp.ENVS:
            if env_name in target_file_names[0]:
                title = env_name if smooth_interval == 1 else env_name + " smooth_plot({})".format(smooth_interval)
                fig.suptitle(title + "seed{}".format(seeds[i]))
        for ax, l in zip(axis, labels):
            ax.set_title(l)
            ax.set_xlabel("steps")
            ax.set_ylabel(l)

        # main plot
        for ax, l in zip(axis, labels):
            y = d[l]
            pre_y = pre_d[l]
            x = np.arange(len(y))

            x, y = smooth_plot(x, y, smooth_interval)
            ax.plot(x, y, lw=1, color=cmap(i))

            # calc knack
            _min = pre_y.min()
            _max = pre_y.max()
            knack_thresh = 0.8
            normed_y = (y - _min) / (_max - _min)
            thresh_line = knack_thresh * (_max - _min) + _min
            for _x, _normed_y, _y in zip(x, normed_y, y):
                if _normed_y > knack_thresh:
                    ax.plot(_x, _y, marker='o', markersize=1, color='k')
            ax.plot(x, np.tile(thresh_line, len(x)), lw=1, color='k')
        # save
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, "seed{}".format(seeds[i]) + save_mode)
        plt.savefig(save_name)

        i += 1

    # save
    with open(os.path.join(save_dir, "data_path.txt"), 'w') as f:
        [f.writelines(tar_f_name+'\n') for tar_f_name in target_file_names]
        [f.writelines(tar_f_name + '\n') for tar_f_name in pre_target_file_names]
    print(save_dir)
    # plt.show()


def knack_percent_to_all_steps(data_root_dir, save_dir, save_mode=".pdf", **kwargs):
    """
    1episode中の何％がコツとみなされたのかプロットする。

    :param data_root_dir:
    :param file_place_depth:
    :param target_epoch:
    :param save_dir:
    :param save_mode:
    :param kwargs:
    :return:
    全データ読み込み
    seedごとに整理
    epoch順にソート
    コツの計算

    """
    # plot setting
    plt.style.use("mystyle2")
    os.makedirs(save_dir, exist_ok=True)
    smooth_interval = 1

    # get file names
    extension = "seed*"
    folder_reg = os.path.join(data_root_dir, extension)
    folder_names = glob(folder_reg)  # path/to/seedX

    parser = re.compile(r'.*_epoch(\d+)\.npz')
    parser2 = re.compile(r'.*seed(\d+).*')
    for folder_name in folder_names:
        seed = int(parser2.match(folder_name).group(1))
        files = glob(os.path.join(folder_name, "*/*.npz"))
        # sort by epoch
        files = sorted(files, key=lambda x: int(parser.match(x).group(1)))
        data = [np.load(f) for f in files]  # 同じシード違うエポックのデータが入る

        # コツの閾値計算、そのエポックはコツが全ステップに対して何％あったか
        labels = ["knack", "knack_kurtosis"]
        _min = {labels[0]: 0, labels[1]: 0}
        _max = {labels[0]: 1, labels[1]: 1}
        knack_thresh = 0.8
        knack_percent = {labels[0]: [], labels[1]: []}
        for d in data:

            # main plot
            for l in labels:
                y = d[l]
                x = np.arange(len(y))

                normed_y = (y - _min[l]) / (_max[l] - _min[l])
                knack_or_not = y > normed_y
                knack_percent[l].append(knack_or_not.sum() / len(x))

                _min[l] = y.min()
                _max[l] = y.max()
            # save
            with open(os.path.join(save_dir, "data_path.txt"), 'a') as f:
                [f.writelines(tar_f_name + '\n') for tar_f_name in files]

        # plot
        fig, axis = plt.subplots(2)
        cp = CP()
        for env_name in cp.ENVS:
            if env_name in files[0]:
                title = env_name if smooth_interval == 1 else env_name + " smooth_plot({})".format(smooth_interval)
                fig.suptitle(title + "seed{}".format(seed))
        epochs = [int(parser.match(x).group(1)) for x in files]
        for ax, l in zip(axis, labels):
            ax.set_title(l)
            ax.set_xlabel("epoch")
            ax.set_ylabel(l)
            ax.plot(epochs, knack_percent[l])

        save_name = os.path.join(save_dir, "knack_percent_in_epoch_seed{}".format(seed) + save_mode)
        plt.savefig(save_name)

    print(save_dir)

    # plt.show()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default=None)
    parser.add_argument('--file_place_depth', type=int, default=None)
    parser.add_argument('--target_epoch', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_mode', type=str, default=".pdf")

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    # plot_importance_episode(**args)
    knack_percent_to_all_steps(**args)
