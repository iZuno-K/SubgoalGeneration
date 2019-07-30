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
    """
    # get file names
    extension = "*" + ".npz"
    depth = "*/" * file_place_depth
    file_reg = os.path.join(data_root_dir, depth) + extension
    file_names = glob(file_reg)

    # get target file_names
    target_file_names = []
    parser = re.compile(r'.*_epoch(\d+)\.npz')
    for file_name in file_names:
        if int(parser.match(file_name).group(1)) == target_epoch:
            target_file_names.append(file_name)

    # load data
    data = [np.load(tar_f_name) for tar_f_name in target_file_names]

    # plot
    plt.style.use("mystyle2")
    smooth_interval = 1

    fig, axis = plt.subplots(2)
    # plot variance and kurtosis
    labels = ["knack", "knack_kurtosis"]
    # plot title
    cp = CP()
    for env_name in cp.ENVS:
        if env_name in target_file_names[0]:
            title = env_name if smooth_interval == 1 else env_name + " smooth_plot({})".format(smooth_interval)
            fig.suptitle(title)
    for ax, l in zip(axis, labels):
        ax.set_title(l)
        ax.set_xlabel("steps")
        ax.set_ylabel(l)
    # plot data
    for d in data:
        for ax, l in zip(axis, labels):
            y = d[l]
            x = np.arange(len(y))

            x, y = smooth_plot(x, y, smooth_interval)
            ax.plot(x, y)

    # save
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir, "importance_change_in_episode" + save_mode)
    plt.savefig(save_name)
    plt.show()


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
    plot_importance_episode(**args)

