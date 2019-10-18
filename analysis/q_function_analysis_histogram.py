import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from glob import glob
from misc.plotter.return_plotter import plot_train_and_eval
import yaml
import tensorboardX as tbx
from algorithms.knack_based_policy import KnackBasedPolicy
from pathlib import Path

def draw_histogram(data, metric, save_path=None, tensorboard=False):
    """
    epoch0_2001.npzが対象

    全部のコツ度の分布を見る
    配列を読み込む (num_epoch, num_steps_in_an_epoch)
    マージして1次元にする
    histogramを描く

    コツとみなした箇所のコツ度の分布を見る
    配列を読み込む (num_epoch, num_steps_in_an_epoch)
    for epoch
        閾値取得
        閾値を超えるインデックスを取得．
        その部分を抽出して配列に格納
    表示


    draw histogramas of
    knack kurtosis value
    sign variance
    sign variance of knack
    :param data:
    :param save_path:
    :return:
    """
    keys = list(data.keys())  # ['knack_kurtosis', 'knack_or_not', 'signed_variance', 'variance', 'diff_to_meam']

    # draw figure
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../misc/plotter/drawconfig.mplstyle"))
    if metric is not None:
        fig, axis = plt.subplots(nrows=3, ncols=2, figsize=(6, 4.5))
        axis = axis.flatten()
    else:
        fig, axis = plt.subplots(ncols=2, figsize=(6, 2))
    # fig, axis = plt.subplots(nrows=2, ncols=2)
    bins = 50
    # for time series histogram
    if tensorboard:
        writer = tbx.SummaryWriter(os.path.join(os.path.dirname(save_path), "time_series_hist"))


    # draw knack_kurtosis
    kurtosises = data["knack_kurtosis"].squeeze()
    _data = np.concatenate(kurtosises)  # (num_epoch, num_steps_in_an_epoch)
    axis[0].hist(_data, bins=bins)
    axis[0].set_title("knack_kurtosis")

    # draw sign_variance
    signed_variances = data["signed_variance"].squeeze()
    _data = np.concatenate(signed_variances)  # (num_epoch, num_steps_in_an_epoch)
    axis[1].hist(_data, bins=bins)
    axis[1].set_title("signed_variance")

    # draw sign_variance_knack on knack
    kurtosises_on_knack = []
    signed_variances_on_knack = []
    kurtosises_not_on_knack = []
    signed_variances_not_on_knack = []


    metric = 'kurtosis' if metric == "Knack-exploration" else metric

    i = 0
    if metric is not None:
        knack_thresh = data["current_knack_thresh"]
        for kurtosis, signed_variance, thresh in zip(kurtosises, signed_variances, knack_thresh):
            knack = KnackBasedPolicy.calc_knack_value_by_metric({'kurtosis': kurtosis, 'signed_variance': signed_variance}, metric).squeeze()

            kurtosis_on_knack = kurtosis[knack > thresh]  # extract values larger than threshold
            signed_variance_on_knack = signed_variance[knack > thresh]  # extract values larger than threshold
            kurtosises_on_knack.append(kurtosis_on_knack)
            signed_variances_on_knack.append(signed_variance_on_knack)

            kurtosises_not_on_knack.append(kurtosis[knack < thresh])  # extract values larger than threshold
            signed_variances_not_on_knack.append(signed_variance[knack < thresh])  # extract values larger than threshold

            if tensorboard:
                writer.add_histogram("kurtosis", kurtosis, i)
                writer.add_histogram("signed_variance", signed_variance, i)
                if len(kurtosis_on_knack) > 0:
                    writer.add_histogram("kurtosis_on_knack", kurtosis_on_knack, i)
                if len(signed_variance_on_knack) > 0:
                    writer.add_histogram("signed_variance_on_knack", signed_variance_on_knack, i)
            i += 1

        kurtosises_on_knack = np.concatenate(kurtosises_on_knack)
        axis[2].hist(kurtosises_on_knack, bins=bins)
        axis[2].set_title("kurtosis_on_knack")
        print("exploitation ratio: {}".format(len(kurtosises_on_knack) / len(np.concatenate(kurtosises))))

        signed_variances_on_knack = np.concatenate(signed_variances_on_knack)
        axis[3].hist(signed_variances_on_knack, bins=bins)
        axis[3].set_title("signed_variance_on_knack")

        axis[4].hist(np.concatenate(kurtosises_not_on_knack), bins=bins)
        axis[4].set_title("kurtosis_not_on_knack")

        axis[5].hist(np.concatenate(signed_variances_not_on_knack), bins=bins)
        axis[5].set_title("signed_variance_not_on_knack")

        data_on_knack = {"knack_kurtosis": kurtosises_on_knack, "signed_variance": signed_variances_on_knack}
    else:
        data_on_knack = None

    for ax in axis.flatten():
        ax.set_yscale("log")
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    _save_path = os.path.join(os.path.dirname(save_path), "kurtosis_variance_correlation.png")
    # _save_path = None
    draw_kurtosis_variance_correlation(data, data_on_knack, save_path=_save_path)

    if tensorboard:
        writer.close()


def draw_kurtosis_variance_correlation(data, data_on_knack, save_path=None):
    keys = list(data.keys())  # ['knack_kurtosis', 'knack_or_not', 'signed_variance', 'variance', 'diff_to_meam']

    # draw figure
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../misc/plotter/drawconfig.mplstyle"))
    fig, axis = plt.subplots(ncols=2, figsize=(6, 3))
    # axis = axis.flatten()

    knack_kurtosis = data['knack_kurtosis']
    knack_kurtosis = np.concatenate(knack_kurtosis)  # (num_epoch, num_steps_in_an_epoch)
    sign_variance_knack = data['signed_variance']
    sign_variance_knack = np.concatenate(sign_variance_knack)

    axis[0].scatter(knack_kurtosis, sign_variance_knack, s=1)
    axis[0].set_xlabel('kurtosis')
    axis[0].set_ylabel('signed_variance')
    axis[0].set_title("all states")

    if data_on_knack is not None:
        axis[1].scatter(data_on_knack["knack_kurtosis"], data_on_knack["signed_variance"], s=1)
        axis[1].set_xlabel('kurtosis')
        axis[1].set_ylabel('signed_variance')
        axis[1].set_title("states on knack")
    else:
        axis[1].scatter(knack_kurtosis, sign_variance_knack, s=1)
        axis[1].set_xlabel('kurtosis')
        axis[1].set_ylabel('signed_variance')
        axis[1].set_ylim(-6, 6)
        axis[1].set_title("zoom")

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=None)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    # load data
    print(args["root_path"])
    data_path = os.path.join(args["root_path"], "array/epoch0_2000.npz")
    data = np.load(data_path)

    # plot histogram
    print("drawing histogram...")
    save_path = os.path.join(args["root_path"], "histogram.pdf")
    metric = Path(args["root_path"]).parts[-3]
    if metric == "GMMPolicy" or metric == "EExploitation":
        metric = None
    draw_histogram(data, metric, save_path)

    # # plot return
    print("plotting reward...")
    log_file = os.path.join(args["root_path"], "log.csv")
    save_path = os.path.join(args["root_path"], "return_plot.pdf")
    plot_train_and_eval(log_file=log_file, smooth=50, save_path=save_path)

    # # draw movie
    # from experiments.gym_experiment import main, eval_render
    # from sac.envs import GymEnv
    # print("drawing movie")
    # yaml_file = os.path.join(args["root_path"], "hyparam.yaml")
    # with open(yaml_file, "r") as f:
    #     hypara = yaml.load(f)
    # hypara["eval_model"] = os.path.join(args["root_path"], "model")
    # hypara["trial"] = None
    # env_id = args["root_path"].split('/')[-2]
    # print(env_id)
    # hypara["env"] = GymEnv(env_id)
    # algorithm = main(**hypara)
    # save_path = os.path.join(args["root_path"], "movie.mp4")
    # eval_render(algorithm, eval_model=hypara["eval_model"], save_path=save_path)
