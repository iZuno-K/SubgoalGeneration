import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from glob import glob
from misc.plotter.return_plotter import plot_train_and_eval
from experiments.gym_experiment import main, eval_render
import yaml
from sac.envs import GymEnv
import tensorboardX as tbx


def draw_histogram(data, save_path=None):
    """
    draw histogramas of
    knack kurtosis value
    sign variance
    sign variance of knack
    :param data:
    :param save_path:
    :return:
    """
    keys = list(data.keys())  # ['knack_kurtosis', 'knack_or_not', 'sign_variance_knack', 'variance', 'diff_to_meam']

    # draw figure
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../misc/plotter/drawconfig.mplstyle"))
    fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(6, 3))
    axis = axis.flatten()
    bins = 50

    # draw knack_kurtosis
    k = "knack_kurtosis"
    _data = data[k]
    if len(_data.shape) == 2:  # (num_epoch, num_steps_in_an_epoch)
        _data = np.concatenate(_data)
    axis[0].hist(_data, bins=bins)
    axis[0].set_title(k)

    # draw sign_variance
    k = "sign_variance_knack"
    _data = data[k]
    if len(_data.shape) == 2:  # (num_epoch, num_steps_in_an_epoch)
        _data = np.concatenate(_data)
    axis[1].hist(_data, bins=bins)
    axis[1].set_title(k)

    # draw sign_variance_knack on knack
    kurtosis_on_knack = data["knack_kurtosis"][data['knack_or_not']]
    sign_variance_on_knack = data["sign_variance_knack"][data['knack_or_not']]
    axis[3].hist(kurtosis_on_knack, bins=bins)
    axis[3].set_title("kurtosis_on_knack")

    axis[4].hist(sign_variance_on_knack, bins=bins)
    axis[4].set_title("sign_variance_on_knack")

    # diff to mean
    diff_to_mean = data["diff_to_meam"]  # shape:(data_num, action_sample_num)
    axis[2].hist(diff_to_mean.mean(axis=1))
    axis[2].set_title("diff_to_mean_mean")

    greater_than_mean = (diff_to_mean >= 0).sum(axis=1)
    smaller_than_mean = (diff_to_mean < 0).sum(axis=1)
    axis[5].hist((greater_than_mean - smaller_than_mean))
    axis[5].set_title("diff_to_mean_count")

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def draw_histogram2(data, save_path=None):
    """
    epoch0_2001.npzが対象

    全部のコツ度の分布を見る
    配列を読み込む (num_epoch, num_steps_in_an_epoch)
    マージして1次元にする
    histogramを描く

    コツとみなした箇所のコツ度の分布を見る
    配列を読み込む (num_epoch, num_steps_in_an_epoch)
    for epoch
        過去のepochからmin maxを算出
        それで現在のepochのコツの値を正則化．
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
    keys = list(data.keys())  # ['knack_kurtosis', 'knack_or_not', 'sign_variance_knack', 'variance', 'diff_to_meam']

    # draw figure
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../misc/plotter/drawconfig.mplstyle"))
    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(6, 3))
    # fig, axis = plt.subplots(nrows=2, ncols=2)
    axis = axis.flatten()
    bins = 50
    # for time series histogram
    writer = tbx.SummaryWriter(os.path.join(os.path.dirname(save_path), "time_series_hist"))


    # draw knack_kurtosis
    k = "knack_kurtosis"
    _data = data[k]
    _data = np.concatenate(_data)  # (num_epoch, num_steps_in_an_epoch)
    axis[0].hist(_data, bins=bins)
    axis[0].set_title(k)

    # draw sign_variance
    k = "signed_variance"
    _data = data[k]
    _data = np.concatenate(_data)  # (num_epoch, num_steps_in_an_epoch)
    axis[1].hist(_data, bins=bins)
    axis[1].set_title(k)

    # draw sign_variance_knack on knack
    metric = "knack_kurtosis"  # min maxを計算してコツかどうか判定を行う
    knacks = data[metric]
    signed_variances = data["signed_variance"]
    _min = 0.
    _max = 1.
    kurtosises_on_knack = []
    signed_variances_on_knack = []
    knack_thresh = 0.8
    i = 0
    for knack, signed_variance in zip(knacks, signed_variances):
        normalized_knack = (knack - _min) / (_max - _min)
        kurtosis_on_knack = knack[normalized_knack > knack_thresh]  # extract values larger than threshold
        signed_variance_on_knack = signed_variance[normalized_knack > knack_thresh]  # extract values larger than threshold
        kurtosises_on_knack.append(kurtosis_on_knack)
        signed_variances_on_knack.append(signed_variance_on_knack)

        writer.add_histogram("kurtosis", knack, i)
        writer.add_histogram("signed_variance", signed_variance, i)
        if len(kurtosis_on_knack) > 0:
            writer.add_histogram("kurtosis_on_knack", kurtosis_on_knack, i)
        if len(signed_variance_on_knack) > 0:
            writer.add_histogram("signed_variance_on_knack", signed_variance_on_knack, i)

        _min = np.min(knack)
        _max = np.max(knack)
        i += 1

    axis[2].hist(np.concatenate(kurtosises_on_knack), bins=bins)
    axis[2].set_title("kurtosis_on_knack")

    axis[3].hist(np.concatenate(signed_variances_on_knack), bins=bins)
    axis[3].set_title("sign_variance_on_knack")


    for ax in axis.flatten():
        ax.set_yscale("log")
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=None)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    # load data
    print(args["root_path"])
    data_path = os.path.join(args["root_path"], "array/epoch0_2001.npz")
    data = np.load(data_path)

    # plot histogramqq
    print("drawing histogram...")
    save_path = os.path.join(args["root_path"], "histogram.pdf")
    draw_histogram2(data, save_path)

    # plot return
    print("plotting histogram...")
    log_file = os.path.join(args["root_path"], "log.csv")
    save_path = os.path.join(args["root_path"], "return_plot.pdf")
    plot_train_and_eval(log_file=log_file,smooth=50, save_path=save_path)

    # draw movie
    print("drawing movie")
    yaml_file = os.path.join(args["root_path"], "hyparam.yaml")
    with open(yaml_file, "r") as f:
        hypara = yaml.load(f)
    hypara["eval_model"] = os.path.join(args["root_path"], "model")
    hypara["trial"] = None
    env_id = args["root_path"].split('/')[-2]
    print(env_id)
    hypara["env"] = GymEnv(env_id)
    algorithm = main(**hypara)
    save_path = os.path.join(args["root_path"], "movie.mp4")
    eval_render(algorithm, eval_model=hypara["eval_model"], save_path=save_path)
