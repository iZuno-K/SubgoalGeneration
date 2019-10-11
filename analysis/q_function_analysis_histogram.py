import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from glob import glob

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
    axis[0].hist(data[k], bins=bins)
    axis[0].set_title(k)

    # draw sign_variance
    k = "sign_variance_knack"
    axis[1].hist(data[k], bins=bins)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    # load data
    data = np.load(args["data_path"])

    draw_histogram(data)

