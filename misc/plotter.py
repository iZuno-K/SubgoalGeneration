import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import json

def maze_plot(map, v_table, variances):
    """
    map and values must be the same size
    :param map:
    :param values:
    :return:
    """
    # sphinx_gallery_thumbnail_number = 2

    fig, (ax1, ax2) = plt.subplots(1, 2)

    im = ax1.imshow(v_table, cmap='Reds')
    # Loop over data dimensions and create text annotations.
    for i in range(v_table.shape[0]):
        for j in range(v_table.shape[1]):
            text = ax1.text(j, i, map[i][j],
                           ha="center", va="center", color="black")

    im = ax2.imshow(variances, cmap='Reds')
    # Loop over data dimensions and create text annotations.
    for i in range(variances.shape[0]):
        for j in range(variances.shape[1]):
            text = ax2.text(j, i, map[i][j],
                           ha="center", va="center", color="black")

    ax1.set_title("V(s)")
    ax2.set_title('state-importance')

    fig.tight_layout()
    plt.show()

def continuous_maze_plot(root_dir):
    save_path = os.path.join(root_dir, 'graphs')
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(root_dir, 'log.json')
    map_files = glob(os.path.join(root_dir, 'maps'))
    plot_log(log_file, save_path=save_path)

def log_reader(log_file):
    """decode my log format"""
    data = dict(
        total_step=[],
        mean_return=[],
        q_loss=[],
        v_loss=[],
        policy_loss=[]
    )
    with open(log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        l = line.replace('\n', '')
        dic = json.loads(l)
        for key in data.keys():
            data[key].append(dic[key])

    return data

def plot_log(log_file, save_path=None):
    data = log_reader(log_file)
    total_steps = data.pop('total_step')
    ylabels = {'mean_return': 'mean return', 'q_loss': 'loss', 'v_loss': 'loss', 'policy_loss': 'loss', }
    plt.style.use('mystyle2')
    fig, axes = plt.subplots(2, 2, sharex='col')
    for i, key in enumerate(data.keys()):
        axes[int(i/2), i % 2].set_title(key)
        axes[int(i/2), i % 2].set_ylabel(ylabels[key])
        if int(i/2) == 1:
            axes[int(i/2), i % 2].set_xlabel('total steps')
        axes[int(i/2), i % 2].plot(total_steps, data[key])

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'log.pdf'))

    plt.show()


