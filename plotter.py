import matplotlib.pyplot as plt
import numpy as np


def maze_plot(map, values):
    """
    map and values must be the same size
    :param map:
    :param values:
    :return:
    """
    # sphinx_gallery_thumbnail_number = 2

    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap='Reds')

    # Loop over data dimensions and create text annotations.
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = ax.text(j, i, map[i][j],
                           ha="center", va="center", color="b")

    ax.set_title("title")
    fig.tight_layout()
    plt.show()

