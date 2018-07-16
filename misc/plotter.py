import matplotlib.pyplot as plt
import numpy as np


def maze_plot(map, v_table, variances):
    """
    map and values must be the same size
    :param map:
    :param values:
    :return:
    """
    # sphinx_gallery_thumbnail_number = 2

    fig, (ax1, ax2) = plt.subplots(1,2)

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

