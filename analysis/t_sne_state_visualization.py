import numpy as np
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import multiprocessing as mp
import os

import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import patches, animation


LIGHT_INTERVAL = 10
EPOCH_STEPS = 1000


def load_saved_file(light, dim):
    if light:
        if dim == 2:
            save_npz = '/home/isi/karino/t_sne_light.npz'
        elif dim == 3:
            save_npz = '/home/isi/karino/t_sne_light_3d.npz'
        else:
            raise NotImplementedError
    else:
        if dim == 2:
            save_npz = '/home/isi/karino/t_sne.npz'
        elif dim == 3:
            save_npz = '/home/isi/karino/t_sne_3d.npz'
        else:
            raise NotImplementedError
    return save_npz

def t_sne(light=False, dim=2):
    """
    1. データの読み込み
    - Defaultのもの
    - kurtosis入れたもの
    2. t-sneを持ちいた圧縮表現の計算
    - データを後で復元できる形に結合
    - t-sneの計算
    - データの復元
    とりあえず描画

    次，データを時系列順に描画
    :return:
    """
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../misc/plotter/drawconfig.mplstyle"))

    # データ読み込み
    default_path = "/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/GMMPolicy/Walker2d-v2/seed3/array/epoch0_2000.npz"
    kurtois_path = "/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/Knack-exploration/Walker2d-v2/seed3/array/epoch0_2000.npz"
    classes = [default_path, kurtois_path]
    labels = ['Default', 'kurtosis']
    color = ['red', 'blue']

    # save file configuration
    save_npz = load_saved_file(light, dim)

    # load and reshpae data
    print("loading and reshaping ...")
    if os.path.exists(save_npz):
        data = np.load(save_npz)['arr_0']
    else:
        if light:
            data = [np.load(_file)['states'][::LIGHT_INTERVAL] for _file in classes]  # shape (len(clases), epoch, epoch_steps, obs_shape)
        else:
            data = [np.load(_file)['states'] for _file in classes]  # shape (len(clases), epoch, epoch_steps, obs_shape)
        shapes = [d.shape for d in data]  # all shapes may be the same
        data = [np.concatenate(d, axis=0) for d in data]  # flatten the data (len(clases), epoch * epoch_steps, obs_shape)
        # shape reconstruction test
        # a = np.arange(2*3*4).reshape(2,3,4)
        # np.concatenate(a, axis=0).reshape(a.shape) - a
        data = np.concatenate(data, axis=0)  # (len(clases) * epoch * epoch_steps, obs_shape)
        print(data.shape)

        # embedding by t-sne
        print("calc t-sne ...")
        start = time.time()
        dim_embedding = dim
        # if light:
        #     tsne = TSNE(n_components=dim_embedding, n_jobs=mp.cpu_count(), verbose=True, n_iter=500)
        # else:
        tsne = TSNE(n_components=dim_embedding, n_jobs=mp.cpu_count(), verbose=True, n_iter=1000)
        data = tsne.fit_transform(data)
        # data = TSNE(n_components=dim_embedding, random_state=0).fit_transform(data)  # (len(clases) * epoch * epoch_steps, obs_shape)
        data = data.reshape(len(classes), -1, dim_embedding)  # shape (len(classes), epoch * epoch_steps, dim_embedding)
        print(data.shape)
        print("calc time is: {}".format(time.time() - start))
        # print("calc time per data is: {}".format((time.time() - start) / len()))
        np.savez_compressed(save_npz, data)

    print("plotting ...")
    if dim == 2:
        for i in range(len(classes)):
            plt.scatter(data[i, :, 0], data[i, :, 1], c=color[i], label=labels[i], s=0.01)
    elif dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(len(classes)):
            ax.plot(data[i, :, 0], data[i, :, 1], data[i, :, 2], marker="o", linestyle='None', c=color[i], label=labels[i], markersize=0.01)
    plt.legend()
    # plt.savefig("/home/karino/Desktop/tsne.jpg")
    # plt.savefig("/home/isi/karino/tsne.jpg")
    if light:
        plt.savefig("/home/isi/karino/tsne_light.jpg")
        plt.show()
    else:
        plt.show()
        plt.savefig("/home/isi/karino/tsne.jpg")


def time_series_t_sne(light=True, dim=2):
    """
    1時刻分のデータを取り出す
    表示する
    動画にする
    :return:
    """
    # save file
    # save_npz = load_saved_file(light, dim)
    save_npz = "/home/karino/t_sne_light.npz"
    data = np.load(save_npz)['arr_0']  # (classes, num_epoch * epoch_steps, dim)
    data = data.reshape(len(data), -1, EPOCH_STEPS, dim)  # (classes, num_epoch, epoch_steps, dim)
    # data = data[:, :10]
    print(data.shape)

    # file_name = '/home/isi/karino/test.mp4'
    file_name = '/home/karino/test.mp4'
    # make_anim(data, file_name)

    ani = AnimationMaker(data)
    ani.save()
    # ani.save(file_name)

def make_anim(data, filename):
    fig = plt.figure()
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
                    bottom=False, left=False, right=False, top=False)
    plt.tight_layout()
    frames = []
    labels = ['Default', 'kurtosis']
    color = ['red', 'blue']

    print("make frames")
    for i in range(data.shape[1]):
        scat = []
        for j, d in enumerate(data):
            _ = plt.scatter(d[i, :, 0], d[i, :, 1], c=color[j], label=labels[j], s=0.01)
            scat.append(_)
            plt.legend()
        frames.append(scat)

    print("make animation")
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
    # ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    print("save animation")
    ani.save(filename, writer="ffmpeg")
    plt.close()


class AnimationMaker(object):
    def __init__(self, data):
        self.data = data  # (classes, num_epoch, epoch_steps, dim)
        self.labels = ['Default', 'kurtosis']
        self.color = ['red', 'blue']
        # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        self.fig, self.ax = plt.subplots()

        self.ani = animation.FuncAnimation(self.fig, self.update, interval=50, frames=data.shape[1] * data.shape[2],
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        self.scat = []
        t = 0
        for d, c, l, in zip(self.data, self.color, self.labels): # (classes, shape)
            _ = self.ax.scatter(d[t, :, 0], d[t, :, 1], c=c, s=2, cmap="red", label=l)
            self.scat.append(_)
        # self.scat = self.ax.scatter(x, y, c=c, s=0.1, cmap="jet", edgecolor="k")
        return self.scat

    def update(self, t):
        epoch = int(t / EPOCH_STEPS)
        for scat, d in zip(self.scat, self.data):
            # set positions
            # scat.set_offsets(np.concatenate(d[:t+1], axis=0))
            scat.set_offsets(d[epoch, :t % EPOCH_STEPS + 1])

            # set colors
            # scat.set_array(data[keys[j]])

        return self.scat

    def save(self, save_path=None):
        """
        draw animation by calling updatefig func
        :param str save_path: path to save created animation
        :return:
        """
        if save_path is not None:
            self.ani.save(save_path, writer='ffmpeg')
        else:
            plt.show()

if __name__ == '__main__':
    # t_sne(light=True, dim=2)
    time_series_t_sne(light=True, dim=2)

