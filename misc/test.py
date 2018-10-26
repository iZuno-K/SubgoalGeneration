import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from misc.plotter import map_reshaper
from glob import glob
from environments.continuous_space_maze import ContinuousSpaceMaze

def animation_test():

    def f(x, y):
        return np.sin(x) + np.cos(y)

    class Anim(object):
        def __init__(self):
            self.fig, self.axes = plt.subplots(2, 2)
            self.x = np.linspace(0, 2 * np.pi, 120)
            self.y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

            self.im = self.axes[0, 0].imshow(f(self.x, self.y), animated=True)
            # self.im2 = self.axes[0, 1].imshow(f(self.x, self.y), animated=True)
            c = patches.Circle(xy=[20, 20], radius=25, fc='k', ec='k')
            c2 = patches.Circle(xy=[50, 20], radius=25, fc='k', ec='k')
            self.c = [self.axes[0, 0].add_patch(c), self.axes[0, 0].add_patch(c2)]
            # self.c.extend([self.axes[0, 1].add_patch(c), self.axes[0, 1].add_patch(c2)])

            self.map_files = glob('/home/karino/mount/NormalizedContinuousSpaceMaze-0718-01h-26m-1531844792s/maps/*.npz')
            self.env = ContinuousSpaceMaze()
            d = self.load_map_data(map_file=self.map_files[-1])
            # !!!!!!!!! vmin and vmax are very important. it is not changeable
            self.im2 = self.axes[0, 1].imshow(d['v_map'], animated=True, vmin=0., vmax=1.)

        def updateifig(self, i, *args):
            # global x, y
            print(i)
            self.x += np.pi / 15.
            self.y += np.pi / 20.
            x = self.x
            y = self.y
            self.im.set_array(f(x, y))
            # self.im2.set_array(f(self.x, self.y))
            # c = patches.Circle(xy=[50, 50], radius=50, fc='k', ec='k')
            # c = self.axes[0, 0].add_patch(c)

            data = np.load(self.map_files[i])
            v_map = data['v_map'].reshape(25, 25)
            v_map =v_map - np.min(v_map)
            v_map = v_map / np.max(v_map)
            knack_map = data['knack_map'].reshape(25, 25)
            knack_map_kurtosis = data['knack_map_kurtosis'].reshape(25, 25)

            v_map = map_reshaper(v_map)
            knack_map = map_reshaper(knack_map)
            knack_map_kurtosis = map_reshaper(knack_map_kurtosis)
            if True:
                mask = np.ones([50, 50])
                for i in range(50):
                    for j in range(50):
                        if (np.linalg.norm(np.asarray([j, i]) - self.env.h1.c) <= self.env.h1.r) or (
                                np.linalg.norm(np.asarray([i, j]) - self.env.h2.c) <= self.env.h2.r):
                            mask[j, i] = 0.

                v_map[mask == 0] = np.min(v_map)
                knack_map[mask == 0] = np.min(knack_map)
                knack_map_kurtosis[mask == 0] = np.min(knack_map_kurtosis)
            # data = self.load_map_data(self.map_paths[-i], is_mask=True)
            self.v_map = v_map
            self.im2.set_array(self.v_map)
            print(np.min(v_map), np.max(v_map))

            # self.im2.set_data(v_map)

            parts = [self.im, self.im2]
            parts.extend(self.c)
            return parts

        def animate(self):
            ani = animation.FuncAnimation(self.fig, self.updateifig, interval=50, blit=True)
            plt.show()

        def load_map_data(self, map_file, is_mask=True):
            data = np.load(map_file)
            v_map = data['v_map'].reshape(25, 25)
            knack_map = data['knack_map'].reshape(25, 25)
            knack_map_kurtosis = data['knack_map_kurtosis'].reshape(25, 25)

            v_map = map_reshaper(v_map)
            knack_map = map_reshaper(knack_map)
            knack_map_kurtosis = map_reshaper(knack_map_kurtosis)
            if is_mask:
                mask = np.ones([50, 50])
                for i in range(50):
                    for j in range(50):
                        if (np.linalg.norm(np.asarray([j, i]) - self.env.h1.c) <= self.env.h1.r) or (
                                np.linalg.norm(np.asarray([i, j]) - self.env.h2.c) <= self.env.h2.r):
                            mask[j, i] = 0.

                v_map[mask == 0] = np.min(v_map)
                knack_map[mask == 0] = np.min(knack_map)
                knack_map_kurtosis[mask == 0] = np.min(knack_map_kurtosis)

            map_data = {'v_map': v_map, 'knack_map': knack_map, 'knack_map_kurtosis': knack_map_kurtosis}
            return map_data

    a = Anim()
    a.animate()


def plot_test():
    """
    Test the order of matplot imshow
    Test the coordinate

    order:
    arr[0][0], arr[0][1], ...
    arr[1][0], arr[1][0], ...
    ...

    !!! The coordinate have -0.5 offset !!!
    The coordinate origin is upper left
    if we use relatively adjust coordinate, - [0.5, 0.5]

    """
    fig, ax = plt.subplots(1)
    im = np.arange(25).reshape(5, 5)
    print(im)

    offset = np.array([0.5, 0.5])
    center = np.array([2, 3])  # [x, y]
    c = patches.Circle(xy=center-offset, radius=2)
    ax.add_patch(c)

    # white means small, black does large value
    ax.imshow(im, cmap='binary')

    plt.show()


def test():
    a = 1
    print("Hello:{}".format(a))


if __name__ == '__main__':
    # animation_test()
    # plot_test()
    test()