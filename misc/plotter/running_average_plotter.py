import argparse
from glob import glob
import os
import matplotlib.pyplot as plt
from matplotlib import patches, animation
import numpy as np

from environments.continuous_space_maze import ContinuousSpaceMaze
from misc.plotter.experienced_states_plotter import normalize, map_reshaper, TotalExperienceAnimationMaker

"""
基本的には経験マップと同じ、移動平均をとるだけえ
"""


class RunningAveragePlotter(TotalExperienceAnimationMaker):
    def __init__(self, root_dir, average_times=20, exclude_fault=0):
        super(RunningAveragePlotter, self).__init__(root_dir)
        self.average_times = average_times  # calc average of average_times of data
        self.maps_data = np.zeros([3, self.average_times, self.resolution, self.resolution])  # values for heat-map
        self.save_name = 'running_average_in' + str(self.average_times) + '.mp4'
        self.exclude_fault = bool(exclude_fault)

    def updateifig(self, i):
        """
        calculate running average through time
        :param i:
        :return:
        """
        # load knack value
        map_data = self.load_map_data(self.map_paths[self.counter * self.frame_skip])  # v_map, knack_map, knack_map_kurtosis  (3, 50, 50)
        self.maps_data = np.concatenate((self.maps_data[:, 1:, :, :], map_data[:, np.newaxis, :, :]), axis=1)  # concatenate the last n-1 data and a new data for averaging
        map_data = np.mean(self.maps_data, axis=1)  # calc average

        if self.exclude_fault:
            mask = self.load_positive_states(self.map_paths[self.counter * self.frame_skip]) > 0
        else:
            # experienced states data is saved twice more than map
            experienced_states = []
            for i in range(min(0, self.counter - 1) * self.frame_skip * 2, self.counter * self.frame_skip * 2):
                experienced_states.extend(np.load(self.experienced_states_kancks_paths[i])['states'])  # (steps, states_dim)
            experienced_states = np.array(experienced_states, dtype=np.int32).T  # (states_dim, steps)
            visit_count_hist, xedges, yedges = np.histogram2d(x=experienced_states[0], y=experienced_states[1], bins=self.resolution,
                                                              range=[sorted(self.range[0]), sorted(self.range[1])])
            self.states_visit_counts += visit_count_hist

            # normalize among only experienced states
            mask = self.states_visit_counts > 0

        mask = mask.T  # mask[x][y] -> mask[y][x] for map_data[y][x]
        for i in range(len(map_data)):
            _min = np.min(map_data[i][mask])
            _max = np.max(map_data[i][mask])
            map_data[i] = (map_data[i] - _min) / _max
            map_data[i] = map_data[i] * mask
            map_data[i] = map_data[i]

        # update heat-map
        self.im[0, 0].set_array(mask)
        for im, arr in zip(self.im.flatten()[1:], map_data):  #
            im.set_array(arr)

        # make return list to matplot
        parts = self.im.flatten().tolist()
        parts.extend(self.circles)

        self.counter += 1

        return parts

    def load_map_data(self, path):
        """
        load data to visualize from path
        :param str path: path to data file
        :return numpy.ndarray: shape=(3, 50, 50)=[v_map, knack_map, knack_map_kurtosis]
        """
        data = np.load(path)
        v_map = data['q_1_moment'].reshape(25, 25)
        knack_map = data['knack_map'].reshape(25, 25)
        knack_map_kurtosis = data['knack_map_kurtosis'].reshape(25, 25)

        # normalize array into (0., 1.) to visualize
        v_map = normalize(v_map)
        knack_map = normalize(knack_map)
        knack_map_kurtosis = normalize(knack_map_kurtosis)

        v_map = map_reshaper(v_map)
        knack_map = map_reshaper(knack_map)
        knack_map_kurtosis = map_reshaper(knack_map_kurtosis)

        return  np.array([v_map, knack_map, knack_map_kurtosis])

    def load_positive_states(self, path):
        """
        load histogram of succeeded trajectory
        :param str path: path to .npz
        :return numpy.ndarray:
        """
        data = np.load(path)
        if self.exclude_fault:
            if "visit_count" in data.keys():
                positive_states = data["visit_count"]  # (50, 50)
                return positive_states
            else:
                raise AssertionError("{} does not have key `visit_count`".format(path))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--average-times', type=int, default=20)
    parser.add_argument('--exclude-fault', type=int, default=0)
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    animator = RunningAveragePlotter(root_dir=args['root_dir'], average_times=args['average_times'])
    save_path = os.path.join(args['root_dir'], 'graphs')
    os.makedirs(save_path, exist_ok=True)
    animator.animate(save_path=save_path)
