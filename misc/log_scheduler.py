import psutil
import os
import csv
import numpy as np
import random


class LogScheduler(object):
    def __init__(self):
        self.log_dir = None
        self.exist_ok = None

        self._csv_file = None
        self._csv_header = []
        self._csv_data = {}
        self.add_num_csv = [0, 0]

        self._array_file = None
        self._array_keys = []
        self._array_data = {}
        self.add_num_array = [0, 0]  # add number from previous write to current add. Use this to make save-file name

        self._compress_flag = True
        self.mem_limit = psutil.virtual_memory().total * 0.8

        self.save_array_flag = True

    # logger config
    def set_log_dir(self, log_dir, exist_ok=False):
        self.log_dir = log_dir
        self.exist_ok = exist_ok
        os.makedirs(log_dir, exist_ok=exist_ok)

    # save array flag
    def set_save_array_flag(self, flag):
        self.save_array_flag = flag

    def set_memory_limit_by_ratio(self, mem_limit_ratio):
        if mem_limit_ratio > 1 or mem_limit_ratio < 0:
            raise AssertionError("mem_limit_ratio should be [0.0, 1.0]")
        self.mem_limit = psutil.virtual_memory().total * mem_limit_ratio

    def set_compress(self, compress_flag):
        self._compress_flag = compress_flag

    @staticmethod
    def _add(pool, keys, counter, data):
        if type(data) == dict:
            for k, v in data.items():
                if k not in keys:
                    keys.append(k)
                    pool[k] = []
                pool[k].append(v)
        elif hasattr(data, "len"):
            for k, v in zip(keys, data):
                pool[k].append(v)
        else:
            raise AssertionError("type(array) should be dict or array like object. Actual* type(array)={}".format(type(data)))
        counter[1] += 1

    # csv logger config
    def make_csv(self, filename, overwrite_ok=False):
        """
        :param filename:*.csv
        :return:
        """
        self._csv_file = filename
        if not overwrite_ok:
            if os.path.exists(filename):
                raise AssertionError("File exists. (overwrite_ok={})".format(overwrite_ok))
        # reset the file
        # f = open(self._csv_file, 'w')
        # f.close()

    def add_csv_headers(self, headers):
        """
        :param headers: list of str
        :return:
        """
        self._csv_header = headers
        for k in self._csv_header:
            self._csv_data[k] = []
        if self._csv_file is None:
            raise AssertionError("You have to call make_csv before add_csv_headers")
        else:
            with open(self._csv_file, 'a') as f:
                writer = csv.writer(f, lineterminator='\n', delimiter=',')
                headers = self._csv_data.keys()
                writer.writerow(headers)

    def add_csv_data(self, data):
        """
        :param dict or list data:
        :return:
        """
        raise NotImplementedError("currently this method has bag (save the oldest value of list rather than the latest one)")
        self._add(self._csv_data, self._csv_header, self.add_num_csv, data)

    # array logger config
    def set_array_file_name(self, file):
        """
        :param str file: log name is file+(self.add_num[0]_self.add_num[1])
        :return:
        """
        array_dir = os.path.join(self.log_dir, "array")
        os.makedirs(array_dir, exist_ok=self.exist_ok)
        self._array_file = os.path.join(array_dir, file)

    def set_array_keys(self, keys):
        """
        :param list of str keys: names of each array
        :return:
        """
        self._array_keys = keys
        for k in self._array_keys:
            self._array_data[k] = []

    def add_array_data(self, data):
        """
        :param dict or list data:
        :return:
        """
        if self.save_array_flag:
            self._add(self._array_data, self._array_keys, self.add_num_array, data)

    def write(self, force=False):
        memory_used = psutil.virtual_memory().used

        if self.add_num_csv[1] > self.add_num_csv[0]:
            if self._csv_file is None:
                self._csv_file = os.path.join(self.log_dir, "log.csv")
            if self._csv_header == []:
                self._csv_header = self._csv_data.keys()
            if not os.path.exists(self._csv_file):
                with open(self._csv_file, 'w') as f:
                    writer = csv.writer(f, lineterminator='\n', delimiter=',')
                    writer.writerow(self._csv_data.keys())

            with open(self._csv_file, 'a') as f:
                writer = csv.writer(f, lineterminator='\n', delimiter=',')
                writer.writerows(list(zip(*self._csv_data.values())))
                self._csv_data = {k: [] for k in self._csv_header}
                if force:
                    f.flush()
                else:
                    if random.random() < 0.01:
                        f.flush()

        if (memory_used > self.mem_limit or force) and self.save_array_flag:
            if self.add_num_array[1] > self.add_num_array[0]:
                if self._array_file is None:
                    os.makedirs(os.path.join(self.log_dir, "array"), exist_ok=True)
                    self._array_file = os.path.join(self.log_dir, "array", "epoch")
                if self._array_keys == []:
                    self._array_keys = self._array_data.keys()

                filename = self._array_file + "{}_{}.npz".format(self.add_num_array[0], self.add_num_array[1])
                if self._compress_flag:
                    np.savez_compressed(filename, **self._array_data)
                else:
                    np.savez(filename, **self._array_data)
                for k in self._array_keys:
                    self._array_data[k] = []
                self.add_num_array[0] = self.add_num_array[1]

    def force_write(self):
        """
        You must call this at the end of your program
        """
        self.write(force=True)

    def __del__(self):
        try:
            self.force_write()
        except:
            pass

_logger = LogScheduler()


def get_logger():
    return _logger
