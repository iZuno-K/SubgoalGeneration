import json
import numpy as np
import os

_my_data = dict(
    total_step=0,
    total_episode=0,
    mean_return=[],
    q_loss=[],
    v_loss=[],
    policy_loss=[],
)

DEFAULT = "/home/isi/karino/master/SubgoalGeneration/data3/"
# DEFAULT = "/home/karino/tmp/"
_my_log_root_dir = DEFAULT
_my_log_file_name = "log.json"
_my_data_file = None
_my_log_parent_dir = None
_my_map_log_dir = None

class MyJsonLogger(object):
    def __init__(self, file_name):
        self.file = open(file_name, 'w')

    def write(self):
        _my_data['mean_return'] = np.mean(_my_data['mean_return'])
        _my_data['q_loss'] = np.mean(_my_data['q_loss'])
        _my_data['v_loss'] = np.mean(_my_data['v_loss'])
        _my_data['policy_loss'] = np.mean(_my_data['policy_loss'])

        for k, v in sorted(_my_data.items()):
            if hasattr(v, 'dtype'):
                v = v.tolist()
                _my_data[k] = float(v)
        self.file.write(json.dumps(_my_data) + '\n')
        self.file.flush()

        data_reset()

    def close(self):
        self.file.close()

def make_log_dir(path):
    global _my_log_parent_dir
    _my_log_parent_dir = path
    os.makedirs(_my_log_parent_dir, exist_ok=True)
    global _my_map_log_dir
    _my_map_log_dir = os.path.join(_my_log_parent_dir, 'maps')
    os.makedirs(_my_map_log_dir)
    global _my_data_file
    _my_data_file = open(os.path.join(_my_log_parent_dir, _my_log_file_name), 'w')


def write():
    _my_data['mean_return'] = np.mean(_my_data['mean_return'])
    _my_data['q_loss'] = np.mean(_my_data['q_loss'])
    _my_data['v_loss'] = np.mean(_my_data['v_loss'])
    _my_data['policy_loss'] = np.mean(_my_data['policy_loss'])

    for k, v in sorted(_my_data.items()):
        if hasattr(v, 'dtype'):
            v = v.tolist()
            _my_data[k] = float(v)
    # print("log file is:{}".format(_my_log_file_name))
    _my_data_file.write(json.dumps(_my_data) + '\n')
    _my_data_file.flush()

    data_reset()

def close():
    _my_data_file.close()

def data_append(key, val):
    _my_data[key].append(val)

def data_update(key, val):
    _my_data[key] = val

def data_reset():
    _my_data['mean_return'] = []
    _my_data['q_loss'] = []
    _my_data['v_loss'] = []
    _my_data['policy_loss'] = []
