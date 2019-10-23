import numpy as np
from glob import glob
import os
import re

def merge_array_files(path):
    array_files = glob(os.path.join(path, '*.npz'))
    if len(array_files) > 1:
        p = re.compile(r'.*epoch(\d+)_(\d+)\.npz')
        epochs = [list(map(int, p.match(f).groups())) for f in array_files]
        min_epoch = epochs[0][0]
        max_epoch = epochs[-1][1]
        print('min: {}, max: {}'.format(min_epoch, max_epoch))
        save_file = os.path.join(path, 'epoch{}_{}.npz'.format(min_epoch, max_epoch))

        array_files = sorted(array_files, key=lambda x: int(p.match(x).group(1)))

        # load all data
        print("loading ...")
        _dict = np.load(array_files[0])
        print("loading done")
        keys = list(_dict.keys())
        # keys = []
        # for k in _keys:
        #     if 'qf' not in k:
        #         keys.append(k)
        # print(keys)
        shapes = {k: _dict[k].shape for k in keys}
        n = max_epoch - min_epoch
        save_data = {}
        for k in keys:
            s = list(shapes[k])
            s[0] = n
            save_data.update({k: np.zeros(s)})
        # save_data = {k: np.zeros((n,) + shapes[k]) for k in keys}
        print("data reshaping ...")
        print(array_files)
        for f in array_files:
            print(f)
            _dict = np.load(f)
            start_epoch, end_epoch = list(map(int, p.match(f).groups()))
            print(start_epoch, end_epoch)
            for k in keys:
                print(k)
                save_data[k][start_epoch:end_epoch] = _dict[k]
        print("data reshaping done")

        # save
        print("saving with compression ...")
        np.savez_compressed(save_file, **save_data)
        print("saving with compression done")

        for f in array_files:
            os.remove(f)

if __name__ == '__main__':
    file = "/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/small_variance/Walker2d-v2/seed2/array"
    merge_array_files(file)