import numpy as np
from glob import glob
import os
import re

def merge_array_files(path):
    array_files = glob(os.path.join(path, '*.npz'))
    if len(array_files) > 1:
        p = re.compile(r'.*epoch(\d+)_(\d+)\.npz')
        epochs = [list(map(int, p.match(f).groups())) for f in array_files]
        start = epochs[0][0]
        end = epochs[-1][1]

        a = np.load(array_files[0])