import sys
import os
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from threading import Thread
import time
# sys.path.append(os.path.expanduser('~/pycharm-debug.egg'))

"""
debug_histogram
debug_threading_for_save
"""

def debug_histogram(debug=False):
    """
    :param debug:bool, debug or not
    """
    if not debug:
        return
    else:
        print('debug file is {}'.format(__file__))
        plt.style.use('mystyle3')
        xedges = np.arange(6) - 0.5
        yedges = np.arange(6) - 0.5
        test_array = [[1, 3]] * 3
        test_array.extend([[1, 4]]*5)
        x, y = zip(*test_array)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))

        fig, ax = plt.subplots()
        ax.imshow(H.T)
        ax.scatter(x=x, y=y, c='y', s=10)
        plt.show()

def debug_threading_for_save(debug=False):
    if not debug:
        return
    else:
        def _main():
            for i in range(10):
                data = np.arange(i+1)
                now = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
                print('main: {}'.format(now))
                if i % 3 == 1:
                    thread = Thread(target=save_thread, args=(data, 1))
                    thread.start()

        def save_thread(data, a):
            time.sleep(1)
            print('save_thread: shape is {}'.format(data.shape))

        class Test():
            def __init__(self):
                self.ndarr = np.zeros(5)
                self.scalar = 0
            def update(self):
                self.ndarr += 10
                self.scalar += 10
            def printing(self):
                def threaded_printing(a, b):
                    time.sleep(1)
                    print('threaded:{}, {}'.format(a, b))
                # ndarray should be copied
                thread_test = Thread(target=threaded_printing, kwargs={'a': self.scalar, 'b': self.ndarr.copy()})
                thread_test.start()

        def _main2():
            t = Test()
            for i in range(5):
                t.printing()
                t.update()

        print('debug file is {}'.format(__file__))
        # _main()
        _main2()


if __name__ == '__main__':
    # debug_histogram(debug=True)
    debug_threading_for_save(debug=True)