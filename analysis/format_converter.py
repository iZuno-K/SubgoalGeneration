import os
from glob import glob
import numpy as np
import re
import argparse


def array_converter(root_dir):
    """
    保存形式を途中で変えたので，それを合わせるために使う．
    experienced/_epoch0.npz _epoch1.npz ...  _epoch2001.npz --> array/epoch0_2001.npz

    保存先の設定
    保存先にすでにファイルがあったら上書きしないようにreturn

    読み込み先の設定
    全データの読み込み
    整形
    保存
    """

    save_path = os.path.join(root_dir, "array")
    load_path = os.path.join(root_dir, "experienced")
    load_files = glob(os.path.join(load_path, "*.npz"))

    # set save_file
    p = re.compile(r".*_epoch(\d+).npz")
    epoch_nums = [int(p.match(name).group(1)) for name in load_files]
    min_epoch = min(epoch_nums)
    max_epoch = max(epoch_nums)
    n = "epoch{}_{}.npz".format(min_epoch, max_epoch + 1)  # +1 due to my mistake at logger
    save_file = os.path.join(save_path, n)

    # check exist
    if os.path.exists(save_file):
        print("return since the file exists: {}".format(save_file))
        return
    os.makedirs(save_path, exist_ok=False)

    # load all data
    load_files = sorted(load_files, key=lambda x: int(p.match(x).group(1)))
    print("loading ...")
    dicts = [np.load(f) for f in load_files]
    print("loading done")
    keys = list(dicts[0].keys())
    shapes = {k: dicts[0][k].shape for k in keys}
    n = len(dicts)
    save_data = {k: np.zeros((n,) + shapes[k]) for k in keys}

    print("data reshaping ...")
    for i, d in enumerate(dicts):
        for k in keys:
            save_data[k][i] = d[k]
    print("data reshaping done")

    # add description
    p = re.compile(r".*_(epoch\d+).npz")
    descriptions = [p.match(name).group(1) for name in load_files]
    save_data["descriptions"] = descriptions

    # save
    print("saving with compression ...")
    np.savez_compressed(save_file, **save_data)
    print("saving with compression done")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=None)
    return vars(parser.parse_args())


if __name__ == '__main__':
    # args = parse_args()
    # array_converter(args["root_dir"])

    for seed in range(1, 11):
        a = "/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95/HalfCheetah-v2/seed{}".format(seed)
        print(a)
        array_converter(a)


