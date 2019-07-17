import os
import os.path as osp
from glob import glob
from shutil import copy2
import tarfile
import argparse

# receive root_dir
# retrieve log file in the same depth dir
# copy file to save the same hierarchy
# compress


def retrieve_logfile(root_dir, depth, save_dir):
    star = '*/' * depth
    log_file_templates = osp.join(root_dir, star, 'log.json')
    print(log_file_templates)
    log_file_names = glob(log_file_templates)
    print(log_file_names[:10])

    print("retrieve files...")
    for log_file_name in log_file_names:
        dir_name = osp.dirname(log_file_name)
        _save_dir = osp.join(save_dir, osp.relpath(dir_name, root_dir))  # save the same hierarchy
        os.makedirs(_save_dir, exist_ok=True)
        _save_file_name = osp.join(_save_dir, log_file_name.split('/')[-1])
        copy2(log_file_name, _save_dir)

    print("compressing...")
    # remove the last /
    if save_dir[-1] == '/':
        save_dir = save_dir[:-2]
    print(save_dir)
    tar_name = save_dir + ".tar.gz"
    print(tar_name)
    archive = tarfile.open(tar_name, mode='w:gz')
    archive.add(save_dir)
    archive.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    retrieve_logfile(**args)

#