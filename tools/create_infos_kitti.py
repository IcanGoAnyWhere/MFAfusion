from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos
import yaml
import os
from pathlib import Path
from easydict import EasyDict
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/kitti_dataset_LI.yaml',
                    help='specify the config of dataset')
parser.add_argument('--func', type=str, default='create_kitti_infos', help='')

args = parser.parse_args()


dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
ROOT_DIR = ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()


split_path = ROOT_DIR / 'data' / 'collect_xinzhen_32/ImageSets'
list = []
for root ,dirs, files in os.walk(Path.joinpath(ROOT_DIR,'data/collect_xinzhen_32/training/image_2')):
    for file in files:
        list.append(file)
sample_id_list = np.arange(len(list))
sample_id_list = ','.join(str(i) for i in sample_id_list)
sample_id_list = sample_id_list.split(',')
with open(split_path / "test.txt", "w") as f:
    for i in range(len(sample_id_list)):
        sample_id_list[i] = sample_id_list[i].zfill(6)
        f.write(sample_id_list[i] + '\n')
with open(split_path / "train.txt", "w") as f:
    for i in range(len(sample_id_list)):
        sample_id_list[i] = sample_id_list[i].zfill(6)
        f.write(sample_id_list[i] + '\n')
with open(split_path / "val.txt", "w") as f:
    for i in range(len(sample_id_list)):
        sample_id_list[i] = sample_id_list[i].zfill(6)
        f.write(sample_id_list[i] + '\n')


create_kitti_infos(
    dataset_cfg=dataset_cfg,
    class_names=['Car', 'Pedestrian', 'Cyclist'],
    data_path=ROOT_DIR / 'data' / 'collect_xinzhen_32',
    save_path=ROOT_DIR / 'data' / 'collect_xinzhen_32'
)