from pcdet.datasets.nuscenes.nuscenes_dataset import create_nuscenes_info
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
import yaml
import argparse
from pathlib import Path
from easydict import EasyDict
from pcdet.utils import common_utils

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--cfg_file', type=str, default='cfgs/dataset_configs/nuscenes_dataset_LI.yaml',
                    help='specify the config of dataset')
parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
args = parser.parse_args()

if args.func == 'create_nuscenes_infos':
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    dataset_cfg.VERSION = args.version
    create_nuscenes_info(
        version=dataset_cfg.VERSION,
        data_path=ROOT_DIR / 'data' / 'nuscenes',
        save_path=ROOT_DIR / 'data' / 'nuscenes',
        max_sweeps=dataset_cfg.MAX_SWEEPS,
    )

    nuscenes_dataset = NuScenesDataset(
        dataset_cfg=dataset_cfg, class_names=None,
        root_path=ROOT_DIR / 'data' / 'nuscenes',
        logger=common_utils.create_logger(), training=True
    )
    nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)