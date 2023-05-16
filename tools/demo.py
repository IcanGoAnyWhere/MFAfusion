import argparse
import glob
import time
from pathlib import Path


import open3d
from visual_utils import open3d_vis_utils as V
OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import numpy as np
import torch
import os
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.kitti.kitti_dataset_demo import KittiDataset_demo
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from img2avi import img2avi


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pv_rcnn.yaml',
    #                     help='specify the config for demo')
    # parser.add_argument('--data_path', type=str, default='../data/kitti',
    #                     help='specify the point cloud data file or directory')

    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/pv_rcnn_nuscenes.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/nuscenes',
                        help='specify the point cloud data file or directory')

    # '../output/kitti_models/VPfusionRCNN_kitti/default/ckpt/softmax_55_4096.pth'
    parser.add_argument('--ckpt', type=str,
                        default='../output/kitti_models/pv_rcnn/default/ckpt/checkpoint_epoch_16.pth',
                        help='specify the pretrained model')

    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    if cfg['DATA_CONFIG'].DATASET == 'KittiDataset':
        demo_dataset = KittiDataset_demo(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), logger=logger
        )
    else:
        demo_dataset = NuScenesDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), logger=logger
        )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)

    checkpoint = torch.load(args.ckpt)

    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    file_dir = args.data_path +'/capture/'
    if os.path.exists(file_dir) is False:
        os.mkdir(file_dir)
    with torch.no_grad():
        vis = open3d.visualization.Visualizer()
        vis.create_window(window_name = 'results', width = 800, height=600,  visible=True)
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                 points=data_dict['points'][:, 1:],vis = vis,root_path=args.data_path,
                ref_boxes=None,
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            # time.sleep(1)

            vis.capture_screen_image(file_dir+"%s"%idx +".jpg",do_render=True)
            vis.clear_geometries()

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
    vis.destroy_window()
    img2avi(file_dir)
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
