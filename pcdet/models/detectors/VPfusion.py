from .detector3d_template import DetectorFusionTemplate
from .. import backbones_2d

class VPfusion(DetectorFusionTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'IMG_backbone_2d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head', 'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()

    def build_IMG_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('IMG_BACKBONE_2D', None) is None:
            return None, model_info_dict

        IMG_backbone_2d_module = backbones_2d.__all__[self.model_cfg.IMG_BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.IMG_BACKBONE_2D
        )
        model_info_dict['module_list'].append(IMG_backbone_2d_module)
        model_info_dict['num_img_features'] = IMG_backbone_2d_module.num_img_features
        return IMG_backbone_2d_module, model_info_dict

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
