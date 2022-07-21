import numpy as np
import torch
import torch.nn as nn


BatchNorm2d = nn.BatchNorm2d
def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)

class ImgBackbone(nn.Module):
    def __init__(self, model_cfg, stride = 1):
        super().__init__()
        self.model_cfg = model_cfg

        inplanes_1 = self.model_cfg.IMG_CHANNELS[0]
        outplanes_1 = self.model_cfg.IMG_CHANNELS[1]
        self.conv1 = nn.Sequential(
            conv3x3(inplanes_1, outplanes_1, stride),
            BatchNorm2d(outplanes_1),
            nn.ReLU(inplace=True),
            conv3x3(outplanes_1, outplanes_1, 2 * stride)
        )

        inplanes_2 = self.model_cfg.IMG_CHANNELS[1]
        outplanes_2 = self.model_cfg.IMG_CHANNELS[2]
        self.conv2 = nn.Sequential(
            conv3x3(inplanes_2, outplanes_2, stride),
            BatchNorm2d(outplanes_2),
            nn.ReLU(inplace=True),
            conv3x3(outplanes_2, outplanes_2, 2 * stride)
        )

        inplanes_3 = self.model_cfg.IMG_CHANNELS[2]
        outplanes_3 = self.model_cfg.IMG_CHANNELS[3]
        self.conv3 = nn.Sequential(
            conv3x3(inplanes_3, outplanes_3, stride),
            BatchNorm2d(outplanes_3),
            nn.ReLU(inplace=True),
            conv3x3(outplanes_3, outplanes_3, 2 * stride)
        )

        inplanes_4 = self.model_cfg.IMG_CHANNELS[3]
        outplanes_4 = self.model_cfg.IMG_CHANNELS[4]
        self.conv4 = nn.Sequential(
            conv3x3(inplanes_4, outplanes_4, stride),
            BatchNorm2d(outplanes_4),
            nn.ReLU(inplace=True),
            conv3x3(outplanes_4, outplanes_4, 2 * stride)
        )

        self.num_img_features = outplanes_4


    def forward(self, batch_dict):
        x_conv1_bs = []
        x_conv2_bs = []
        x_conv3_bs = []
        x_conv4_bs = []

        for bs in range(batch_dict['batch_size']):
            x = batch_dict['images'][bs,:,:,:].unsqueeze(0)
            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            x_conv1_bs.append(x_conv1)
            x_conv2_bs.append(x_conv2)
            x_conv3_bs.append(x_conv3)
            x_conv4_bs.append(x_conv4)

        x_conv1_cat = torch.cat(x_conv1_bs, dim=0)
        x_conv2_cat = torch.cat(x_conv2_bs, dim=0)
        x_conv3_cat = torch.cat(x_conv3_bs, dim=0)
        x_conv4_cat = torch.cat(x_conv4_bs, dim=0)

        batch_dict['multi_img_features'] =  {
                'x_conv1': x_conv1_cat,
                'x_conv2': x_conv2_cat,
                'x_conv3': x_conv3_cat,
                'x_conv4': x_conv4_cat,
            }

        return batch_dict



