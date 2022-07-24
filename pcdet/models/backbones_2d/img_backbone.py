import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock,conv1x1,conv3x3
from torchvision._internally_replaced_utils import load_state_dict_from_url


class ResNet(nn.Module):
    def __init__(
        self,
        block: BasicBlock,
        layers: [2, 2, 2, 2],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
        norm_layer = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_conv1 = self.layer1(x)
        x_conv2 = self.layer2(x_conv1)
        x_conv3 = self.layer3(x_conv2)
        x_conv4 = self.layer4(x_conv3)


        return x_conv1,x_conv2,x_conv3,x_conv4


class ImgBackbone(nn.Module):
    def __init__(self, model_cfg, stride = 1):
        super().__init__()
        self.model_cfg = model_cfg

        self.ImgResnet = ResNet(BasicBlock,[2, 2, 2, 2])
        model_dict = self.ImgResnet.state_dict()
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        self.ImgResnet.load_state_dict(pretrained_dict)

        outplanes_4 = self.model_cfg.IMG_CHANNELS[4]
        self.num_img_features = outplanes_4


    def forward(self, batch_dict):


        x_conv1_bs = []
        x_conv2_bs = []
        x_conv3_bs = []
        x_conv4_bs = []

        for bs in range(batch_dict['batch_size']):
            x = batch_dict['images'][bs,:,:,:].unsqueeze(0)

            x_conv1,x_conv2,x_conv3,x_conv4 = self.ImgResnet(x)
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



