# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.registry import MODELS

from thop import profile
from thop import clever_format
from torchinfo import summary

import time


@MODELS.register_module()
class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to
            [dict(type='Kaiming', layer='ConvTranspose2d'),
             dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)].
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=[
                     dict(type='Kaiming', layer='ConvTranspose2d'),
                     dict(
                         type='Constant',
                         layer='NaiveSyncBatchNorm2d',
                         val=1.0)
                 ]):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (List[torch.Tensor]): Multi-level features with 4D Tensor in
                (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]


# input_map1 = torch.rand(1, 64, 160, 160).cuda(2)
# input_map2 = torch.rand(1, 128, 80, 80).cuda(2)
# input_map3 = torch.rand(1, 256, 40, 40).cuda(2)

# model = SECONDFPN(in_channels=[64, 128, 256], out_channels=[128, 128, 128]).cuda(2)
# output_map = model([input_map1, input_map2, input_map3])


# macs, params = profile(model, inputs=[[input_map1, input_map2, input_map3]])
# macs *= 2
# macs, params = clever_format([macs, params], "%.3f")
# print("FLOPs:", macs)
# print("Params:", params)

# t1 = time.time()
# test_times = 100
# for i in range(test_times):
#     output = model([input_map1, input_map2, input_map3])
# t2 = time.time()
# print("fps:", (1 / ((t2 - t1) / test_times)))

# Trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f'Trainable params: {Trainable_params / 1e6}M')