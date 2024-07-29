import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
import torch.distributed as dist
from torch import Tensor
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS, TASK_UTILS
import os
from timm.models.layers import DropPath
from mmdet3d.models.modal_fusion.mhca import FeatureResizer
from mmdet3d.models.modal_fusion.LPCF import LPCF
from mmdet3d.models.modal_fusion.position_encodings import LearnedPositionalEncoding3D, PositionEmbeddingSine1D


class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    
    K is the number of superpatches, therefore hops equals res // K.
    """
    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
            )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
            
        x_j = x - x
        for i in range(self.K, H, self.K):
            x_c = x - torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
            x_j = torch.max(x_j, x_c)
        for i in range(self.K, W, self.K):
            x_r = x - torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)
        

class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, drop_path=0.0, K=2):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  # out_channels back to 1x}
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

       
    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x


class GatingGraphFusion(nn.Module):
    def __init__(self, feature_dim, text_dim=768):
        super().__init__()
        self.fc    = nn.Linear(text_dim, feature_dim)
        self.beta  = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.grapher = MRConv4d(in_channels=feature_dim, out_channels=feature_dim)

    def forward(self, x, text_embedding):
        x = self.grapher(x)

        gating_factors = torch.sigmoid(self.fc(text_embedding))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        # f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = x
        f = f * gating_factors          # 2) (soft) feature routing based on text
        
        f = f + x
        return f
        # gating_factors = torch.sigmoid(self.fc(text_embedding))
        # gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        # f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        # f = f * gating_factors          # 2) (soft) feature routing based on text
        # f = self.grapher(x)
        # f = f + x
        # return f


@MODELS.register_module()
class ThreeStageGGF(nn.Module):
    def __init__(self, text_dim, feature_dim_list, dropout=0.001):
        super().__init__()
        self.text_dim = text_dim
        self.feature_dim_list = feature_dim_list

        self.mhca_s3 = GatingGraphFusion(feature_dim=feature_dim_list[0], text_dim=text_dim)
        self.mhca_s4 = GatingGraphFusion(feature_dim=feature_dim_list[1], text_dim=text_dim)
        self.mhca_s5 = GatingGraphFusion(feature_dim=feature_dim_list[2], text_dim=text_dim)

    def forward(self, pts_feats, text_feat):
        pts_fusion = (self.mhca_s3(pts_feats[0], text_feat), self.mhca_s4(pts_feats[1], text_feat), self.mhca_s5(pts_feats[2], text_feat))
        
        return pts_fusion


# s3 = torch.randn(1, 64, 160, 160).cpu()
# s4 = torch.randn(1, 128, 80, 80).cpu()
# s5 = torch.randn(1, 256, 40, 40).cpu()
# k = torch.randn(1, 30, 768).cpu()
# k_pool = torch.randn(1, 768).cpu()
# k_mask = torch.BoolTensor(1, 30).cpu()

# model = ThreeStageGGF(text_dim=768, feature_dim_list=[64, 128, 256]).cpu()
# output = model((s3, s4, s5), k_pool)
# print(len(output))