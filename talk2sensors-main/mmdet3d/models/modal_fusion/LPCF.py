import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
import torch.distributed as dist
from torch import Tensor
from mmdet3d.registry import MODELS


class LPCF(nn.Module):
    def __init__(self, feature_dim, text_dim=768):
        super().__init__()
        self.fc    = nn.Linear(text_dim, feature_dim)
        self.beta  = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, feature_dim, 1, 1)), requires_grad=True)

    def forward(self, x, text_embedding):
        gating_factors = torch.sigmoid(self.fc(text_embedding))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        f = x * self.gamma + self.beta  # 1) learned feature scaling/modulation
        f = f * gating_factors          # 2) (soft) feature routing based on text
        return f + x


@MODELS.register_module()
class FourStageLPCF(nn.Module):
    def __init__(self, text_dim, feature_dim_list, dropout=0.001):
        super().__init__()
        self.text_dim = text_dim
        self.feature_dim_list = feature_dim_list

        self.mhca_s2 = LPCF(feature_dim=feature_dim_list[0], text_dim=text_dim)
        # self.mhca_s3 = LPCF(feature_dim=feature_dim_list[1], text_dim=text_dim)
        # self.mhca_s4 = LPCF(feature_dim=feature_dim_list[2], text_dim=text_dim)
        # self.mhca_s5 = LPCF(feature_dim=feature_dim_list[3], text_dim=text_dim)

    def forward(self, pts_feats, text_feat):
        pts_fusion = self.mhca_s2(pts_feats[0], text_feat)
        
        return pts_fusion


@MODELS.register_module()
class ThreeStageLPCF(nn.Module):
    def __init__(self, text_dim, feature_dim_list, dropout=0.001):
        super().__init__()
        self.text_dim = text_dim
        self.feature_dim_list = feature_dim_list

        self.mhca_s3 = LPCF(feature_dim=feature_dim_list[0], text_dim=text_dim)
        self.mhca_s4 = LPCF(feature_dim=feature_dim_list[1], text_dim=text_dim)
        self.mhca_s5 = LPCF(feature_dim=feature_dim_list[2], text_dim=text_dim)

    def forward(self, pts_feats, text_feat):
        pts_fusion = (self.mhca_s3(pts_feats[0], text_feat), self.mhca_s4(pts_feats[1], text_feat), self.mhca_s5(pts_feats[2], text_feat))
        
        return pts_fusion


@MODELS.register_module()
class TwoStageLPCF(nn.Module):
    def __init__(self, text_dim, feature_dim_list, dropout=0.001):
        super().__init__()
        self.text_dim = text_dim
        self.feature_dim_list = feature_dim_list

        self.mhca_s4 = LPCF(feature_dim=feature_dim_list[0], text_dim=text_dim)
        self.mhca_s5 = LPCF(feature_dim=feature_dim_list[1], text_dim=text_dim)

    def forward(self, pts_feats, text_feat):
        pts_fusion = (self.mhca_s4(pts_feats[0], text_feat), self.mhca_s5(pts_feats[1], text_feat))
        
        return pts_fusion


# s3 = torch.randn(1, 64, 160, 160).cuda()
# s4 = torch.randn(1, 128, 80, 80).cuda()
# s5 = torch.randn(1, 256, 40, 40).cuda()
# k = torch.randn(1, 768).cuda()
# k_mask = torch.BoolTensor(1, 30).cuda()

# model = ThreeStageLPCF(text_dim=768, feature_dim_list=[64, 128, 256]).cuda()
# output = model((s3, s4, s5), k)
# print(output[0].shape)