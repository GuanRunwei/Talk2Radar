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
from mmdet3d.models.modal_fusion.mhca import FeatureResizer
from mmdet3d.models.modal_fusion.LPCF import LPCF
from mmdet3d.models.modal_fusion.position_encodings import LearnedPositionalEncoding3D, PositionEmbeddingSine1D


class LocalGlobalFusion(nn.Module):
    def __init__(self, key_dim, d_model, nhead, dropout=0.001):
        super().__init__()
        self.key_dim = key_dim
        self.d_model = d_model
        self.key_linear = FeatureResizer(input_feat_size=key_dim, output_feat_size=d_model, dropout=dropout)
        self.value_linear = FeatureResizer(input_feat_size=key_dim, output_feat_size=d_model, dropout=dropout)

        self.global_fusion = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.local_fusion = LPCF(feature_dim=d_model, text_dim=key_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, image_feat, query_pos, text_feat, text_feat_pool, text_mask):
        local_fusion_feat = self.local_fusion(image_feat, text_feat_pool)

        B, C, H, W = image_feat.shape
        memory_new = self.key_linear(self.with_pos_embed(text_feat, None)).permute(1, 0, 2)
        value = self.value_linear(text_feat)

        global_fusion_feat = self.global_fusion(query=self.with_pos_embed(image_feat, query_pos).flatten(2).permute(2, 0, 1),
                                   key=memory_new,
                                   value=value.permute(1, 0, 2), attn_mask=None,
                                   key_padding_mask=text_mask)[0]

        global_fusion_feat = global_fusion_feat.reshape(B, C, H, W)
        final_fusion_feat = global_fusion_feat + local_fusion_feat + image_feat
        return final_fusion_feat


@MODELS.register_module()
class ThreeStageLGF(nn.Module):
    def __init__(self, text_dim, feature_dim_list, nhead_list=[4, 4, 8], dropout=0.001):
        super().__init__()
        self.key_dim = text_dim
        self.d_model_list = feature_dim_list

        self.lgf_s3 = LocalGlobalFusion(key_dim=self.key_dim, d_model=self.d_model_list[0], nhead=nhead_list[0], dropout=dropout)
        self.lgf_s4 = LocalGlobalFusion(key_dim=self.key_dim, d_model=self.d_model_list[1], nhead=nhead_list[1], dropout=dropout)
        self.lgf_s5 = LocalGlobalFusion(key_dim=self.key_dim, d_model=self.d_model_list[2], nhead=nhead_list[2], dropout=dropout)

        self.q_pos_s3 = LearnedPositionalEncoding3D(num_feats=int(self.d_model_list[0] // 2))
        self.q_pos_s4 = LearnedPositionalEncoding3D(num_feats=int(self.d_model_list[1] // 2))
        self.q_pos_s5 = LearnedPositionalEncoding3D(num_feats=int(self.d_model_list[2] // 2))

    def forward(self, pts_feats, text_feat, text_feat_pool, text_feat_mask):
        pts_fusion_s3 = self.lgf_s3(pts_feats[0], self.q_pos_s3(pts_feats[0]), text_feat, text_feat_pool, text_feat_mask) 
        pts_fusion_s4 = self.lgf_s4(pts_feats[1], self.q_pos_s4(pts_feats[1]), text_feat, text_feat_pool, text_feat_mask) 
        pts_fusion_s5 = self.lgf_s5(pts_feats[2], self.q_pos_s5(pts_feats[2]), text_feat, text_feat_pool, text_feat_mask)
        
        return (pts_fusion_s3, pts_fusion_s4, pts_fusion_s5)


# s3 = torch.randn(1, 64, 160, 160).cpu()
# s4 = torch.randn(1, 128, 80, 80).cpu()
# s5 = torch.randn(1, 256, 40, 40).cpu()
# k = torch.randn(1, 30, 768).cpu()
# k_pool = torch.randn(1, 768).cpu()
# k_mask = torch.BoolTensor(1, 30).cpu()

# model = ThreeStageLGF(text_dim=768, feature_dim_list=[64, 128, 256]).cpu()
# output = model((s3, s4, s5), k, k_pool, k_mask)
# print(len(output))