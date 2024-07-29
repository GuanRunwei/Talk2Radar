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
from mmdet3d.models.modal_fusion.position_encodings import LearnedPositionalEncoding3D, PositionEmbeddingSine1D
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, key_dim, d_model, nhead, dropout=0.001):
        super().__init__()
        self.key_dim = key_dim
        self.d_model = d_model
        self.key_linear = FeatureResizer(input_feat_size=key_dim, output_feat_size=d_model, dropout=dropout)
        self.value_linear = FeatureResizer(input_feat_size=key_dim, output_feat_size=d_model, dropout=dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.key_pos = None


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None
                ):
        B, C, H, W = tgt.shape
        
        memory_new = self.key_linear(self.with_pos_embed(memory, pos)).permute(1, 0, 2)
        value = self.value_linear(memory)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos).flatten(2).permute(2, 0, 1),
                                   key=memory_new,
                                   value=value.permute(1, 0, 2), attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt2 = tgt2.reshape(B, C, H, W)
        # tgt = tgt + tgt2
        return tgt2


@MODELS.register_module()
class ThreeStageMHCA(nn.Module):
    def __init__(self, text_dim, feature_dim_list, nhead_list=[4, 4, 8], dropout=0.001):
        super().__init__()
        self.key_dim = text_dim
        self.d_model_list = feature_dim_list

        self.mhca_s3 = MultiHeadCrossAttention(key_dim=self.key_dim, d_model=self.d_model_list[0], nhead=nhead_list[0], dropout=dropout)
        self.mhca_s4 = MultiHeadCrossAttention(key_dim=self.key_dim, d_model=self.d_model_list[1], nhead=nhead_list[1], dropout=dropout)
        self.mhca_s5 = MultiHeadCrossAttention(key_dim=self.key_dim, d_model=self.d_model_list[2], nhead=nhead_list[2], dropout=dropout)

        self.q_pos_s3 = LearnedPositionalEncoding3D(num_feats=int(self.d_model_list[0] // 2))
        self.q_pos_s4 = LearnedPositionalEncoding3D(num_feats=int(self.d_model_list[1] // 2))
        self.q_pos_s5 = LearnedPositionalEncoding3D(num_feats=int(self.d_model_list[2] // 2))

    def forward(self, pts_feats, text_feat, text_feat_mask):
        pts_fusion_s3 = self.mhca_s3(pts_feats[0], text_feat, text_feat_mask, query_pos=self.q_pos_s3(pts_feats[0])) 
        pts_fusion_s4 = self.mhca_s4(pts_feats[1], text_feat, text_feat_mask, query_pos=self.q_pos_s4(pts_feats[1])) 
        pts_fusion_s5 = self.mhca_s5(pts_feats[2], text_feat, text_feat_mask, query_pos=self.q_pos_s5(pts_feats[2]))
        
        return (pts_fusion_s3, pts_fusion_s4, pts_fusion_s5)


# s3 = torch.randn(1, 64, 160, 160).cpu()
# s4 = torch.randn(1, 128, 80, 80).cpu()
# s5 = torch.randn(1, 256, 40, 40).cpu()
# k = torch.randn(1, 30, 768).cpu()
# k_mask = torch.BoolTensor(1, 30).cpu()

# model = ThreeStageMHCA(text_dim=768, feature_dim_list=[64, 128, 256]).cpu()
# output = model((s3, s4, s5), k, k_mask)
# print(len(output))
    