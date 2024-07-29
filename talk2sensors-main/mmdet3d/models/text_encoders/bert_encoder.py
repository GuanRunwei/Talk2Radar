import torch
import torch.nn as nn
import transformers
import math
from typing import Dict, List, Optional, Sequence
from mmdet3d.registry import MODELS


@MODELS.register_module()
class TextEncoder(nn.Module):
    def __init__(self, pretrained_path: Optional[dict] = None):
        super().__init__()

        self.tokenizer = None
        self.text_encoder = None
        try:
            self.text_encoder = transformers.AlbertModel.from_pretrained(pretrained_path)
        except:
            print("keep consistent between model name and pretrained path")

        freeze_text_encoder = True
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

    def forward_text(self, text_queries, device):
        # tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, truncation=True, return_tensors='pt').to(
        #     device)
        # tokenized_queries = self.tokenizer.batch_encode_plus(text_queries, max_length=10, padding= 'max_length', truncation = 'longest_first', return_tensors='pt').to(
        #     device)
        text_queries = {
            'input_ids': text_queries['input_ids'].to(device),
            'token_type_ids': text_queries['token_type_ids'].to(device),
            'attention_mask': text_queries['attention_mask'].to(device)
        }

        encoded_text = self.text_encoder(**text_queries)
        # Transpose memory because pytorch's attention expects sequence first
        text_features = encoded_text.last_hidden_state.clone()
        text_features_pool = encoded_text.pooler_output.clone()
        text_pad_mask = text_queries['attention_mask'].ne(1).bool()
        return text_features, text_features_pool, text_pad_mask

    def forward(self, sentence_query, src=None):
        text_features, text_features_pooler, text_pad_mask = self.forward_text(sentence_query, src[0][0].device)
        return text_features, text_features_pooler, text_pad_mask

