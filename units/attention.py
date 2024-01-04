import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_num, dropout=0):
        super().__init__()
        assert embed_dim % head_num == 0
        self.head_num = head_num
        self.head_dim = embed_dim // head_num
        self.qkv_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        # batch * head * patch_num * dim
        attn = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(q.shape[-1])
        if mask is not None:
            if mask.dtype == torch.bool:
                attn.masked_fill_(mask, -np.inf)
            else:
                attn += mask
        attn = self.dropout(torch.softmax(attn, dim=-1))
        return torch.matmul(attn, v)

    def forward(self, query, key, value, mask=None):
        batch = query.shape[0]
        query = self.qkv_layers[0](query).reshape(batch, -1, self.head_num, self.head_dim).transpose(1, 2)
        key = self.qkv_layers[1](key).reshape(batch, -1, self.head_num, self.head_dim).transpose(1, 2)
        value = self.qkv_layers[2](value).reshape(batch, -1, self.head_num, self.head_dim).transpose(1, 2)
        value = self._scaled_dot_product_attention(query, key, value, mask)
        return value.transpose(1, 2).reshape(batch, -1, self.head_num * self.head_dim)