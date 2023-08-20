import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
from .Layers import *




class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.key_act = LIFSpike()
        self.queries = nn.Linear(emb_size, emb_size)
        self.query_act = LIFSpike()
        self.values = nn.Linear(emb_size, emb_size)
        self.value_act = LIFSpike()
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.fc_act = LIFSpike()

    def forward(self, queries: Tensor, keys:Tensor, values: Tensor) -> Tensor:
        _, T, _, _ = queries.shape
        queries = rearrange(self.queries(queries.flatten(0, 1)), "(b t) n (h d) -> b t h n d",t = T, h=self.num_heads)
        queries = self.query_act(queries)
        queries = rearrange(queries, 'b t h n d -> (b t) h n d')
        keys = rearrange(self.keys(keys.flatten(0, 1)), "(b t) n (h d) -> b t h n d",t = T, h=self.num_heads)
        keys = self.key_act(keys)
        keys = rearrange(keys, 'b t h n d -> (b t) h n d')
        values = rearrange(self.values(values.flatten(0, 1)), "(b t) n (h d) -> b t h n d",t = T, h=self.num_heads)
        values = self.value_act(values)
        values = rearrange(values, 'b t h n d -> (b t) h n d')

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        # if mask is not None:
        #     fill_value = torch.finfo(torch.float32).min
        #     energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        
        out = rearrange(out, "(b t) w c -> b t w c", t = T)
        out = self.fc_act(out)
        return out