import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, repeat
from .Layers import *






class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, expansion * emb_size)
        self.act1 = LIFSpike()
        self.dropout = nn.Dropout(drop_p)
        self.fc2 = nn.Linear(expansion * emb_size, emb_size)
        self.act2 = LIFSpike()

    def forward(self, x):
        B, T, W, C = x.shape
        x = rearrange(x, 'b t w c -> (b t) w c')
        x = self.fc1(x)
        x = rearrange(x, '(b t) w c -> b t w c', t = T)
        x = self.act1(x)
        x = rearrange(x, 'b t w c -> (b t) w c')
        x = self.fc2(self.dropout(x))
        x = self.act2(rearrange(x, '(b t) w c -> b t w c', t = T))
        # x = rearrange(x, '(b t) w c -> b t w c', t = T)
        return x
        