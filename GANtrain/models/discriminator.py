import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np
from .Base_Layers.Layers import *
from einops import rearrange, reduce, repeat
from einops.layers.torch import Reduce
from .Base_Layers.Attention import MultiHeadAttention
from .Base_Layers.FeedForward import FeedForwardBlock


class Dis_EncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attention = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.dropout1 = nn.Dropout(drop_p)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feedforward = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        self.dropout2 = nn.Dropout(drop_p)

    # discriminator only has temporal attention
    def forward(self, x):
        
        B, T, W, C = x.shape
        # temporal attention
        res1 = x
        x = self.norm1(x.flatten(0, 1))
        x = rearrange(x, '(b t) w c -> b t w c', b = B)
        x = self.attention(x, x, x)
        x = self.dropout1(x.flatten(0, 1))
        x = rearrange(x, '(b t) w c -> b t w c', b = B)
        x += res1

        # feedforward block
        res2 = x
        x = self.norm2(x.flatten(0, 1))
        x = rearrange(x, '(b t) w c -> b t w c', b = B)
        x = self.feedforward(x)
        x = self.dropout2(x.flatten(0, 1))
        x = rearrange(x, '(b t) w c -> b t w c', b = B)
        x += res2
        return x







class Dis_Encoder(nn.Sequential):
    def __init__(self, depth=1, **kwargs):
        super().__init__(*[Dis_EncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

    
class PatchEmbedding_Linear(nn.Module):
    def __init__(self, in_channels = 21,  emb_size = 100, seq_length = 1024):

        super().__init__()

        self.projection = nn.Linear(in_channels, emb_size)
        self.act = LIFSpike()

        self.positions = nn.Parameter(torch.randn(seq_length , emb_size))

    def forward(self, x: Tensor) -> Tensor:

        x = self.projection(x)

        # position
        x += self.positions
        x = self.act(x)
        return x        
        
        
class Discriminator(nn.Module):
    def __init__(self, args,
                 emb_size=50, 
                 n_classes=1, 
                 **kwargs):
        super(Discriminator, self).__init__()
        self.seq_len = args.seq_len
        self.time_seg = args.time_seg
        in_channels = args.label_dim + args.channels
        self.patch = PatchEmbedding_Linear(in_channels, emb_size, self.seq_len)
        self.blocks1 = Dis_Encoder(args.depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5,**kwargs)
        self.blocks2 = Dis_Encoder(args.depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)
        self.embedding = nn.Embedding(8, args.label_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        # embedding block
        label_emb = self.embedding(labels).unsqueeze(1).repeat(1, self.seq_len, 1)
        x = torch.cat([x, label_emb], dim=-1) # use CGAN
        x = self.patch(x) 
        
        # attention block
        x = rearrange(x, 'b (t w) c -> b t w c', t = self.time_seg[0]) # temporal segmentation
        x = self.blocks1(x) 
        x = rearrange(x, 'b t w c -> b (t w) c')
        x = rearrange(x, 'b (t w) c -> b t w c', t = self.time_seg[1]) # repartition
        x = self.blocks2(x) 
        
        x = rearrange(x, 'b t w c -> b (t w) c')
        output = self.classifier(x)
        return self.sigmoid(output)
        