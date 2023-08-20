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


class Generator(nn.Module):
    def __init__(self,args,embed_dim=50):
        super(Generator, self).__init__()
        self.channels = args.channels
        self.latent_dim = args.latent_dim + args.label_dim
        self.seq_len = args.seq_len
        self.embed_dim = embed_dim
        self.depth = args.depth
        self.attn_drop_rate = args.attn_drop_rate
        self.forward_drop_rate = args.forward_drop_rate
        self.time_seg = args.time_seg
        self.num_heads = args.num_heads

        self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.act = LIFSpike() # LIFSpike is the firing function of LIF neuron
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.blocks1 = Gen_Encoder(
                         depth=self.depth,
                         emb_size = self.embed_dim,
                         drop_p = self.attn_drop_rate,
                         forward_drop_p=self.forward_drop_rate,
                         seq_len = self.seq_len,
                         time_emb = self.time_seg[0]
                        )
        self.blocks2 = Gen_Encoder(
                         depth=self.depth,
                         emb_size = self.embed_dim,
                         drop_p = self.attn_drop_rate,
                         forward_drop_p=self.forward_drop_rate,
                         seq_len = self.seq_len,
                         time_emb = self.time_seg[1]
                        )
        self.deconv = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.channels, 1, 1, 0)
        )
        self.embedding = nn.Embedding(8, args.label_dim)
        self.relu = nn.ReLU()

    def forward(self, z, labels):
        # embedding block
        label_emb = self.embedding(labels)
        z = torch.cat([z, label_emb], dim=-1) # use CGAN
        x = self.l1(z).view(-1, self.seq_len, self.embed_dim) 
        B, W, C = x.shape
       
        x = x + (self.pos_embed)
        x = self.act(x)
        
        # attention block
        x = rearrange(x, 'b (t w) c -> b t w c', t = self.time_seg[0]) # temporal segmentation
        x = self.blocks1(x) 
        x = rearrange(x, 'b t w c -> b (t w) c')
        x = rearrange(x, 'b (t w) c -> b t w c', t = self.time_seg[1]) # repartition
        x = self.blocks2(x)

        output = self.deconv(x.permute(0, 1, 3, 2).flatten(0, 1).contiguous())
        
        output = rearrange(output, '(b t) c w -> b t w c', t = self.time_seg[1])
        output = rearrange(output, 'b t w c -> b (t w) c')
        output = self.relu(output)
        return output
    
    
class Gen_EncoderBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 time_emb,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5,
                 seq_len = 400,
                 factor = 20
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.dropout1 = nn.Dropout(drop_p)
        self.attention = MultiHeadAttention(emb_size, num_heads, drop_p)
        
        
        self.router = nn.Parameter(torch.randn(time_emb, factor ,emb_size ))
        self.dim_sender = MultiHeadAttention(emb_size, num_heads, dropout = drop_p)
        self.dim_receiver = MultiHeadAttention(emb_size, num_heads, dropout = drop_p)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout2 = nn.Dropout(drop_p)
        self.layer1 = nn.Conv1d(seq_len//time_emb, emb_size, 1,1,0)
        self.layer2 = nn.Conv1d( emb_size,seq_len//time_emb,1,1,0)
        
        
        self.dropout3 = nn.Dropout(drop_p)
        self.norm3 = nn.LayerNorm(emb_size)
        self.feedforward = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
    # generator has temporal and spatial attention
    def forward(self, x):

        router = repeat(self.router, 't factor c -> b t factor c', b = x.shape[0])
        B, T, W, C = x.shape
        
        
        # temporal attention
        res1 = x
        x = self.norm1(x.flatten(0, 1))
        x = rearrange(x, '(b t) w c -> b t w c', b = B)
        x = self.attention(x,x,x)
        x = self.dropout1(x.flatten(0, 1))
        x = rearrange(x, '(b t) w c -> b t w c', b = B)
        x += res1
        
        # spatial attention
        res2 = x
        x = self.norm2(x.flatten(0, 1))
        x = rearrange(self.layer1(x), '(b t) w c -> b t c w', b = B)
        dim_buffer = self.dim_sender(router, x, x)
        dim_receive = self.dim_receiver(x, dim_buffer, dim_buffer)
        dim_receive = dim_receive.flatten(0, 1)
        x = rearrange(dim_receive, 't c w -> t w c')
        x = rearrange(self.layer2(x), '(b t) w c -> b t w c', t = T)
        x += res2
        
        # feedforward block
        res3 = x
        x = self.norm3(x.flatten(0, 1))
        x = rearrange(x, '(b t) w c -> b t w c', b = B)
        x = self.feedforward(x)
        x = self.dropout3(x.flatten(0, 1))
        x = rearrange(x, '(b t) w c -> b t w c', b = B)
        x += res3

        return x






class Gen_Encoder(nn.Sequential):
    def __init__(self, depth=1, **kwargs):
        super().__init__(*[Gen_EncoderBlock(**kwargs) for _ in range(depth)])       
        