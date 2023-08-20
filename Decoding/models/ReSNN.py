import torch
from torch import nn
from torch.autograd import Variable
from .Base_Layers.Layers import *
from einops import rearrange, reduce, repeat
from .Base_Layers.LIFRNN import LIFConvLSTM

class ReSNN(nn.Module):
    def __init__(self, T, input_size):
        super(ReSNN, self).__init__()
        self.LIFRNN = LIFConvLSTM(T,input_size,128)
        self.act = LIFSpike()
        self.fc = nn.Linear(128, 8)

    def forward(self, x):
        B = x.shape[0]
        x = rearrange(self.LIFRNN(x), 'b c t -> b t c')

        x = self.act(x)
        x = rearrange(x, 'b t c -> (b t) c')
        x = rearrange(self.fc(x), '(b t) c -> b t c', b = B)

        return x