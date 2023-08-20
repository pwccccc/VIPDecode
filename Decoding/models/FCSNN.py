import random
from .Base_Layers.Layers import *
from torch.nn import functional as F
from einops import rearrange, reduce, repeat


class FCSNN(nn.Module):
    def __init__(self, T, input_size):
        super(FCSNN, self).__init__()
        self.fc1= SeqToANNContainer(nn.Linear(input_size, 256))
        self.gelu = SeqToANNContainer(nn.GELU())
        self.ta = TimeAttention(2, ratio=1)
        self.sa = SpatialAttention(256, ratio=16)
        self.act2 = LIFSpike()
        self.fc2 = SeqToANNContainer(nn.Linear(256, 8))

    def forward(self, x):
        # x = self.ta1(x) * x
        t = x.shape[1]
        x = self.fc1(x)
        x = self.gelu(x)
        x = rearrange(x,'b (i t) c -> b i (t c)', i = 2)
        x = self.ta(x) * x
        x = rearrange(x,'b i (t c) -> b (i t) c', t = t//2)
        x = self.sa(x) * x
        # out = self.ca(out) * out
        x = self.act2(x)
        
        x = self.fc2(x)

        return x

class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return F.log_softmax(avgout, dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        avgout = self.sharedMLP(self.avg_pool(x))
        # maxout = self.sharedMLP(self.max_pool(x))
        # out = self.sigmoid(avgout + maxout)
        # out = x.transpose(1,2)
        return F.log_softmax(avgout, dim = 1).squeeze(2).unsqueeze(1)





