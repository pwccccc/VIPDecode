import torch
from torch import nn
from einops import rearrange, reduce, repeat

class LSTM(nn.Module):
    def __init__(self,T, input_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                       hidden_size=128,
                       num_layers=1,
                       dropout=0.5,
                       bidirectional=False,
                       bias=True,
                       batch_first=True)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(128, 8)

    def forward(self, x):
        output, hn = self.lstm(x)
        hn = hn[0].transpose(0, 1)
        hn = hn[:, -1, :]
        x = self.dropout(hn)
        x = self.fc(x)

        return x

