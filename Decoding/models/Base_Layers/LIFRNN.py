import torch
from torch import nn
from torch.autograd import Variable
from .Layers import *
from einops import rearrange, reduce, repeat
scale = 0.0  # 膜电位初始化的噪音，0是0初始化

torch.manual_seed(666)  # 随机种子
torch.cuda.manual_seed_all(666)

# temporal and spatial attention
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class LIFConvLSTM(nn.Module):
    def __init__(self,
                 input_range,
                 inputSize,
                 hiddenSize,
                 kernel_size = 1,
                 bias=True,
                 decay=0.3,
                 onlyLast=True,
                 dropOut=0,
                 useBatchNorm=False,
                 useLayerNorm=False,):
        super().__init__()

        self.onlyLast = onlyLast
        self.input_range = input_range
        self.network = nn.Sequential()
        self.network = LIFConvLSTMCell(input_range=input_range // 2,
                                        inputSize=inputSize,
                                        hiddenSize=hiddenSize,
                                        kernel_size=kernel_size,
                                        bias=bias,
                                        dropOut=dropOut,
                                        useBatchNorm=useBatchNorm,
                                        useLayerNorm=useLayerNorm,
                                        decay=decay)

    def forward(self, data, hidden_state=None):

        torch.cuda.empty_cache()
        data = rearrange(data.transpose(1,2), 'b c (t1 t2) -> b c t1 t2' , t1 = self.input_range // 2) 
        
        self.network.reset()

        for step in range(list(data.size())[-1]):
            out = data[:, :, :, step]
            out = self.network(out)
            output = out

            if step == 0:
                temp = list(output.size())
                temp.append(data.size()[-1])
                outputsum = torch.zeros(temp)
                if outputsum.device != data.device:
                    outputsum = outputsum.to(data.device)

            outputsum[:, :, :, step] = output

        if self.onlyLast:
            return output
        else:
            return outputsum

class LIFConvLSTMCell(nn.Module):
    def __init__(self,
                 input_range,
                 inputSize,
                 hiddenSize,
                 kernel_size,
                 dropOut=0,
                 bias=True,
                 useBatchNorm=False,
                 useLayerNorm=False,
                 decay=0.3):
        """
        :param inputSize:(Num) number of input
        :param hiddenSize:(Num) number of output
        :param spikeAcFun:handle of recurrent spike firing function
        :param dropOut: 0~1 unused
        :param useBatchNorm:
        :param useLayerNorm:
        """
        super().__init__()
        self.input_range = input_range
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None
        self.act = LIFSpike()


        self.convlstmcell = ConvLSTMCell(input_size=input_range,
                                                  input_dim=inputSize,
                                                  hidden_dim=hiddenSize,
                                                  kernel_size=kernel_size,
                                                  bias=bias)


        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(hiddenSize)
            self.BNLayerc = nn.BatchNorm1d(hiddenSize)

        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm(hiddenSize)
            self.Lnormc = nn.LayerNorm(hiddenSize)

 
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        self.c = None
        self.h = None

    def forward(self, input, init_v=None):
        """
        :param input:
        :param init_v:
        :return:
        """
        self.batchSize = input.size()[0]

        torch.cuda.empty_cache()

        if input.device != self.convlstmcell.conv.weight.device:
            input = input.to(self.convlstmcell.conv.weight.device)

        if self.h is None:
            if init_v is None:
                self.h = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     self.input_range,
                                     device=input.device)

                self.c = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     self.input_range,
                                     device=input.device)

            else:
                self.h, self.c = init_v.clone()

  
        u, self.c = self.convlstmcell(input_tensor=input,
                                      cur_state=[self.h, self.c])
        self.h = self.decay * u * (1 - self.act(u))    # important
        x = u

        return x

    def reset(self):
        self.c = None
        self.h = None


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv_att = CBAMLayer(channel=self.hidden_dim, reduction=4, spatial_kernel=7)

        self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv,
                                             self.hidden_dim,
                                             dim=1)
        cc_f = self.conv_att(cc_f)

        i = torch.sigmoid(cc_i)

        del cc_i
        torch.cuda.empty_cache()

        f = torch.sigmoid(cc_f)

        del cc_f
        torch.cuda.empty_cache()

        o = torch.sigmoid(cc_o)

        del cc_o
        torch.cuda.empty_cache()

        g = torch.tanh(cc_g)

        del cc_g
        torch.cuda.empty_cache()

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        del f, i, g, o, c_cur,cur_state,input_tensor,h_cur
        torch.cuda.empty_cache()

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size,
                                     self.hidden_dim,
                                     self.input_size)).cuda(),
                Variable(torch.zeros(batch_size,
                                     self.hidden_dim,
                                     self.input_size)).cuda())


