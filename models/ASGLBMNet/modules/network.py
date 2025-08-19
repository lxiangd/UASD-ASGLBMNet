import torch
import math
import numpy
from .BMamba import BMamba1D

class LayerScale(torch.nn.Module):
    def __init__(self, dims, input_size, Layer_scale_init=1.0e-5):
        super().__init__()
        if dims == 1:
            self.layer_scale = torch.nn.Parameter(torch.ones(input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 2:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,input_size)*Layer_scale_init, requires_grad=True)
        elif dims == 3:
            self.layer_scale = torch.nn.Parameter(torch.ones(1,1,input_size)*Layer_scale_init, requires_grad=True)
    
    def forward(self, x):
        return x*self.layer_scale

class Masking(torch.nn.Module):
    def __init__(self, input_dim, Activation_mask='Sigmoid', **options):
        super(Masking, self).__init__()
        
        self.options = options
        if self.options['concat_opt']:
            self.pw_conv = torch.nn.Conv1d(input_dim*2, input_dim, 1, stride=1, padding=0)

        if Activation_mask == 'Sigmoid':
            self.gate_act = torch.nn.Sigmoid()
        elif Activation_mask == 'ReLU':
            self.gate_act = torch.nn.ReLU()
            

    def forward(self, x, skip):
   
        if self.options['concat_opt']:
            y = torch.cat([x, skip], dim=-2)
            y = self.pw_conv(y)
        else:
            y = x
        y = self.gate_act(y) * skip

        return y


class FFN(torch.nn.Module):
    def __init__(self, in_channels, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.LayerNorm(in_channels),
            torch.nn.Linear(in_channels, in_channels*6))
        self.net2 = torch.nn.Sequential(
            torch.nn.GLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(in_channels*3, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)

    def forward(self, x):
        y = self.net1(x)         
        y = self.net2(y)          
        return x + self.Layer_scale(y)


class BMAM(torch.nn.Module):
    """BMAM: 基于 Mamba 的全局注意力 (原 EGA)."""
    def __init__(self, in_channels: int, num_mha_heads: int):
        super().__init__()
        self.attention = BMamba1D(hidden_dim=in_channels, d_state=num_mha_heads)

    def forward(self, x: torch.Tensor):
        x = self.attention(x)     
        x = x.permute(0, 2, 1)    
        return x




class LTCAM(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, dropout_rate, Layer_scale_init=1.0e-5):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear1 = torch.nn.Linear(in_channels, in_channels*2)
        self.glu = torch.nn.GLU()
        self.dw_conv_1d = torch.nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels)
        self.linear2 = torch.nn.Linear(in_channels, 2*in_channels)
        self.bn = torch.nn.BatchNorm1d(2*in_channels)
        self.linear3 = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Linear(2*in_channels, in_channels),
            torch.nn.Dropout(dropout_rate))
        self.Layer_scale = LayerScale(dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init)

    def forward(self, x):
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.glu(y)              # [B,T,2C] -> gate -> [B,T,C]
        y = y.permute(0,2,1)         # [B,C,T]
        y = self.dw_conv_1d(y)       # depthwise conv
        y = y.permute(0,2,1)         # [B,T,C]
        y = self.linear2(y)          # [B,T,2C]
        y = y.permute(0,2,1)         # [B,2C,T]
        y = self.bn(y)
        y = y.permute(0,2,1)         # [B,T,2C]
        y = self.linear3(y)          # [B,T,C]
        return x + self.Layer_scale(y)
    
class GLBM(torch.nn.Module):

    def __init__(self, in_channels: int, num_mha_heads: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.ega = BMAM(in_channels=in_channels, num_mha_heads=num_mha_heads)
        self.global_gcfn = FFN(in_channels=in_channels, dropout_rate=dropout_rate)
        self.cla = LTCAM(in_channels=in_channels, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.local_gcfn = FFN(in_channels=in_channels, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor):
        x = self.ega(x)             # -> [B,T,N]
        x = self.global_gcfn(x)     # [B,T,N]
        x = self.cla(x)             # [B,T,N]
        x = self.local_gcfn(x)      # [B,T,N]
        x = x.permute(0, 2, 1)      # -> [B,N,T]
        return x
    