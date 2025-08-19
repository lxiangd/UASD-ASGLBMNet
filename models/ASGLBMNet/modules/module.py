import sys
sys.path.append('../')
import torch.nn as nn
import torch.nn.functional as F
import torch
import warnings
warnings.filterwarnings('ignore')

from .network import *


class AudioEncoder(torch.nn.Module):
    def __init__(self, enc_conf: dict, proj_conf: dict):
        super().__init__()
        self.enc_conv = torch.nn.Conv1d(
            in_channels=enc_conf['in_channels'],
            out_channels=enc_conf['out_channels'],
            kernel_size=enc_conf['kernel_size'],
            stride=enc_conf['stride'],
            groups=enc_conf['groups'],
            bias=enc_conf['bias'])
        self.enc_act = torch.nn.GELU()
        self.proj_norm = torch.nn.GroupNorm(num_groups=1, num_channels=proj_conf['num_channels'], eps=1e-8)
        self.proj_conv = torch.nn.Conv1d(
            in_channels=proj_conf['in_channels'],
            out_channels=proj_conf['out_channels'],
            kernel_size=proj_conf['kernel_size'],
            bias=proj_conf['bias'])

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        enc_feat = self.enc_conv(x)
        enc_feat = self.enc_act(enc_feat)
        proj = self.proj_norm(enc_feat)
        proj = self.proj_conv(proj)
        return proj, enc_feat

class Enclayer(torch.nn.Module):

    def __init__(self, glbm_conf: dict, down_conv_layer: dict, down_conv=True):
        super().__init__()

        class DownConvLayer(torch.nn.Module):
            def __init__(self, in_channels: int, samp_kernel_size: int):
                super().__init__()
                self.down_conv = torch.nn.Conv1d(
                    in_channels=in_channels, out_channels=in_channels,
                    kernel_size=samp_kernel_size, stride=2,
                    padding=(samp_kernel_size-1)//2, groups=in_channels)
                self.BN = torch.nn.BatchNorm1d(num_features=in_channels)
                self.gelu = torch.nn.GELU()

            def forward(self, x: torch.Tensor):
                y = self.down_conv(x)
                y = self.BN(y)
                y = self.gelu(y)
                return y

    
        self.glbm_1 = GLBM(**glbm_conf)
        self.glbm_2 = GLBM(**glbm_conf)
        self.downconv = DownConvLayer(**down_conv_layer) if down_conv else None

    def forward(self, x: torch.Tensor): 
        x = self.glbm_1(x)
        x = self.glbm_2(x)
        skip = x
        if self.downconv:
            x = self.downconv(x)
        return x, skip

class DecLayer(torch.nn.Module):

    def __init__(self, num_spks: int, glbm_conf: dict):  
        super().__init__()
        self.glbm_1 = GLBM(**glbm_conf)
        self.glbm_2 = GLBM(**glbm_conf)
        self.glbm_3 = GLBM(**glbm_conf)

    def forward(self, x: torch.Tensor):
        x = self.glbm_1(x)
        x = self.glbm_2(x)
        x = self.glbm_3(x)
        skip = x
        return x, skip

  


class middleLayer(torch.nn.Module):
    def __init__(self, in_channels: int, num_spks: int): 
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 4*in_channels, kernel_size=1),
            torch.nn.GLU(dim=-2),
            torch.nn.Conv1d(2*in_channels, in_channels, kernel_size=1))
        self.norm = torch.nn.GroupNorm(1, in_channels, eps=1e-8)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.norm(x)
        return x


class Enc_Dec(torch.nn.Module):
    def __init__(self, num_stages: int, enc_stage: dict, middle_layer: dict, simple_fusion:dict, dec_stage: dict):
        super().__init__()
        self.num_stages = num_stages

        # Contracting
        self.enc_stages = torch.nn.ModuleList([
            Enclayer(**enc_stage, down_conv=True) for _ in range(self.num_stages)
        ])
        self.bottleneck_G = Enclayer(**enc_stage, down_conv=False)
        self.middle_block = middleLayer(**middle_layer)

        # Expanding
        self.simple_fusion = torch.nn.ModuleList([])
        self.dec_stages = torch.nn.ModuleList([])
        for _ in range(self.num_stages):
            self.simple_fusion.append(torch.nn.Conv1d(
                in_channels=simple_fusion['out_channels']*2,
                out_channels=simple_fusion['out_channels'], kernel_size=1))
            self.dec_stages.append(DecLayer(num_spks=dec_stage['num_spks'], glbm_conf=dec_stage['glbm_conf']))
    
    def forward(self, input: torch.Tensor):
        x, _ = self.pad_signal(input)
        skip = []
        for idx in range(self.num_stages):
            x, skip_ = self.enc_stages[idx](x)
            skip_ = self.middle_block(skip_)
     
            skip.append(skip_)
        x, _ = self.bottleneck_G(x)
        x = self.middle_block(x) # B, 2F, T
        
        each_stage_outputs = []
        # Temporal Expanding Part
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = torch.nn.functional.upsample(x, skip[idx_en].shape[-1])
            x = torch.cat([x,skip[idx_en]],dim=1)
            x = self.simple_fusion[idx](x)
            x, _ = self.dec_stages[idx](x)
        
        last_stage_output = x 
        return last_stage_output, each_stage_outputs
    
    def pad_signal(self, input: torch.Tensor):
        if input.dim() == 1: input = input.unsqueeze(0)
        elif input.dim() not in [2, 3]: raise RuntimeError("Input can only be 2 or 3 dimensional.")
        elif input.dim() == 2: input = input.unsqueeze(1)
        L = 2**self.num_stages
        batch_size = input.size(0)  
        ndim = input.size(1)
        nframe = input.size(2)
        padded_len = (nframe//L + 1)*L
        rest = 0 if nframe%L == 0 else padded_len - nframe
        if rest > 0:
            pad = torch.autograd.Variable(torch.zeros(batch_size, ndim, rest)).type(input.type()).to(input.device)
            input = torch.cat([input, pad], dim=-1)
        return input, rest


class AudioDecoder(torch.nn.Module):

    def __init__(self, out_layer_conf: dict, dec_conf: dict, masking: bool = False):
        super().__init__()
        self.masking = masking
        in_channels_enc = out_layer_conf['in_channels']
        proj_channels = out_layer_conf['out_channels']
        self.expand = torch.nn.Sequential(
            torch.nn.Linear(proj_channels, 4*proj_channels),
            torch.nn.GLU(),
            torch.nn.Linear(2*proj_channels, in_channels_enc))
        self.mask_block = Masking(in_channels_enc, Activation_mask="ReLU", concat_opt=None)
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels=dec_conf['in_channels'],
            out_channels=dec_conf['out_channels'],
            kernel_size=dec_conf['kernel_size'],
            stride=dec_conf['stride'],
            bias=dec_conf['bias'])

    def forward(self, sep_feature: torch.Tensor, encoder_feature: torch.Tensor):
        sep_feature = sep_feature[..., :encoder_feature.shape[-1]]
        x = sep_feature.permute(0, 2, 1)
        x = self.expand(x)
        x = x.permute(0, 2, 1)
        if self.masking:
            x = self.mask_block(x, encoder_feature)
        audio = self.deconv(x)
        return audio.squeeze(1)