import torch
import numpy as np

from math import ceil
from itertools import permutations
from torchaudio.transforms import MelScale

from mir_eval.separation import bss_eval_sources
from typing import List, Type, Any, Callable, Optional, Union

# 导入scores中的SI-SNR计算函数
from ..scores import cal_sisnr_torch

# Utility functions
def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def l1norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim, p=1)


class PIT_SISNR_mag(torch.nn.Module):
    """PIT SISNR loss in magnitude domain."""
    
    def __init__(self, frame_length=512, frame_shift=128, window='hann', 
                 num_stages=1, num_spks=2, scale_inv=True, mel_opt=False):
        """Initialize PIT SISNR magnitude loss module."""
        super(PIT_SISNR_mag, self).__init__()
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.window = window
        self.num_stages = num_stages
        self.num_spks = num_spks
        self.scale_inv = scale_inv
        self.mel_opt = mel_opt
        
        # Register window buffer  
        if window == 'hann':
            window_tensor = torch.hann_window(frame_length)
            self.register_buffer("window_tensor", window_tensor)
        else:
            self.window_tensor = None
        
        if mel_opt:
            self.mel_fb = MelScale(n_mels=80, sample_rate=16000, n_stft=int(frame_length / 2) + 1)
    
    def stft_magnitude(self, x):
        """Compute STFT magnitude using torch.stft."""
        window_param = self.window_tensor if self.window_tensor is not None else self.window
        x_stft = torch.stft(x, self.frame_length, self.frame_shift, self.frame_length, 
                           window_param, return_complex=True)
        magnitude = torch.abs(x_stft).transpose(-2, -1)
        return magnitude
    
    def forward(self, estims, targets, input_sizes=None, **kwargs):
        """Calculate forward propagation."""
        eps = 1.0e-12
        
        def _STFT_Mag_SDR_loss():
            loss_for_permute = []
            
            mix = estims
            src = targets
            mix_zm = mix - torch.mean(mix, dim=-1, keepdim=True)
            src_zm = src - torch.mean(src, dim=-1, keepdim=True)
            
            if self.scale_inv:
                scale = torch.sum(mix_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps)
                src_zm = torch.clamp(scale, min=1e-2) * src_zm
            
            mix_zm_mag = self.stft_magnitude(mix_zm)
            src_zm_mag = self.stft_magnitude(src_zm)
            
            if self.mel_opt:
                mix_zm_mag = self.mel_fb(mix_zm_mag)
                src_zm_mag = self.mel_fb(src_zm_mag)
            
            utt_loss = -20 * torch.log10(eps + l2norm(l2norm(src_zm_mag)) / (l2norm(l2norm(mix_zm_mag - src_zm_mag)) + eps))
            loss_for_permute.append(utt_loss)
            return sum(loss_for_permute)
        
        pscore = _STFT_Mag_SDR_loss() 
        
        if input_sizes is not None:
            num_utts = input_sizes.shape[0]
        else:
            num_utts = len(estims)
        
        return torch.sum(pscore) / num_utts


class PIT_SISNR_time(torch.nn.Module):
    """SISNR loss in time domain."""
    
    def __init__(self, num_spks=1, scale_inv=True):
        """Initialize PIT SISNR time loss module."""
        super(PIT_SISNR_time, self).__init__()
        self.scale_inv = scale_inv

    def forward(self, estims, targets, input_sizes=None, **kwargs):
        """Calculate forward propagation."""
        eps = 1.0e-8
        
       
        sisnr_value = cal_sisnr_torch( targets, estims)
        return -sisnr_value

