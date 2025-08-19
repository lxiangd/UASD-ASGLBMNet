import torch
import torch.nn.functional as F
import numpy as np
import mir_eval.separation
import fast_bss_eval

'''DPRNN计算方式'''
def cal_sisnr_torch(s,x, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))

    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    sisnr=20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
 
    return torch.mean(sisnr)



def cal_simple_sdr_torch(targets, estims, eps=1e-10):
    """
    简单的SDR计算，作为fast_bss_eval的备用方案
    使用基本的信号功率比计算
    
    Args:
        targets: 目标信号 tensor [batch, time]
        estims: 估计信号 tensor [batch, time]
        eps: 数值稳定性参数
    
    Returns:
        sdr: 简单SDR值
    """
    import torch
    
    # 确保输入是tensor
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)
    if not isinstance(estims, torch.Tensor):
        estims = torch.tensor(estims, dtype=torch.float32)
    
    # 计算信号功率
    signal_power = torch.sum(targets ** 2, dim=-1) + eps
    # 计算噪声功率（估计信号与目标信号的差）
    noise_power = torch.sum((estims - targets) ** 2, dim=-1) + eps
    
    # 计算SDR
    sdr = 10 * torch.log10(signal_power / noise_power + eps)
    
    return torch.mean(sdr)

def cal_sdr_torch(estimated, reference):
    """
    计算SDR (Signal-to-Distortion Ratio)
    使用稳定的数值计算方法
    """

    # 确保输入张量在同一设备上，并且是浮点类型
    if estimated.device != reference.device:
        estimated = estimated.to(reference.device)
    
    estimated = estimated.float()
    reference = reference.float()
    
    # 检查输入是否有效
    if torch.isnan(estimated).any() or torch.isnan(reference).any():
        print("输入包含NaN，使用简化SDR计算")
        return cal_simple_sdr_torch(estimated, reference)
    
    if torch.isinf(estimated).any() or torch.isinf(reference).any():
        print("输入包含Inf，使用简化SDR计算")
        return cal_simple_sdr_torch(estimated, reference)
    
    # 添加小量避免零值
    eps = 1e-8
    estimated = estimated + eps * torch.randn_like(estimated)
    reference = reference + eps * torch.randn_like(reference)
    
    # 使用 fast_bss_eval 库计算 SDR
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(
        estimated.detach().cpu().numpy(),
        reference.detach().cpu().numpy()
    )
    
    # 检查结果是否有效
    if np.isnan(sdr[0]) or np.isinf(sdr[0]):
        print("fast_bss_eval返回无效值，使用简化SDR计算")
        return cal_simple_sdr_torch(estimated, reference)
        
    return torch.tensor(sdr[0], dtype=torch.float32)
    
  


def cal_snrseg_torch(clean_wavs, enhanced_wavs, mixture_wavs=None, frame_length=400):
    """
    使用torch计算分段SNR (Segmental Signal-to-Noise Ratio)
    
    Args:
        clean_wavs: 干净信号 tensor [batch, time]
        enhanced_wavs: 增强信号 tensor [batch, time]
        mixture_wavs: 混合信号 tensor [batch, time] (可选，用于计算improvement)
        frame_length: 帧长度 (默认400样本，对应25ms@16kHz)
    
    Returns:
        snrseg: 分段SNR值
    """
    # 确保输入是2D tensor [batch, time]
    if clean_wavs.dim() == 1:
        clean_wavs = clean_wavs.unsqueeze(0)
    elif clean_wavs.dim() == 3:  # [batch, channels, time]
        clean_wavs = clean_wavs.squeeze(1)  # 去掉通道维度，假设是单通道
        
    if enhanced_wavs.dim() == 1:
        enhanced_wavs = enhanced_wavs.unsqueeze(0)
    elif enhanced_wavs.dim() == 3:  # [batch, channels, time]
        enhanced_wavs = enhanced_wavs.squeeze(1)  # 去掉通道维度，假设是单通道
    
    batch_size, signal_length = clean_wavs.shape
    
    # 计算帧数
    num_frames = signal_length // frame_length
    
    if num_frames == 0:
        # 如果信号太短，直接计算整体SNR
        return cal_sdr_torch(clean_wavs, enhanced_wavs)
    
    # 分段计算SNR
    snr_segments = []
    
    for i in range(num_frames):
        start_idx = i * frame_length
        end_idx = start_idx + frame_length
        
        clean_frame = clean_wavs[:, start_idx:end_idx]
        enhanced_frame = enhanced_wavs[:, start_idx:end_idx]
        
        # 计算当前帧的SNR
        signal_power = torch.sum(clean_frame ** 2, dim=1)
        noise_power = torch.sum((clean_frame - enhanced_frame) ** 2, dim=1)
        
        frame_snr = 10 * torch.log10(signal_power / (noise_power + 1e-8) + 1e-8)
        
        # 限制SNR范围在[-10, 35]dB
        frame_snr = torch.clamp(frame_snr, min=-10, max=35)
        
        snr_segments.append(frame_snr)
    
    # 计算平均分段SNR
    snrseg = torch.mean(torch.stack(snr_segments))
    
    return snrseg

def cal_sisnri(clean_wavs, enhanced_wavs, mixture_wavs):
    """     
    使用torch计算SI-SNR improvement
    
    Args:
        clean_wavs: 干净信号 tensor [batch, time]
        enhanced_wavs: 增强信号 tensor [batch, time]
        mixture_wavs: 混合信号 tensor [batch, time]
    
    Returns:
        sisnri: SI-SNR improvement值 (标量)
    """
    # 计算增强信号的SI-SNR
    sisnr_enhanced = cal_sisnr_torch(clean_wavs, enhanced_wavs)
    
    # 计算混合信号的SI-SNR
    sisnr_mixture = cal_sisnr_torch(clean_wavs, mixture_wavs)
    
    # 计算improvement并求平均值
    sisnri = torch.mean(sisnr_enhanced - sisnr_mixture)
    
    return sisnri

def cal_sdri(clean_wavs, enhanced_wavs, mixture_wavs):
    """
    使用mir_eval的bss_eval计算SDR improvement

    Args:
        clean_wavs: 干净信号 tensor [batch, time]
        enhanced_wavs: 增强信号 tensor [batch, time]
        mixture_wavs: 混合信号 tensor [batch, time]

    Returns:
        sdri: SDR improvement值
    """
    # 计算增强信号的SDR
    sdr_enhanced = cal_sdr_torch(clean_wavs, enhanced_wavs)

    # 计算混合信号的SDR
    sdr_mixture = cal_sdr_torch(clean_wavs, mixture_wavs)

    # 计算improvement
    sdri = sdr_enhanced - sdr_mixture

    return sdri

def cal_snrsegi(clean_wavs, enhanced_wavs, mixture_wavs, frame_length=400):
    """
    使用torch计算分段SNR improvement
    
    Args:
        clean_wavs: 干净信号 tensor [batch, time]
        enhanced_wavs: 增强信号 tensor [batch, time]
        mixture_wavs: 混合信号 tensor [batch, time]
        frame_length: 帧长度
    
    Returns:
        snrsegi: 分段SNR improvement值
    """
    # 计算增强信号的分段SNR
    snrseg_enhanced = cal_snrseg_torch(clean_wavs, enhanced_wavs, frame_length=frame_length)
    
    # 计算混合信号的分段SNR
    snrseg_mixture = cal_snrseg_torch(clean_wavs, mixture_wavs, frame_length=frame_length)
    
    # 计算improvement
    snrsegi = snrseg_enhanced - snrseg_mixture
    
    return snrsegi

# 为了保持向后兼容，保留原有的函数名
def cal_sisnri_batch(clean_wavs, enhanced_wavs, mixture_wavs, FS=16000):
    """Calculate batch SI-SNR improvement using torch"""
    return cal_sisnri(clean_wavs, enhanced_wavs, mixture_wavs)

def cal_sdri_batch(clean_wavs, enhanced_wavs, mixture_wavs, FS=16000):
    """Calculate batch SDR improvement using mir_eval"""
    return cal_sdri(clean_wavs, enhanced_wavs, mixture_wavs)

def cal_snrsegi_batch(clean_wavs, enhanced_wavs, mixture_wavs, FS=16000):
    """Calculate batch segmental SNR improvement using torch"""
    return cal_snrsegi(clean_wavs, enhanced_wavs, mixture_wavs)
