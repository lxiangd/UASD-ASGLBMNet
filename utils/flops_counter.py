import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Union
import numpy as np

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. Install with: pip install thop")


def count_flops_and_macs(model: nn.Module, input_shape: tuple, device: str = 'cpu'):
    """
    使用thop计算模型的FLOPs和MACs
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状
        device: 设备类型
    
    Returns:
        tuple: (flops, macs)
    """
    if not THOP_AVAILABLE:
        print("thop不可用，将返回0。请使用 'pip install thop' 安装。")
        return 0, 0
    
    model = model.to(device)
    model.eval()
    
    # 创建随机输入
    dummy_input = torch.randn(*input_shape).to(device)
    
    try:
        # 使用thop计算MACs和参数
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        # FLOPs ≈ 2 * MACs
        flops = macs * 2
        
        return int(flops), int(macs)
        
    except Exception as e:
        print(f"使用thop计算FLOPs失败: {e}")
        return 0, 0


def format_flops_macs(flops: int, macs: int) -> tuple:
    """
    格式化FLOPs和MACs数值
    
    Args:
        flops: FLOPs数值
        macs: MACs数值
    
    Returns:
        tuple: (formatted_flops, formatted_macs)
    """
    if not THOP_AVAILABLE:
        return "N/A", "N/A"
    try:
        flops_str, macs_str = clever_format([flops, macs], "%.3f")
        return flops_str, macs_str
    except Exception as e:
        print(f"格式化FLOPs/MACs失败: {e}")
        return "N/A", "N/A"


def cal_total_params(model: nn.Module) -> int:
    """
    计算模型总参数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        int: 总参数量
    """
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # 示例用法
    
    # 创建一个简单的模型进行测试
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32, 1)
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = x.mean(dim=-1)  # Global average pooling
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    input_shape = (1, 1, 1024)  # (batch_size, channels, length)
    
    total_params = cal_total_params(model)
    flops, macs = count_flops_and_macs(model, input_shape)
    flops_str, macs_str = format_flops_macs(flops, macs)
    
    print("=" * 50)
    print("模型摘要")
    print("=" * 50)
    print(f"总参数量: {total_params / 1e6:.2f}M ({total_params:,})")
    print(f"FLOPs: {flops_str}")
    print(f"MACs: {macs_str}")
    print("=" * 50)
