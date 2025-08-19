#!/usr/bin/env python3
"""
模型测试脚本 - 加载训练好的模型进行测试并保存结果
"""

import os
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import utils
from dataloader import create_dataloader
from config.config import Config

# 设置环境变量抑制警告
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.parallel")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

def test_model(cfg, model_path, output_dir):
    """
    测试模型函数
    
    Args:
        cfg: 配置对象
        model_path: 模型权重文件路径
        output_dir: 输出目录
    """
    
    # 设置GPU设备
    if 'gpu_ids' in cfg.train:
        gpu_ids = cfg.train['gpu_ids']
        if isinstance(gpu_ids, (int, str)):
            gpu_ids = [int(gpu_ids)]
        elif isinstance(gpu_ids, (list, tuple)):
            gpu_ids = [int(i) for i in gpu_ids]
        if gpu_ids and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_ids[0]}')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(cfg.model['device'] if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 创建模型
    print("创建模型...")
    model = utils.get_arch(cfg)
    
    # 加载模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查checkpoint格式
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
        epoch = checkpoint.get('epoch', 'unknown')
        best_sisnr = checkpoint.get('best_sisnr', 'unknown')
        print(f"模型训练轮次: {epoch}, 最佳SISNR: {best_sisnr}")
    else:
        model_state_dict = checkpoint
        print("直接加载模型状态字典")
    
    # 过滤掉thop添加的total_ops和total_params键
    filtered_state_dict = {}
    for key, value in model_state_dict.items():
        if not (key.endswith('.total_ops') or key.endswith('.total_params')):
            filtered_state_dict[key] = value
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    total_params = utils.cal_total_params(model)
    print(f'模型总参数: {total_params} ({total_params / 1000000.0:.2f} M)')
    
    
    # 计算 FLOPs 和 MACs
    print("计算模型 FLOPs 和 MACs...")
   
    # 使用和数据加载器相同的输入形状
    input_shape = (1, cfg.dataset['chunk_size'])  # (batch_size, length)
    flops, macs = utils.count_flops_and_macs(model, input_shape, str(device))
    flops_str, macs_str = utils.format_flops_macs(flops, macs)
    print(f'模型 FLOPs: {flops} ({flops_str})')
    print(f'模型 MACs: {macs} ({macs_str})')
    
    
    # 创建数据加载器
    print("创建测试数据加载器...")
    
    class OptCompat:
        def __init__(self, cfg):
            self.chunk_size = cfg.dataset['chunk_size']
            self.batch_size = 2  # 测试时使用较小batch size
            self.noisy_dirs_for_train = cfg.dataset['paths']['train']
            self.noisy_dirs_for_valid = cfg.dataset['paths']['valid']
            self.noisy_dirs_for_test = cfg.dataset['paths']['test']
    
    opt = OptCompat(cfg)
    test_loader = create_dataloader(opt, mode='test')
    print(f"测试集大小: {len(test_loader)} 批次")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建保存音频的子目录
    audio_dir = output_dir / "audio_results"
    audio_dir.mkdir(exist_ok=True)
    
    mix_dir = audio_dir / "mix"
    clean_dir = audio_dir / "clean"
    enhanced_dir = audio_dir / "enhanced"
    
    mix_dir.mkdir(exist_ok=True)
    clean_dir.mkdir(exist_ok=True)
    enhanced_dir.mkdir(exist_ok=True)
    
    # 打开测试日志文件
    log_file = output_dir / "test.log"
    log_fp = open(log_file, 'w')
    
    print("开始测试...")
    log_fp.write(f"模型测试开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fp.write(f"模型路径: {model_path}\n")
    log_fp.write(f"测试设备: {device}\n")
    log_fp.write(f"模型参数量: {total_params} ({total_params / 1000000.0:.2f} M)\n")
    log_fp.write(f"模型 FLOPs: {flops} ({flops_str})\n")
    log_fp.write(f"模型 MACs: {macs} ({macs_str})\n")
    log_fp.write(f"测试集大小: {len(test_loader)} 批次\n")
    log_fp.write("-" * 80 + "\n")
    
    # 初始化指标统计
    total_sisnr = 0.0
    total_snrseg = 0.0
    total_sdr = 0.0
    total_samples = 0
    
    # 获取损失函数用于计算指标
    loss_calculator = utils.get_loss(cfg)
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (targets, inputs) in enumerate(tqdm(test_loader, desc="测试进度")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 模型推理
            outputs = model(inputs)
            
            # 处理模型输出
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]  # 如果返回多个值，取第一个
            
            # 确保输出维度正确
            if hasattr(outputs, 'dim'):
                if outputs.dim() == 3 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)  # [B, 1, T] -> [B, T]
            else:
                # 如果是其他类型，尝试转换为tensor
                outputs = torch.tensor(outputs) if not isinstance(outputs, torch.Tensor) else outputs
            
            # 计算评估指标
            batch_sisnr = utils.cal_sisnri_batch(targets, outputs, inputs)
            batch_snrseg = utils.cal_snrsegi_batch(targets, outputs, inputs)
            batch_sdr = utils.cal_sdri_batch(targets, outputs, inputs)
            
            total_sisnr += batch_sisnr.item() * inputs.size(0)
            total_snrseg += batch_snrseg.item() * inputs.size(0)
            total_sdr += batch_sdr.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # 保存音频文件 (每个批次保存前几个样本)
            batch_size = inputs.size(0)
            for i in range(min(batch_size, 3)):  # 每批次最多保存3个样本
                sample_idx = batch_idx * batch_size + i
                
                # 转换为numpy数组并确保在CPU上
                mix_audio = inputs[i].cpu().numpy()
                clean_audio = targets[i].cpu().numpy()
                enhanced_audio = outputs[i].cpu().numpy()
                
                # 保存为.npy文件
                np.save(mix_dir / f"mix_{sample_idx:04d}.npy", mix_audio)
                np.save(clean_dir / f"clean_{sample_idx:04d}.npy", clean_audio)
                np.save(enhanced_dir / f"enhanced_{sample_idx:04d}.npy", enhanced_audio)
            
            # 记录批次结果
            if batch_idx % 10 == 0:  # 每10个批次记录一次
                avg_sisnr = total_sisnr / total_samples
                avg_snrseg = total_snrseg / total_samples
                avg_sdr = total_sdr / total_samples
                print(f"批次 [{batch_idx}/{len(test_loader)}] - "
                      f"当前平均SISNR: {avg_sisnr:.4f}, 当前平均SNRseg: {avg_snrseg:.4f}, 当前平均SDR: {avg_sdr:.4f}")
                
                log_fp.write(f"批次 [{batch_idx}/{len(test_loader)}] - "
                           f"SISNR: {batch_sisnr:.4f}, SNRseg: {batch_snrseg:.4f}, SDR: {batch_sdr:.4f}\n")
                log_fp.flush()
    
    # 计算最终平均指标
    avg_sisnr = total_sisnr / total_samples
    avg_snrseg = total_snrseg / total_samples
    avg_sdr = total_sdr / total_samples
    
    test_time = time.time() - start_time
    
    # 输出最终结果
    print("\n" + "="*80)
    print("测试完成!")
    print(f"测试样本数: {total_samples}")
    print(f"模型参数量: {total_params} ({total_params / 1000000.0:.2f} M)")
    print(f"模型 FLOPs: {flops_str}")
    print(f"模型 MACs: {macs_str}")
    print(f"平均 SI-SNR improvement: {avg_sisnr:.6f} dB")
    print(f"平均 SNRseg improvement: {avg_snrseg:.6f} dB")
    print(f"平均 SDR improvement: {avg_sdr:.6f} dB")
    print(f"测试总耗时: {test_time:.2f} 秒")
    print(f"每样本平均耗时: {test_time/total_samples:.4f} 秒")
    print("="*80)
    
    # 记录最终结果到日志
    log_fp.write("\n" + "="*80 + "\n")
    log_fp.write("测试完成!\n")
    log_fp.write(f"测试样本数: {total_samples}\n")
    log_fp.write(f"模型参数量: {total_params} ({total_params / 1000000.0:.2f} M)\n")
    log_fp.write(f"模型 FLOPs: {flops_str}\n")
    log_fp.write(f"模型 MACs: {macs_str}\n")
    log_fp.write(f"平均 SI-SNR improvement: {avg_sisnr:.6f} dB\n")
    log_fp.write(f"平均 SNRseg improvement: {avg_snrseg:.6f} dB\n")
    log_fp.write(f"平均 SDR improvement: {avg_sdr:.6f} dB\n")
    log_fp.write(f"测试总耗时: {test_time:.2f} 秒\n")
    log_fp.write(f"每样本平均耗时: {test_time/total_samples:.4f} 秒\n")
    log_fp.write(f"测试结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fp.write("="*80 + "\n")
    
    log_fp.close()
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"- 测试日志: {log_file}")
    print(f"- 音频结果: {audio_dir}")
    print(f"  - 混合音频: {mix_dir}")
    print(f"  - 干净音频: {clean_dir}")
    print(f"  - 增强音频: {enhanced_dir}")
    
    return avg_sisnr, avg_snrseg

def main():
    """主函数"""
    model = 'convtasnet'  # 默认模型名称
    parser = argparse.ArgumentParser(description='模型测试脚本')
    parser.add_argument('--config', type=str, default=f'config/shipsear/{model}.yaml', 
                       help='配置文件路径')
    parser.add_argument('--model', type=str, default=f'log/shipsear/{model.upper()}/models/best.pt',
                       help='模型权重文件路径')
    parser.add_argument('--output', type=str, default=f'test_results/{model.upper()}/test_unseennoise',
                       help='输出目录')
    
    args = parser.parse_args()
    
    
    # 加载配置
    if args.config is None:
        config_path = Config.get_default_config()
    else:
        config_path = args.config
    
    cfg = Config(config_path)
    
    print(f"配置文件: {config_path}")
    print(f"模型文件: {args.model}")
    print(f"输出目录: {args.output}")
    
    # 开始测试
    try:
        avg_sisnr, avg_snrseg = test_model(cfg, args.model, args.output)
        print(f"\n测试成功完成!")
        print(f"最终结果 - SISNR: {avg_sisnr:.4f} dB, SNRseg: {avg_snrseg:.4f} dB")
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
