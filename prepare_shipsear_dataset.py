#!/usr/bin/env python3
"""\
数据集合成与混合样本自动生成脚本 (Shipsear 数据集专用增强格式)
==================================================================

整体流程概览
------------
1. 读取 clean (目标/船舶) 原始 WAV：对每个文件按 `segment_length` 秒切片，步长为 `segment_step` 秒，生成大量重叠/非重叠片段，统一重采样到 `target_sr`。
2. 读取 noise 噪声 WAV：全部转换为单声道并保存为 `.npy` 以便快速随机访问。
3. Train/Valid 划分：对“clean 分段列表”做 8:2 随机划分（逐 segment 级别，而不是逐原文件）。
4. 混合样本生成：对每个 clean segment，按给定 SNR 序列 (snr_min→snr_max, 步长 snr_step) 随机抖动形成实际 SNR；从噪声集中随机抽一个噪声文件，再随机裁一段同长度片段，与 clean 线性混合得到 mixture，并保存 (clean / noise / mixture) 三份 `.npy`。
5. 样本命名：`<clean原文件名>_<segment_idx>_snr<目标SNR>` 例如 `ship001_0005_snr-05.00`。
6. 结果组织：
     output_dir/
             clean_npy/               # 所有 clean 的 .npy 片段及 clean_segments_info.csv
             noise_npy/               # 所有 noise 的 .npy 及 noise_files_info.csv
             train/ <sample_id>/clean.npy|noise.npy|mixture.npy
             valid/ <sample_id>/...
             mix_samples_info.csv     # 汇总所有生成样本的元信息（含实际 SNR ）


主要参数说明 (命令行参数)
------------------------
--clean_dir              原始 clean wav 路径。
--noise_dir              原始 noise wav 路径。
--output_dir             输出根目录，内部会创建 train/ valid/ 及 *_npy。
--snr_min / --snr_max    期望 SNR 范围 (dB)，闭区间；每个整档会再加入 (0, snr_step) 的随机抖动形成 `actual_snr`。
--snr_step               SNR 档位步长 (dB)。
--segment_length         对原始 clean 文件进行初次切片的窗口长度 (秒)。
--segment_step           初次切片窗口步长 (秒)，可 < segment_length 以形成重叠。
--clean_segment_length   最终用于混合的片段长度 (秒)。若 < segment_length 则会在切片内部再次随机截取；可不同于初次切片窗口减少冗余。
--target_sr              统一重采样采样率 (Hz)。
--num_workers            预留参数（当前大部分处理为串行/简单多进程占位，可后续扩展）。


使用示例
---------
python prepare_shipsear_dataset.py \\
    --clean_dir /path/clean \\
    --noise_dir /path/noise \\
    --output_dir /path/demucs_format_16k \\
    --snr_min -15 --snr_max 0 --snr_step 5 \\
    --segment_length 10 --segment_step 1 --clean_segment_length 5

"""

import os
import random
import numpy as np
import torch
import torchaudio as ta
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
from scipy.io import wavfile

def load_audio(file_path, target_sr=16000):
    """加载音频文件并重采样到目标采样率"""
    try:
        audio, sr = ta.load(file_path)
        if sr != target_sr:
            audio = ta.functional.resample(audio, sr, target_sr)
        return audio
    except Exception as e:
        print(f"无法加载音频文件 {file_path}: {e}")
        return None

def add_noise_to_clean(clean_audio, noise_audio, snr_db):
    """将噪声添加到干净信号中，达到指定的信噪比"""
    # 转换为numpy，但不进行标准化
    if isinstance(clean_audio, torch.Tensor):
        clean_audio_np = clean_audio.numpy()
    else:
        clean_audio_np = clean_audio.copy()
    
    if isinstance(noise_audio, torch.Tensor):
        noise_audio_np = noise_audio.numpy()
    else:
        noise_audio_np = noise_audio.copy()
    
    # 计算原始信号功率
    clean_power = np.mean(clean_audio_np ** 2)
    noise_power = np.mean(noise_audio_np ** 2)
    
    # 计算需要的噪声增益以达到指定SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_gain = np.sqrt(clean_power / (noise_power * snr_linear))
    
    # 调整噪声强度
    scaled_noise = noise_gain * noise_audio_np
    
    # 添加噪声
    noisy_audio = clean_audio_np + scaled_noise
    
    # 返回原始clean（不标准化）和调整后的噪声
    return noisy_audio, clean_audio_np, scaled_noise

def extract_random_segment(audio, segment_length_seconds, sample_rate):
    """从音频中随机提取指定长度的片段"""
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    
    segment_length_samples = int(segment_length_seconds * sample_rate)
    audio_length = audio.shape[1] if len(audio.shape) > 1 else len(audio)
    
    if audio_length <= segment_length_samples:
        # 如果音频长度不足，重复音频直到足够长
        repeats = int(np.ceil(segment_length_samples / audio_length)) + 1
        audio = np.tile(audio, repeats)
        audio_length = len(audio)
    
    # 随机选择起始位置
    max_start = audio_length - segment_length_samples
    start_pos = random.randint(0, max_start)
    end_pos = start_pos + segment_length_samples
    
    # 提取片段
    segment = audio[start_pos:end_pos] if len(audio.shape) == 1 else audio[:, start_pos:end_pos]
    return segment, start_pos, end_pos

def save_numpy_audio(path, audio_data):
    """保存音频数据为numpy格式"""
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.numpy()
    np.save(str(path), audio_data)


def collect_and_save_clean_audio(clean_dir, output_dir, segment_length=10, segment_step=2, target_sr=16000, num_workers=None):
    """收集并保存干净音频文件为npy格式，先分段再随机划分训练验证集"""
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    # 创建clean音频保存目录
    clean_npy_dir = Path(output_dir) / "clean_npy"
    clean_npy_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在分段信息文件
    csv_path = clean_npy_dir / "clean_segments_info.csv"
    if csv_path.exists():
        print("检测到已存在的分段信息文件，直接加载...")
        clean_csv = pd.read_csv(csv_path)
        return clean_csv
    
    # 收集所有需要处理的文件
    all_files = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]
    
    # 创建所有分段的信息列表
    all_segments = []
    
    print("生成所有分段信息...")
    for file_name in tqdm(all_files, desc="处理音频文件分段"):
        # 加载音频
        file_path = Path(clean_dir) / file_name
        audio = load_audio(file_path, target_sr)
        
        # 生成分段文件名
        file_stem = Path(file_name).stem.replace('.wav', '')
        segment_dir = clean_npy_dir / file_stem
        segment_dir.mkdir(parents=True, exist_ok=True)
        
        if audio is None:
            continue
        
        # 转换为单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 按segment_length分段，使用segment_step作为步长
        audio_length_seconds = audio.shape[1] / target_sr
        segment_length_samples = int(segment_length * target_sr)
        segment_step_samples = int(segment_step * target_sr)
        
        # 计算可以生成的分段数量
        max_start_sample = audio.shape[1] - segment_length_samples
        if max_start_sample < 0:
            continue  # 音频太短，跳过
        
        # 生成所有可能的分段起始位置
        start_positions = list(range(0, max_start_sample + 1, segment_step_samples))
        
        # 为每个分段创建独立的信息
        for seg_idx, start_sample in enumerate(start_positions):
            end_sample = start_sample + segment_length_samples
            segment_audio = audio[:, start_sample:end_sample]
            
           
            #保存为npy文件

            npy_path = clean_npy_dir / f"{file_stem}/{seg_idx:04d}.npy"

            save_numpy_audio(npy_path, segment_audio)

            segment_info = {
                'save_path': str(npy_path),
                'original_file': file_stem,
                'segment_idx': seg_idx,
            }
            all_segments.append(segment_info)
    
    print(f"总共生成 {len(all_segments)} 个分段")
    #保存为csv文件
    clean_csv = pd.DataFrame(all_segments)
    clean_csv.to_csv(clean_npy_dir / "clean_segments_info.csv", index=False)
    return clean_csv

def get_and_save_noise_files(noise_dir, output_dir, target_sr=16000):
    """获取噪声文件并保存为npy格式"""
    noise_files = list(Path(noise_dir).glob("*.wav"))
    print(f"找到 {len(noise_files)} 个噪声文件")
    
    # 创建噪声保存目录
    noise_npy_dir = Path(output_dir) / "noise_npy"
    noise_npy_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在噪声信息文件
    csv_path = noise_npy_dir / "noise_files_info.csv"
    if csv_path.exists():
        print("检测到已存在的噪声信息文件，直接加载...")
        noise_csv = pd.read_csv(csv_path)
        return noise_csv
    
    saved_noise_files = []
    
    for noise_file in tqdm(noise_files, desc="保存噪声文件"):
        # 加载噪声音频
        audio = load_audio(noise_file, target_sr)
        if audio is None:
            continue
        
        # 转换为单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 生成保存路径
        npy_filename = f"{noise_file.stem}.npy"
        npy_path = noise_npy_dir / npy_filename
        
        # 保存为npy
        save_numpy_audio(npy_path, audio)
        
        saved_noise_files.append({
            'save_path': str(npy_path),
            'original_path': str(noise_file),
            'duration': audio.shape[1] / target_sr
        })
    
    print(f"成功保存 {len(saved_noise_files)} 个噪声文件")
    # 保存噪声文件信息
    noise_csv = pd.DataFrame(saved_noise_files)
    noise_csv.to_csv(csv_path, index=False)
    return noise_csv


def fast_save_wav(path, audio_tensor, sr):
    audio_np = audio_tensor.squeeze().cpu().numpy()
    wavfile.write(str(path), sr, audio_np)

def create_training_samples_from_npy(clean_info_csv, noise_csv, output_dir, 
                                    segment_length=10, segment_step=2, clean_segment_length=5, target_sr=16000, 
                                    snr_range=(-20, 10), snr_step=5):
    """从npy分段文件创建训练样本"""
    
    # 创建输出目录
    train_dir = Path(output_dir) / "train"
    valid_dir = Path(output_dir) / "valid"
    if train_dir.exists() or valid_dir.exists():
        import shutil
        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(valid_dir, ignore_errors=True)
    train_dir.mkdir(parents=True)
    valid_dir.mkdir(parents=True)

    # 随机划分训练集和验证集
    train_segments = clean_info_csv.sample(frac=0.8, random_state=42)
    valid_segments = clean_info_csv.drop(train_segments.index)
    
    print(f"训练集分段数: {len(train_segments)}")
    print(f"验证集分段数: {len(valid_segments)}")
    
    # 用于保存所有生成的混合样本信息
    all_mix_samples = []

    def process_segment_set(segments_df, output_dir, set_name):
        """处理分段集合（训练集或验证集）"""
        sample_counter = 0
        mix_samples = []
        
        for idx, segment_info in tqdm(segments_df.iterrows(), total=len(segments_df), desc=f"处理{set_name}"):
            # 加载clean分段npy文件
            try:
                clean_segment = np.load(segment_info['save_path'])
                if len(clean_segment.shape) == 1:
                    clean_segment = clean_segment.reshape(1, -1)
            except Exception as e:
                print(f"无法加载clean分段 {segment_info['save_path']}: {e}")
                continue
            
            # 如果需要更短的片段，从segment中随机提取
            if clean_segment_length < segment_length:
                clean_segment, _, _ = extract_random_segment(
                    clean_segment, clean_segment_length, target_sr)
            
            # 为每个SNR级别生成样本
            for snr_db in range(snr_range[0], snr_range[1] + 1, snr_step):
                snr = snr_db + np.random.random() * snr_step
            
                # 随机选择噪声文件并提取片段
                noise_info = noise_csv.sample(n=1).iloc[0]  # 随机选择一个噪声文件
                try:
                    noise_audio = np.load(noise_info['save_path'])
                    if len(noise_audio.shape) == 1:
                        noise_audio = noise_audio.reshape(1, -1)
                except Exception as e:
                    print(f"无法加载噪声音频 {noise_info['save_path']}: {e}")
                    continue
                
                # 从噪声中随机提取片段
                noise_segment, _, _ = extract_random_segment(
                    noise_audio, clean_segment_length, target_sr)
                
                # 添加噪声（包含标准化）
                noisy_audio, clean_final, noise_final = add_noise_to_clean(
                    clean_segment, noise_segment, snr)
                
               
                # 创建样本ID
                sample_id = f"{segment_info['original_file']}_{segment_info['segment_idx']:04d}_snr{snr_db:+06.2f}"
                
                # 创建样本目录
                sample_dir = output_dir / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存音频数据
                save_numpy_audio(sample_dir / "clean.npy", clean_final)
                save_numpy_audio(sample_dir / "noise.npy", noise_final)
                save_numpy_audio(sample_dir / "mixture.npy", noisy_audio)

           
                
                # 记录样本信息
                sample_info = {
                    'sample_id': sample_id,
                    'set_type': set_name,
                    'clean_path': str(sample_dir / "clean.npy"),
                    'noise_path': str(sample_dir / "noise.npy"),
                    'mixture_path': str(sample_dir / "mixture.npy"),
                    'original_clean_file': segment_info['original_file'],
                    'segment_idx': segment_info['segment_idx'],
                    'original_noise_file': Path(noise_info['save_path']).stem,
                    'snr_db': snr_db,
                    'actual_snr': snr,
                    'duration_seconds': clean_segment_length
                }
                mix_samples.append(sample_info)
                
                sample_counter += 1
        
        print(f"生成了 {sample_counter} 个样本")
        return sample_counter, mix_samples
    
    # 处理训练集和验证集
    print("处理训练集...")
    train_count, train_mix_samples = process_segment_set(train_segments, train_dir, "train")
    all_mix_samples.extend(train_mix_samples)
    
    print("处理验证集...")
    valid_count, valid_mix_samples = process_segment_set(valid_segments, valid_dir, "valid")
    all_mix_samples.extend(valid_mix_samples)
    
    # 保存混合样本信息到CSV
    mix_csv = pd.DataFrame(all_mix_samples)
    mix_csv_path = Path(output_dir) / "mix_samples_info.csv"
    mix_csv.to_csv(mix_csv_path, index=False)
    print(f"混合样本信息已保存到: {mix_csv_path}")
    print(f"总共生成 {len(all_mix_samples)} 个混合样本")
    
    return len(all_mix_samples)



def main():
    parser = argparse.ArgumentParser(description="预处理船舶数据集 - 优化版本")
    parser.add_argument("--clean_dir", type=str, 
                       default="data/clean",
                       help="包含船舶音频的目录")
    parser.add_argument("--noise_dir", type=str,
                       default="data/noise",
                       help="噪声文件目录")
    parser.add_argument("--output_dir", type=str,
                       default="data/data_format_16k",
                       help="输出目录")
    parser.add_argument("--snr_min", type=float, default=-15,
                       help="最小信噪比(dB)")
    parser.add_argument("--snr_max", type=float, default=0,
                       help="最大信噪比(dB)")
    parser.add_argument("--snr_step", type=int, default=5,
                       help="信噪比间隔(dB)")
    parser.add_argument("--segment_length", type=float, default=10,
                       help="clean音频分段长度(秒)")
    parser.add_argument("--segment_step", type=float, default=1,
                       help="分段步长(秒)，用于控制分段之间的重叠")
    parser.add_argument("--clean_segment_length", type=float, default=5,
                       help="最终训练样本长度(秒)，从seg")
    parser.add_argument("--target_sr", type=int, default=16000,
                       help="目标采样率")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="并行进程数")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("开始预处理船舶数据集（新版本）...")
    print(f"干净音频目录: {args.clean_dir}")
    print(f"噪声目录: {args.noise_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"信噪比范围: {args.snr_min}dB 到 {args.snr_max}dB，间隔 {args.snr_step}dB")
    print(f"clean分段长度: {args.segment_length}秒")
    print(f"分段步长: {args.segment_step}秒")
    print(f"最终样本长度: {args.clean_segment_length}秒")
    print(f"采样率: {args.target_sr}Hz")
    print(f"并行进程数: {args.num_workers}")
    
    start_time = time.time()
    
    
    
    # 步骤1: 收集clean音频文件信息并保存为npy格式，按长度排序划分训练验证集
    print("\n步骤1: 收集并保存clean音频文件...")
    clean_info_csv  = collect_and_save_clean_audio(
        args.clean_dir,
        args.output_dir,
        segment_length=args.segment_length,
        segment_step=args.segment_step,
        target_sr=args.target_sr,
        num_workers=args.num_workers
    )

    # 步骤2: 保存噪声文件为npy格式
    print("\n步骤2: 保存噪声文件...")
    noise_info_csv = get_and_save_noise_files(args.noise_dir, args.output_dir, args.target_sr)

    # 步骤3: 创建训练样本
    print("\n步骤3: 创建训练样本...")
    all_samples = create_training_samples_from_npy(
        clean_info_csv,
        noise_info_csv,
        args.output_dir,
        segment_length=args.segment_length,
        segment_step=args.segment_step,
        clean_segment_length=args.clean_segment_length,
        target_sr=args.target_sr,
        snr_range=(args.snr_min, args.snr_max),
        snr_step=args.snr_step
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n数据集预处理完成！总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

if __name__ == "__main__":
    main() 