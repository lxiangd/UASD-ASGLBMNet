from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import random

class DeepShipDataset(Dataset):
    """船舶声音数据集加载器 - 适配新的数据格式"""
    
    def __init__(self, opt, mode='train'):
        """
        初始化数据集
        
        参数:
            opt: 配置参数
            mode: 'train', 'valid' 或 'test'
        """
        self.chunk_size = opt.chunk_size
        self.mode = mode
        
        # 确定数据目录
        if mode == 'train':
            data_dir = opt.noisy_dirs_for_train
        elif mode == 'valid':
            data_dir = opt.noisy_dirs_for_valid
        else:
            data_dir = opt.noisy_dirs_for_test
            
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"数据目录不存在: {self.data_dir}")
        
        # 检查segments目录是否存在
        segments_dir = self.data_dir 
        if not segments_dir.exists():
            raise ValueError(f"segments目录不存在: {segments_dir}")
        
        # 获取所有片段目录
        self.segment_dirs = []
        for segment_dir in segments_dir.iterdir():
            if segment_dir.is_dir():
                # 检查是否包含所需的所有文件
                clean_file = segment_dir / "clean.npy"
                noise_file = segment_dir / "noise.npy"
                mixture_file = segment_dir / "mixture.npy"
                
                if clean_file.exists() and noise_file.exists() and mixture_file.exists():
                    self.segment_dirs.append(segment_dir)
        
        if not self.segment_dirs:
            raise ValueError(f"在{segments_dir}中没有找到有效的片段数据")
        
        # 读取数据集信息文件（如果存在）
        info_file = self.data_dir.parent / "dataset_info.csv"
        self.info_dict = {}
        if info_file.exists():
            self.info_df = pd.read_csv(info_file)
            # 创建查找字典
            for _, row in self.info_df.iterrows():
                segment_id = row['segment_id']
                self.info_dict[segment_id] = {
                    'snr_db': row['snr_db'],
                    'ship_type': row['ship_type'],
                    'original_file': row['original_file'],
                    'segment_idx': row['segment_idx'],
                    'duration': row['duration'],
                    'sample_rate': row['sample_rate'],
                    'noise_file': row['noise_file']
                }
        
        print(f"<{mode.capitalize()} dataset>")
        print(f"Found {len(self.segment_dirs)} segment directories")
        print(f"Data directory: {self.data_dir}")
    
    def _load_audio_file(self, file_path):
        """加载numpy格式的音频文件"""
        try:
            audio_data = np.load(str(file_path))
            # #标准化
            # audio_data = (audio_data - audio_data.mean()) / (audio_data.std() + 1e-5)
            # 确保音频是1D数组
            if len(audio_data.shape) > 1:
                audio_data = audio_data.squeeze()
            return audio_data
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None
    
    def _process_audio(self, audio_data):
        """处理音频数据，确保长度符合要求"""
        if audio_data is None:
            # 如果加载失败，返回零信号
            return np.zeros(self.chunk_size)
        
        # 确保音频长度符合要求
        if len(audio_data) < self.chunk_size:
            # 如果音频太短，循环填充
            audio_data = np.pad(audio_data, (0, self.chunk_size - len(audio_data)), mode='wrap')
        elif len(audio_data) > self.chunk_size:
            # 如果音频太长，随机截取
            start = random.randint(0, len(audio_data) - self.chunk_size)
            audio_data = audio_data[start:start + self.chunk_size]
        # # 归一化
        # audio_data = audio_data / np.max(np.abs(audio_data))
        return audio_data
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.segment_dirs)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        
        segment_dir = self.segment_dirs[idx]
        segment_id = segment_dir.name
        
        # 加载音频数据
        clean_path = segment_dir / "clean.npy"
        noise_path = segment_dir / "noise.npy"
        mixture_path = segment_dir / "mixture.npy"
        
        # 检查文件是否存在
        if not clean_path.exists() or not noise_path.exists() or not mixture_path.exists():
            print(f"警告: 文件缺失 in {segment_dir}")
            # 返回零数据
            zero_data = np.zeros(self.chunk_size)
            return torch.from_numpy(zero_data).float(), torch.from_numpy(zero_data).float()
            
        clean = self._load_audio_file(clean_path)
        noise = self._load_audio_file(noise_path)
        mixture = self._load_audio_file(mixture_path)
        
        # 检查数据是否为空
        if clean is None or noise is None or mixture is None:
            print(f"警告: 数据加载失败 in {segment_dir}")
            zero_data = np.zeros(self.chunk_size)
            return torch.from_numpy(zero_data).float(), torch.from_numpy(zero_data).float()
            
        # # 处理音频数据
        # clean = self._process_audio(clean_data)
        # noise = self._process_audio(noise_data)
        # mixture = self._process_audio(mixture_data)
        
        # 检查数据是否包含NaN或无穷大
        if np.any(np.isnan(clean)) or np.any(np.isnan(mixture)):
            print(f"警告: 数据包含NaN in {segment_dir}")
            zero_data = np.zeros(self.chunk_size)
            return torch.from_numpy(zero_data).float(), torch.from_numpy(zero_data).float()
        
        if np.any(np.isinf(clean)) or np.any(np.isinf(mixture)):
            print(f"警告: 数据包含无穷大 in {segment_dir}")
            zero_data = np.zeros(self.chunk_size)
            return torch.from_numpy(zero_data).float(), torch.from_numpy(zero_data).float()
            
        # 转换为tensor
        clean = torch.from_numpy(clean).float()
        noise = torch.from_numpy(noise).float()
        mixture = torch.from_numpy(mixture).float()
        
        
        # 返回训练器期望的格式：(targets, inputs)
        # targets: 干净信号 (clean)
        # inputs: 带噪信号 (mixture)
        return clean, mixture
            
      

def create_dataloader(opt, mode='train'):
    """
    创建数据加载器
    
    参数:
        opt: 配置参数
        mode: 'train', 'valid' 或 'test'
    """
    dataset = DeepShipDataset(opt, mode)
    
    # 训练模式使用随机采样，验证和测试模式按顺序采样
    shuffle = (mode == 'train')
    
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

