import swanlab
import os
import shutil

class SwanLabLogger:
    def __init__(self, project_name="underwater_denoising", run_dir=None, swanlog_dir=None,config=None):
        """
        初始化SwanLab Logger
        
        Args:
            project_name: swanlab项目名称
            run_name: 运行名称，如果不指定会自动生成
            config: 训练配置，会被记录到swanlab
        """
        self.project_name = project_name
        self.run_name = run_dir
        
        # 初始化swanlab
        swanlab.init(
            project=project_name,
            name=run_dir,  # 使用正确的参数名
            config=config
        )
        
    def log_train_loss(self, loss_type, train_loss, step):
        """记录训练损失"""
        swanlab.log({f'train_{loss_type}_loss': train_loss}, step=step)

    def log_valid_loss(self, loss_type, valid_loss, step):
        """记录验证损失"""
        swanlab.log({f'valid_{loss_type}_loss': valid_loss}, step=step)

    def log_score(self, metrics_name, metrics, step):
        """记录评估指标"""
        swanlab.log({metrics_name: metrics}, step=step)

    def log_wav(self, noisy_wav, clean_wav, enhanced_wav, step):
        """记录音频文件（可选功能）"""
        # swanlab支持音频记录，但需要将tensor转换为numpy
        try:
            import numpy as np
            
            # 转换为numpy数组
            if hasattr(noisy_wav, 'cpu'):
                noisy_wav = noisy_wav.cpu().numpy()
            if hasattr(clean_wav, 'cpu'):
                clean_wav = clean_wav.cpu().numpy()
            if hasattr(enhanced_wav, 'cpu'):
                enhanced_wav = enhanced_wav.cpu().numpy()
                
            swanlab.log({
                'noisy_wav': swanlab.Audio(noisy_wav, sample_rate=16000),
                'clean_target_wav': swanlab.Audio(clean_wav, sample_rate=16000),
                'enhanced_wav': swanlab.Audio(enhanced_wav, sample_rate=16000)
            }, step=step)
        except Exception as e:
            print(f"Warning: Could not log audio files: {e}")

    def log_model(self, model_path, epoch):
        """记录模型文件"""
        try:
            # 获取swanlab的运行目录
            run_dir = swanlab.get_run_dir()
            if run_dir:
                # 定义模型保存的目标目录
                save_dir = os.path.join(run_dir, "files", "models")
                os.makedirs(save_dir, exist_ok=True)
                
                # 复制模型文件
                shutil.copy(model_path, save_dir)
                print(f"模型已复制到SwanLab目录: {os.path.join(save_dir, os.path.basename(model_path))}")
                
                swanlab.log({'saved_model_epoch': epoch}, step=epoch)
            else:
                print("Warning: SwanLab运行目录未找到，无法保存模型。")
        except Exception as e:
            print(f"Warning: Could not save model to swanlab: {e}")
    
    def finish(self):
        """结束swanlab记录"""
        swanlab.finish()

# 为了兼容性，创建一个Writer类的别名
Writer = SwanLabLogger
