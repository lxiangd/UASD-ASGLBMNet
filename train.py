#!/usr/bin/env python3
"""
多GPU训练脚本 - 使用DataParallel
"""

import os
import argparse
import warnings
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import utils
import random
import numpy as np
import time
from dataloader import create_dataloader
from config.config import Config
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

# 抑制PyTorch弃用警告
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.parallel")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

def train_worker(cfg):
    """训练工作函数"""
    
    # 设置GPU设备
    if 'gpu_ids' in cfg.train:
        gpu_ids = cfg.train['gpu_ids']
        # 支持int、str、list三种格式
        if isinstance(gpu_ids, (str)):
            if ',' in gpu_ids:
                gpu_ids = [int(i) for i in gpu_ids.split(',')]
            else:
                gpu_ids = [int(gpu_ids)]
        elif isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        elif isinstance(gpu_ids, (list, tuple)):
            gpu_ids = [int(i) for i in gpu_ids]
        if gpu_ids and torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_ids[0]}')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(cfg.model['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    # 创建模型
    print("创建模型...")
    model = utils.get_arch(cfg)
    model = model.to(device)
    dataset_name = cfg.dataset.get('name', 'default_dataset')

     # 创建日志目录
    log_dir = cfg.output['log_dir']
    model_dir = cfg.output['model_dir']
    
    utils.mkdir(log_dir)
    utils.mkdir(model_dir)

    # 加载预训练模型（如果配置中启用）
    start_epoch = 1  # 默认从第1个epoch开始
    if hasattr(cfg, 'pretrain') and cfg.pretrain.get('enabled', False):
        # 自动构建预训练模型路径
        if 'model_path' in cfg.pretrain and cfg.pretrain['model_path']:
            # 如果手动指定了路径，使用指定路径
            pretrain_path = cfg.pretrain['model_path']
        else:
            # 自动构建路径：使用相同架构的best.pt
            pretrain_path = os.path.join(model_dir, "best.pt")
        
        if os.path.exists(pretrain_path):
            print(f"加载预训练模型: {pretrain_path}")
            checkpoint = torch.load(pretrain_path, map_location=device)
            # 获取实际的模型（去掉DataParallel包装）
            model_to_load = model.module if hasattr(model, 'module') else model
            
            # 过滤掉不需要的键（如thop添加的total_ops, total_params等）
            model_state_dict = checkpoint['model']
            filtered_state_dict = {}
            for key, value in model_state_dict.items():
                if not (key.endswith('.total_ops') or key.endswith('.total_params')):
                    filtered_state_dict[key] = value
            
            model_to_load.load_state_dict(filtered_state_dict, strict=False)
            
            # 加载epoch信息
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
                print(f"从epoch {start_epoch}继续训练")
            
            print("预训练模型加载成功!")
        else:
            print(f"警告：预训练模型文件不存在: {pretrain_path}")
            print("将从头开始训练")

    if len(gpu_ids) > 1:
        print(f"检测到 {len(gpu_ids)} 个GPU，使用DataParallel")
        
      
        model = DataParallel(model,device_ids=gpu_ids)

        
      
    else:
        print("使用单GPU训练")
    
    total_params = utils.cal_total_params(model.module if hasattr(model, 'module') else model)
    print(f'模型总参数: {total_params} ({total_params / 1000000.0:.2f} M)')
    
    # 计算 FLOPs 和 MACs
    print("计算模型 FLOPs 和 MACs...")
    try:
        # 使用和数据加载器相同的输入形状
        input_shape = (1, cfg.dataset['chunk_size'])  # (batch_size, length)
        actual_model = model.module if hasattr(model, 'module') else model
        flops, macs = utils.count_flops_and_macs(actual_model, input_shape, str(device))
        flops_str, macs_str = utils.format_flops_macs(flops, macs)
        print(f'模型 FLOPs: {flops} ({flops_str})')
        print(f'模型 MACs: {macs} ({macs_str})')
    except Exception as e:
        print(f"计算 FLOPs 和 MACs 时出错: {e}")
        flops, macs = 0, 0
        flops_str, macs_str = "N/A", "N/A"
    
    # 不再强制 model.cuda()，避免与 DataParallel 设备列表不一致
    
    # 定义损失类型和优化器
    trainer, validator = utils.get_train_mode(cfg)
    loss_calculator = utils.get_loss(cfg)
    
    # 获取优化器配置
    opt_cfg = cfg.train['optimizer']
    optimizer_name = opt_cfg['name']
    optimizer_params = opt_cfg[optimizer_name]
    
    # 创建优化器
    optimizer_class = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_class(
        model.parameters(),
        **optimizer_params
    )
    
    # 如果加载了预训练模型，还需要加载优化器状态
    if hasattr(cfg, 'pretrain') and cfg.pretrain.get('enabled', False):
        # 重新构建预训练模型路径以加载优化器状态
        if 'model_path' in cfg.pretrain and cfg.pretrain['model_path']:
            pretrain_path = cfg.pretrain['model_path']
        else:
            pretrain_path = os.path.join("log", cfg.model['arch'], "models", "best.pt")
        
        if os.path.exists(pretrain_path):
            checkpoint = torch.load(pretrain_path, map_location=device)
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("优化器状态加载成功!")
                except Exception as e:
                    print(f"优化器状态加载失败，使用默认状态: {e}")
    
    # 获取调度器配置
    sched_cfg = cfg.train['scheduler']
    scheduler_name = sched_cfg['name'] 
    scheduler_params = sched_cfg[scheduler_name]
    
    # 创建学习率调度器
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
    scheduler = scheduler_class(optimizer, **scheduler_params)
    
    # 创建数据加载器
    print("创建数据加载器...")
    
    # 创建兼容的opt对象
    class OptCompat:
        def __init__(self, cfg):
            # 从cfg中提取所需的属性
            self.chunk_size = cfg.dataset['chunk_size']
            self.batch_size = cfg.train['batch_size']
            self.noisy_dirs_for_train = cfg.dataset['paths']['train']
            self.noisy_dirs_for_valid = cfg.dataset['paths']['valid']
            self.noisy_dirs_for_test = cfg.dataset['paths']['test']
    
    opt = OptCompat(cfg)
    train_loader = create_dataloader(opt, mode='train')
    valid_loader = create_dataloader(opt, mode='valid')
    
    print(f"训练集大小: {len(train_loader)} 批次")
    print(f"验证集大小: {len(valid_loader)} 批次")
    
   
    
    # 初始化SwanLab Logger
    swanlab_config = {
        'project_name':  f"{cfg.output['swanlab']['project_name']}",
        'run_dir': f"{cfg.model['arch']}",
        'swanlog_dir': log_dir,

    }


    # 准备swanlab配置
    swanlab_train_config = {
        # 'model': cfg.model['arch'],
        # 'batch_size': cfg.train['batch_size'],
        # 'epochs': cfg.train['epochs'],
        # 'gpu_ids': cfg.train['gpu_ids'],
        # 'learning_rate': cfg.train['optimizer'][cfg.train['optimizer']['name']]['lr'],
        # 'chunk_size': cfg.dataset['chunk_size'],
        # 'dataset': cfg.dataset['name'],
        'total_params_M': round(total_params / 1000000.0, 2),
        'flops': flops_str,
        'macs': macs_str,
        'full_config': cfg.config  # 使用config属性获取配置字典
    }

    writer = utils.Writer(project_name=swanlab_config['project_name'], run_dir=swanlab_config['run_dir'], swanlog_dir=swanlab_config['swanlog_dir'], config=swanlab_train_config)
    
    train_log_fp = open(os.path.join(model_dir, 'train_log.txt'), 'a')
    
    print(f"日志目录: {log_dir}")
    print(f"当前时间: {cfg.timestamp}")
    
    # 开始训练
    best_sisnr = float('-inf')
    print('开始训练...')
    
    # 如果加载了预训练模型，同步加载最佳SISNR
    if hasattr(cfg, 'pretrain') and cfg.pretrain.get('enabled', False):
        if 'model_path' in cfg.pretrain and cfg.pretrain['model_path']:
            pretrain_path = cfg.pretrain['model_path']
        else:
            pretrain_path = os.path.join("log", cfg.model['arch'], "models", "best.pt")
        
        if os.path.exists(pretrain_path):
            checkpoint = torch.load(pretrain_path, map_location=device)
            if 'best_sisnr' in checkpoint:
                best_sisnr = checkpoint['best_sisnr']
                print(f"加载最佳SISNR: {best_sisnr:.6f}")
    
    patience = cfg.train.get('early_stop_patience', cfg.train['early_stop_patience'])  # 早停容忍度
    epochs_no_improve = 0
    stop_training = False

    for epoch in range(start_epoch, cfg.train['epochs'] + 1):
        if stop_training:
            print(f"早停触发，{patience}个epoch未提升，提前终止训练。")
            break
        st_time = time.time()

        # 训练
        train_loss = trainer(model, train_loader, loss_calculator, optimizer,
                           writer, epoch, device, cfg)

        # 验证
        valid_loss, sisnr, snrseg = validator(model, valid_loader, loss_calculator,
                                                None, writer, epoch, device, cfg)

        # 保存最佳模型
        if sisnr > best_sisnr:
            best_sisnr = sisnr
            save_path = os.path.join(model_dir, f'chkpt_{epoch}.pt')
            best_path = os.path.join(model_dir, 'best.pt')
            
            # 保存模型时获取实际的模型（去掉DataParallel包装）
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_data = {
                'model': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_sisnr': best_sisnr
            }
            
            # 保存epoch checkpoint
            torch.save(checkpoint_data, save_path)
            print(f"保存最佳模型: {save_path}")
            
            # 保存best.pt（用于预训练加载）
            torch.save(checkpoint_data, best_path)
            print(f"更新最佳模型: {best_path}")
            
            epochs_no_improve = 0
            
            # 记录最佳模型到swanlab
            if writer:
                writer.log_model(save_path, epoch)
        else:
            epochs_no_improve += 1
            print(f"SISNR未提升，已连续{epochs_no_improve}个epoch未提升")
            if epochs_no_improve >= patience:
                stop_training = True

        epoch_time = time.time() - st_time
        print('EPOCH[{}] T {:.6f} |  V {:.6f}  takes {:.3f} seconds'
                .format(epoch, train_loss, valid_loss, epoch_time))
        print('SISNRi {:.6f} |  SNRsegi {:.6f}'.format(sisnr,  snrseg))

        # 写入日志
        train_log_fp.write('EPOCH[{}] T {:.6f} |  V {:.6f}  takes {:.3f} seconds\n'
                            .format(epoch, train_loss, valid_loss, epoch_time))
        train_log_fp.write('SISNRi {:.6f} |  SNRsegi {:.6f}\n'.format(sisnr,  snrseg))
        train_log_fp.flush()

        # 更新学习率
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss)  # 使用验证损失作为指标
        else:
            scheduler.step()  # 其他调度器直接step

    # 训练完成
    print('训练完成!')
    train_log_fp.close()
    
    # 关闭swanlab
    if writer:
        writer.finish()

    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'final_model.pt')
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'model': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch if stop_training else cfg.train['epochs'],
        'best_sisnr': best_sisnr
    }, final_model_path)
    print(f"保存最终模型: {final_model_path}")
    
    # 记录最终模型到swanlab
    if writer:
        writer.log_model(final_model_path, epoch if stop_training else cfg.train['epochs'])

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/shipsear/ASGLBMNet.yaml', help='配置文件路径 (可选, 若提供 --model 将自动推断)')
    parser.add_argument('--pretrain', type=str, default=False, help='启用预训练模型加载 (true/false)')

    args = parser.parse_args()

    # 选择配置

    config_path = args.config or Config.get_default_config()

    cfg = Config(config_path)

    # 处理预训练开关
    if args.pretrain == 'true' or args.pretrain is True:
        if not hasattr(cfg, 'pretrain'):
            cfg.pretrain = {'enabled': True}
        else:
            cfg.pretrain['enabled'] = True
        print('通过命令行参数启用预训练模型加载')

    train_worker(cfg)

if __name__ == "__main__":
    main() 