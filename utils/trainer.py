import torch
from tqdm import tqdm
from .scores import cal_sisnri_batch, cal_sdri_batch, cal_snrsegi_batch


######################################################################################################################
#                                               train loss function                                                  #
######################################################################################################################
def unified_phase(model, data_loader, loss_calculator, optimizer, writer, epoch, device, cfg, phase='train', loss_type='time'):
    """
    统一训练/验证阶段
    phase: 'train' 或 'valid'
    """
    is_train = phase == 'train'
    model.train() if is_train else model.eval()
    total_loss = 0
    other_loss=[]
    metrics = {'sisnr':0,  'snrseg':0}
    
    with torch.set_grad_enabled(is_train):
        # 设置进度条描述，显示当前epoch和阶段
        phase_desc = f"Epoch {epoch:3d} [{phase.upper():5s}]"
        pbar = tqdm(data_loader, desc=phase_desc, ncols=100)
        
        for targets, inputs in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
                
            outputs = handle_model_output(model(inputs), loss_type)

            # 损失计算分支
            if loss_type == 'mrstft':
                sc_loss, mag_loss = loss_calculator[0](outputs, targets)
                mae_loss = loss_calculator[1](outputs, targets)
                loss = sc_loss + mag_loss + mae_loss
            elif loss_type == 'pit-sisnri':
                loss = loss_calculator(outputs, targets,inputs)
            elif loss_type in ['sisnr_timemag', 'adaptive_sisnr_timemag']:
                # 新的混合损失函数返回(loss, details)
                loss_result = loss_calculator(outputs, targets)
                if isinstance(loss_result, tuple):
                    loss, loss_details = loss_result
                    other_loss.append(loss_details)
                else:
                    loss = loss_result
            else:
                loss = loss_calculator(outputs, targets)
            
            # 反向传播和优化
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 记录结果
            total_loss += loss.item()
            
            # 验证阶段计算指标(sdr计算很慢)
            if not is_train:
                metrics['sisnr'] += cal_sisnri_batch(targets, outputs, inputs).item()
                # metrics['sdr'] += cal_sdri_batch(targets, outputs, inputs).item()
                metrics['snrseg'] += cal_snrsegi_batch(targets, outputs, inputs).item()

    # 统一处理日志记录
    avg_loss = total_loss / len(data_loader)
    if other_loss:
        avg_other_loss = {}
        if loss_type in ['sisnr_timemag', 'adaptive_sisnr_timemag'] :
            avg_other_loss['time_loss'] = 0
            avg_other_loss['mag_loss'] = 0
        for loss_dict in other_loss:
            for k, v in loss_dict.items():
                if k not in avg_other_loss:
                    avg_other_loss[k] = 0
                avg_other_loss[k] += v
        for k in avg_other_loss:
            avg_other_loss[k] /= len(other_loss)
        if writer and is_train:
            for key, value in avg_other_loss.items():
                writer.log_train_loss(f"loss/{key}", value, epoch)
    if is_train:
        if writer:
            writer.log_train_loss('total', avg_loss, epoch)
        return avg_loss
    else:
        avg_metrics = {k: v/len(data_loader) for k,v in metrics.items()}
        if writer:
            writer.log_valid_loss('total', avg_loss, epoch)
            for name, value in avg_metrics.items():
                writer.log_score(name.upper(), value, epoch)
            # writer.log_wav(inputs[0], targets[0], outputs[0], epoch)

        # 返回验证阶段期望的4个值
        return avg_loss, avg_metrics['sisnr'],avg_metrics['snrseg']


# 模型输出处理函数
def handle_model_output(outputs, loss_type):
    if isinstance(outputs, dict):  # HTDemucs格式处理
        return outputs['clean'].squeeze(1)
    elif isinstance(outputs, list):  # SepReformer格式处理
        return outputs[0]
    return outputs.squeeze(1)