from models import get_registered_model, MODEL_REGISTRY

# ----------------------------- Model Init ----------------------------- #
def get_arch(cfg):
    arch = cfg.model['arch']
    print(f'You choose {arch} ...')

    # 允许 arch 中带有 condition 后缀 (例如 ASGLBMNet_no_gcfn)
    base_arch = arch.split('_')[0]
    model_cls = get_registered_model(base_arch)
    if model_cls is None:
        raise Exception(f"Arch error! 未注册的模型: {base_arch}. 请在 models/__init__.py 中注册或使用 @register_model 装饰器。")

    # 支持两种初始化方式: 1) 直接传 cfg  2) 传拆分好的参数 (如 CONVTASNET)
    if base_arch.upper() == 'CONVTASNET' and base_arch in cfg.model:
        model = model_cls(**cfg.model[base_arch])
    else:
        # 默认把整个 cfg 传入 (当前 ASGLBMNet 如此)
        model = model_cls(cfg)
    return model


# ------------------------- Trainer / Validator ------------------------ #
def get_train_mode(cfg):
    # loss_type: 既支持 train.loss 也支持 model.loss.type
    if 'loss' in cfg.train:
        loss_type = cfg.train['loss']
    elif 'loss' in cfg.model:
        loss_type = cfg.model['loss']['type']
    else:
        raise Exception('Loss type not found in config!')

    print(f'You choose loss type {loss_type} ...')
    from .trainer import unified_phase

    def trainer(model, data_loader, loss_calculator, optimizer, writer, epoch, device, cfg, **kwargs):
        return unified_phase(model, data_loader, loss_calculator, optimizer, writer, epoch, device, cfg, phase='train', loss_type=loss_type, **kwargs)

    def validator(model, data_loader, loss_calculator, optimizer, writer, epoch, device, cfg, **kwargs):
        return unified_phase(model, data_loader, loss_calculator, optimizer, writer, epoch, device, cfg, phase='valid', loss_type=loss_type, **kwargs)

    return trainer, validator


# ------------------------------ Loss Init ----------------------------- #
LOSS_REGISTRY = {}

def register_loss(name: str):
    def decorator(fn_or_cls):
        LOSS_REGISTRY[name.lower()] = fn_or_cls
        return fn_or_cls
    return decorator

# 预注册基础 loss
import torch
from torch.nn import L1Loss
from torch.nn.functional import mse_loss as _mse_loss

LOSS_REGISTRY['l1'] = L1Loss
LOSS_REGISTRY['l2'] = lambda: _mse_loss  # functional 直接返回

@register_loss('sisnr')
def _sisnr_loss_factory():
    from .loss.loss import PIT_SISNR_time
    return PIT_SISNR_time()


def get_loss(cfg):
    # 获取 loss 操作名称
    if 'loss' in cfg.train:
        loss_oper = cfg.train['loss']
    elif 'loss' in cfg.model:
        loss_oper = cfg.model['loss']['type']
    else:
        raise Exception('Loss operation not found in config!')

    key = loss_oper.lower()
    if key not in LOSS_REGISTRY:
        raise Exception(f'Loss operation error! 未注册的 loss: {loss_oper}. 请在 utils/model_init.py 中注册或在相应 __init__ 中调用 register_loss。')

    factory = LOSS_REGISTRY[key]
    obj = factory() if callable(factory) and not isinstance(factory, torch.nn.Module) else factory()
    # 如果 obj 仍是函数 (如 mse), 直接返回
    if isinstance(obj, torch.nn.Module):
        device = torch.device(cfg.train['device'] if torch.cuda.is_available() else 'cpu')
        obj = obj.to(device)
    print(f'You choose loss operation with {loss_oper} ...')
    return obj


# --------------------------- Utility functions ------------------------ #
def cal_total_params(model):
    return sum(p.numel() for p in model.parameters())

__all__ = [
    'get_arch', 'get_train_mode', 'get_loss', 'cal_total_params',
    'register_loss', 'LOSS_REGISTRY'
]
