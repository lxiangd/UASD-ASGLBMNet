"""模型注册中心。

新增模型最简流程:
1. 在 models/<YourModel>/<YourModel>.py 中实现类并使用 @register_model("YourModel") 装饰。
2. 新建 config/<dataset>/<YourModel>.yaml (model.arch: YourModel)。
3. 运行: python train.py --model YourModel --dataset <dataset>

无需再手动编辑此文件。
"""

MODEL_REGISTRY = {}

def register_model(name: str):
	def decorator(cls):
		MODEL_REGISTRY[name.lower()] = cls
		return cls
	return decorator

def get_registered_model(name: str):
	return MODEL_REGISTRY.get(name.lower())

# 导入已存在模型文件以触发装饰器注册
from .ASGLBMNet.ASGLBMNet import ASGLBMNet  # noqa: F401
from .convtasnet.convtasnet import Conv_TasNet  # noqa: F401

__all__ = [
	'MODEL_REGISTRY', 'register_model', 'get_registered_model'
]