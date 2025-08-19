import os
import yaml
from datetime import datetime

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d")
        self.config['timestamp'] = timestamp

        # 更新输出路径 (模型名不再附加 condition 后缀)
        self.config['output']['log_dir'] = os.path.join(
            self.config['output']['log_dir'],
            self.config['dataset']['name'],
            f"{self.config['model']['arch']}"
        )
        self.config['output']['model_dir'] = os.path.join(
            self.config['output']['log_dir'],
            self.config['output']['model_dir']
        )
    
    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    @staticmethod
    def get_default_config():
        """获取默认配置文件路径"""
        config_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(config_dir, 'default.yaml') 

    @staticmethod
    def auto_from(model_name: str, dataset: str = 'shipsear'):
        """根据模型名与数据集自动构造 yaml 路径.
        约定: config/<dataset>/<ModelName>.yaml
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(base_dir, dataset, f'{model_name}.yaml')
        if not os.path.exists(cand):
            raise FileNotFoundError(f'未找到配置文件: {cand}')
        return cand