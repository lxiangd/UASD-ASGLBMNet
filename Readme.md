
## 📂 目录结构

```
UASD-ASGLBMNet/
	train.py                  # 训练入口
	test.py                   # 测试 / 推理脚本 (可自行扩展)
	options.py                # 可能的参数辅助（如后续拓展）
	config/                   # 配置文件
		shipsear/ASGLBMNet.yaml
		shipsear/convtasnet.yaml
	models/                   # 模型实现与注册 (@register_model)
		ASGLBMNet/ASGLBMNet.py
		convtasnet/convtasnet.py
	utils/                    # 工具集合 (loss/metrics/logger/flops/...)
	dataloader/               # 数据加载逻辑
	loss/                     # （项目自带 loss 可扩展）
	data/                     # 原始 clean / noise (你放这里)
	generate_dummy_signals.py # 生成纯音测试数据
	prepare_shipsear_dataset.py # 将 clean/noise 合成为多 SNR 训练集
	requirements.txt
	Readme.md
```

## 🚀 快速开始

### 1. 创建环境
```bash
conda create -n asglbmnet python=3.10 -y
conda activate asglbmnet
pip install -r requirements.txt
```

### 2. 准备数据

方式 A: 使用你自己的原始数据：
```
data/
	clean/   *.wav (目标/干净信号)
	noise/   *.wav (噪声信号)
```


### 3. 构建标准训练集格式
使用多 SNR 合成脚本：
```bash
python prepare_shipsear_dataset.py \
	--clean_dir data/clean \
	--noise_dir data/noise \
	--output_dir data/demucs_format_16k \
	--snr_min -15 --snr_max 0 --snr_step 5 \
	--segment_length 10 --segment_step 1 --clean_segment_length 5
```
生成后结构示例：
```
data/demucs_format_16k/
	clean_npy/ ...
	noise_npy/ ...
	train/<sample_id>/clean.npy|noise.npy|mixture.npy
	valid/<sample_id>/...
	mix_samples_info.csv
```
在 yaml 中设置：
```yaml
dataset:
	paths:
		train: data/demucs_format_16k/train
		valid: data/demucs_format_16k/valid
		test:  data/demucs_format_16k/valid   # 若暂时没有单独 test，可复用
```

### 4. 训练
```bash
python train.py --config config/shipsear/ASGLBMNet.yaml
```
启用预训练（若 `log/<arch>/models/best.pt` 已存在）：
```bash
python train.py --config config/shipsear/ASGLBMNet.yaml --pretrain true
```

### 5. 测试 / 推理
（根据你的 test.py 实现调整）
```bash
python test.py --model_path path/to/best.pt --config config/shipsear/ASGLBMNet.yaml
```

## 🧩 模型注册与扩展

### 新增模型
1. 新建目录: `models/MyNet/`，创建 `MyNet.py`
2. 在文件中：
```python
from models import register_model
@register_model("MyNet")
class MyNet(torch.nn.Module):
		def __init__(self, cfg): ...
		def forward(self, x): ...
```
3. 新建配置: `config/shipsear/MyNet.yaml`，设定 `model.arch: "MyNet"`
4. 训练: `python train.py --model MyNet`


### 新增损失函数
1. 在 `utils/model_init.py` 中注册：
```python
from utils.model_init import register_loss
@register_loss('my_loss')
def _my_loss_factory():
		return MyLossClass()
```
2. 在 yaml 中：
```yaml
model:
	loss:
		type: my_loss
```


### 切换已有损失
支持: `l1`, `l2`, `sisnr` (PIT_SISNR_time)，以及你自行注册的。



## 📊 指标 / 日志
- SISNRi / SNRsegi 在验证阶段计算
- 自动记录最佳模型到 `log/<dataset>/<arch>/models/best.pt`
- 支持 SwanLab，swanlab使用见官网




## 参考代码
https://github.com/dmlguq456/SepReformer
