import sys
sys.path.append('../')

import torch
import warnings
warnings.filterwarnings('ignore')

from .modules.module import *
from models import register_model


@register_model("ASGLBMNet")
class ASGLBMNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model = cfg.model

        def get_attr(obj, name):
            if isinstance(obj, dict):
                return obj[name]
            return getattr(obj, name)
        audio_enc = get_attr(model, 'audio_enc')
        module_separator = get_attr(model, 'module_enc_dec')
        audio_dec = get_attr(model, 'audio_dec')
        self.num_stages = get_attr(model, 'num_stages')

        # 构建子模块
        self.audio_encoder = AudioEncoder(audio_enc['layer1'], audio_enc['layer2'])
        self.enc_dec = Enc_Dec(**module_separator)
        self.audio_decoder = AudioDecoder(audio_dec['layer1'], audio_dec['layer2'])

    def forward(self, x: torch.Tensor):
        projected_feature, encoder_output = self.audio_encoder(x)
        last_stage_output, _ = self.enc_dec(projected_feature)
        audio = self.audio_decoder(last_stage_output, encoder_output)
        return audio