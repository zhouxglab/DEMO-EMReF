import functools
import logging
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from model.model_utils import init_weights
from ops.argparser import argparser
params = argparser()
config = params['config']
if  "infer_MR" in config:
    from model.hma_arch_v2 import HMANet3D

elif "infer_HR" or "infer_ET" in config:
    from model.hma_arch import HMANet3D

else:
    raise ValueError(f"Unsupported config path: {config}") 

import torch
import torch.nn as nn

def define_G(opt):
    model_opt = opt['model']
   
    netG =  HMANet3D(
                    img_size=model_opt['HMANet']['img_size'],
                    patch_size=1,
                    in_chans=model_opt['HMANet']['in_chans'],
                    embed_dim=model_opt['HMANet']['embed_dim'],
                    depths=model_opt['HMANet']['depths'],
                    num_heads=model_opt['HMANet']['num_heads'],
                    window_size=model_opt['HMANet']['window_size'],
                    interval_size=model_opt['HMANet']['interval_size'],
                    mlp_ratio=model_opt['HMANet']['mlp_ratio'],
                    drop_rate=0.1,
                    attn_drop_rate=0,
                    drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm,
                    img_range=model_opt['HMANet']['img_range'],
                    upsampler=model_opt['HMANet']['upsampler'],
                    resi_connection=model_opt['HMANet']['resi_connection']
                    )

    init_weights(netG, init_type='orthogonal')  # 对网络 netG 进行正交初始化，以确保模型在训练时能够更稳定地传递和更新梯度
    return netG
