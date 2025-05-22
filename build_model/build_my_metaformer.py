import os
import sys
sys.path.insert(0, os.getcwd())
from functools import partial
import torch
import torch.nn as nn
from timm.models.registry import register_model

from models.my_metaformer import MetaFormer
from models.my_metaformer import RandomMixing, Pooling, SepConv, Attention, MlpHead


'''仅修改decoder部分, 实现四个stage多尺度特征融合'''
def caformer_s18_msf(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return model


def caformer_b36_in21ft1k_msf(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return model


def caformer_b36_384_in21ft1k_msf(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    return model


'''在实现decoder多尺度特征融合的基础上加入EVA02的结构配置'''
# 基本的caformer
def caformer_s18_eva02(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        eva02=True,
        **kwargs)

    return model


def caformer_b36_eva02(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        eva02=True,
        **kwargs)
    return model


'''全部使用注意力的mataformer, 18表示深度, 同时使用eva02结构和多尺度融合'''
def fulla_former_s18_eva02(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[Attention, Attention, Attention, Attention],
        head_fn=MlpHead,
        eva02=True,
        **kwargs)

    return model
