#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/06 20:49:16
@Author : weiyutao
@File : TMLo.py
"""
import torch
import torch.nn as nn
import sys
from typing import (
    Optional,
    Callable
)




from whoami.tool.transformer.model_configs import TLMoModelConfig
from whoami.tool.transformer.components import Dropout
from whoami.tool.transformer.components import LayerNormBase
from whoami.tool.transformer.components import Activation
from whoami.tool.transformer.components import BufferCache
from whoami.tool.transformer.components import RotaryEmbedding


class TLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """
    
    def __init__(
        self, 
        layer_id: int, 
        config: TLMoModelConfig,
        cache: BufferCache 
    ):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )

        self.__cache = cache
        assert config.d_model % config.n_heads == 0
        
        self._activation_checkpoint_fn: Optional[Callable] = None
        
        # why residual_dropout? not attention_dropout?
        # Dropout
        self.dropout = Dropout(config.residual_dropout)
        
        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            assert config.effective_n_kv_heads is not None
            # 注意力层的输出Key的维度不会受effective_n_kv_heads的影响
            # 每个查询头的维度(config.d_model // config.n_heads)*KV服务的查询头的数量(effective_n_kv_heads)就等于一个key应该归一化的尺寸size
            # 但是这里注意，虽然注意力层归一化的size不受effective_n_kv_heads的影响，我们依然要保证k归一化和q归一化的尺度一致
            # 何为一致？
            # 两种选择？归一化应用于完整的QKV投影输出(size=d_model)，分头归一化（size=d_model//n_heads）
            # 但是一般的自注意力输出归一化的维度是：
            # Q SIZE=d_model
            # K size=(d_model//n_heads)*effective_n_kv_heads
            # 标准transformer一般不对V进行归一化
            # 一般是对注意力输出归一化，而不需要对注意力矩阵（参数）归一化
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine
            )
            self.q_norm = LayerNormBase.build(
                config,
                elementwise_affine=config.attention_layer_norm_with_affine
            )

        if config.clip_qkv is not None:
            assert config.clip_qkv > 0
        
        # Activcation function
        self.act: Activation = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0
        
        # 两个投影层，一个投影注意力机制，一个投影前馈（目的是为了保证transformer模块的输入和输出维度一致）

        # 上下文：
            # 输入 -> 多头自注意力 -> 拼接多头输出 -> self.attn_out -> 残差 -> 层归一化
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )
        
        # 上下文：
            # 输入 -> FF扩展层 -> 激活函数 -> self.ff_out -> 残差连接 -> 层归一化
        # 为什么要考虑激活函数的output_multiplier？
        # 因为hidden_size是固定的，而不同的激活函数有不同的output_multiplier
        # 而激活函数的输出维度是output_multiplier*self.hidden_size
        # 一般的激活函数output_multiplier是1，也即激活函数的输出维度即hidden_size
        # 但是swiglu激活函数不同，他的输出维度一般是hidden_size的一半，因此在进入attn_out
        # 层的输入不同的激活函数维度不同。
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device
        )
        
        # private customer attribution _is_residual, what means is residual fr this feed forward.
        self.ff_out._is_residual = True
        
        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)
            
    
        