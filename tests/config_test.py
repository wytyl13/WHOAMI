#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/01 22:25:04
@Author : weiyutao
@File : config_test.py
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional


# 引入被测试的模块
from whoami.tool.transformer.types import LayerNormType
from whoami.tool.transformer.types import ActivationType
from whoami.tool.transformer.model_configs import TLMoModelConfig

from whoami.tool.transformer.components import TLMoConfigurationError
from whoami.tool.transformer.types import ActivationCheckpointStrategy

from whoami.tool.transformer.components import (
    Dropout, 
    LayerNormBase, 
    LayerNorm, 
    RMSLayerNorm,
    Activation,
    GELU,
    ReLU,
    SwiGLU,
    RotaryEmbedding,
    BufferCache,
)
from whoami.tool.transformer.blocks import TLMoBlock
from whoami.tool.transformer.models import TLMo


# 设置基本配置，可以在多个测试中复用
@pytest.fixture
def base_config():
    return TLMoModelConfig(
        vocab_size=1000,
        d_model=64,
        n_heads=8,
        n_layers=2,
        mlp_ratio=4.0,
        layer_norm_eps=1e-5,
        layer_norm_type=LayerNormType.default,
        activation_type=ActivationType.gelu,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        residual_dropout=0.1,
        max_sequence_length=512,
        init_device="cpu",
        include_bias=True,
        bias_for_layer_norm=True,
        layer_norm_with_affine=True,
        effective_n_kv_heads=8,
        attention_layer_norm=True,
        attention_layer_norm_with_affine=True,
        rope=True,
        alibi=False,
        flash_attention=False,
        block_group_size=1
    )
    


# 准备测试数据
@pytest.fixture
def test_data():
    batch_size = 2
    seq_length = 10
    d_model = 64
    n_heads = 8
    
    test_input = torch.randn(batch_size, seq_length, d_model)
    buffer_cache = BufferCache()
    
    return {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "d_model": d_model,
        "n_heads": n_heads,
        "test_input": test_input,
        "buffer_cache": buffer_cache
    }
    

# 正确的测试函数，使用夹具作为参数
def test_main(base_config, test_data):
    # 现在你可以访问夹具返回的值
    print(base_config)
    print(test_data["test_input"].shape)
    

# ========== 测试Dropout组件 ==========
class TestDropout:
    
    def test_dropout_train_mode(self, test_data):
        """测试Dropout在训练模式下的行为"""
        drop = Dropout(p=0.5)
        drop.train()
        output_train = drop(test_data["test_input"])
        
        # 验证输出形状相同
        assert output_train.shape == test_data["test_input"].shape
        
        # 由于dropout在训练模式下是随机的，我们不能确切地测试值，但可以检查是否有变化
        assert not torch.allclose(output_train, test_data["test_input"])
    
    def test_dropout_eval_mode(self, test_data):
        """测试Dropout在评估模式下的行为"""
        drop = Dropout(p=0.5)
        drop.eval()
        output_eval = drop(test_data["test_input"])
        
        # 验证评估模式下输出与输入相同
        torch.testing.assert_close(output_eval, test_data["test_input"])
    
    def test_dropout_zero_probability(self, test_data):
        """测试p=0.0的Dropout"""
        drop_zero = Dropout(p=0.0)
        drop_zero.train()
        output_zero = drop_zero(test_data["test_input"])
        
        # 验证p=0时输出与输入相同
        torch.testing.assert_close(output_zero, test_data["test_input"])
        