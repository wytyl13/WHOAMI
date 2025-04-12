#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/10 20:54:37
@Author : weiyutao
@File : model_config_test.py
"""
from whoami.tool.transformer.types import ActivationType, LayerNormType
from whoami.tool.transformer.model_configs import ModelConfig
from whoami.tool.transformer.model_configs import TLMoModelConfig


class TestTLMoModelConfig:
    """测试TLMoModelConfig类的行为和属性"""
    def test_default_values(self):
        """测试配置类的默认值是否符合预期"""
        config = TLMoModelConfig()
        
        # 检查基本模型参数
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_layers == 12
        assert config.mlp_ratio == 4
        assert config.vocab_size == 50257
        assert config.embedding_size == 50257
        assert config.max_sequence_length == 1024
        
        # 检查注意力机制配置
        assert config.n_kv_heads is None
        assert config.alibi is False
        assert config.rope is True
        assert config.flash_attention is True
        assert config.clip_qkv is None
        
        # 检查dropout配置
        assert config.attention_dropout == 0.1
        assert config.embedding_dropout == 0.1
        assert config.residual_dropout == 0.1
        
        # 检查规范化和激活配置
        assert config.layer_norm_eps == 1e-05
        assert config.layer_norm_type == LayerNormType.default
        assert config.activation_type == ActivationType.swiglu
        assert config.layer_norm_with_affine is True
        assert config.attention_layer_norm_with_affine is True
        assert config.attention_layer_norm is False
        assert config.include_bias is True
        assert config.bias_for_layer_norm is None
        
        # 检查其他配置
        assert config.block_group_size == 1
        assert config.init_device is None
        assert config.rope_full_precision is True

    def test_custom_values(self):
        """测试自定义配置参数是否生效"""
        custom_config = TLMoModelConfig(
            d_model=1024,
            n_heads=16,
            n_layers=24,
            n_kv_heads=4,
            activation_type=ActivationType.gelu,
            max_sequence_length=2048,
            rope=False,
            alibi=True
        )
        
        assert custom_config.d_model == 1024
        assert custom_config.n_heads == 16
        assert custom_config.n_layers == 24
        assert custom_config.n_kv_heads == 4
        assert custom_config.activation_type == ActivationType.gelu
        assert custom_config.max_sequence_length == 2048
        assert custom_config.rope is False
        assert custom_config.alibi is True

    def test_effective_n_kv_heads(self):
        """测试effective_n_kv_heads方法的返回值"""
        # 测试默认情况（n_kv_heads为None）
        config1 = TLMoModelConfig(n_heads=12, n_kv_heads=None)
        assert config1.effective_n_kv_heads() == 12
        
        # 测试MQA情况（n_kv_heads=1）
        config2 = TLMoModelConfig(n_heads=12, n_kv_heads=1)
        assert config2.effective_n_kv_heads() == 1
        
        # 测试GQA情况（1 < n_kv_heads < n_heads）
        config3 = TLMoModelConfig(n_heads=12, n_kv_heads=4)
        assert config3.effective_n_kv_heads() == 4
        
        # 测试MHA情况（n_kv_heads = n_heads）
        config4 = TLMoModelConfig(n_heads=8, n_kv_heads=8)
        assert config4.effective_n_kv_heads() == 8

    def test_mlp_hidden_size_behavior(self):
        """测试mlp_hidden_size的行为"""
        # 当未指定mlp_hidden_size时，应该使用mlp_ratio和d_model计算
        config1 = TLMoModelConfig(d_model=768, mlp_ratio=4, mlp_hidden_size=None)
        assert config1.mlp_hidden_size is None  # 直接属性应该是None
        
        # 当指定mlp_hidden_size时，应该使用指定值
        config2 = TLMoModelConfig(d_model=768, mlp_ratio=4, mlp_hidden_size=3072)
        assert config2.mlp_hidden_size == 3072

    def test_model_inheritance(self):
        """测试TLMoModelConfig是否正确继承自ModelConfig"""
        config = TLMoModelConfig()
        assert isinstance(config, ModelConfig)

    def test_mutually_exclusive_position_embeddings(self):
        """测试互斥的位置嵌入配置"""
        # 注意：这个测试只是检查是否可以设置互斥的配置，但实际上，
        # 在生产代码中，应该有逻辑来防止同时启用rope和alibi
        config1 = TLMoModelConfig(rope=True, alibi=False)
        print()
        print(config1)
        assert config1.rope is True
        assert config1.alibi is False
        config2 = TLMoModelConfig(rope=False, alibi=True)
        assert config2.rope is False
        assert config2.alibi is True