#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/10
@Author : weiyutao
@File : test_layer_norm.py
"""
import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))  # 根据实际目录层级调整
sys.path.insert(0, project_root)

from whoami.tool.transformer.model_configs.TLMo import TLMoModelConfig
from whoami.tool.transformer.types import LayerNormType
from whoami.tool.transformer.components import LayerNormBase, LayerNorm, RMSLayerNorm


class TestLayerNorm:
    """测试LayerNorm相关类"""

    @pytest.fixture
    def config(self):
        """创建测试用的配置对象"""
        return TLMoModelConfig(
            d_model=768,
            layer_norm_eps=1e-5,
            layer_norm_with_affine=True,
            include_bias=True,
            bias_for_layer_norm=None,
            init_device=None,
        )

    def test_layer_norm_initialization(self, config):
        """测试LayerNorm的初始化"""
        # 测试默认初始化
        ln = LayerNorm(config)
        assert ln.normalized_shape == (config.d_model, )
        assert ln.eps == config.layer_norm_eps
        assert ln.weight is not None
        assert ln.bias is not None
        assert ln.weight.shape == torch.Size([config.d_model])
        assert ln.bias.shape == torch.Size([config.d_model])
        assert torch.allclose(ln.weight, torch.ones_like(ln.weight))
        assert torch.allclose(ln.bias, torch.zeros_like(ln.bias))
        assert ln.low_precision is False

        # 测试自定义大小
        custom_size = 256
        ln = LayerNorm(config, size=custom_size)
        assert ln.normalized_shape == (custom_size, )
        assert ln.weight.shape == torch.Size([custom_size])
        assert ln.bias.shape == torch.Size([custom_size])

        # 测试无仿射变换
        ln = LayerNorm(config, elementwise_affine=False)
        assert ln.weight is None
        assert ln.bias is None

        # 测试低精度设置
        ln = LayerNorm(config, low_precision=True)
        assert ln.low_precision is True

    def test_rms_layer_norm_initialization(self, config):
        """测试RMSLayerNorm的初始化"""
        # 测试默认初始化
        rms = RMSLayerNorm(config)
        assert rms.normalized_shape == config.d_model
        assert rms.eps == config.layer_norm_eps
        assert rms.weight is not None
        assert rms.bias is not None
        assert rms.weight.shape == torch.Size([config.d_model])
        assert rms.bias.shape == torch.Size([config.d_model])
        assert torch.allclose(rms.weight, torch.ones_like(rms.weight))
        assert torch.allclose(rms.bias, torch.zeros_like(rms.bias))

        # 测试自定义大小
        custom_size = 256
        rms = RMSLayerNorm(config, size=custom_size)
        assert rms.normalized_shape == custom_size
        assert rms.weight.shape == torch.Size([custom_size])
        assert rms.bias.shape == torch.Size([custom_size])

        # 测试无仿射变换
        rms = RMSLayerNorm(config, elementwise_affine=False)
        assert rms.weight is None
        assert rms.bias is None

    def test_reset_parameters(self, config):
        """测试参数重置功能"""
        # 创建实例并修改参数
        ln = LayerNorm(config)
        ln.weight.data.fill_(2.0)
        ln.bias.data.fill_(1.0)
        
        # 确认修改生效
        assert torch.allclose(ln.weight, torch.full_like(ln.weight, 2.0))
        assert torch.allclose(ln.bias, torch.full_like(ln.bias, 1.0))
        
        # 重置参数
        ln.reset_parameters()
        
        # 确认重置后恢复默认值
        assert torch.allclose(ln.weight, torch.ones_like(ln.weight))
        assert torch.allclose(ln.bias, torch.zeros_like(ln.bias))

    def test_layer_norm_forward(self, config):
        """测试LayerNorm的前向传播"""
        batch_size = 2
        seq_len = 10
        d_model = config.d_model
        
        # 创建一个输入张量
        x = torch.randn(batch_size, seq_len, d_model)
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = ((x - x_mean) ** 2).mean(dim=-1, keepdim=True)
        x_std = torch.sqrt(x_var + config.layer_norm_eps)
        expected_output = (x - x_mean) / x_std  # 标准化但没有仿射变换
        
        # 创建无仿射变换的LayerNorm实例
        ln = LayerNorm(config, elementwise_affine=False)
        output = ln(x)
        
        # 验证输出是否符合预期
        assert output.shape == x.shape
        assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-5)
        
        # 创建有仿射变换的LayerNorm实例
        ln = LayerNorm(config)
        ln.weight.data.fill_(2.0)  # 缩放因子为2
        ln.bias.data.fill_(1.0)    # 偏移为1
        
        output = ln(x)
        expected_output_affine = expected_output * 2.0 + 1.0  # 应用仿射变换
        
        # 验证带仿射变换的输出
        assert torch.allclose(output, expected_output_affine, rtol=1e-5, atol=1e-5)

    def test_rms_layer_norm_forward(self, config):
        """测试RMSLayerNorm的前向传播"""
        batch_size = 2
        seq_len = 10
        d_model = config.d_model
        
        # 创建一个输入张量
        x = torch.randn(batch_size, seq_len, d_model)
        # RMS标准化计算
        variance = (x ** 2).mean(dim=-1, keepdim=True)
        expected_output = x * torch.rsqrt(variance + config.layer_norm_eps)
        
        # 创建无仿射变换的RMSLayerNorm实例
        rms = RMSLayerNorm(config, elementwise_affine=False)
        output = rms(x)
        
        # 验证输出是否符合预期
        assert output.shape == x.shape
        assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-5)
        
        # 创建有仿射变换的RMSLayerNorm实例
        rms = RMSLayerNorm(config)
        rms.weight.data.fill_(2.0)  # 缩放因子为2
        rms.bias.data.fill_(1.0)    # 偏移为1
        
        output = rms(x)
        expected_output_affine = expected_output * 2.0 + 1.0  # 应用仿射变换
        
        # 验证带仿射变换的输出
        assert torch.allclose(output, expected_output_affine, rtol=1e-5, atol=1e-5)

    def test_low_precision_layer_norm(self, config):
        """测试低精度LayerNorm"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping low precision test")
            
        batch_size = 2
        seq_len = 10
        d_model = config.d_model
        
        # 创建一个输入张量
        x = torch.randn(batch_size, seq_len, d_model, device='cuda')
        
        # 创建标准LayerNorm和低精度LayerNorm
        ln_normal = LayerNorm(config, low_precision=False).cuda()
        ln_low = LayerNorm(config, low_precision=True).cuda()
        
        # 正常精度下比较输出
        output_normal = ln_normal(x)
        output_low = ln_low(x)
        
        # 两者输出应该非常接近
        assert torch.allclose(output_normal, output_low, rtol=1e-3, atol=1e-3)
        
        # 测试自动混合精度下的行为
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                output_normal_amp = ln_normal(x)
                output_low_amp = ln_low(x)
                
                # 验证低精度模式在自动混合精度下的行为
                assert output_low_amp.dtype in [torch.float16, torch.bfloat16]
                # 即使在自动混合精度下，输出也应该相似
                assert torch.allclose(output_normal_amp, output_low_amp, rtol=1e-2, atol=1e-2)

    def test_build_function(self, config):
        """测试build工厂方法"""
        # 测试默认类型
        config.layer_norm_type = LayerNormType.default
        ln = LayerNormBase.build(config=config)
        assert isinstance(ln, LayerNorm)
        assert ln.low_precision is False
        
        # 测试低精度类型
        config.layer_norm_type = LayerNormType.low_precision
        ln = LayerNormBase.build(config)
        assert isinstance(ln, LayerNorm)
        assert ln.low_precision is True
        
        # 测试RMS类型
        config.layer_norm_type = LayerNormType.rms
        ln = LayerNormBase.build(config)
        assert isinstance(ln, RMSLayerNorm)
        
        # 测试未知类型
        config.layer_norm_type = "unknown"
        with pytest.raises(NotImplementedError):
            LayerNormBase.build(config)

    def test_cast_if_autocast_enabled(self, config):
        """测试_cast_if_autocast_enabled方法"""
        ln = LayerNorm(config)
        
        # 创建测试张量
        tensor = torch.randn(10, 10)
        
        # 没有启用自动混合精度的情况
        result = ln._cast_if_autocast_enabled(tensor)
        assert result is tensor  # 应该返回原始张量
        
        # 由于难以直接测试autocast环境，这里仅测试基本功能
        # 完整测试需要在实际GPU环境下进行

    def test_comparison_with_torch_layer_norm(self, config):
        """将自定义LayerNorm与PyTorch内置LayerNorm比较"""
        batch_size = 2
        seq_len = 10
        d_model = config.d_model
        
        # 创建输入张量
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 创建自定义和PyTorch LayerNorm
        custom_ln = LayerNorm(config)
        torch_ln = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        
        # 确保权重相同
        torch_ln.weight.data.copy_(custom_ln.weight.data)
        torch_ln.bias.data.copy_(custom_ln.bias.data)
        
        # 比较输出
        custom_output = custom_ln(x)
        torch_output = torch_ln(x)
        
        # 输出应该非常接近
        assert torch.allclose(custom_output, torch_output, rtol=1e-6, atol=1e-6)

    def test_cuda_support(self, config):
        """测试在CUDA设备上的行为"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping CUDA test")
            
        batch_size = 2
        seq_len = 10
        d_model = config.d_model
        
        # 创建GPU上的输入张量
        x = torch.randn(batch_size, seq_len, d_model, device='cuda')
        
        # 创建Layer Norm并移至GPU
        ln = LayerNorm(config).cuda()
        rms = RMSLayerNorm(config).cuda()
        
        # 测试前向传播
        ln_output = ln(x)
        rms_output = rms(x)
        
        # 验证输出是否在GPU上
        assert ln_output.device.type == 'cuda'
        assert rms_output.device.type == 'cuda'
        
        # 验证输出形状
        assert ln_output.shape == x.shape
        assert rms_output.shape == x.shape

    def test_gradient_flow(self, config):
        """测试梯度流动"""
        batch_size = 2
        seq_len = 10
        d_model = config.d_model
        
        # 创建需要梯度的输入
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        # 测试LayerNorm梯度
        ln = LayerNorm(config)
        ln_out = ln(x)
        ln_loss = ln_out.sum()
        ln_loss.backward()
        
        # 验证梯度不为None且非零
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None
        
        # 重置梯度
        x.grad.zero_()
        
        # 测试RMSLayerNorm梯度
        rms = RMSLayerNorm(config)
        rms_out = rms(x)
        rms_loss = rms_out.sum()
        rms_loss.backward()
        
        # 验证梯度不为None且非零
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        assert rms.weight.grad is not None
        assert rms.bias.grad is not None

    def test_different_input_shapes(self, config):
        """测试不同的输入形状"""
        # 创建层归一化实例
        ln = LayerNorm(config)
        rms = RMSLayerNorm(config)
        
        # 测试不同的输入形状
        shapes = [
            (768,),                # 1D
            (10, 768),             # 2D
            (5, 10, 768),          # 3D
            (2, 5, 10, 768),       # 4D
        ]
        
        for shape in shapes:
            x = torch.randn(*shape)
            
            # 测试LayerNorm
            ln_out = ln(x)
            assert ln_out.shape == x.shape
            
            # 测试RMSLayerNorm
            rms_out = rms(x)
            assert rms_out.shape == x.shape
