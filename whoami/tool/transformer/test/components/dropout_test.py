#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/10
@Author : weiyutao
@File : test_dropout.py
"""
import sys
import os
import pytest
import torch
import numpy as np
from whoami.tool.transformer.components import Dropout


class TestDropout:
    """测试自定义Dropout实现"""

    def test_init(self):
        """测试初始化参数"""
        # 测试默认概率
        dropout = Dropout()
        assert dropout.p == 0.5
        assert not dropout.inplace

        # 测试自定义概率
        dropout = Dropout(p=0.3, inplace=True)
        assert dropout.p == 0.3
        assert dropout.inplace

    def test_zero_prob(self):
        """测试p=0时的行为（不应该有丢弃）"""
        dropout = Dropout(p=0.0)
        
        # 创建一个测试输入张量
        x = torch.ones(100, 100)
        x_orig = x.clone()
        
        # 在训练模式下应用dropout
        dropout.train()
        y = dropout(x)
        
        # 当p=0时，输出应该与输入完全相同
        assert torch.all(torch.eq(y, x_orig))
        assert torch.all(torch.eq(y, x))  # 确保输入也没有被修改
        
        # 在评估模式下应用dropout
        dropout.eval()
        y = dropout(x)
        
        # 评估模式下应该也与输入完全相同
        assert torch.all(torch.eq(y, x_orig))

    def test_training_mode(self):
        """测试训练模式下的dropout行为"""
        torch.manual_seed(42)  # 设置随机种子以确保结果可重现
        
        p = 0.5
        dropout = Dropout(p=p)
        dropout.train()  # 设置为训练模式
        
        # 创建一个测试输入张量
        x = torch.ones(1000, 1000)
        y = dropout(x)
        
        # 检查是否有元素被丢弃（设置为0）
        zero_elements = (y == 0).sum().item()
        total_elements = y.numel()
        dropout_rate = zero_elements / total_elements
        
        # 由于随机性，我们检查dropout率是否接近p值（允许一定的误差）
        assert abs(dropout_rate - p) < 0.01
        
        # 检查非零元素是否被正确缩放 (1/(1-p))
        nonzero_values = y[y > 0]
        expected_value = 1.0 / (1.0 - p)
        assert torch.allclose(nonzero_values, torch.tensor(expected_value), rtol=1e-5)

    def test_eval_mode(self):
        """测试评估模式下的dropout行为（不应该有丢弃）"""
        dropout = Dropout(p=0.5)
        dropout.eval()  # 设置为评估模式
        
        # 创建一个测试输入张量
        x = torch.ones(100, 100)
        x_orig = x.clone()
        y = dropout(x)
        
        # 在评估模式下，输出应该与输入完全相同
        assert torch.all(torch.eq(y, x_orig))

    def test_different_probabilities(self):
        """测试不同的丢弃概率"""
        torch.manual_seed(42)  # 设置随机种子以确保结果可重现
        
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dropout = Dropout(p=p)
            dropout.train()  # 设置为训练模式
            
            # 创建一个较大的测试输入张量，以获得更准确的统计结果
            x = torch.ones(1000, 1000)
            y = dropout(x)
            
            # 检查丢弃率是否接近p值
            zero_elements = (y == 0).sum().item()
            total_elements = y.numel()
            dropout_rate = zero_elements / total_elements
            
            # 允许5%的误差范围
            assert abs(dropout_rate - p) < 0.05
            
            # 检查非零元素是否被正确缩放
            if p < 1.0:  # 避免除以零
                nonzero_values = y[y > 0]
                expected_value = 1.0 / (1.0 - p)
                assert torch.allclose(nonzero_values, torch.tensor(expected_value), rtol=1e-5)

    def test_inplace_operation(self):
        """测试inplace=True的行为"""
        torch.manual_seed(42)  # 设置随机种子以确保结果可重现
        
        # 使用inplace=True
        dropout = Dropout(p=0.5, inplace=True)
        dropout.train()
        
        x = torch.ones(100, 100)
        x_orig = x.clone()
        y = dropout(x)
        
        # 验证x已被修改（inplace操作）
        assert not torch.all(torch.eq(x, x_orig))
        
        # 验证y与修改后的x相同（因为它们引用同一内存）
        assert torch.all(torch.eq(y, x))
        
        # 对比使用inplace=False
        dropout = Dropout(p=0.5, inplace=False)
        dropout.train()
        
        x = torch.ones(100, 100)
        x_orig = x.clone()
        y = dropout(x)
        
        # 验证x未被修改
        assert torch.all(torch.eq(x, x_orig))
        
        # 验证y与x不同
        assert not torch.all(torch.eq(y, x))

    def test_dropout_shape(self):
        """测试dropout保持输入形状不变"""
        torch.manual_seed(42)
        
        input_shapes = [
            (10,),  # 1D
            (10, 20),  # 2D
            (10, 20, 30),  # 3D
            (5, 10, 15, 20)  # 4D
        ]
        
        dropout = Dropout(p=0.5)
        dropout.train()
        
        for shape in input_shapes:
            x = torch.ones(*shape)
            y = dropout(x)
            assert y.shape == shape

    def test_backward_pass(self):
        """测试梯度流过dropout层"""
        torch.manual_seed(42)
        
        for p in [0.0, 0.3, 0.5]:
            dropout = Dropout(p=p)
            dropout.train()
            
            # 创建需要梯度的输入
            x = torch.ones(10, 10, requires_grad=True)
            y = dropout(x)
            loss = y.sum()
            loss.backward()
            
            # 检查梯度是否正确传播
            # 对于p=0，所有梯度都应该是1
            # 对于p>0，梯度应该是0（被丢弃的元素）或1/(1-p)（保留的元素）
            if p == 0.0:
                assert torch.all(x.grad == 1.0)
            else:
                # 检查梯度是否为0或1/(1-p)
                expected_grad = 1.0 / (1.0 - p)
                zero_grads = (x.grad == 0.0)
                nonzero_grads = (x.grad == expected_grad)
                assert torch.all(zero_grads | nonzero_grads)
                
                # 检查零梯度的比例是否接近p
                zero_grad_ratio = zero_grads.float().mean().item()
                assert abs(zero_grad_ratio - p) < 0.1