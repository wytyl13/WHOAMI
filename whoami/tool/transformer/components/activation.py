#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/06 19:21:16
@Author : weiyutao
@File : activation.py
"""
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import (
    cast,
)
import torch.nn.functional as F
from whoami.tool.transformer.model_configs.TLMo import TLMoModelConfig
from whoami.tool.transformer.types import ActivationType


class Activation(nn.Module):
    """Activation Function for any transformer model."""
    def __init__(self, config: TLMoModelConfig):
        super().__init__()
        self.config = config
        
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented
    
    
    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError
    
    
    def build(cls, config: TLMoModelConfig) -> 'Activation':
        if config.activation_type == ActivationType.gelu:
            # cast function just static type transform.
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=True))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)


class GELU(nn.GELU):
    # Why define this attribution? In order to cast to Activation.
    @property
    def output_multiplier(self) -> float:
        return 1.0
    
    
class ReLU(nn.ReLU):
    def output_multiplier(self) -> float:
        return 1.0
    
    
class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=1)
        return F.silu(gate) * x
    
    
    @property
    def output_multiplier(self) -> float:
        # 输出维度占比输入维度
        return 0.5

