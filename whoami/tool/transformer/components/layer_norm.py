#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/06 19:24:28
@Author : weiyutao
@File : layer_norm.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from typing import (
    Optional
)


from whoami.tool.transformer.model_configs.TLMo import TLMoModelConfig
from whoami.tool.transformer.types import LayerNormType


class LayerNormBase(nn.Module): 
    
    def __init__(
        self,
        config: TLMoModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True
    ):
        super().__init__()
        self.config = config
        self.eps = config.layer_norm_eps
        self.normalized_shape = (size or config.d_model, )
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            # Init weight default is 1, means not scale.初始化无缩放
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=self.config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                # Init bias default is zero, means not bias.初始化无偏移
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=self.config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            # By explicity registering as None, rather than not declaring it at all, you can
            # keep the model architecture consistent and make it easy to subsequently check the mudule's parameter list.
            # This code will define self.weight = None, self.bias = None
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)
            
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    # cls只是一个约定，如果使用了classmethod装饰器，则第一个参数默认为类而不是实例，不管第一个参数名称是什么
    # 如果没有使用classmethod装饰器定义该方式，然后直接使用类去调用该方法，那么传递的第一个参数会默认为是传递给cls
    # 那么就会报错，因为config是必传参数，会报错没有传递config参数
    # 但是如果你指定了config参数去传递，那么会报错不是类方法    
    @classmethod    
    def build(cls, config: TLMoModelConfig, size: Optional[int] = None, **kwargs) -> 'LayerNormBase':
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: {config.layer_norm_type}")


    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            # with torch.cuda.amp.autocast():, if used it, torch.is_autocast_enabled() == True
            # with torch.cpu.amp.autocast():, if used it, torch.is_autocast_cpu_enabled() == True
            # torch.cuda.amp.autocast(dtype=torch.bfloat16):， user can specific the precision.
            # CPU 的默认自动混合精度类型是 torch.bfloat16 (BFloat16 格式)
            # CUDA (GPU) 的默认自动混合精度类型是 torch.float16 (半精度浮点数)
            # This can be used for all step in layer norm.
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor
            
            
    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
            


class LayerNorm(LayerNormBase):
    """
    The default class LayerNorm implementation which can optionally run in low precision.
    """
    def __init__(
        self, 
        config: TLMoModelConfig, 
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None
    ):
        super().__init__(config, size, elementwise_affine=elementwise_affine)
        self.low_precision = low_precision
    
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor: 
        if self.low_precision:
            # 首先获取当前的上下文精度进行精度转换
            # 然后在禁用自动混合精度上下文中进行归一化操作，使得精度可控，但是low_precision不一定总是低精度，只是说控制了精度的改变
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                # 因为在禁用自动混合精度上下文中，因此精度使用的是downcast_x的精度
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps
                )
        else:
            # 并不在禁用自动混合精度上下文中进行归一化操作
            # 因此不会使用固定的精度，还是使用上下文的精度，更不可控
            print(f"normalized_shape: --------------------------- {self.normalized_shape}")
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)
    


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class: `LayerNorm` implementation
    """
    def __init__(
        self,
        config: TLMoModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config=config, size=size, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            # 禁用自动混合精度以后会采用输入的精度去计算（如果不做显示更改）
            # 精度可控，全程使用高精度float32
            
            # 为确保计算结果的一致性和稳定性，强制使用全精度计算
            # RMS选择保守策略，始终提升到float32去计算归一化操作，而标准的归一化操作则不同，通过 _cast_if_autocast_enabled() 方法有更灵活的精度控制
            # 也禁用了自动混合精度，但是并没有显式提高精度，标准归一化允许用户使用low_precision平衡精度和效率
            og_dtype = x.dtype # 存储输入张亮的原始数据类型
            x = x.to(torch.float32)
            
            # x(2, 4), x.mean(-1, keepdim=False)=(2,), x.mean(-1, keepdim=True)=(2, 1) 
            variance = x.pow(2).mean(-1, keepdim=True) # 实际计算的是平方均值，不是方差
            x = x * torch.rsqrt(variance + self.eps) # 倒数平方根 torch.sqrt(x)是x的平方根x^(1/2), 而torch.rsqrt(x)=1/(x)^(1/2)
            x = x.to(og_dtype)
            
        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x
            
            
            
            