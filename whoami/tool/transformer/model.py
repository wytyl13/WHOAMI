#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/03/09 19:49:11
@Author : weiyutao
@File : model.py
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Callable
from functools import partial
import torch.nn.functional as F
from abc import abstractmethod
import sys



if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")


from whoami.tool.transformer.model_config import TransformerModelConfig
from whoami.tool.transformer.exceptions import TLMoCOnfigurationError
from whoami.tool.base.base_tool import BaseTool
from whoami.tool.transformer.check_point_config import ActivationCheckpointStrategy
from whoami.tool.transformer.model_config import LayerNormType

def activation_checkpoint_function(cfg: TransformerModelConfig):
    
    # 检查点是否保存和恢复随机数生成器的状态，用于在反向传播时候用于恢复前向传播输出结果和之前的前向传播随机数一致
    # 如果是false则会提高性能
    preserve_rng_state = not (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )

    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint, 
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    """


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        The default value of inplace is False.
        Unless you are really under memeory pressure, using the default value 
        of False is a safer choice.
        """
        
        output = input if self.p == 0.0 else F.dropout(input, self.p. self.training, self.inplace)
        return output


class LayerNormBase(nn.Module): 
    
    def __init__(
        self,
        config: TransformerModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True
    ):
        super().__init__()
        self.config = config
        self.eps = config.layer_norm_eps
        self.normalized_shape = (size or config.d_model)
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
        
        
    def build(cls, config: TransformerModelConfig, size: Optional[int] = None, **kwargs) -> 'LayerNormBase':
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
        config: TransformerModelConfig, 
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
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)
    


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class: `LayerNorm` implementation
    """
    def __init__(
        self,
        config: TransformerModelConfig,
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
            
            
class TLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """
    
    def __init__(
        self, 
        layer_id: int, 
        config: TransformerModelConfig,
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



            

class TLMo(nn.Module):
    config: Optional[TransformerModelConfig] = None
    def __init__(self, transformer_config: TransformerModelConfig):
        super().__init__()
        self.config = transformer_config
        
        if self.config.alibi and self.config.flash_attention:
            raise TLMoCOnfigurationError("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise TLMoCOnfigurationError("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise TLMoCOnfigurationError("embedding size should be at least as big as vocab size.")
            elif self.config.embedding_size % 128 != 0:
                import warnings
                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointStrategy] = None
        # callable: The function can be callable with ()
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)
        
        
        # ensuer layers numbers
        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise TLMoCOnfigurationError('n_layers must be divisible by block group size!')
        
        # Flash attention, it is a memory-efficient attention computation algorithm that
        # significantly reduces the memory footprint and improves the computational speed of
        # the attention mechanism through the chunking and recomputation strategies.
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Reduced memory footprint, support for longer sequences
        # Sacrificing some speed for memory efficiency. 
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.config.embedding_size or self.config.vocab_size, self.config.d_model, device=self.config.init_device
                ),
                emb_drop=Dropout(self.config.embedding_dropout),
                ln_f=LayerNorm.build(self.config)
            )
        )
        
        blocks = [TLMoBlock.build(i, self.config, self.__cache) for i in range(self.config.n_layers)]
        
    def _run(self, *args, **kwargs):
        raise NotImplementedError


        
        
