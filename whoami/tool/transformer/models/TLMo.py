#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/03/09 19:49:11
@Author : weiyutao
@File : model.py
"""
import torch
import torch.nn as nn
from typing import (
    Optional, 
    Callable,
)
from functools import partial


from whoami.tool.transformer.model_configs.TLMo import TLMoModelConfig
from whoami.tool.transformer.components import TLMoConfigurationError
from whoami.tool.transformer.types import ActivationCheckpointStrategy
from whoami.tool.transformer.components import Dropout
from whoami.tool.transformer.blocks import TLMoBlock

def activation_checkpoint_function(cfg: TLMoModelConfig):
    
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


    

class TLMo(nn.Module):
    config: Optional[TLMoModelConfig] = None
    def __init__(self, transformer_config: TLMoModelConfig):
        super().__init__()
        self.config = transformer_config
        
        if self.config.alibi and self.config.flash_attention:
            raise TLMoConfigurationError("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise TLMoConfigurationError("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise TLMoConfigurationError("embedding size should be at least as big as vocab size.")
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
            raise TLMoConfigurationError('n_layers must be divisible by block group size!')
        
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


        
        
