#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/03/09 19:44:48
@Author : weiyutao
@File : model_config.py
"""
from typing import Dict, Optional, Union


from whoami.configs.model_config import ModelConfig


class TransformerModelConfig(ModelConfig):
    """
    TLMo (model) configuration
    """


    # Note that the defaults for these attributes are equivalent to the base GPT2 model.
    
    d_model: int = 768
    """
    The hidden size of the model.
    """
    
    n_heads: int = 12
    """
    The number of self-attention heads.
    """

    n_kv_heads: Optional[int] = None
    """
    The number of heads to use for keys and values. Defaults to `n_heads`.
    Set this to ``None`` or ``n_heads`` for normal multi-head attention.
    Set this to 1 for multi-query attention.
    Set it to some in-between value for LLama2-style grouped query attention.
    """

    clip_qkv: Optional[float] = None
    """
    Clip QKV to this value when set.
    """
    
    n_layers: int = 12
    """
    The number of layers/blocks.
    """

    mlp_ratio: int = 4
    """
    The ratio of the inner MLP dimensionality to ``d_model``.
    This is only used when ``mlp_hidden_size`` is not set.
    """

    alibi: bool = False
    """
    If ``True``, use ALiBi embeddings. mutually exclusive with ``rope``
    """

    rope: bool = True
    """
    Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``
    """

    flash_attention: bool = True
    """
    If ``True``, use ``FlashAttention``.
    """
    
    embedding_size: int = 50257
    
    vocab_size: int = 50257
    
    attention_dropout: float = 0.1
    """
    The dropout probability within the attention modules.
    """

    embedding_dropout: float = 0.1
    """
    The dropout probability for embedding.
    """
    
    residual_dropout: float = 0.1
    """
    The dropout probabiliyty for the MLP and attention output within each block.
    """
    
    block_group_size: int = 1
    """
    The number of blocks to group together into a single parent block.
    This has no affect on the number of parameters in the model and is only used to wrap 
    groups of blocks together with a single FSDP wrapper during training.
    If you are already experiencing OOM issues, try reducing block_group_size to get a finer-grained memory management.
    If there is enough memory but training is slow, try increasing the block_group size to reduce communication overheads.
    Notice that block_group_size is valid for both multicard and distributed during training.
    """
    
    init_device: Optional[str] = None
    """
    The torch device to use when initializing the model parameters, "cpu", "cuda:0" and so on.
    """

    layer_norm_eps: float = 1e-05