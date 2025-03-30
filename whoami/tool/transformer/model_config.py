#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/03/09 19:44:48
@Author : weiyutao
@File : model_config.py
"""
from typing import Dict, Optional, Union


from whoami.configs.model_config import ModelConfig
from whoami.utils.utils import StrEnum


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swigle"


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to Pytorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """
    
    rms = "rms"
    """
    An RMSNorm implementation. When using torch.compile this is probably the fastest implementation.
    """

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
    X ->(w_q) Query, 
    X ->(w_k) Key
    X ->(w_v) Value
    
    
    b_head = 8
    n_kv_heads = 4
    X ->(w_q) Query   8
    X ->(w_k) Key (2)
    X ->(w_v) Value (2)
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
    The dropout probability for the MLP and attention output within each block.
    This dropout is dedicated to be used for the out of MLP layer and attention.
    Notice it is not the attention matrix.
    The attention matrix: (Q @ K.T) / sqrt(d_k), d_k = d_model / n_head.
    The attention probs = softmax(The attention matrix)
    attention_dropout(The attention probs)
    
    attention_output = attention_probs @ V
    
    x_ = x + residual_dropout(attention_output)
    
    x__ = layer_norm(x_)
    x___ = mlp(x__)
    x____ = x__ + residual_dropout(x___)
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
    
    
    attention_layer_norm_with_affine: bool = True
    """
    Toggle affine transform for the QK norms.
    """
    
    
    layer_norm_with_affine: bool = True
    """
    Whether to include bias and weight parameters for the layer norms.
    This is only affects layer norms that are immediately followed by a linear layer in the forward pass,
    so everything expect QK-norms. To turn off affines for QK norms as well, set attribute attention_layer_norm_with_affine to False.
    """
    
    bias_for_layer_norm: Optional[bool] = None
    """
    Whether or not to include bias parameters in layer norm.
    This is separate from the include_bias parameter, because of a ROCm crash when biases are disabled in layer norm.
    When this is None (the default), it inherits the setting from include_bias.
    """
    
    include_bias: bool = True
    """
    Whether or not to include bias parameters in linear layers.
    In PaLM, they got rid of all bias terms because they found that large models tend to have near 0 bias terms anyway.
    """
    
    layer_norm_type: LayerNormType = LayerNormType.default
    """
    The layernorm implementation to use.
    """
    
    
    mlp_hidden_size: Optional[int] = None
    """
    Set the exact hidden size for the MLP. Otherwise the inner MLP hidden size will be set to `mlp_ratio * d_model`.
    """
    
    attention_layer_norm: Optional[bool] = False
    """
    Apply layer norm to the keys and queries within the attention mechanism.
    This can help stabilize training.
    """


    activation_type: ActivationType = ActivationType.swiglu
    """
    The activation function to use within the MLP layers.
    """

    
    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """


    def effective_n_kv_heads(self) -> int:
        """
        QKV
        if effective_n_kv_heads == n_heads, MHA 
        if effective_n_kv_heads == 1, MQA
        if 1 < effective_n_kv_heads < n_heads, GQA
        MHA: 
            在标准多头注意力中
            每个查询头Q都有自己匹配的KV，每组QKV是完全独立的，每个注意力头可以独立地关注输入的不同方面
        MQA:
            一个KV头服务所有查询头
        GQA:
            一个KV头服务多个查询头
        这样一来，KV头服务的查询头越多，可学习的参数越少
        但是注意：这只是影响自注意力层的可学习参数，不影响输出
        """
        if self.n_kv_heads is None:
            # default n_heads.
            return self.n_heads
        else:
            return self.n_kv_heads
    