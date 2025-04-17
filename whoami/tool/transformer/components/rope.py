#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/06 20:56:20
@Author : weiyutao
@File : rope.py

---------------------------------------------------------------------------------------------------------------------
Absolute postion embedding
f(xi, i){qkv} := W(xi + pi){qkv}
p{i, 2t} = sin(k/10000^(2t/d))
p{i, 2t+1} = cos(k/10000^(2t/d))
d is d_model what is the embedding dimension, t is range from 0 to d/2, k is the token position.
i is euqal to k.
why sin cos? 
By using 10000^(2t/d), position coding has different frequencies in different dimensions. Lower dimensions
correspond to low-frequency signals and higher dimensions correspond to high-frequency signals. This
allows the model to capture location dependencies at different scales.

Now suppose that d_model is 5.
if use the same position coding.
p1 = 1 1 1 1 1
p2 = 2 2 2 2 2
p3 = 3 3 3 3 3
1 Unavalibility of multi-scale information. Because have the same dimension value for each embeding of token. 
2 Poor generalization ability. just like the diff in pos1 and pos2, pos100 and pos101, they are same. And for long sequences,
subtle differences in distant locations may be over-amplified. Training-Reasoning inconsistency,  if the model has only
seen sequences of length 500(position codes 1-500) during training, when it enconters position 600, the codes for this position(all) 600
are far away in the feature space from any of the codes that the training has seen, and the model may not be able to process them effectively.
Gradient problem, as  the position value increase, without proper normalization, it may lead to a huge difference in the gradient size at
different positions, affecting the optimization process.

In comparison, sine-cosine position coding offers the following advantages:
1 periodicity: due to the periodicity of the sine and cosine functions, the encoding of even unseen locations in training will
fall within the feature space already familiar to the model. 
2 Interpolation properties: sine-cosine encoding creates a smooth interpolation between positions, allowing the model to be 
more easily generalized to intermediate positions. sin(5)=-0.96, sin(5.5)=-0.71, sin(6)=-0.28. It can be seen that the coded value of position
5.5 does fall between position 5 and 6, creatint a smooth transition. This smooth interpolation property is important for the model to learn
positional relationships because Continuity(the model can treat positions as continuous variables instead of discrte labels), 
interpolation capability(even if the training data contains only certain positions, the model can efficiently handle any position between these positions) 
and smoothed gradient(the gradient between neighboring positions varies smoothly during back-propagation, which facilitates optimization).
3 Linear combination property: the encoding of any position can be expressed as a linear combination of the encodings of other positions, which gives the model
a strong inductive bias.
---------------------------------------------------------------------------------------------------------------------


---------------------------------------------------------------------------------------------------------------------
relative position embedding.
fq(xm):=Wq@xm
fk(xn, n):=Wk(xn+pkr)
fv(xn, n):=Wv(xn+pvr)
Notice that adsolute position embedding apply all position embedding for qkv
But the relative position embedding apply all position embedding for kv, not q, just like the expression above.
Not all relative position embedding like this. some apply for qkv.

what is pkr, pvr?
r is the relative position value, just like "who am i", the second token am have the relative position {max(2-1, +kmax) min(2-1, -kmax)} 
to the first token who, if the (-kmax, +kmax) is (-2, 2), then r is equal to 1.
what the value for the pkr? Supposed that we have one trainable weight what name is relative position embedding table, what dimension is
(2*kmax, d_model), why d_model, because the hidden size and embedding size, why 2*kmax? because the index is greater than zero or equal to zero.
we should index the relaive position from the relative position embedding table. The index must be greater than zero or equal to zero.
And the minvalue of r is -kmax, we should use the index value as r+kmax to get the relative position embedding from the relative position embedding table.
And the table will be optimized during bp. Then pkr = relative_position_embedding[r+kmax], what the dimension of relative_position_embedding is (2*kmax, d_model),
what the r+kmax is range from 0 to 2*kmax, what the r is range from -kmax to kmax, what the r is equal to the clip(current_token_position-target_token_position, -kmax, +kmax)
---------------------------------------------------------------------------------------------------------------------








"""
import torch
import torch.nn as nn
from typing import (
    Tuple
)

from whoami.tool.transformer.model_configs import TLMoModelConfig
from whoami.tool.transformer.components import BufferCache


def _non_meta_init_device(config: TLMoModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    

class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings (RoPE)
    """     
    def __init__(self, config: TLMoModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))
    
    
    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
        
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # the dimension of q, k is (batch_size, num_heads, sequence_length, head_dim)
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k
        
        with torch.autocast(q.device.type, enabled=False):
            # 禁用自动混合精度
            query_len, key_len = q_.shape[-2], k_.shape[-2]

            
        pass    
