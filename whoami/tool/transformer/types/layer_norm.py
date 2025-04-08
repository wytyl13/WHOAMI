#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/06 20:34:31
@Author : weiyutao
@File : layer_norm.py
"""
from whoami.tool.transformer.utils import StrEnum


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