#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/06 20:23:54
@Author : weiyutao
@File : activation.py
"""

from whoami.tool.transformer.utils import StrEnum


class ActivationType(StrEnum):
    """Activation Type for any transformer model."""
    gelu = "gelu"
    relu = "relu"
    swiglu = "swigle"