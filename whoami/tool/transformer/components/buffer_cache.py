#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/06 21:00:29
@Author : weiyutao
@File : buffer_cache.py
"""


import torch
import sys

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    """