#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/07 11:18
@Author  : weiyutao
@File    : utils.py
"""

from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)