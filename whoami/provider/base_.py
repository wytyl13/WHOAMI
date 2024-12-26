#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/26 16:50
@Author  : weiyutao
@File    : base_.py
"""

from typing import (
    TypeVar
)
from sqlalchemy.orm import DeclarativeBase


# 定义基类
class Base(DeclarativeBase):
    pass

# 定义泛型类型变量
ModelType = TypeVar("ModelType", bound=Base)