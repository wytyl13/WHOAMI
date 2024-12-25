#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/24 11:36
@Author  : weiyutao
@File    : base_provider.py
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator, ValidationError
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
)

from whoami.utils.log import Logger

class BaseProvider(ABC, BaseModel):
    name: Optional[str] = None
    logger: Optional[Logger] = None
    
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型
    
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
    
    @model_validator(mode="before")
    @classmethod
    def set_name_if_empty(cls, values):
        if "name" not in values or not values["name"]:
            values["name"] = cls.__name__
        return values
    
    @model_validator(mode="before")
    @classmethod
    def set_logger_if_empty(cls, values):
        if "logger" not in values or not values["logger"]:
            values["logger"] = Logger(cls.__name__)
        return values