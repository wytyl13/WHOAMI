#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/30 16:52
@Author  : weiyutao
@File    : base_tool.py
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
    Any,
    Type,
    Literal,
    overload,
)
from whoami.utils.log import Logger


class BaseTool(ABC, BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    args_schema: Optional[BaseModel] = None
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
    
    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Implement the function in child class.
        """
    
    def args(self):
        return self.model_json_schema(self.args_schema)['properties']
    
    
    def model_json_schema(
        self, 
        cls: Type[BaseModel],
        mode: Literal['validation', 'serialization'] = 'validation'
    ) -> dict[str, Any]:
        """
        为模型生成 JSON Schema，包含 Field 的 description。

        Args:
            cls: 要生成 Schema 的模型类
            mode: Schema 的模式 ('validation' 或 'serialization')

        Returns:
            dict: 生成的 JSON Schema
        """

        if cls is BaseModel:
            raise AttributeError('不能直接在 BaseModel 上调用，必须使用其子类')

        schema = {
            'type': 'object',
            'properties': {},
            'required': []
        }

        for field_name, field in cls.__annotations__.items():
            # 获取字段基本信息
            field_schema = {'type': 'string'}  # 默认为string类型
            
            # 获取字段额外信息（如description）
            if hasattr(cls, '__fields__'):
                field_info = cls.__fields__[field_name]
                if hasattr(field_info, 'description'):
                    field_schema['description'] = field_info.description
            
            schema['properties'][field_name] = field_schema
            
            # 如果字段没有默认值，则为必需字段
            if not hasattr(cls, field_name) or getattr(cls, field_name) is ...:
                schema['required'].append(field_name)

        return schema

    def _get_field_schema(self, field_type: Type) -> dict[str, Any]:
        """生成字段的 schema"""
        type_mapping = {
            str: {'type': 'string'},
            int: {'type': 'integer'},
            float: {'type': 'number'},
            bool: {'type': 'boolean'},
            list: {'type': 'array', 'items': {}},
            dict: {'type': 'object'}
        }
        
        return type_mapping.get(field_type, {'type': 'object'})
        