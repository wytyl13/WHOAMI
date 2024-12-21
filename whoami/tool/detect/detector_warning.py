#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/18 16:46
@Author  : weiyutao
@File    : detector_warning.py
"""
from abc import ABC, abstractmethod
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
from pydantic import BaseModel, model_validator, ValidationError
import os
import time

from whoami.utils.log import Logger
from whoami.utils.utils import Utils

utils = Utils()
ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class DetectorWarning(BaseModel, ABC):
    name: Optional[str] = None
    logger: Optional[Logger] = None
    config_path: Optional[str] = None
    config: Optional[dict] = None
    pre_warning_time: Optional[float] = None
    warning_gap: Optional[int] = None
    warning_infomation: any = None
    
    class Config:
        arbitrary_types_allowed = True
        
    @abstractmethod
    def __init__(
        self, 
        name: Optional[str] = None,
        config_path: Optional[str] = None,
        warning_gap: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            config_path=config_path,
            warning_gap=warning_gap,
        )
        self.config_path = config_path if config_path is not None else self.config_path
        self.warning_gap = warning_gap if warning_gap is not None else self.warning_gap
        self.config = utils.read_yaml(self.config_path) if self.config_path is not None else self.config
        if 'warning_gap' not in self.config and self.warning_gap is None:
            raise ValueError("warning_gap must not be null!")
        if self.warning_gap is None:
            self.warning_gap = self.config['warning_gap']
        if hasattr(self, 'logger'):  # 检查是否已经初始化
            return
    
    def __str__(self):
        return f"{self.__class__.__name__} --- config_path: {self.config_path}, pre_warning_time: {self.pre_warning_time}, warning_gap: {self.warning_gap}, url_str_flag: {self.url_str_flag}"
     
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
    
    @model_validator(mode="after")
    @classmethod
    def _valid_config_after(cls, data: any):
        """model_validator after will exec after the construction function __init__"""
        """valid model and model path and init them implemented by inherited class"""
        """but i have found this valid will exec before __init__ function."""
        """but the init value in model_validator after will not overwrite the init value in construct function."""
        if "config_path" not in data or not data.config_path:
            data.config_path = os.path.join(ROOT_DIRECTORY, 'default_config.yaml')
        return data
    
    @abstractmethod
    def customer_send_warning(self, warning_information):
        """send warning function implemented by inherited class."""
    
    def warning(self, warning_information):
        """the warning information implemented by inherited class."""
        if self.pre_warning_time is None or (time.time() - self.pre_warning_time >= self.warning_gap):
            self.customer_send_warning(warning_information)
            self.pre_warning_time = time.time()
        return True
            
        
        
        
        
