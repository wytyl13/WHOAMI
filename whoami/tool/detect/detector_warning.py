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
    ClassVar
)
from pydantic import BaseModel, model_validator, ValidationError
import os
import time
import threading

from whoami.utils.log import Logger
from whoami.configs.detector_config import DetectorConfig

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class DetectorWarning(BaseModel, ABC):
    name: Optional[str] = None
    logger: Optional[Logger] = None
    config_path: Optional[str] = None
    config: Optional[dict] = None
    pre_warning_time: Optional[float] = None
    warning_gap: Optional[int] = None
    warning_infomation: any = None
    lock: ClassVar[threading.Lock] = threading.Lock()
    last_warning_times: ClassVar[Dict[str, float]] = {}
    
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
        self.config = DetectorConfig.from_file(self.config_path).__dict__ if self.config_path is not None else self.config

        if 'warning_gap' not in self.config and self.warning_gap is None:
            raise ValueError("warning_gap must not be null!")
        if self.warning_gap is None:
            self.warning_gap = self.config['warning_gap']
        if hasattr(self, 'logger'):  # 检查是否已经初始化
            return
            
    @abstractmethod
    def tostring(self):
        return {
            "name": self.name,
            "config_path": self.config_path, 
            "pre_warning_time": self.pre_warning_time, 
            "warning_gap": self.warning_gap, 
            "warning_infomation": self.warning_infomation, 
        }
        
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
            data.config_path = os.path.join(ROOT_DIRECTORY, '../../configs/yaml/detect_config_case.yaml')
        return data
    
    @abstractmethod
    def customer_send_warning(self, *args, **kwargs):
        """send warning function implemented by inherited class."""
    
    def warning(self, *args, **kwargs):
        """the warning information implemented by inherited class."""
        # 使用 .get() 方法获取值，优先使用warning函数中传递的pre_warning_time和warning_gap参数
        warning_gap = kwargs.pop('warning_gap', None)
        topic = kwargs.pop('topic', None)
        # 如果没有传递这两个参数，直接使用类自身的
        if warning_gap is None:
            warning_gap = self.warning_gap
        current_time = time.time()
        try:
            with self.lock:
                pre_warning_time = self.last_warning_times.get(topic, 0)
                if current_time - pre_warning_time >= warning_gap:
                    self.last_warning_times[topic] = current_time
                    self.customer_send_warning(*args, **kwargs)
                    return True
        except Exception as e:
            error_info = f"fail to send warning information {str(e)}"
            self.logger.info(error_info)
            return False
        return False 
    
    def warning_back(self, *args, **kwargs):
        """the warning information implemented by inherited class."""
        if self.pre_warning_time is None or (time.time() - self.pre_warning_time >= self.warning_gap):
            self.customer_send_warning(*args, **kwargs)
            self.pre_warning_time = time.time()
        return True
            
        
        
        
        
