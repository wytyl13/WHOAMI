#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/18 10:11
@Author  : weiyutao
@File    : detect.py
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
import numpy as np
from pydantic import BaseModel, model_validator, ValidationError
from ultralytics import YOLO, RTDETR

from whoami.utils.log import Logger

class Detector(BaseModel, ABC):
    name: Optional[str] = None
    model: Optional[Union[YOLO, RTDETR]] = None
    model_path: Optional[str] = None
    class_list: Optional[list[int]] = None
    conf: Optional[float] = None
    logger: Optional[Logger] = None
    
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型
    
    @abstractmethod
    def __init__(
        self, 
        name: Optional[str] = None,
        model: Optional[Union[YOLO, RTDETR]] = None, 
        model_path: Optional[str] = None,
        class_list: Optional[list[int]] = None,
        conf: Optional[float] = None,
        logger: Optional[Logger] = None
    ):
        super().__init__(name=name, model=model, model_path=model_path, class_list=class_list, conf=conf)
        self._valid_model()
        if hasattr(self, 'logger'):  # 检查是否已经初始化
            return
        
    def __str__(self):
        return f"{self.__class__.__name__} -- name: {self.name}, model_path: {self.model_path}, class_list: {self.class_list}, conf: {self.conf}"
    
    def _valid_model(self):
        if self.model is None and self.model_path is None:
            raise ValueError('model, model_path must not be null!')
        return True
        
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
    @abstractmethod
    def valid_model_model_path(cls, data: any):
        """valid model and model path and init them implemented by inherited class"""
        return data
    
    @abstractmethod
    def set_model(self, model_path):
        """set model used model_path implemented by inherited class"""
    
    @abstractmethod
    def predict(
        self, 
        image: Optional[Union[str, np.ndarray]] = None
    ):
        """predict the image implemented by inherited class"""
    
    
    
    