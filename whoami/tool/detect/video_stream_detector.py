#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/18 16:43
@Author  : weiyutao
@File    : video_stream_detector.py
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator, ValidationError, field_validator, validator
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
import inspect
import os
import cv2
import time
import json

from whoami.tool.detect.detector_warning import DetectorWarning
from whoami.tool.detect.detector import Detector
from whoami.utils.log import Logger
from whoami.utils.utils import Utils
from whoami.configs.detector_config import DetectorConfig

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROGRAM_ROOT_DIRECTORY = os.path.abspath(os.path.join(ROOT_DIRECTORY, "../../"))

class VideoStreamDetector(BaseModel, ABC):
    name: Optional[str] = None
    detector_warning: Optional[DetectorWarning] = None
    device_sn: Optional[str] = None
    detector: Optional[Detector] = None
    stream_url: Optional[str] = None
    url_str_flag: Optional[str] = None
    config_path: Optional[str] = None
    config: Optional[dict] = None
    sql_connection: any = None
    topic_name: Optional[str] = None
    logger: Optional[Logger] = None
    
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型
    
    @abstractmethod
    def __init__(
        self, 
        name: Optional[str] = None,
        detector_warning: Optional[DetectorWarning] = None,
        device_sn: Optional[str] = None,
        detector: Optional[Detector] = None,
        stream_url: Optional[str] = None,
        url_str_flag: Optional[str] = None,
        config_path: Optional[str] = None,
        topic_name: Optional[str] = None,
    ):
        super().__init__(
           name=name, 
           detector_warning=detector_warning, 
           device_sn=device_sn, 
           stream_url=stream_url, 
           url_str_flag=url_str_flag,
           config_path=config_path, 
           detector=detector,
           topic_name=topic_name,
        )
        # self.url_str_flag = url_str_flag
        # 注意如果以下初始化方法中使用了子类的构造函数中的变量，那么将达不到你想要的效果
        # 因为父类的构造函数会在子类的构造函数之前执行。解决办法：在子类的构造函数中初始化
        # 最终的解决办法是不再在子类中定义该特殊变量，因为他需要再父类的init构造函数中用到
        self._valid_init(config_path, topic_name, detector)
        self._valid_detector_warning(detector_warning)
    
    
    def set_device_sn(self, device_sn):
        self.device_sn = device_sn
        self._valid_detector_warning()
        
    def set_topic_name(self, topic_name):
        self.topic_name = topic_name
        self._valid_detector_variable_init()
    
    def set_url_str_flag(self, url_str_flag):
        self.url_str_flag = url_str_flag
        self._valid_sql_connection_init()
        self._valid_detector_warning()
       
    def get_video_stream_url(self, device_sn: Optional[str] = None):
        """
        if you want to get video stream url by requesting one post api, you
        should overwrite this method. or you will fail to process this class.
        """
        pass
        
    # def __str__(self):
    #     return f"{self.__class__.__name__} --- name: {self.name}, detector_warning: {self.detector_warning}, device_sn: {self.device_sn}, detector: {self.detector}, stream_url: {self.stream_url}, config_path: {self.config_path}, sql_connection: {self.sql_connection}, topic_name: {self.topic_name}"
    
    def _valid_init(self, config_path, topic_name, detector):
        """notice the valid init order"""
        self._valid_config_init(config_path)
        self._valid_sql_connection_init()
        self._valid_topic_name_init(topic_name)
        self._valid_detector_variable_init(detector)
    
    def _valid_sql_connection_init(self):
        if not self._check_function_code('get_sql_connection') and 'sql' in self.config:
            raise ValueError("you should overwrite the function get_sql_connection when you hava passed the sql config in your config file!")
        if self.__class__._check_function_code('get_sql_connection'):
            self.sql_connection = self.get_sql_connection()
            if not self.__class__._check_function_code('check_sql_video_stream_status'):
                raise ValueError("you should overwrite the function check_sql_video_stream_status when you hava passed the sql config in your config file!")
            if not self.__class__._check_function_code('update_sql_video_stream_status'):
                raise ValueError("you should overwrite the function update_sql_video_stream_status when you hava passed the sql config in your config file!")
        else:
            self.sql_connection = None
        return True
    
    def _valid_config_init(self, config_path):
        self.config_path = config_path if config_path is not None else self.config_path
        self.config = DetectorConfig.from_file(self.config_path).__dict__ if self.config_path is not None else self.config
        return True
    
    def _valid_topic_name_init(self, topic_name):
        self.topic_name = topic_name
        if self.topic_name is None:
            raise ValueError("topic must not be null!")
    
    def _valid_detector_variable_init(self, detector: Detector = None):
        """vaild and init the variable class_list and conf in detector class."""
        """conf can not be null, must less than 1.0 and greater than 0.0"""
        """class_list can be none, none means it is an empty list: []"""
        self.detector = detector if detector is not None else self.detector
        
        if self.detector is None:
            if "model_path" in self.config and self.topic_name in self.config["model_path"]:
                self.detector = self._init_detector(os.path.join(PROGRAM_ROOT_DIRECTORY, self.config["model_path"][self.topic_name]))
            else:
                raise ValueError("fail to init model in detector")
        
        if self.detector.class_list is None:
            if "class_list" in self.config:
                self.detector.class_list = self.config["class_list"][self.topic_name]
            # else:
            #     raise ValueError("the class_list in detector class or class_list in config must not be none!")
        if self.detector.conf is None:
            if "conf" in self.config:
                if self.device_sn in self.config["conf"]:
                    self.detector.conf = self.config["conf"][self.device_sn][self.topic_name]
                elif 'default' in self.config["conf"]:
                    self.detector.conf = self.config["conf"]["default"][self.topic_name]
                elif self.topic_name in self.config["conf"]:
                    self.detector.conf = self.config[self.topic_name]
                else:
                    raise ValueError("the conf in detector class or conf in config must not be none!")
            else:
                raise ValueError("the conf in detector class or conf in config must not be none!")
        return True
    
    @abstractmethod  
    def _init_detector(self, model_path):
        """init detector implemented by inherited class."""
    
    @abstractmethod 
    def _valid_detector_warning(self, detector_warning):
        """init detector warning implemented by inherited class."""
    
    """this field_validator code has dropped because we must valid and init these
        two class instance after initing the construct function.
    """
    # @field_validator('detector', mode='before')
    # def validate_detector(cls, value):
    #     if value is None and not isinstance(value, Detector):
    #         raise TypeError("detector must be an instance of Detector.")
    #     return value
    
    # @field_validator('detector_warning', mode='before')
    # def validate_warning(cls, value):
    #     """field_validator is suitable to valid single one field."""
    #     """it will be exec after initing one fields but before finishing to init all the fields"""
    #     if value is not None and not isinstance(value, DetectorWarning):
    #         raise TypeError("detector_warning must be an instance of DetectorWarning.")
    #     return value
    
    @model_validator(mode="before")
    @classmethod
    def set_name_if_empty(cls, value):
        """model_validator before is suitable to valid between all field."""
        """it will be exec after initing all fields but before the construct function __init__"""
        if "name" not in value or not value["name"]:
            value["name"] = cls.__name__
        return value
    
    @abstractmethod
    def tostring(self):
        return {
            "name": self.name,
            "detector_warning": self.detector_warning, 
            "device_sn": self.device_sn, 
            "detector": self.detector, 
            "stream_url": self.stream_url, 
            "config_path": self.config_path, 
            "sql_connection": self.sql_connection, 
            "topic_name": self.topic_name, 
            "url_str_flag": self.url_str_flag
        }
    
    @model_validator(mode="before")
    @classmethod
    def set_logger_if_empty(cls, value):
        """before意味着该验证逻辑会在所有属性初始化之前执行，当然也会在init函数之前执行"""
        """也就是意味着这个验证逻辑会在读取硬编码之前"""
        """before __str__"""
        if "logger" not in value or not value["logger"]:
            value["logger"] = Logger(cls.__name__)
        return value
    
    @model_validator(mode="after")
    @classmethod
    def _valid_device_sn_stream_url(cls, data: any):
        """after意味着该验证逻辑会在所有属性初始化以后但是在init函数之前执行"""
        """after __str__"""
        if data.device_sn is None and data.stream_url is None:
            error_info = 'Either device_sn or stream_url must be provided!'
            raise ValueError(error_info)
        if data.stream_url is None and not cls._check_function_code('get_video_stream_url'):
            error_info = 'Either stream_url attribution or get_video_stream_url method must be provided!'
            raise ValueError(error_info)
        return data
    
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
    
    @classmethod
    def _check_function_code(cls, func_name: str):
        try:
            method = getattr(cls, func_name)
            if method:
                source = inspect.getsource(method).strip()
        except Exception as e:
            error_info = 'fail to exec _check_function_code!'
            raise RuntimeError(error_info) from e
        if source.endswith("pass"):
            return False
        else:
            return True
    
    @model_validator(mode="before")
    @classmethod
    @abstractmethod
    def _valid_before(cls, values):
        """model_validator after will exec after the construction function __init__"""
        """valid model and model path and init them implemented by inherited class"""
        return values
    
    @model_validator(mode="after")
    @classmethod
    @abstractmethod
    def _valid_after(cls, data: any):
        """model_validator after will exec after the construction function __init__"""
        """valid model and model path and init them implemented by inherited class"""
        return data

    def get_sql_connection(self):
        """overwrite the sql method, notice if the sql method based on the special attribution of child class. you should 
        reset the sql after change the special attribution.
        """
        pass
    
    def check_sql_video_stream_status(self):
        pass
    
    def update_sql_video_stream_status(self):
        pass
    
    def truncate_sql_table(self):
        pass
    
    @abstractmethod
    def get_real_topic_list(self):
        """get real topic list implemented by inherited class"""
        
    @abstractmethod
    def get_topic_based_topic_name(self):
        """get real topic list implemented by inherited class"""
    
    @abstractmethod
    def get_warning_information(self, results):
        """set the warning information based on the predict result implemented by inherited class."""

    def process(self):
        # self.logger.info(self.detector)
        # common logical code. 
        # init stream url
        # need not to handle this exception, if happend, stop the process directly.
        self.stream_url = self.stream_url if self.stream_url else self.get_video_stream_url()
        topic = self.get_topic_based_topic_name()
        
        def recursive_process():
            """recursive recall this function if the running process occurs error!"""
            
            # need not to handle this exception, if happend, stop the process directly.
            real_topic_list = self.get_real_topic_list()
            cap = cv2.VideoCapture(self.stream_url)
            last_sample_time = time.perf_counter()
            
            if not cap.isOpened():
                self.logger.error(f"fail to open stream_url: {self.stream_url}")
                if topic in real_topic_list:
                    real_topic_list.remove(topic)
                # need not handle this exception, if happend, stop the process directly.
                self.update_sql_video_stream_status(real_topic_list)
                self.stream_url = self.stream_url if self.stream_url else self.get_video_stream_url()
                return recursive_process()
            
            # # 是否可以直接访问子类独有的属性？面向对象的多态设计思想是否可用？明天尝试
            # 不需要这个参数，直接为none即可
            # self.detector_warning.stream_url = self.stream_url
            
            # init the detector warning
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.logger.error(f"fail to read each fram from stream url: {self.stream_url}!")
                    if topic in real_topic_list:
                        real_topic_list.remove(topic)
                    self.update_sql_video_stream_status(real_topic_list)
                    self.stream_url = self.stream_url if self.stream_url else self.get_video_stream_url()
                    return recursive_process()
                
                if frame is None or frame.size == 0:
                    self.logger.error(f"fail to read each frame from stream url: {self.stream_url}")
                    if topic in real_topic_list:
                        real_topic_list.remove(topic)
                    self.update_sql_video_stream_status(real_topic_list)
                    self.stream_url = self.stream_url if self.stream_url else self.get_video_stream_url()
                    return recursive_process()
                current_time = time.perf_counter()
                if current_time - last_sample_time >= (0.5 / 100):
                    task_id = self.stream_url + topic
                    self.logger.info(f"{task_id} is running!")

                    # break process if predict image exception occurs.
                    results = self.detector.predict(frame)
                    
                    if topic not in real_topic_list:
                        real_topic_list.append(topic)
                        # break process if update sql data exception occurs.
                        self.update_sql_video_stream_status(real_topic_list)
                    
                    # continue process if warning information exception occurs.
                    try:
                        warning_flag, warning_information = self.get_warning_information(results)
                        if warning_flag:
                            self.detector_warning.warning(warning_information)
                    except Exception as e:
                        continue
        return recursive_process()
        
