#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/01 16:34
@Author  : weiyutao
@File    : video_stream_detector_test.py
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
    List,
)
import inspect
import os
import cv2
import time
import json
import gc
import queue

from whoami.tool.detect.detector_warning import DetectorWarning
from whoami.tool.detect.detector import Detector
from whoami.utils.log import Logger
from whoami.utils.utils import Utils
from whoami.configs.detector_config import DetectorConfig
from whoami.tool.detect.resource_manager import ResourceManager

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
    resource_manager: Optional[ResourceManager] = None
    
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
        # 初始化资源管理器
        self.resource_manager = ResourceManager.get_instance()
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
    
    
    def _video_reader_worker(self, task_id, topic_id, sampling_interval=0.3):
        # 视频流读取工作线程
        self.logger.info(f"Starting video reader for {task_id} with URL: {stream_url}")
        frame_queue = self.resource_manager.get_frame_queue(topic_id)

        # 无限循环，直到任务被标记为停止
        while self.resource_manager.running and task_id in self.resource_manager.task_info and self.resource_manager.task_info[task_id]['active']:
            try:
                # 打开视频流
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    self.logger.error(f"Failed to open stream: {stream_url}")
                    time.sleep(1)  # 等待一段时间再重试
                    continue
                
                # 获取视频信息
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.logger.info(f"Stream {task_id} opened: {width}x{height} @ {fps}fps")
                
                # 根据采样间隔计算需要跳过的帧数
                skip_frames = max(1, int(fps * sampling_interval))
                frame_count = 0
                
                # 读取视频帧
                while cap.isOpened() and self.resource_manager.running and task_id in self.resource_manager.task_info and self.resource_manager.task_info[task_id]['active']:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        self.logger.error(f"Failed to read frame from: {stream_url}")
                        break
                    frame_count += 1
                    # 根据采样间隔决定是否处理当前帧
                    if frame_count % skip_frames == 0:
                        # 如果队列已满，移除最旧的帧以确保处理最新的帧
                        if frame_queue.full():
                            try:
                                frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        # 添加帧到队列，包含元数据
                        metadata = {
                            'timestamp': time.time(),
                            'task_id': task_id,
                            'frame_id': frame_count,
                            'device_sn': self.device_sn,
                            'topic_name': self.topic_name
                        }
                        
                        try:
                            frame_queue.put((frame.copy(), metadata), block=False)
                        except queue.Full:
                            # 队列已满，跳过当前帧
                            pass
                # 视频流读取结束或出错，释放资源
                cap.release()
                # 等待一段时间再重新尝试连接
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in video reader for {task_id}: {str(e)}")
                time.sleep(1)
        self.logger.info(f"Video reader for {task_id} stopped")


    def _inference_worker(self, topic_name):
        # 模型推理工作线程
        self.logger.info(f"Starting inference worker for model: {topic_name}")
        
        # 获取当前模型对应的任务队列和结果队列
        task_queue = self.resource_manager.task_queues[topic_name]
        result_queue = self.resource_manager.result_queues[topic_name]
        batch_size = self.resource_manager.batch_sizes[topic_name]
        
        while self.resource_manager.running:
            try:
                batch_frames = []
                batch_metadata = []

                try:
                    start_time = time.time()
                    frame, metadata = task_queue.get(timeout=0.5)
                    batch_frames.append(frame)
                    batch_metadata.append(metadata)
                except queue.Empty:
                    continue
                
                # 非阻塞的尝试获取更多帧以填满批次
                while len(batch_frames) < batch_size:
                    try:
                        frame, metadata = task_queue.get_nowait()
                        batch_frames.append(frame)
                        batch_metadata.append(metadata)
                    except queue.Empty:
                        break
                
                if batch_frames:
                    batch_start_time = time.time()
                    self.logger.info(f"Processing batch of {len(batch_frames)} frames for model {topic_name}")
                    model = self.resource_manager.get_model(topic_name)
                    batch_results = model.predict_batch(batch_frames)
                    batch_end_time = time.time()
                    self.logger.info(f"Batch inference completed in {batch_end_time - batch_start_time:.3f}s")
                    # 将结果与元数据对应
                    for i, result in enumerate(batch_results):
                        result_queue.put((result, batch_metadata[i]))
            except Exception as e:
                self.logger.error(f"Error in inference worker for {topic_name}: {str(e)}")
                time.sleep(0.1)
        self.logger.info(f"Inference worker for model {topic_name} stopped")
        
    def _result_processer_worker(self, topic_name):
        # 结果处理工作线程
        self.logger.info(f"Starting result processor for model: {topic_name}")
        
        # 获取当前模型对应的结果队列
        result_queue = self.resource_manager.result_queues[topic_name]
        while self.resource_manager.running:
            try:
                # 从结果队列获取结果
                try:
                    result, metadata = result_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                task_id = metadata['task_id']
                device_sn = metadata['device_sn']
                topic_name = metadata['topic_name']
                
                # 检查任务是否还活跃
                if task_id not in self.resource_manager.task_info or not self.resource_manager.task_info[task_id]['active']:
                    continue
                
                # 处理警告信息
                try:
                    warning_flag, warning_information = self.get_warning_information(result)
                    if warning_flag:
                        # 如果检测到警告，发送警告
                        self.detector_warning.warning(warning_information)
                        self.logger.info(f"Warning sent for task {task_id}: {warning_information}")
                except Exception as e:
                    self.logger.error(f"Error processing result for {task_id}: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error in result processor for {topic_name}: {str(e)}")
                time.sleep(0.1)
                
    def _frame_collector_worker(self):
        """帧收集工作线程，从各个视频流队列收集帧并分配到相应的模型任务队列"""
        self.logger.info("Starting frame collector worker")
        while self.resource_manager.running:
            try:
                # 检查所有活跃的视频流
                for stream_url, frame_queue in self.resource_manager.frame_queues.items():
                    # 非阻塞地尝试获取帧
                    try:
                        frame, metadata = frame_queue.get_nowait()
                        task_id = metadata['task_id']
                        
                        # 检查任务是否还活跃
                        if task_id in self.resource_manager.task_info and self.resource_manager.task_info[task_id]['active']:
                            # 根据topic_name确定模型类型
                            topic_name = metadata['topic_name']
                            model_type = topic_name  # 根据您的应用可能需要映射
                            
                            # 将帧添加到对应模型的任务队列
                            task_queue = self.resource_manager.task_queues[model_type]
                            
                            # 如果队列已满，移除最旧的帧
                            if task_queue.full():
                                try:
                                    task_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            
                            # 添加帧到任务队列
                            try:
                                task_queue.put((frame, metadata), block=False)
                            except queue.Full:
                                # 队列已满，跳过
                                pass
                    
                    except queue.Empty:
                        # 队列为空，继续检查下一个
                        continue
                
                # 短暂休眠以避免CPU过载
                time.sleep(0.01)
            
            except Exception as e:
                self.logger.error(f"Error in frame collector: {str(e)}")
                time.sleep(0.1)
        
        self.logger.info("Frame collector worker stopped")
        
    def _database_updater_worker(self, update_interval=3.0):
        """数据库状态更新工作线程"""
        self.logger.info("Starting database updater worker")
        
        last_update_time = time.time()
        
        while self.resource_manager.running:
            try:
                current_time = time.time()
                
                # 每隔指定的时间更新一次数据库
                if current_time - last_update_time >= update_interval:
                    # 收集当前活跃的任务
                    active_topics = {}
                    for task_id, task_info in self.resource_manager.task_info.items():
                        if task_info['active']:
                            device_sn = task_info['device_sn']
                            topic = task_info['topic_name']
                            if device_sn not in active_topics:
                                active_topics[device_sn] = []
                            
                            # 获取完整的topic字符串
                            full_topic = self.get_topic_based_topic_name()
                            
                            if full_topic not in active_topics[device_sn]:
                                active_topics[device_sn].append(full_topic)
                    
                    # 更新数据库状态
                    for device_sn, topics in active_topics.items():
                        try:
                            self.update_sql_video_stream_status(topics)
                            self.logger.info(f"Updated database for device {device_sn} with topics: {topics}")
                        except Exception as e:
                            self.logger.error(f"Error updating database for device {device_sn}: {str(e)}")
                    
                    last_update_time = current_time
                
                # 短暂休眠
                time.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Error in database updater: {str(e)}")
                time.sleep(1)
        
        self.logger.info("Database updater worker stopped")
            
    def process(self):
        self.logger.info("Starting video stream detector process")
        self.stream_url = self.stream_url if self.stream_url else self.get_video_stream_url()
        if not self.stream_url:
            self.logger.error("No video stream URL available")
            return False
        # 获取完整的topic字符串
        topic = self.get_topic_based_topic_name()
        task_id = self.device_sn + topic
        
        # 注册任务
        self.resource_manager.register_task(
            task_id=task_id,
            detector_instance=self.detector,
            topic_name=self.topic_name,
            device_sn=self.device_sn,
            stream_url=self.stream_url
        )
        # 初始化和检查数据库状态
        try:
            real_topic_list = self.get_real_topic_list()
            if topic not in real_topic_list:
                real_topic_list.append(topic)
            self.update_sql_video_stream_status(real_topic_list)
        except Exception as e:
            self.logger.error(f"Error initializing database status: {str(e)}")
            return False
        
        # 启动视频流读取线程
        self.resource_manager.stream_executor.submit(
            self._video_reader_worker,
            task_id,
            self.stream_url,
            0.3  # 采样间隔，可以根据需要调整
        )
        
        # 启动帧收集器线程（如果尚未启动）
        if not hasattr(self.resource_manager, '_frame_collector_running'):
            self.resource_manager._frame_collector_running = True
            self.resource_manager.stream_executor.submit(self._frame_collector_worker)
            
        # 启动模型推理线程（如果该模型类型尚未有推理线程）
        if self.topic_name not in getattr(self.resource_manager, '_inference_workers', {}):
            self.resource_manager._inference_workers = getattr(self.resource_manager, '_inference_workers', {})
            self.resource_manager._inference_workers[self.topic_name] = True
            self.resource_manager.inference_executor.submit(self._inference_worker, self.topic_name)
            
        # 启动结果处理线程（如果该模型类型尚未有结果处理线程）
        if self.topic_name not in getattr(self.resource_manager, '_result_processors', {}):
            self.resource_manager._result_processors = getattr(self.resource_manager, '_result_processors', {})
            self.resource_manager._result_processors[self.topic_name] = True
            self.resource_manager.result_executor.submit(self._result_processor_worker, self.topic_name)
            
        # 启动数据库更新线程（如果尚未启动）
        if not hasattr(self.resource_manager, '_db_updater_running'):
            self.resource_manager._db_updater_running = True
            self.resource_manager.stream_executor.submit(self._database_updater_worker)
            
        self.logger.info(f"Process started for task {task_id}")
        return True
                    

                

