#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/01 15:58
@Author  : weiyutao
@File    : resource_manager.py
"""
import threading
from collections import defaultdict
import queue
from concurrent.futures import ThreadPoolExecutor

class ResourceManager():
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instantask_infoce
    
    def __init__(self):
        
        # 数据存储队列
        self.frame_queues = {} # 帧队列
        self.task_queues = defaultdict(lambda: queue.Queue(maxsize=100)) # 任务模型队列
        self.result_queues = defaultdict(lambda: queue.Queue(maxsize=100)) # 结果队列

        # 线程
        self.stream_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="Stream") # 视频流线程池
        self.inference_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="Inference") # 模型推理线程池
        self.result_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="Result") # 结果处理线程池

        self.active_streams = {} # 追踪活跃视频流
        self.model_cache = {} # 模型实例缓存，避免重复加载
        self.running = False # 是否停止所有线程
        
        self.batch_size = defaultdict(lambda: 8) # 设置批处理大小
        self.task_info = {} # 记录任务信息

    def set_running(self, state: bool):
        self.running = state
    
    def get_frame_queue(self, topic_id):
        if topic_id not in self.frame_queues:
            self.frame_queues[topic_id] = queue.Queue(maxsize=100)
        return self.frame_queues[topic_id]
    
    def get_model(self, topic_name):
        if topic_name not in self.model_cache:
            self.model_cache[topic_name] = self.load_model(topic_name)
        return self.model_cache[topic_name]
    
    def _load_model(self, topic_name):
        """abstract method what need to implement in child class."""
        return None

    def register_task(self, task_id, detector_instance, topic_name, device_sn, stream_url):
        self.task_info[task_id] = {
            'detector': detector_instance,
            'topic_name': topic_name,
            'device_sn': device_sn,
            'stream_url': stream_url,
            'active': True
        }
    
    def remove_task(self, task_id):
        if task_id in self.task_info:
            self.task_info[task_id]['active'] = False
            
    
    def set_batch_size(self, topic_name, batch_size):
        self.batch_size[topic_name] = batch_size
        
    def shutdown(self):
        """关闭所有线程池和资源"""
        self.running = False
        self.stream_executor.shutdown(wait=False)
        self.inference_executor.shutdown(wait=False)
        self.result_executor.shutdown(wait=False)
        self.frame_queues.clear()
        self.task_queues.clear()
        self.result_queues.clear()
        self.active_streams.clear()
        self.model_cache.clear()
        self.task_info.clear()
        


        
        
        
        
        
        
        

