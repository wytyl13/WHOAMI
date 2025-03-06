#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/04 11:01
@Author  : weiyutao
@File    : consumer_tool_pool.py
"""
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
    Generic,
    TypeVar,
    Any,
    Type,
    List
)
from abc import ABC, abstractmethod
from queue import Queue

from whoami.tool.base.base_tool import BaseTool
from whoami.tool.detect.ultralitics_detector import UltraliticsDetector
from whoami.tool.base.model_info import ModelInfo
from whoami.tool.detect.detector import Detector

class ConsumerToolPool(BaseTool):

    pools: dict = None
    locks: dict = None

    def __init__(self, model_paths: Dict[str, ModelInfo], max_pool_size=5):
        super().__init__()
        """
        初始化检测器对象池
        
        :param model_paths: 模型路径字典 {topic: model_info}
        :param max_pool_size: 每个模型最大实例数
        """
        self.pools = {}
        self.locks = {}
        
        # 为每个模型创建线程安全的对象池
        for topic_name, model_info in model_paths.items():
            self.pools[topic_name] = Queue(maxsize=max_pool_size)
            self.locks[topic_name] = threading.Lock()
            
            # 预先创建实例
            for _ in range(max_pool_size):
                consumer_tool = model_info.init_model()
                self.pools[topic_name].put(consumer_tool)
    
    

    def get_consumer_tool(self, topic_name):
        """
        获取指定主题的检测器实例
        
        :param topic: 检测器主题
        :return: 检测器实例
        """
        if topic_name not in self.pools:
            raise ValueError(f"No detector pool for topic: {topic_name}")
        
        # 从池中获取实例
        detector: Detector = self.pools[topic_name].get()
        return detector
    
    def release_consumer_tool(self, topic_name, consumer_tool):
        """
        将检测器实例返回到池中
        
        :param topic: 检测器主题
        :param detector: 检测器实例
        """
        if topic_name not in self.pools:
            raise ValueError(f"No detector pool for topic: {topic_name}")
        
        # 将实例放回池中
        self.pools[topic_name].put(consumer_tool)
        self.logger.info(f"Released tool for {topic_name} back to pool")

    def _run(self, *args, **kwargs):
        pass