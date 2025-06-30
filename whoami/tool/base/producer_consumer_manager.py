#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/04 10:29
@Author  : weiyutao
@File    : producer_consumer_manager.py
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
from pydantic import BaseModel, Field

from whoami.tool.base.base_tool import BaseTool
from whoami.provider.base_ import ModelType
from whoami.tool.base.consumer_tool_pool import ConsumerToolPool
from whoami.tool.base.thread_safe_dict import ThreadSafeDict
from whoami.tool.base.memory_manager import MemoryMonitor
from whoami.tool.base.bound_thread_pool import BoundedThreadPoolExecutor

class ProducerConsumerManager(BaseTool):

    active_production_lines: Dict[str, Any] = Field(default_factory=ThreadSafeDict)
    production_line_locks: Dict[str, Any] = Field(default_factory=ThreadSafeDict)
    production_line_stop_flags: Dict[str, Any] = Field(default_factory=ThreadSafeDict)
    # active_production_lines: dict[str, Type[ModelType]] = Field(default_factory=dict)
    # production_line_locks: Dict[str, threading.Lock] = Field(default_factory=dict)
    # production_line_stop_flags: Dict[str, bool] = Field(default_factory=dict)

    memory_monitor: MemoryMonitor = Field(default_factory=MemoryMonitor) # 内存管理

    producer_pool: ThreadPoolExecutor = None
    # consumer_pool: ThreadPoolExecutor = None
    consumer_pool: BoundedThreadPoolExecutor = None
    production_queue: queue.Queue = None

    consumer_worker_running: bool = True
    consumer_worker_thread: threading.Thread = None
    consumer_tool_pool: Any = None  # Adjust type as needed

    _is_running: bool = None

    def __init__(self, max_producers=20, max_consumers=30, production_queue_size=1000, consumer_tool_pool: ConsumerToolPool = None):
        super().__init__()
        """This is one base class instance what based on the producer and consumer model.
        Produce the product used the producer pool what max size is max_producer, Consume the
        product use the consumers pool what max size is max_consumers, what the product queue size is product_queue_size.
        """

        self.producer_pool = ThreadPoolExecutor(max_workers=max_producers)
        # self.consumer_pool = ThreadPoolExecutor(max_workers=max_consumers)
        self.consumer_pool = BoundedThreadPoolExecutor(max_workers=max_consumers, max_queue_size=30)
        self.production_queue = queue.Queue(maxsize=production_queue_size)

        # set the consumer thread status and start to consume the product use consumer function.
        self.consumer_worker_running = True
        # self.consumer_worker_thread = threading.Thread(target=self._start_consumer_worker)
        # self.consumer_worker_thread.daemon = True
        # self.consumer_worker_thread.start()

        self._is_running = True

        self.consumer_tool_pool: ConsumerToolPool = consumer_tool_pool # set the consumer tool pool if need.
        if self.consumer_tool_pool is None:
            raise ValueError("consumer_tool_pool must not be null!")

        def stop(self):
            self._is_running = False


    def set_consumer_tool_pool(self, consumer_tool_pool):
        self.consumer_tool_pool = consumer_tool_pool
    
    @abstractmethod
    def start_produce_worker(self, *args, **kwargs):
        """start produce work implemented by inherited class"""

    @abstractmethod
    def stop_produce_worker(self, production_id):
        """stop produce work based on production id implemented by inherited class"""
    
    @abstractmethod
    def _start_consumer_worker(self):
        """start consumer worker implemented by inherited class"""

    @abstractmethod
    def shutdown(self):
        """shut dowm the manager and all thread pools"""

