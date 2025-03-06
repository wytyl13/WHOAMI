#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/04 10:29
@Author  : weiyutao
@File    : thread_management.py
"""
from dataclasses import dataclass, field
import threading
import time
from typing import Dict, Optional
import logging

from whoami.tool.base.base_tool import BaseTool
logger = logging.getLogger(__name__)

@dataclass
class ThreadInfo:
    thread: threading.Thread
    stop_event: threading.Event
    start_time: float = field(default_factory=time.time)
    
class ThreadManager():
    def __init__(self):
        self.threads: Dict[str, ThreadInfo] = {}
        self._lock = threading.Lock()
        
    def start_thread(self, thread_id: str, target_func, args=(), kwargs=None):
        """启动一个新线程并管理它"""
        with self._lock:
            if thread_id in self.threads:
                logger.warning(f"Thread {thread_id} already exists. Stopping it first.")
                self.stop_thread(thread_id)
                
            stop_event = threading.Event()
            kwargs = kwargs or {}
            kwargs['stop_event'] = stop_event
            
            thread = threading.Thread(
                target=self._wrap_target,
                args=(target_func, args, kwargs)
            )
            
            thread_info = ThreadInfo(
                thread=thread,
                stop_event=stop_event
            )
            
            self.threads[thread_id] = thread_info
            thread.start()
            logger.info(f"Started thread {thread_id}")
            
    def _wrap_target(self, target_func, args, kwargs):
        """包装目标函数，添加异常处理和清理"""
        try:
            target_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Thread error: {e}")
        finally:
            logger.info("Thread finished execution")
            
    def stop_thread(self, thread_id: str, timeout: float = 5.0) -> bool:
        """停止指定的线程"""
        with self._lock:
            if thread_id not in self.threads:
                logger.warning(f"Thread {thread_id} not found")
                return True
                
            thread_info = self.threads[thread_id]
            thread_info.stop_event.set()
            
            thread_info.thread.join(timeout=timeout)
            is_stopped = not thread_info.thread.is_alive()
            
            if is_stopped:
                del self.threads[thread_id]
                logger.info(f"Successfully stopped thread {thread_id}")
            else:
                logger.error(f"Failed to stop thread {thread_id} within timeout")
                
            return is_stopped
            
    def stop_all_threads(self, timeout: float = 5.0) -> bool:
        """停止所有线程"""
        all_stopped = True
        thread_ids = list(self.threads.keys())
        
        for thread_id in thread_ids:
            if not self.stop_thread(thread_id, timeout):
                all_stopped = False
                
        return all_stopped
        
    def get_running_threads(self) -> Dict[str, ThreadInfo]:
        """获取所有正在运行的线程信息"""
        with self._lock:
            return {
                thread_id: info 
                for thread_id, info in self.threads.items() 
                if info.thread.is_alive()
            }
            
    def cleanup_dead_threads(self):
        """清理已经结束的线程"""
        with self._lock:
            dead_threads = [
                thread_id 
                for thread_id, info in self.threads.items() 
                if not info.thread.is_alive()
            ]
            
            for thread_id in dead_threads:
                del self.threads[thread_id]
                logger.info(f"Cleaned up dead thread {thread_id}")