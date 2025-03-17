#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/06 17:16
@Author  : weiyutao
@File    : memory_manager.py
"""


import tracemalloc
import psutil
import os
import gc

class MemoryMonitor:
    def __init__(self, log_threshold_mb=1024*30):
        self.log_threshold = log_threshold_mb
        self.process = psutil.Process(os.getpid())
    
    def check_memory_usage(self):
        # 获取内存使用情况（以MB为单位）
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if memory_mb > self.log_threshold:
            self.logger.warning(f"High memory usage: {memory_mb:.2f} MB")
            # 可以添加额外的内存释放逻辑
            gc.collect()
        return memory_mb