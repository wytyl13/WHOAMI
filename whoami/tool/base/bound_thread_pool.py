#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/06 18:08
@Author  : weiyutao
@File    : bound_thread_pool.py
"""

from concurrent.futures import ThreadPoolExecutor
import queue

class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers=None, max_queue_size=None, thread_name_prefix=''):
        super().__init__(max_workers, thread_name_prefix)
        self._work_queue = queue.Queue(max_queue_size)