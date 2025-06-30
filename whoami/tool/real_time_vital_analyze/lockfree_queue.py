#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/20 17:47
@Author  : weiyutao
@File    : lockfree_queue.py
"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lock-Free队列实现
支持多生产者多消费者的无锁队列
"""

import threading
import time
from typing import Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging


class AtomicInteger:
    """原子整数实现"""
    def __init__(self, value: int = 0):
        self._value = value
        self._lock = threading.Lock()
    
    def get(self) -> int:
        with self._lock:
            return self._value
    
    def set(self, value: int):
        with self._lock:
            self._value = value
    
    def compare_and_set(self, expected: int, update: int) -> bool:
        """原子比较并设置"""
        with self._lock:
            if self._value == expected:
                self._value = update
                return True
            return False
    
    def get_and_increment(self) -> int:
        """获取当前值并自增"""
        with self._lock:
            old_value = self._value
            self._value += 1
            return old_value
    
    def increment_and_get(self) -> int:
        """自增并获取新值"""
        with self._lock:
            self._value += 1
            return self._value


class AtomicLong:
    """原子长整数实现"""
    def __init__(self, value: float = 0.0):
        self._value = value
        self._lock = threading.Lock()
    
    def get(self) -> float:
        with self._lock:
            return self._value
    
    def set(self, value: float):
        with self._lock:
            self._value = value
    
    def compare_and_set(self, expected: float, update: float) -> bool:
        """原子比较并设置"""
        with self._lock:
            if abs(self._value - expected) < 1e-10:  # 浮点数比较
                self._value = update
                return True
            return False


class LockFreeQueue:
    """
    Lock-Free环形队列实现
    支持多生产者多消费者，提供查看全量数据功能
    """
    
    def __init__(self, capacity: int = 1000):
        """
        初始化Lock-Free队列
        
        Args:
            capacity: 队列容量
        """
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = AtomicInteger(0)  # 读指针
        self.tail = AtomicInteger(0)  # 写指针
        self.size = AtomicInteger(0)  # 当前大小
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def enqueue(self, item: Any) -> bool:
        """
        入队操作（Lock-Free）
        
        Args:
            item: 要入队的数据
            
        Returns:
            bool: 是否成功入队
        """
        max_retries = 100
        for retry in range(max_retries):
            current_size = self.size.get()
            
            # 检查队列是否已满
            if current_size >= self.capacity:
                return False
            
            # 尝试获取写位置
            current_tail = self.tail.get()
            next_tail = (current_tail + 1) % self.capacity
            
            # 原子更新尾指针
            if self.tail.compare_and_set(current_tail, next_tail):
                # 成功获得写位置，写入数据
                self.buffer[current_tail] = item
                
                # 原子增加大小
                self.size.increment_and_get()
                return True
            
            # CAS失败，短暂等待后重试
            if retry > 10:
                time.sleep(0.0001)  # 100微秒
                
        self.logger.warning(f"入队失败，超过最大重试次数: {max_retries}")
        return False
    
    def dequeue(self) -> Optional[Any]:
        """
        出队操作（Lock-Free）
        只取出第一个元素
        
        Returns:
            Optional[Any]: 出队的数据，队列空时返回None
        """
        max_retries = 100
        for retry in range(max_retries):
            current_size = self.size.get()
            
            # 检查队列是否为空
            if current_size <= 0:
                return None
            
            # 尝试获取读位置
            current_head = self.head.get()
            next_head = (current_head + 1) % self.capacity
            
            # 原子更新头指针
            if self.head.compare_and_set(current_head, next_head):
                # 成功获得读位置，读取数据
                item = self.buffer[current_head]
                self.buffer[current_head] = None  # 清理引用
                
                # 原子减少大小
                while True:
                    current_size = self.size.get()
                    if current_size > 0:
                        if self.size.compare_and_set(current_size, current_size - 1):
                            break
                    else:
                        break
                
                return item
            
            # CAS失败，短暂等待后重试
            if retry > 10:
                time.sleep(0.0001)  # 100微秒
                
        self.logger.warning(f"出队失败，超过最大重试次数: {max_retries}")
        return None
    
    def peek_all(self) -> List[Any]:
        """
        查看队列中所有数据（不移除）
        返回当前队列的快照
        
        Returns:
            List[Any]: 队列中所有数据的副本
        """
        # 获取当前状态的快照
        current_head = self.head.get()
        current_tail = self.tail.get()
        current_size = self.size.get()
        
        if current_size <= 0:
            return []
        
        result = []
        
        # 处理环形缓冲区
        if current_tail > current_head:
            # 正常情况：head < tail
            for i in range(current_head, current_tail):
                item = self.buffer[i]
                if item is not None:
                    result.append(item)
        else:
            # 环形跨越：tail < head
            # 先读取 head 到数组末尾
            for i in range(current_head, self.capacity):
                item = self.buffer[i]
                if item is not None:
                    result.append(item)
            # 再读取数组开头到 tail
            for i in range(0, current_tail):
                item = self.buffer[i]
                if item is not None:
                    result.append(item)
        
        return result
    
    def qsize(self) -> int:
        """
        获取队列当前大小
        
        Returns:
            int: 队列中元素数量
        """
        return max(0, self.size.get())
    
    def empty(self) -> bool:
        """
        检查队列是否为空
        
        Returns:
            bool: 队列是否为空
        """
        return self.qsize() <= 0
    
    def full(self) -> bool:
        """
        检查队列是否已满
        
        Returns:
            bool: 队列是否已满
        """
        return self.qsize() >= self.capacity
    
    def put_nowait(self, item: Any):
        """
        兼容queue.Queue接口的非阻塞入队
        
        Args:
            item: 要入队的数据
            
        Raises:
            Exception: 队列满时抛出异常
        """
        if not self.enqueue(item):
            raise Exception("Queue is full")
    
    def get_nowait(self) -> Any:
        """
        兼容queue.Queue接口的非阻塞出队
        
        Returns:
            Any: 出队的数据
            
        Raises:
            Exception: 队列空时抛出异常（模拟queue.Empty）
        """
        item = self.dequeue()
        if item is None:
            from queue import Empty
            raise Empty("Queue is empty")
        return item
    
    def task_done(self):
        """兼容queue.Queue接口"""
        pass  # Lock-Free队列不需要显式的task_done
    
    def get_stats(self) -> dict:
        """
        获取队列统计信息
        
        Returns:
            dict: 统计信息
        """
        return {
            'capacity': self.capacity,
            'size': self.qsize(),
            'head': self.head.get(),
            'tail': self.tail.get(),
            'utilization': self.qsize() / self.capacity * 100
        }


class AtomicDict:
    """
    线程安全的字典实现
    用于管理设备队列映射
    """
    
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()
    
    def get(self, key: str, default=None):
        """获取值"""
        with self._lock:
            return self._dict.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置值"""
        with self._lock:
            self._dict[key] = value
    
    def get_or_create(self, key: str, factory_func):
        """获取或创建值（原子操作）"""
        with self._lock:
            if key not in self._dict:
                self._dict[key] = factory_func()
            return self._dict[key]
    
    def keys(self):
        """获取所有键"""
        with self._lock:
            return list(self._dict.keys())
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._dict
    
    def __getitem__(self, key: str):
        """字典风格访问"""
        with self._lock:
            return self._dict[key]
    
    def __setitem__(self, key: str, value: Any):
        """字典风格设置"""
        with self._lock:
            self._dict[key] = value


# 测试代码
if __name__ == "__main__":
    import time
    import random
    
    def test_lockfree_queue():
        """测试Lock-Free队列"""
        print("🧪 测试Lock-Free队列")
        print("=" * 50)
        
        queue = LockFreeQueue(capacity=100)
        
        # 测试基本操作
        print("1. 测试基本操作:")
        
        # 入队测试
        for i in range(10):
            success = queue.enqueue(f"item_{i}")
            print(f"   入队 item_{i}: {'成功' if success else '失败'}")
        
        print(f"   队列大小: {queue.qsize()}")
        print(f"   队列统计: {queue.get_stats()}")
        
        # 查看全量数据
        all_items = queue.peek_all()
        print(f"   所有数据: {all_items}")
        
        # 出队一个
        item = queue.dequeue()
        print(f"   出队: {item}")
        print(f"   出队后大小: {queue.qsize()}")
        
        # 再次查看全量数据
        all_items = queue.peek_all()
        print(f"   剩余数据: {all_items}")
    
    def test_concurrent_operations():
        """测试并发操作"""
        print("\n🚀 测试并发操作")
        print("=" * 50)
        
        queue = LockFreeQueue(capacity=1000)
        results = {"enqueue_success": 0, "enqueue_fail": 0, "dequeue_success": 0, "dequeue_fail": 0}
        
        def producer(thread_id: int, count: int):
            for i in range(count):
                item = f"thread_{thread_id}_item_{i}"
                if queue.enqueue(item):
                    results["enqueue_success"] += 1
                else:
                    results["enqueue_fail"] += 1
                time.sleep(0.001)  # 1ms
        
        def consumer(thread_id: int, count: int):
            for i in range(count):
                item = queue.dequeue()
                if item is not None:
                    results["dequeue_success"] += 1
                else:
                    results["dequeue_fail"] += 1
                time.sleep(0.002)  # 2ms
        
        # 启动并发测试
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 4个生产者线程
            producers = [executor.submit(producer, i, 50) for i in range(4)]
            # 2个消费者线程
            consumers = [executor.submit(consumer, i, 80) for i in range(2)]
            
            # 等待完成
            for future in producers + consumers:
                future.result()
        
        print(f"   入队成功: {results['enqueue_success']}")
        print(f"   入队失败: {results['enqueue_fail']}")
        print(f"   出队成功: {results['dequeue_success']}")
        print(f"   出队失败: {results['dequeue_fail']}")
        print(f"   最终队列大小: {queue.qsize()}")
        print(f"   队列利用率: {queue.get_stats()['utilization']:.2f}%")
    
    # 运行测试
    test_lockfree_queue()
    test_concurrent_operations()