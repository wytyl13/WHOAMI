#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/20 18:16
@Author  : weiyutao
@File    : fixed_size_queue.py
"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版固定大小滑动队列实现
- 不使用填充值
- 新数据从后面进入，超过容量时最老的数据从前面出去
- 严格按照插入顺序：先进先出(FIFO)
"""

import threading
from typing import Any, List, Optional
import logging


class FixedSizeSlidingQueue:
    """
    简化版固定大小滑动窗口队列
    - 固定最大容量
    - 新数据append到末尾
    - 超过容量时，最老的数据(index 0)被移除
    - 严格FIFO：先进先出
    """
    
    def __init__(self, queue_capacity: int = 60):
        """
        初始化固定大小滑动队列
        
        Args:
            queue_capacity: 队列最大容量
        """
        if queue_capacity <= 0:
            raise ValueError("容量必须大于0")
            
        self.queue_capacity = queue_capacity
        self.buffer = []  # 空列表开始
        self._lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"初始化固定大小滑动队列，容量: {queue_capacity}")
    
    def enqueue(self, item: Any) -> Optional[Any]:
        """
        入队操作：新数据添加到末尾
        
        Args:
            item: 要入队的数据
            
        Returns:
            Optional[Any]: 如果队列满了，返回被挤出的最老数据；否则返回None
        """
        with self._lock:
            # 新数据添加到末尾
            self.buffer.append(item)
            
            removed_item = None
            # 如果超过容量，移除最老的数据（index 0）
            if len(self.buffer) > self.queue_capacity:
                removed_item = self.buffer.pop(0)  # 移除第一个元素（最老的数据）
                self.logger.debug(f"队列满，移除最老数据: {removed_item}")
            
            self.logger.debug(f"入队成功: {item}, 队列大小: {len(self.buffer)}")
            return removed_item
    
    def dequeue(self) -> Optional[Any]:
        """
        出队操作：从前面取出最老的数据
        
        Returns:
            Optional[Any]: 取出的数据，如果队列为空返回None
        """
        with self._lock:
            if not self.buffer:
                self.logger.debug("队列为空，无法取出数据")
                return None
            
            # 从前面取出最老的数据
            oldest_item = self.buffer.pop(0)
            self.logger.debug(f"出队: {oldest_item}, 剩余大小: {len(self.buffer)}")
            return oldest_item
    
    def peek_all(self) -> List[Any]:
        """
        查看队列中所有数据（不移除）
        
        Returns:
            List[Any]: 队列中所有数据的副本 [oldest, ..., newest]
        """
        with self._lock:
            return self.buffer.copy()
    
    def peek_newest(self, count: int = 1) -> List[Any]:
        """
        查看最新的N个数据
        
        Args:
            count: 要查看的数据个数
            
        Returns:
            List[Any]: 最新的N个数据
        """
        with self._lock:
            if count <= 0:
                return []
            return self.buffer[-count:]
    
    def peek_oldest(self, count: int = 1) -> List[Any]:
        """
        查看最老的N个数据
        
        Args:
            count: 要查看的数据个数
            
        Returns:
            List[Any]: 最老的N个数据
        """
        with self._lock:
            if count <= 0:
                return []
            return self.buffer[:count]
    
    def qsize(self) -> int:
        """
        获取队列当前大小
        
        Returns:
            int: 队列当前元素个数
        """
        with self._lock:
            return len(self.buffer)
    
    def empty(self) -> bool:
        """
        检查队列是否为空
        
        Returns:
            bool: 队列是否为空
        """
        with self._lock:
            return len(self.buffer) == 0
    
    def full(self) -> bool:
        """
        检查队列是否已满
        
        Returns:
            bool: 队列是否已达到最大容量
        """
        with self._lock:
            return len(self.buffer) >= self.queue_capacity
    
    def clear(self):
        """
        清空队列
        """
        with self._lock:
            self.buffer.clear()
            self.logger.info("队列已清空")
    
    def get_stats(self) -> dict:
        """
        获取队列统计信息
        
        Returns:
            dict: 队列统计信息
        """
        with self._lock:
            return {
                'queue_capacity': self.queue_capacity,
                'current_size': len(self.buffer),
                'is_empty': len(self.buffer) == 0,
                'is_full': len(self.buffer) >= self.queue_capacity,
                'usage_ratio': len(self.buffer) / self.queue_capacity * 100,
                'oldest_item': self.buffer[0] if self.buffer else None,
                'newest_item': self.buffer[-1] if self.buffer else None
            }
    
    # 兼容queue.Queue接口
    def put_nowait(self, item: Any):
        """兼容queue.Queue接口"""
        self.enqueue(item)
    
    def get_nowait(self) -> Any:
        """兼容queue.Queue接口"""
        item = self.dequeue()
        if item is None:
            from queue import Empty
            raise Empty()
        return item
    
    def task_done(self):
        """兼容queue.Queue接口"""
        pass


class FixedSizeAtomicDict:
    """
    管理多个固定大小队列的管理器
    """
    
    def __init__(self, queue_capacity: int = 60):
        self.queue_capacity = queue_capacity
        self._queues = {}
        self._lock = threading.RLock()
    
    def get_or_create(self, key: str, queue_capacity: Optional[int] = None) -> FixedSizeSlidingQueue:
        """获取或创建固定大小队列"""
        with self._lock:
            if key not in self._queues:
                cap = queue_capacity if queue_capacity is not None else self.queue_capacity
                self._queues[key] = FixedSizeSlidingQueue(queue_capacity=cap)
            return self._queues[key]
    
    def get(self, key: str) -> Optional[FixedSizeSlidingQueue]:
        """获取队列"""
        with self._lock:
            return self._queues.get(key)
    
    def remove(self, key: str) -> bool:
        """移除队列"""
        with self._lock:
            if key in self._queues:
                del self._queues[key]
                return True
            return False
    
    def keys(self) -> List[str]:
        """获取所有键"""
        with self._lock:
            return list(self._queues.keys())
    
    def get_all_stats(self) -> dict:
        """获取所有队列的统计信息"""
        with self._lock:
            stats = {}
            for key, queue in self._queues.items():
                stats[key] = queue.get_stats()
            return stats


# 测试代码
if __name__ == "__main__":
    def test_basic_operations():
        """测试基本操作"""
        print("🧪 测试基本操作")
        print("=" * 50)
        
        # 创建容量为5的队列
        queue = FixedSizeSlidingQueue(queue_capacity=5)
        
        print("1. 初始状态:")
        print(f"   队列内容: {queue.peek_all()}")
        print(f"   是否为空: {queue.empty()}")
        print(f"   当前大小: {queue.qsize()}")
        print()
        
        print("2. 依次添加数据 [1, 2, 3, 4, 5]:")
        for i in range(1, 6):
            removed = queue.enqueue(i)
            print(f"   添加 {i}: {queue.peek_all()}, 被移除: {removed}")
        print()
        
        print("3. 继续添加数据 [6, 7, 8] (会挤出老数据):")
        for i in range(6, 9):
            removed = queue.enqueue(i)
            print(f"   添加 {i}: {queue.peek_all()}, 被移除: {removed}")
        print()
        
        print("4. 出队操作:")
        while not queue.empty():
            item = queue.dequeue()
            print(f"   出队: {item}, 剩余: {queue.peek_all()}")
    
    def test_sliding_window():
        """测试滑动窗口效果"""
        print("\n🔄 测试滑动窗口效果")
        print("=" * 50)
        
        queue = FixedSizeSlidingQueue(queue_capacity=4)
        
        print("模拟数据流:")
        data_stream = [10, 20, 30, 40, 50, 60, 70]
        
        for i, data in enumerate(data_stream):
            removed = queue.enqueue(data)
            print(f"时刻 {i+1}: 新数据={data}, 队列={queue.peek_all()}, 被挤出={removed}")
            
            # 显示统计信息
            stats = queue.get_stats()
            print(f"         统计: 大小={stats['current_size']}/{stats['queue_capacity']}, "
                  f"最老={stats['oldest_item']}, 最新={stats['newest_item']}")
            print()
    
    def test_peek_operations():
        """测试查看操作"""
        print("🔍 测试查看操作")
        print("=" * 50)
        
        queue = FixedSizeSlidingQueue(queue_capacity=6)
        
        # 添加一些数据
        for i in range(1, 8):  # 1-7
            queue.enqueue(i)
        
        print(f"队列内容: {queue.peek_all()}")
        print(f"最老的2个: {queue.peek_oldest(2)}")
        print(f"最新的3个: {queue.peek_newest(3)}")
        print(f"统计信息: {queue.get_stats()}")
    
    # 运行测试
    test_basic_operations()
    test_sliding_window()
    test_peek_operations()