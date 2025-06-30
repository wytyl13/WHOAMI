#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/20 17:57
@Author  : weiyutao
@File    : socket_server_manager.py
"""


import threading
import queue
from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    List
)
import time
from threading import Lock
import numpy as np


from whoami.tool.base.producer_consumer_manager import ProducerConsumerManager
from whoami.tool.base.consumer_tool_pool import ConsumerToolPool
from whoami.tool.real_time_vital_analyze.socket_server import SocketServer
from whoami.tool.real_time_vital_analyze.lockfree_queue import LockFreeQueue, AtomicDict, AtomicLong

from whoami.tool.real_time_vital_analyze.fixed_size_queue import FixedSizeSlidingQueue, FixedSizeAtomicDict
from whoami.neural_network.rnn.lstm_engine import LSTMEngine
from whoami.neural_network.rnn.auto_encoders import RecurrentAutoencoder
from whoami.tool.real_time_vital_analyze.peak_state import RealTimeStateMonitor
from whoami.tool.real_time_vital_analyze.sleep_data_state_storage import SleepDataStateStorage
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.real_time_vital_analyze.sleep_data_state import SleepDataState

model_params = {
    'seq_len': 20,
    'n_features': 2,
    'embedding_dim': 128
}

engine = LSTMEngine(
    model_class=RecurrentAutoencoder,
    model_params=model_params,
    seq_len=20,
    n_features=2,
    threshold=5,
    normalized_flag=0
)
engine.setup(model_path='/work/ai/WHOAMI/whoami/neural_network/checkpoint_epoch_32_2dimensions_20000_no_normalized.pth', scaler_path='/work/ai/WHOAMI/whoami/neural_network/rnn/training_scaler.pkl')

monitor = RealTimeStateMonitor(
    off_bed_threshold=0.1,          # 离床阈值
    apnea_threshold=0.5,            # 呼吸暂停阈值
    activation_threshold=1.0,       # 峰值检测激活阈值
    rise_factor=1.5,
    peak_factor=2.0,
    min_peak_duration=1.0,
    min_peak_height=5.0
)

sql_provider = SqlProvider(
    model=SleepDataState, 
    sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml",
)

# 创建存储管理器
storage = SleepDataStateStorage(
    single_insert_db=sql_provider.add_record,
    batch_insert_db=sql_provider.bulk_insert_with_update,
    buffer_duration=60.0,
    min_interval=10.0,
    max_interval=30.0
)


class SocketServerManager(ProducerConsumerManager):
    """socketserver管理类

    Args:
        ProducerConsumerManager (_type_): _description_
    """
    classified_queues: Optional[AtomicDict] = AtomicDict()
    socket_servers: Dict[int, SocketServer] = {}
    consumer_queue_size: int = 30
    space_rate: float = 0.01
    latest_real_time_data: Dict[str, Any] = {} # 字典对应的值为浮点数
    latest_real_time_label: Dict[str, Any] = {}  # 字典对应值为元祖(异常状态boolean, 重构损失值)
    last_submit_times: Dict[str, Any] = {}
    min_submit_interval: float = 1.0
    device_queue_capacity: Optional[int] = None
    performance_stats: Optional[Dict] = None
    sliding_window_size: Optional[int] = None  # 滑动窗口大小
    fill_value: Optional[Any] = 0
    submit_time_lock: Optional[threading.Lock] = None
    injected_data: Optional[List] = None
    
    def __init__(
        self, 
        max_producers: int = 20,
        max_consumers: int = 30,
        production_queue_size: int = 1000,
        consumer_tool_pool: ConsumerToolPool = None,
        consumer_queue_size: Optional[int] = None,
        space_rate: Optional[float] = None,
        sliding_window_size: int = 20, 
        fill_value: Any = (0, 0),
        injected_data: Optional[List] = None
    ):
        super().__init__(
            max_producers=max_producers,
            max_consumers=max_consumers,
            production_queue_size=production_queue_size,
            consumer_tool_pool=consumer_tool_pool
        )
        self.sliding_window_size = sliding_window_size
        self.fill_value = fill_value
        
        self.consumer_queue_size = consumer_queue_size if consumer_queue_size is not None else self.consumer_queue_size
        self.space_rate = space_rate if space_rate is not None else self.space_rate
        
        self.min_submit_interval = 0.1  # 实时处理，缩短间隔
        
        # 使用固定大小队列字典
        self.classified_queues = FixedSizeAtomicDict(
            queue_capacity=sliding_window_size,
        )
        
        # 时间戳管理（原子操作）
        self.last_submit_times = {}  # device_id -> AtomicLong
        self.submit_time_lock = threading.Lock()  # 用于时间戳字典的线程安全
        
        # 性能统计
        self.performance_stats = {
            'total_enqueue': 0,
            'total_consume': 0,
            'total_devices': 0
        }
        
        
        self.injected_data = injected_data
        
        self.logger.info("SocketServerManager initialized")
    
    
    def get_or_create_queue(self, device_id: str) -> queue.Queue:
        """
        获取或创建设备的固定大小滑动队列
        
        Args:
            device_id: 设备ID
            
        Returns:
            FixedSizeSlidingQueue: 设备对应的固定大小队列
        """
        device_queue = self.classified_queues.get_or_create(device_id)
        
        # 初始化时间戳管理
        with self.submit_time_lock:
            if device_id not in self.last_submit_times:
                self.last_submit_times[device_id] = AtomicLong(0.0)
                self.logger.info(f"创建设备 {device_id} 的固定大小队列(大小:{self.sliding_window_size})")
        
        return device_queue
    
    
    def _classify_and_store_data(self, parse_data):
        """
        数据分类存储到固定大小滑动队列
        
        Args:
            parse_data: 解析后的数据，最后一个元素是device_id
        """
        device_id = parse_data[-1]
        try:
            # 获取或创建设备队列
            target_queue = self.get_or_create_queue(device_id)
            
            # 入队到滑动窗口（新数据从前面进入）
            target_queue.enqueue(parse_data)
            self.consumer_pool.submit(
                self._process_sliding_window_data, 
                device_id, 
                target_queue
            )
            # 更新最新数据缓存
            self.latest_real_time_data[device_id] = parse_data
            self.performance_stats['total_enqueue'] += 1
            
            self.logger.debug(f"数据已入队到设备 {device_id} 滑动窗口: {parse_data}")
            
        except Exception as e:
            self.logger.error(f"固定大小队列数据存储失败: {e}")
    
    
    def start_socket_server(self, port: int, backlog: int = 5):
        if port in self.socket_servers:
            self.logger.warning(f"Socket server on port {port} already exists")
            return
        
        production_id = f"socket_server_{port}"

        self.production_line_locks[production_id] = threading.Lock()

        self.production_line_stop_flags[production_id] = False
        socket_server = SocketServer(
            port=port,
            data_callback=self._classify_and_store_data,
            backlog=backlog,
            injected_data=self.injected_data
        )
        
        socket_server.start()
        self.socket_servers[port] = socket_server
        
        self.active_production_lines[production_id] = {
            'port': port,
            'socket_server': socket_server
        }
        
        self.logger.info(f"Started socket server on port {port} with production ID {production_id}")


    def _handle_data(self, data: Dict[str, Any]):
        try:
            self.production_queue.put(data)
            self.logger.info(f"product: {data}")
        except Exception as e:
            self.logger.error(f"Error adding data to production queue: {e}")
    
    
    def stop_socket_server(self, port: int):
        if port not in self.socket_servers:
            self.logger.warning(f"No socket server running on port {port}")
            return
        production_id = f"socket_server_{port}"
        with self.production_line_locks[production_id]:
            self.production_line_stop_flags[production_id] = True
            self.socket_servers[port].stop()
            
            if production_id in self.active_production_lines:
                del self.active_production_lines[production_id]


    def start_produce_worker(self, port: int, *args, **kwargs):
        self.start_socket_server(port, *args, **kwargs)
    
    
    def stop_produce_worker(self, production_id: str):
        if production_id in self.active_production_lines:
            port = self.active_production_lines[production_id]['port']
            self.stop_socket_server(port)
    
    
    def _start_consumer_worker(self):
        """
        固定大小队列版本的消费者工作线程
        实现：查看全量数据(60个) + 从末尾消费1个(最老数据) + 前面补充填充值
        """
        self.logger.info("启动固定大小滑动窗口消费者工作线程")
        
        while self.consumer_worker_running:
            consumed_any = False
            
            # 获取当前所有设备ID
            current_device_ids = self.classified_queues.keys()
            
            for device_id in current_device_ids:
                try:
                    # 获取设备队列
                    device_queue = self.classified_queues.get(device_id)
                    if device_queue is None:
                        continue
                    
                    # 检查是否有真实数据（非填充值）
                    real_data_count = device_queue.count_real_data()
                    if real_data_count == 0:
                        self.logger.debug(f"设备 {device_id} 队列中无真实数据，跳过处理")
                        continue
                    
                    # 检查时间间隔
                    current_time = time.time()
                    with self.submit_time_lock:
                        last_submit_atomic = self.last_submit_times.get(device_id)
                        
                        if last_submit_atomic is not None:
                            last_submit_time = last_submit_atomic.get()
                            time_diff = current_time - last_submit_time
                            
                            if time_diff >= self.min_submit_interval:
                                # 原子更新提交时间
                                if last_submit_atomic.compare_and_set(last_submit_time, current_time):
                                    # 提交消费任务
                                    self.consumer_pool.submit(
                                        self._process_sliding_window_data, 
                                        device_id, 
                                        device_queue
                                    )
                                    consumed_any = True
                                    self.logger.info(f"提交设备 {device_id} 滑动窗口消费任务，真实数据: {real_data_count}/{self.sliding_window_size}")
                            else:
                                remaining_time = self.min_submit_interval - time_diff
                                self.logger.debug(f"设备 {device_id} 时间间隔不足，还需等待 {remaining_time:.3f} 秒")
                        else:
                            # 首次处理
                            self.last_submit_times[device_id] = AtomicLong(current_time)
                            self.consumer_pool.submit(
                                self._process_sliding_window_data, 
                                device_id, 
                                device_queue
                            )
                            consumed_any = True
                            self.logger.info(f"首次提交设备 {device_id} 滑动窗口消费任务")
                    
                except Exception as e:
                    self.logger.error(f"设备 {device_id} 消费检查时出错: {e}")
            
            if not consumed_any:
                time.sleep(0.1)  # 短暂等待
    
    
    def _process_sliding_window_data(self, device_id: str, device_queue: FixedSizeSlidingQueue):
        """
        处理滑动窗口数据
        逻辑：查看全量60个数据 -> 从末尾取出1个最老数据进行处理 -> 前面自动补充填充值
        
        Args:
            device_id: 设备ID
            device_queue: 设备的滑动窗口队列
        """
        try:
            # 1. 查看队列中所有数据（60个元素的完整窗口）
            all_window_data = device_queue.peek_all()
            print(f"original data: {all_window_data[-1]}")
            
            timestamp, breath_bpm, breath_line, heart_bpm, heart_line, target_distance, signal_strength, _, body_move_energy, body_move_range, in_bed, device_id = all_window_data[-1]
            
            
            breath_line_heart_line = np.array([breath_line, heart_line, signal_strength]).reshape(1, 3)
            
            # all_window_data = [item[1:8] for item in all_window_data] # 7 dimensions
            # all_window_data = [[item[i] for i in [2, 4, 5, 6, 7, 8]] for item in all_window_data] # 6 dimensions
            all_window_data = [[item[i] for i in [2, 4]] for item in all_window_data] # 2 dimensions
            add_size = self.sliding_window_size - len(all_window_data)
            add_data = [self.fill_value] * add_size
            add_data.extend(all_window_data)
            add_data = np.array(add_data)
            print(f"lstm input data: {add_data[-1]}")
            # 2. 获取数据分布统计
            # data_stats = device_queue.get_data_distribution()
            result, error = engine.predict(data=add_data, return_details=True)
            state = monitor.update(value=error, body_move_energy=body_move_energy, timestamp=timestamp, breath_line_heart_line=breath_line_heart_line)
            state_str = state[0]
            print(result, error, state[0], timestamp)
            
            self.latest_real_time_label[device_id] = (result, error, state_str)
            
            # 数据库缓存
            storage_dict = {
                "device_id": device_id,
                "timestamp": timestamp,
                "breath_bpm": breath_bpm,
                "breath_line": breath_line,
                "heart_bpm": heart_bpm,
                "heart_line": heart_line,
                "reconstruction_error": error,
                "state": state_str
            }
            
            storage.add_data_point(**storage_dict)
            
        except Exception as e:
            self.logger.error(f"设备 {device_id} 滑动窗口数据处理失败: {e}")


    def _process_item_with_context(self, device_id: str, device_queue: LockFreeQueue):
        """
        带上下文的数据处理（Lock-Free版本）
        实现"查看全量数据，只消费第一个"的逻辑
        
        Args:
            device_id: 设备ID
            device_queue: 设备的Lock-Free队列
        """
        try:
            # 1. 获取队列中所有数据的快照（不移除数据）
            all_cached_items = device_queue.peek_all()
            
            if not all_cached_items:
                self.logger.debug(f"设备 {device_id} 队列为空，无数据可处理")
                return
            
            # 2. 只消费第一个数据（从队列中移除）
            consumed_item = device_queue.dequeue()
            
            if consumed_item is None:
                self.logger.warning(f"设备 {device_id} 队列在处理过程中变空")
                return
            
            self.performance_stats['total_dequeue'] += 1
            
            # 3. 记录处理信息
            self.logger.info(f"设备 {device_id} 处理上下文:")
            self.logger.info(f"  缓存数据总数: {len(all_cached_items)}")
            self.logger.info(f"  当前消费数据: {consumed_item}")
            self.logger.info(f"  剩余队列大小: {device_queue.qsize()}")
            
            # 4. 调用消费工具池处理数据（带上下文）
            if hasattr(self, 'consumer_tool_pool') and self.consumer_tool_pool:
                pass
                # # 如果消费工具池支持上下文处理
                # if hasattr(self.consumer_tool_pool, 'process_data_with_context'):
                #     self.consumer_tool_pool.process_data_with_context(
                #         device_id=device_id,
                #         current_data=consumed_item,
                #         cache_context=all_cached_items
                #     )
                # else:
                #     # 降级到普通处理
                #     self.consumer_tool_pool.process_data(consumed_item)
            else:
                pass
                # 默认处理逻辑
                # self._default_process_with_context(device_id, consumed_item, all_cached_items)
                
        except Exception as e:
            self.logger.error(f"设备 {device_id} Lock-Free数据处理失败: {e}")


    def shutdown(self):
        """关闭管理器（接口保持不变）"""
        self.logger.info("Shutting down FixedSizeSocketServerManager...")
        
        # 停止所有Socket服务器
        for port in list(self.socket_servers.keys()):
            self.stop_socket_server(port)
        
        # 停止消费者工作线程
        self.consumer_worker_running = False
        
        # 关闭线程池
        self.producer_pool.shutdown(wait=True)
        self.consumer_pool.shutdown(wait=True)
        
        # 清空生产队列
        while not self.production_queue.empty():
            try:
                self.production_queue.get_nowait()
                self.production_queue.task_done()
            except Exception:
                pass
        
        # 输出最终统计
        final_stats = {
            **self.performance_stats,
            'devices_count': len(self.classified_queues.keys()),
            'windows_stats': self.get_all_windows_stats()
        }
        self.logger.info(f"Final stats: {final_stats}")
        
        self.logger.info("FixedSizeSocketServerManager shutdown complete")


    def get_device_window_stats(self, device_id: str) -> dict:
        """
        获取设备滑动窗口统计信息
        
        Args:
            device_id: 设备ID
            
        Returns:
            dict: 窗口统计信息
        """
        device_queue = self.classified_queues.get(device_id)
        if device_queue:
            return device_queue.get_data_distribution()
        return {}
    
    
    def get_all_windows_stats(self) -> dict:
        """
        获取所有设备滑动窗口统计信息
        
        Returns:
            dict: 所有设备的窗口统计
        """
        return self.classified_queues.get_all_stats()
    
    
    def reset_device_window(self, device_id: str):
        """
        重置设备滑动窗口（用填充值填满）
        
        Args:
            device_id: 设备ID
        """
        device_queue = self.classified_queues.get(device_id)
        if device_queue:
            device_queue.reset_with_fill()
            self.logger.info(f"设备 {device_id} 滑动窗口已重置")
    
    
    def get_performance_stats(self) -> dict:
        """
        获取性能统计信息
        
        Returns:
            dict: 性能统计
        """
        return {
            **self.performance_stats,
            'total_devices': len(self.classified_queues.keys()),
            'queue_stats': self.get_all_queue_stats()
        }
    
    
    def _run(self):
        pass


if __name__ == '__main__':
    from whoami.tool.base.rnn_model_info import RNNModelInfo
    from whoami.neural_network.rnn.model import LSTM
    from whoami.configs.detector_config import DetectorConfig
    CONFIG_PATH = '/work/ai/WHOAMI/whoami/scripts/detect/detect_config.yaml'
    CONFIG = DetectorConfig.from_file(CONFIG_PATH).__dict__
    TOPIC_DICT = CONFIG['topics']
    conf_dict = CONFIG["conf"]
    model_path_dict = CONFIG["model_path"]
    class_list_dict = CONFIG["class_list"]
    topic_list = TOPIC_DICT
    model_paths = {}
    for conf_key, conf_value in conf_dict.items():
        for topic_name in topic_list:
            topic_key = conf_key + topic_name
            model_paths[topic_key] = RNNModelInfo(
                model_path="/work/ai/WHOAMI/whoami/"+model_path_dict[topic_name],
                model_type_class=LSTM,
                classes=class_list_dict[topic_name],
                conf=conf_value[topic_name]
            )
    print(f"model_paths: --------------------------------------\n {model_paths}")
    consumer_tool_pool = ConsumerToolPool(model_paths=model_paths)
    
    
    """
    device_id = "DEV001"
    base_time = time.time()
    test_scenarios = [
        (0, "在床正常", 70, 16),
        (1, "在床正常", 71, 17),
        (2, "清醒", 75, 18),
        (3, "清醒", 74, 17),
        (4, "浅睡眠", 65, 14),
        (5, "浅睡眠", 66, 15),
        (6, "呼吸暂停", 70, 8),        # 异常：会存储前60秒数据
        (7, "呼吸暂停", 72, 6),
        (8, "深睡眠", 60, 12),
        (9, "深睡眠", 58, 13),
        (10, "深睡眠", 58, 13),
        (11, "呼吸暂停", 58, 13),
        (12, "体动", 58, 13),
    ]
    
    print("开始模拟数据流...")
    print("=" * 60)
    injected_data = []
    for i, (offset, state, hr, br) in enumerate(test_scenarios):
        timestamp = base_time + offset
        
        # 模拟波形数据
        breath_line = 1.0
        heart_line = 1.0
        error = 0.2222
        injected_data.append((timestamp, br, breath_line, hr, heart_line, 1.0, 20, 0, 20, 1, 0, device_id))
    """
    
    
    from whoami.provider.sql_provider import SqlProvider
    from whoami.tool.health_report.sx_device_wavve_vital_sign_log_20250522 import SxDeviceWavveVitalSignLog
    sql_provider_test = SqlProvider(
        model=SxDeviceWavveVitalSignLog, 
        sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml",
    )
    result = sql_provider_test.get_record_by_condition(
        condition={"device_sn": "13D2F34920008071211195A907"},
        fields=["create_time", "breath_bpm", "breath_line", "heart_bpm", "heart_line", "distance", "signal_intensity", "state", "body_move_data", "device_sn"],
        date_range={"date_field": "create_time", "start_date": "2025-6-29 19:00:00", "end_date": "2025-6-30 07:00:00"}
    )
    print(result[0])
    
    from datetime import datetime, timezone, timedelta
    def preprocess_query_results_safe(records: List[Dict[str, Any]], 
                                source_timezone: str = 'Asia/Shanghai') -> List[Dict[str, Any]]:
        """更安全的版本 - 明确指定源时区"""
        if not records:
            return []
        
        import pytz
        
        # 使用pytz处理时区（更准确，考虑夏令时等）
        source_tz = pytz.timezone(source_timezone)
        utc_tz = pytz.UTC
        processed_records = []
        
        for record in records:
            if record.get('heart_bpm') is None:
                continue
            
            processed_record = record.copy()
            
            if 'create_time' in processed_record and isinstance(processed_record['create_time'], datetime):
                dt = processed_record['create_time']
                
                if dt.tzinfo is None:
                    # 明确指定原始数据的时区
                    dt_localized = source_tz.localize(dt)
                else:
                    dt_localized = dt
                
                # 转换为UTC时间戳
                processed_record['create_time'] = dt_localized.astimezone(utc_tz).timestamp()
            
            if 'body_move_data' in processed_record and processed_record['body_move_data'] is None:
                processed_record['body_move_data'] = 0
            
            processed_records.append(processed_record)
        
        return processed_records

    
    def check_none_values(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        检查数据中的None值情况
        
        Args:
            records: 要检查的数据列表
            
        Returns:
            Dict: 包含None值统计信息的字典
        """
        if not records:
            return {
                'total_records': 0,
                'has_none': False,
                'none_statistics': {},
                'none_details': []
            }
        
        total_records = len(records)
        none_count_by_field = {}
        records_with_none = []
        
        # 统计每个字段的None值数量
        for i, record in enumerate(records):
            record_has_none = False
            record_none_fields = []
            
            for field, value in record.items():
                if value is None:
                    # 统计字段的None数量
                    if field not in none_count_by_field:
                        none_count_by_field[field] = 0
                    none_count_by_field[field] += 1
                    
                    record_has_none = True
                    record_none_fields.append(field)
            
            # 记录包含None的记录
            if record_has_none:
                records_with_none.append({
                    'record_index': i,
                    'none_fields': record_none_fields,
                    'record_preview': {k: v for k, v in record.items() if k in record_none_fields or k in ['create_time', 'device_sn']}
                })
        
        # 计算百分比
        none_statistics = {}
        for field, count in none_count_by_field.items():
            percentage = (count / total_records) * 100
            none_statistics[field] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        has_none = len(none_count_by_field) > 0
        
        result = {
            'total_records': total_records,
            'has_none': has_none,
            'none_statistics': none_statistics,
            'records_with_none_count': len(records_with_none),
            'none_details': records_with_none[:10]  # 只显示前10个包含None的记录
        }
        
        return result
    

    result = preprocess_query_results_safe(result)
    injected_data = []
    for item in result:
        tuple_data = (
            item.get('create_time', 0),           # 位置0: timestamp
            item.get('breath_bpm', 0),            # 位置1: breath_bpm
            item.get('breath_line', 0),           # 位置2: breath_line
            item.get('heart_bpm', 0),             # 位置3: heart_bpm
            item.get('heart_line', 0),            # 位置4: heart_line
            item.get('distance', 0),              # 位置5: target_distance
            item.get('signal_intensity', 0),      # 位置6: signal_strength
            item.get('state', 0),                 # 位置7: state
            item.get('body_move_data', 0),        # 位置8: body_move_energy (缺失字段)
            0,                                    # 位置9: body_move_range (缺失字段)
            0,                                    # 位置10: in_bed 
            item.get('device_sn', '00000001')     # 位置11: device_id
        )
        injected_data.append(tuple_data)
    
    print(injected_data[0])
    
    # socket_server_manager = SocketServerManager(
    #     max_producers=10,      # 最大生产者数量
    #     max_consumers=15,      # 最大消费者数量
    #     production_queue_size=500,  # 生产队列大小
    #     consumer_tool_pool=consumer_tool_pool,
    #     injected_data=injected_data
    # )

    # socket_server_manager.start_socket_server(port=8888, backlog=5)
    
    # try:
    #     # 让服务器运行一段时间
    #     import time
    #     time.sleep(3600)  # 运行1小时
    # finally:
    #     # 停止特定端口的服务器
    #     socket_server_manager.stop_socket_server(port=8888)
        
    #     # 或者关闭整个管理器及其所有服务器
    #     socket_server_manager.shutdown()