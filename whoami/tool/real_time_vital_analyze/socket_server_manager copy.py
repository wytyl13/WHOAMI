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

from whoami.tool.base.producer_consumer_manager import ProducerConsumerManager
from whoami.tool.base.consumer_tool_pool import ConsumerToolPool
from whoami.tool.real_time_vital_analyze.socket_server import SocketServer

class SocketServerManager(ProducerConsumerManager):
    """socketserver管理类

    Args:
        ProducerConsumerManager (_type_): _description_
    """
    classified_queues: Dict[str, queue.Queue] = {}
    queue_creation_lock: threading.Lock = threading.Lock()
    socket_servers: Dict[int, SocketServer] = {}
    consumer_queue_size: int = 30
    space_rate: float = 0.01
    latest_real_time_data: Dict[str, Any] = {} 
    last_submit_times: Dict[str, Any] = {}
    min_submit_interval: float = 10.0
    submit_time_lock: threading.Lock = Lock()
    def __init__(
        self, 
        max_producers: int = 20,
        max_consumers: int = 30,
        production_queue_size: int = 1000,
        consumer_tool_pool: ConsumerToolPool = None,
        consumer_queue_size: Optional[int] = None,
        space_rate: Optional[float] = None
    ):
        super().__init__(
            max_producers=max_producers,
            max_consumers=max_consumers,
            production_queue_size=production_queue_size,
            consumer_tool_pool=consumer_tool_pool
        )
        self.consumer_queue_size = consumer_queue_size if consumer_queue_size is not None else self.consumer_queue_size
        self.space_rate = space_rate if space_rate is not None else self.space_rate
        self.logger.info("SocketServerManager initialized")
    
    def get_or_create_queue(self, device_id: str) -> queue.Queue:
        if device_id in self.classified_queues:
            return self.classified_queues[device_id]
        
        with self.queue_creation_lock:
            if device_id not in self.classified_queues:
                self.classified_queues[device_id] = queue.Queue(maxsize=1000)
                self.logger.info(f"创建新设备队列：{device_id}")    
            return self.classified_queues[device_id]
    
    
    def _classify_and_store_data(self, parse_data):
        device_id = parse_data[-1]
        try:
            target_queue = self.get_or_create_queue(device_id)
            target_queue.put_nowait(parse_data)
            self.latest_real_time_data[device_id] = parse_data
            self.logger.info(f"数据已分类到设备队列：{device_id}, {parse_data}")
        except queue.Full:
            self.logger.warning(f"设备{device_id}队列已满，丢弃数据")
        except Exception as e:
            self.logger.error(f"分类存储失败：{e}")
    
    
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
            backlog=backlog
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
    
    
    # def _start_consumer_worker(self):
        
    #     while self.consumer_worker_running:
    #         current_time = time.time()
    #         consumed_any = False
            
    #         # 实时获取当前所有设备id
    #         with self.queue_creation_lock:
    #             current_device_ids = list(self.classified_queues.keys())
            
    #         # 添加调试日志
    #         self.logger.info(f"当前有 {len(current_device_ids)} 个设备: {current_device_ids}")
            
    #         for device_id in current_device_ids:
    #             items = []
    #             try:
    #                 with self.queue_creation_lock:
    #                     if device_id not in self.classified_queues:
    #                         self.logger.warning(f"设备 {device_id} 不在队列中，跳过")
    #                         continue
    #                     device_queue = self.classified_queues[device_id]
                    
    #                 # 获取队列信息
    #                 queue_size = device_queue.qsize()
    #                 threshold = self.consumer_queue_size * (1 + self.space_rate)
                    
    #                 # 详细的调试日志
    #                 self.logger.info(f"设备 {device_id}: 队列大小={queue_size}, 阈值={threshold}")
                    
    #                 if queue_size >= threshold:
    #                     self.logger.info(f"设备 {device_id} 队列大小达到阈值，检查时间间隔...")
                        
    #                     with self.submit_time_lock:
    #                         last_submit_time = self.last_submit_times.get(device_id, 0)
    #                         time_diff = current_time - last_submit_time
                            
    #                         self.logger.info(f"设备 {device_id}: 上次提交时间={last_submit_time}, 当前时间={current_time}, 时间差={time_diff:.1f}秒, 最小间隔={self.min_submit_interval}秒")
                            
    #                         if time_diff >= self.min_submit_interval:
    #                             self.logger.info(f"设备 {device_id} 满足时间间隔条件，提交消费任务")
    #                             self.consumer_pool.submit(self._process_item, device_queue)
    #                             self.last_submit_times[device_id] = current_time
    #                             consumed_any = True
    #                             self.logger.info(f"✅ 成功提交设备 {device_id} 消费任务，距离上次提交 {time_diff:.1f} 秒")
    #                         else:
    #                             remaining_time = self.min_submit_interval - time_diff
    #                             self.logger.info(f"⏰ 设备 {device_id} 时间间隔不足，还需等待 {remaining_time:.1f} 秒")
    #                 else:
    #                     self.logger.debug(f"设备 {device_id} 队列大小未达到阈值，跳过消费")
                    
    #                 # 这行日志应该在 try 块里，但在 if 条件外
    #                 self.logger.debug(f"从设备 {device_id} 队列消费数据检查完成")
                    
    #             except queue.Empty:
    #                 self.logger.debug(f"设备 {device_id} 队列为空")
    #                 continue
    #             except Exception as e:
    #                 self.logger.error(f"从设备 {device_id} 队列消费时出错: {e}")

    #         self.logger.info(f"本轮消费检查完成，consumed_any={consumed_any}")
            
    #         if not consumed_any:
    #             time.sleep(1.0)
    
    
    def _start_consumer_worker(self):
        while self.consumer_worker_running:
            current_time = time.time()
            consumed_any = False
            
            # 实时获取当前所有设备id
            with self.queue_creation_lock:
                current_device_ids = list(self.classified_queues.keys())
            for device_id in current_device_ids:
                items = []
                try:
                    with self.queue_creation_lock:
                        if device_id not in self.classified_queues:
                            continue
                        device_queue = self.classified_queues[device_id]
                    if device_queue.qsize() >= (self.consumer_queue_size * (1 + self.space_rate)):
                        with self.submit_time_lock:
                            last_submit_time = self.last_submit_times.get(device_id, 0)
                            time_diff = current_time - last_submit_time
                            if time_diff >= self.min_submit_interval:
                                self.consumer_pool.submit(self._process_item, device_queue)
                                self.last_submit_times[device_id] = current_time
                                consumed_any = True
                                self.logger.info(f"提交设备 {device_id} 消费任务，距离上次提交 {time_diff:.1f} 秒")
                            else:
                                remaining_time = self.min_submit_interval - time_diff
                                self.logger.debug(f"设备 {device_id} 时间间隔不足，还需等待 {remaining_time:.1f} 秒")
                    self.logger.debug(f"从设备 {device_id} 队列消费数据")
                except queue.Empty:
                    continue
            
                except Exception as e:
                    self.logger.error(f"从设备 {device_id} 队列消费时出错: {e}")
    
            if not consumed_any:
                time.sleep(1.0)
                
    def _process_item(self, device_queue: queue.Queue):
        items = []
        try:
            for i in range(self.consumer_queue_size):
                item = device_queue.get_nowait()
                items.append(item)
                device_queue.task_done()
            self.logger.info(len(items))
            self.logger.info(f"consumer: {items}")
            # self.logger.info(f"consumers: {item}")
            # Use the consumer tool pool to process the item
            # if self.consumer_tool_pool:
            #     self.consumer_tool_pool.process_data(item)
            # else:
            #     self.logger.warning("No consumer tool pool specified. Item will not be processed.")
        except queue.Empty:
            self.logger.warning("队列在处理过程中变空了")
        except Exception as e:
            self.logger.error(f"Error processing item: {e}")
            
            
    def shutdown(self):
        for port in list(self.socket_servers.keys()):
            self.stop_socket_server(port)
        
        self.consumer_worker_running = False
        self.producer_pool.shutdown(wait=True)

        self.consumer_pool.shutdown(wait=True)

        while not self.production_queue.empty():
            try:
                self.production_queue.get_nowait()
                self.production_queue.task_done()
            except Exception as e:
                pass
        
        self.logger.info("SocketServerManager shut down")


    
    def _run(self):
        pass
    
    
if __name__ == '__main__':
    from whoami.tool.base.model_info import ModelInfo
    from whoami.configs.detector_config import DetectorConfig
    from whoami.tool.detect.ultralitics_detector import UltraliticsDetector
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
            model_paths[topic_key] = ModelInfo(
                model_path="/work/ai/WHOAMI/whoami/"+model_path_dict[topic_name],
                model_type_class=UltraliticsDetector,
                classes=class_list_dict[topic_name],
                conf=conf_value[topic_name]
            )
    print(f"model_paths: --------------------------------------\n {model_paths}")
    consumer_tool_pool = ConsumerToolPool(model_paths=model_paths)
    
    socket_server_manager = SocketServerManager(
        max_producers=10,      # 最大生产者数量
        max_consumers=15,      # 最大消费者数量
        production_queue_size=500,  # 生产队列大小
        consumer_tool_pool=consumer_tool_pool
    )

    socket_server_manager.start_socket_server(port=8000, backlog=5)
    
    try:
        # 让服务器运行一段时间
        import time
        time.sleep(3600)  # 运行1小时
    finally:
        # 停止特定端口的服务器
        socket_server_manager.stop_socket_server(port=8000)
        
        # 或者关闭整个管理器及其所有服务器
        socket_server_manager.shutdown()