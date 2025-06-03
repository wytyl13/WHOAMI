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
    Optional
)


from whoami.tool.base.producer_consumer_manager import ProducerConsumerManager
from whoami.tool.base.consumer_tool_pool import ConsumerToolPool
from whoami.tool.real_time_vital_analyze.socket_server import SocketServer

class SocketServerManager(ProducerConsumerManager):
    """socketserver管理类

    Args:
        ProducerConsumerManager (_type_): _description_
    """
    
    socket_servers: Optional[Dict[int, SocketServer]] = None

    def __init__(
        self, 
        max_producers: int = 20,
        max_consumers: int = 30,
        production_queue_size: int = 1000,
        consumer_tool_pool: ConsumerToolPool = None
    ):
        super().__init__(
            max_producers=max_producers,
            max_consumers=max_consumers,
            production_queue_size=production_queue_size,
            consumer_tool_pool=consumer_tool_pool
        )
        
        self.socket_servers: Dict[int, SocketServer] = {}
        self.logger.info("SocketServerManager initialized")
    
    
    
    def start_socket_server(self, port: int, backlog: int = 5):
        if port in self.socket_servers:
            self.logger.warning(f"Socket server on port {port} already exists")
            return
        
        production_id = f"socket_server_{port}"

        self.production_line_locks[production_id] = threading.Lock()

        self.production_line_stop_flags[production_id] = False
        socket_server = SocketServer(
            port=port,
            data_callback=self._handle_data,
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
        while self.consumer_worker_running:
            try:
                item = self.production_queue.get(timeout=1)
                self.consumer_pool.submit(self._process_item, item)

                self.production_queue.task_done()
            except queue.Empty:
                continue
        
            except Exception as e:
                self.logger.error(f"Error in consumer worker: {e}")
    
    
    def _process_item(self, item: Tuple[Any]):
        try:
            pass
            # self.logger.info(f"consumers: {item}")
            # Use the consumer tool pool to process the item
            # if self.consumer_tool_pool:
            #     self.consumer_tool_pool.process_data(item)
            # else:
            #     self.logger.warning("No consumer tool pool specified. Item will not be processed.")
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