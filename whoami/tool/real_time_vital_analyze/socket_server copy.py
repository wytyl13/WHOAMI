#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/20 09:46
@Author  : weiyutao
@File    : socket_server.py
"""
import socket
import threading
import time
import struct
import queue

from whoami.tool.agent.base_tool import tool
from whoami.tool.base.producer_consumer_manager import ProducerConsumerManager
from whoami.tool.base.consumer_tool_pool import ConsumerToolPool

class SocketServer(ProducerConsumerManager):
    """one single instance, one port.
    """
    server_socket: socket.socket = None
    port: int = None
    accept_thread: threading.Thread = None

    production_id: str = None

    is_port_running: bool = False
    def __init__(
        self,
        port: int,
        max_producers: int = 5,
        max_consumers: int = 5,
        production_queue_size: int = 100,
        consumer_tool_pool: ConsumerToolPool = None,
        backlog: int = 5
    ):
        """_summary_

        Args:
            port (int): 要监听的端口号
            max_producers (int, optional): 最大生产者线程数. Defaults to 5.
            max_consumers (int, optional): 最大消费者线程数. Defaults to 5.
            production_queue_size (int, optional): 生产队列大小. Defaults to 100.
        """
        super().__init__(
            max_producers=max_producers,
            max_consumers=max_consumers,
            production_queue_size=production_queue_size,
            consumer_tool_pool=consumer_tool_pool,
        )
        self.port = port
        self.production_id = f"socket_server_{port}"

        self.production_line_locks[self.production_id] = threading.Lock()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', port))
        self.server_socket.listen(backlog)
        
        self.is_port_running = True
        
        self.active_production_lines[self.production_id] = {
            'port': port,
            'socket': self.server_socket
        }
        self.start_produce_worker()
        self.logger.info(f"Socket服务器已初始化，监听端口：{port}")

    
    def start_produce_worker(self, *args, **kwars) -> None:
        self.accept_thread = threading.Thread(
            target=self._accept_connections,
        )
        self.accept_thread.daemon = True
        self.accept_thread.start()
        self.logger.info(f"开始监听端口{self.port}，生产线ID:{self.production_id}")

    
    def _accept_connections(self) -> None:
        self.logger.info(self.production_line_stop_flags.get(self.production_id, True))
        while self.is_port_running and not self.production_line_stop_flags.get(self.production_id, True):
            try:
                self.server_socket.settimeout(0.5)
                try:
                    client_socket, addr = self.server_socket.accept()
                    self.producer_pool.submit(
                        self._process_client,
                        client_socket,
                        addr
                    )
                    self.logger.info(f"端口{self.port}接收到连接：{addr}")
                except socket.timeout:
                    # self.logger.error("timeout")
                    continue
            except Exception as e:
                if self.is_port_running:
                    self.logger.error(f"端口{self.port}接受连接时出错: {e}")
                    

    def parse_vital_data_packet(self, hex_data):
        # Add timestamp
        timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
        
        # Convert hex string to bytes if needed
        if isinstance(hex_data, str):
            # Remove spaces if present
            hex_data = hex_data.replace(" ", "")
            data_bytes = bytes.fromhex(hex_data)
        else:
            data_bytes = hex_data
        
        # Parse header
        header = {
            "magic": data_bytes[0],
            "version": data_bytes[1],
            "type": data_bytes[2],
            "cmd": data_bytes[3],
            "req_id": int.from_bytes(data_bytes[4:8], byteorder="big"),
            "timeout": int.from_bytes(data_bytes[8:10], byteorder="big"),
            "content_len": int.from_bytes(data_bytes[10:14], byteorder="big"),
            "func_tag": int.from_bytes(data_bytes[14:16], byteorder="big")
        }
        
        # Check if this is vital data
        if header["func_tag"] == 0x03e8:
            payload = data_bytes[16:]
            
            # Parse the vital data according to the protocol
            # Each float is 4 bytes in IEEE-754 format
            vital_data = {
                "timestamp": timestamp,  # Add timestamp to vital data
                "breath_bpm": struct.unpack('>f', payload[0:4])[0],
                "breath_curve": struct.unpack('>f', payload[4:8])[0],
                "heart_rate_bpm": struct.unpack('>f', payload[8:12])[0],
                "heart_rate_curve": struct.unpack('>f', payload[12:16])[0],
                "target_distance": struct.unpack('>f', payload[16:20])[0],
                "signal_strength": struct.unpack('>f', payload[20:24])[0],
                "valid_bit_id": struct.unpack('>i', payload[24:28])[0],
            }
            
            # Check if we have body movement data (newer protocol versions)
            if len(payload) >= 36:
                vital_data["body_move_energy"] = struct.unpack('>f', payload[28:32])[0]
                vital_data["body_move_range"] = struct.unpack('>f', payload[32:36])[0]
                
            return {"header": header, "vital_data": vital_data, "timestamp": timestamp}
        
        return {"header": header, "data": data_bytes[16:], "timestamp": timestamp}
    
    
    def _process_client(
        self,
        client_socket: socket.socket,
        addr: tuple
    ) -> None:
        try:
            timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
            data = client_socket.recv(4096)
            self.logger.info(data)
            if data:
                
                if len(data) < 16:
                    self.logger.error(f"{timestamp} - 数据包太短: {len(data)} 字节")
                    return
                if data[0] != 0x13:
                    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                    self.logger.error(f"{timestamp} - 无效的魔数: 0x{data[0]:02x}")
                    return
                # 解析功能标签
                func_tag = int.from_bytes(data[14:16], byteorder="big")

                try:
                    # 处理不同类型的数据包
                    if func_tag == 0x03e8:  
                        # 生命体征数据
                        parse_data = self.parse_vital_data_packet(data)
                        if "vital_data" in parse_data:
                            vital = parse_data["vital_data"]
                            valid_status = "0_无效"
                            if vital["valid_bit_id"] == 1:
                                valid_status = "1_呼吸有效"
                            elif vital["valid_bit_id"] == 2:
                                valid_status = "2_呼吸和心率有效"
                            in_bed = vital["signal_strength"] > 0
                            vital["in_bed"] = in_bed
                            vital["valid_status"] = valid_status
                            # 添加到队列
                            self.production_queue.put(vital)  # 这一行是关键
                    elif func_tag == 0x0001:  
                        # 设备ID数据包，忽略
                        return
                    elif func_tag == 0x040f:
                        # 累积体动动量值
                        return
                    else:
                        timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                        self.logger.info(f"{timestamp} - 收到未处理的功能标签: 0x{func_tag:04x}")
                        return
            
                except Exception as e:
                    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                    self.logger.error(timestamp + f"  处理数据包错误: {e}\r\n")
                    return
        except Exception as e:
            self.logger.error(f"处理客户端数据时出错：{e}")
            client_socket.close()
    
    
    def stop_produce_worker(self, production_id: str = None): 
        """停止生产工作

        Args:
            production_id (str, optional): _description_. Defaults to None.
        """
        if production_id is None:
            production_id = self.production_id
            
        if production_id != self.production_id:
            self.logger.error(f"生产线ID不匹配：提供的是{production_id}，而实例的是{self.production_id}")
            return
        
        
        with self.production_line_locks[self.production_id]:
            self.production_line_stop_flags[self.production_id] = True
            self.is_port_running = False
            
            try:
                self.server_socket.close()
            except Exception as e:
                self.logger.info(f"关闭端口{self.port}的socket时出错：{e}")
            
            self.active_production_lines.pop(self.production_id, None)

            self.logger.info(f"已停止端口{self.port}的监听，生产线ID：{self.production_id}")
    
    
    def shutdown(self):
        self.stop_produce_worker()
        self.consumer_worker_running = False
        self.producer_pool.shutdown(wait=True)
        while not self.production_queue.empty():
            try:
                self.production_queue.get_nowait()
                self.production_queue.task_done()
            except Exception as e:
                pass
        
        self.logger.error(f"端口{self.port}的socket服务器已关闭！")
        
    
    def get_production_queue(self) -> queue.Queue:
        return self.production_queue
    
    
    def get_port(self) -> int:
        return self.port
    
    
    def _run(self):
        pass   
    
    def _start_consumer_worker(self):
        pass    
        
        
        
if __name__ == '__main__':
    
    server_8000 = SocketServer(
        port=8000,
        max_producers=5,
        max_consumers=5,
        production_queue_size=50,
        consumer_tool_pool=123
    )
    try:
        while True:
            try:
                # 获取队列中的数据而不是队列对象本身
                data_packet = server_8000.get_production_queue().get(timeout=1)
                print("收到数据包:", data_packet)
                # 处理完毕后标记为已完成
                server_8000.get_production_queue().task_done()
            except queue.Empty:
                print("队列当前为空，等待中...")
                time.sleep(1)
    except KeyboardInterrupt:
        print("停止服务")
        server_8000.shutdown()