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
from typing import (
    Optional,
    Callable,
    Dict,
    Any
)
import numpy as np


from whoami.tool.agent.base_tool import tool


@tool
class SocketServer:
    """one single instance, one port.
    cache all signal data (Tuple) used deque (Double-Ended Queue).
    1 in out bed status diagnostic.
        Index all the signal_strength (fixed threshold) or other fields (reconstruct used deep learning) to diagnostic the in out bed status. 
        index all necessary data and transform to numpy, dtype=float32, call the in_out_bed instance to handle it.
    2 
    """
    server_socket: Optional[socket.socket] = None
    port: Optional[int] = None
    is_running: Optional[bool] = None
    
    def __init__(
        self,
        port: int,
        backlog: int = 5,
        data_callback: Callable[[Dict[str, Any]], None] = None
    ):
        """_summary_

        Args:
            port (int): 要监听的端口号
            max_producers (int, optional): 最大生产者线程数. Defaults to 5.
            max_consumers (int, optional): 最大消费者线程数. Defaults to 5.
            production_queue_size (int, optional): 生产队列大小. Defaults to 100.
        """
        super().__init__()
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', port))
        self.server_socket.listen(backlog)
        
        self.is_running  = False
        self.accept_thread = None
        self.client_threads = []
        self.logger.info(f"Socket server initialized on port {port}")
        self.devices = {}  
        self.data_callback = data_callback      

    def start(self):
        if self.is_running:
            self.logger.warning(f"Socket server on port {self.port} is already running!")
            return
        
        self.is_running = True
        self.accept_thread = threading.Thread(target=self._accept_connections)
        self.accept_thread.daemon = True # 守护线程
        self.accept_thread.start()
        self.logger.info(f"Socket server started on port {self.port}")
        
    
    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        try:
            self.server_socket.close()
        except Exception as e:
            self.logger.error(f"Error closing server socket on port {self.port}: {e}")
        
        if self.accept_thread and self.accept_thread.is_alive():
            self.accept_thread.join(timeout=2)
        self.logger.info(f"Socket server on port {self.port} stopped!")
    
    
    def _accept_connections(self):
        while self.is_running:
            try:
                self.server_socket.settimeout(0.5)
                try:
                    client_socket, addr = self.server_socket.accept() # 阻塞等待直到有新的客户端连接进来
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    self.client_threads.append(client_thread)
                    self.logger.info(f"Accepted connection from {addr} on port {self.port}")
                    self.send_get_radar_id_request(client_socket) # 发送获取当前连接客户端对应到device_sn请求
                except socket.timeout:
                    continue
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Error accepting connection on port {self.port}: {e}")


    def _handle_client(self, client_socket, addr):
        try:
            while self.is_running:
                data = client_socket.recv(4096)
                if not data:
                    break
                parse_data = self._parse_data(data, addr)
                if parse_data:
                    self.logger.info(parse_data)
                    self.data_callback(parse_data)
        except Exception as e:
            self.logger.error(f"Error handling client {addr} on port {self.port}: {e}")
        finally:
            client_socket.close()
            self.logger.info(f"Connection closed with {addr} on port {self.port}")
    
    
    def _parse_device_id(self, data_bytes, addr):
        """Parse device ID data packet (function tag 0x0001)."""
        # 提取设备ID (从payload的5-18字节)
        
        payload = data_bytes[16:]
        
        # 协议文档指定，响应包含13字节的Radar ID
        if len(payload) >= 13:
            radar_id = payload[:13]
            
            # 将Radar ID格式化为十六进制字符串
            radar_id_hex = ''.join([f'{b:02x}' for b in radar_id])
            
        
        # 存储这个addr对应的设备ID
        self.devices[addr] = radar_id_hex
        
        self.logger.info(f"Received device ID: {radar_id_hex} from {addr}")
        return {"device_id": radar_id_hex, "addr": addr}
    
    
    def _parse_data(self, data, addr):
        """Parse received data according to the protocol."""
        try:
            if len(data) < 16:
                self.logger.error(f"Data packet too short: {len(data)} bytes")
                return None
            
            if data[0] != 0x13:
                timestamp = int(time.time())
                self.logger.error(f"{timestamp} - Invalid magic number: 0x{data[0]:02x}")
                return None
            
            # Parse function tag
            func_tag = int.from_bytes(data[14:16], byteorder="big")
            
            # Handle different packet types
            if func_tag == 0x03e8:  # Vital data
                return self._parse_vital_data(data, addr)
            elif func_tag == 0x0001:  # Device ID
                return None  # Skip device ID packets
            elif func_tag == 0x040f:  # Body movement data
                return None  # Skip body movement packets for now
            elif func_tag == 0x0410:  # Device ID
                self._parse_device_id(data, addr)
                return None
            else:
                timestamp = int(time.time())
                self.logger.info(f"{timestamp} - Received unhandled function tag: 0x{func_tag:04x}")
                return None
                
        except Exception as e:
            timestamp = int(time.time())
            self.logger.error(f"{timestamp} - Error parsing data packet: {e}")
            return None
    
    
    def _parse_vital_data(self, data_bytes, addr):
        """Parse vital sign data packet (function tag 0x03e8)."""
        timestamp = int(time.time())
        
        # Parse header
        # header = {
        #     "magic": data_bytes[0],
        #     "version": data_bytes[1],
        #     "type": data_bytes[2],
        #     "cmd": data_bytes[3],
        #     "req_id": int.from_bytes(data_bytes[4:8], byteorder="big"),
        #     "timeout": int.from_bytes(data_bytes[8:10], byteorder="big"),
        #     "content_len": int.from_bytes(data_bytes[10:14], byteorder="big"),
        #     "func_tag": int.from_bytes(data_bytes[14:16], byteorder="big")
        # }
        
        
        # Parse payload
        payload = data_bytes[16:]
        
        # Parse vital data
        # vital_data = {
        #     "timestamp": timestamp,
        #     "breath_bpm": struct.unpack('>f', payload[0:4])[0],
        #     "breath_curve": struct.unpack('>f', payload[4:8])[0],
        #     "heart_rate_bpm": struct.unpack('>f', payload[8:12])[0],
        #     "heart_rate_curve": struct.unpack('>f', payload[12:16])[0],
        #     "target_distance": struct.unpack('>f', payload[16:20])[0],
        #     "signal_strength": struct.unpack('>f', payload[20:24])[0],
        #     "valid_bit_id": struct.unpack('>i', payload[24:28])[0],
        # }
         # Extract values
        # Extract values with reduced precision
        breath_bpm = round(struct.unpack('>f', payload[0:4])[0], 5)  # 保留2位小数
        breath_curve = round(struct.unpack('>f', payload[4:8])[0], 5)  # 保留3位小数
        heart_bpm = round(struct.unpack('>f', payload[8:12])[0], 5)  # 保留2位小数
        heart_curve = round(struct.unpack('>f', payload[12:16])[0], 5)  # 保留3位小数
        target_distance = round(struct.unpack('>f', payload[16:20])[0], 2)  # 保留2位小数
        signal_strength = round(struct.unpack('>f', payload[20:24])[0], 5)  # 保留2位小数
        valid_bit_id = struct.unpack('>i', payload[24:28])[0]  # 整数不需要舍入
        body_move_energy = 0.0
        body_move_range = 0.0
        if len(payload) >= 36:
            body_move_energy = round(struct.unpack('>f', payload[28:32])[0], 5)
            body_move_range = round(struct.unpack('>f', payload[32:36])[0], 2)
        
        # Determine in_bed status and validity
        in_bed = signal_strength > 0
        
        # valid_status = "0_无效"
        # if valid_bit_id == 1:
        #     valid_status = "1_呼吸有效"
        # elif valid_bit_id == 2:
        #     valid_status = "2_呼吸和心率有效"
        
        device_id = self.devices.get(addr, "unknown")
        
        # Define structured array data type
        # dt = np.dtype([
        #     ('timestamp', np.int64),
        #     ('breath_bpm', np.float32),
        #     ('breath_curve', np.float32),
        #     ('heart_rate_bpm', np.float32),
        #     ('heart_rate_curve', np.float32),
        #     ('target_distance', np.float32),
        #     ('signal_strength', np.float32),
        #     ('valid_bit_id', np.int32),
        #     ('body_move_energy', np.float32),
        #     ('body_move_range', np.float32),
        #     ('in_bed', np.bool_),
        #     ('valid_status', 'U20'),
        #     ('port', np.int32),
        #     ('device_id', 'U30')
        # ])
        
        # Create structured array with a single record
        # vital_data = np.array([(
        #     timestamp,
        #     breath_bpm,
        #     breath_curve,
        #     heart_bpm,
        #     heart_curve,
        #     target_distance,
        #     signal_strength,
        #     valid_bit_id,
        #     body_move_energy,
        #     body_move_range,
        #     in_bed,
        #     valid_status,
        #     self.port,
        #     device_id
        # )], dtype=dt)
        return (
            timestamp, 
            breath_bpm, 
            breath_curve, 
            heart_bpm, 
            heart_curve, 
            target_distance, 
            signal_strength, 
            valid_bit_id, 
            body_move_energy, 
            body_move_range, 
            1 if in_bed else 0, 
            device_id
        )
            

    def send_get_radar_id_request(
        self, 
        client_socket, 
        request_type=1
    ):
        """
        发送获取雷达ID请求
        功能标签：0x0410
        request_type: 0=默认请求，1=替代格式1，2=替代格式2
        """
        timestamp = int(time.time())
        
        if request_type == 0:
            # 原始请求格式
            request = bytearray([
                0x13, 0x01,        # 魔数和版本
                0x01, 0x00,        # 类型(0x01=请求)和命令
                0x00, 0x00, 0x00, 0x01,  # 请求ID
                0x00, 0x0A,        # 超时
                0x00, 0x00, 0x00, 0x06,  # 内容长度 (6字节)
                0x04, 0x10,        # 功能标签 (0x0410)
                0x00, 0x00, 0x00, 0x00   # 数据内容(空)
            ])
            print(f"{timestamp} - 发送获取雷达ID请求(格式0)...")
        
        elif request_type == 1:
            # 替代格式1 - 调整类型和命令
            request = bytearray([
                0x13, 0x01,        # 魔数和版本
                0x01, 0x01,        # 类型和命令(调整命令为0x01)
                0x00, 0x00, 0x00, 0x01,  # 请求ID
                0x00, 0x0A,        # 超时
                0x00, 0x00, 0x00, 0x06,  # 内容长度
                0x04, 0x10,        # 功能标签
                0x00, 0x00, 0x00, 0x00   # 数据内容
            ])
            print(f"{timestamp} - 发送获取雷达ID请求(格式1)...")
        
        elif request_type == 2:
            # 替代格式2 - 使用协议文档中准确的格式
            request = bytearray([
                0x13, 0x01,        # 魔数和版本
                0x01, 0x00,        # 类型和命令
                0x00, 0x00, 0x00, 0x02,  # 请求ID(增加)
                0x00, 0x0A,        # 超时
                0x00, 0x00, 0x00, 0x06,  # 内容长度
                0x04, 0x10,        # 功能标签
                0x00, 0x00, 0x00, 0x00   # 数据内容
            ])
            print(f"{timestamp} - 发送获取雷达ID请求(格式2)...")
        
        # 发送请求
        client_socket.send(request)
        self.logger.info(f"{timestamp}  已发送获取雷达ID请求(格式{request_type})\r\n")


    def execute(self):
        pass        


if __name__ == '__main__':
    
    server_8000 = SocketServer(
        port=8000,
    )
    server_8000.start()
    # server_8000.stop()
    
    while True:
        continue
    
    