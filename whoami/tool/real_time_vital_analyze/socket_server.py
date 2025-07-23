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
import pytz
from datetime import datetime, timezone, timedelta

from whoami.tool.agent.base_tool import tool

from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sx_device_wavve_vital_sign_log_20250522 import SxDeviceWavveVitalSignLog
sql_provider_test = SqlProvider(
    model=SxDeviceWavveVitalSignLog, 
    sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml",
)
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
        data_callback: Callable[[Dict[str, Any]], None] = None,
        device_sn_call_back: Callable[[Dict[str, Any]], None] = None,
        injected_data: Optional[list] = None,
        device_sn: Optional[str] = None
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
        self.device_sn_call_back = device_sn_call_back
        self.device_sn = device_sn   
        self.injected_data = injected_data 


    def preprocess_data(self, data):
        try:
            source_tz = pytz.timezone('Asia/Shanghai')
            utc_tz = pytz.UTC
            if data.get('heart_bpm') is None:
                    return None
            processed_record = data.copy()
            
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
        except Exception as e:
            raise ValueError(f"fail to exec preprocess data {str(e)}")
        return processed_record



    def _send_registration_response(self, client_socket, original_data):
        """回复设备注册请求，告诉设备注册成功，可以开始发送数据"""
        try:
            # 从原始请求中提取req_id
            req_id = int.from_bytes(original_data[4:8], byteorder="big")
            
            # 构造回复消息
            response = bytearray([
                0x13, 0x01,                    # magic, version
                0x00, 0x02,                    # type=response(0x00), cmd=response(0x02)
                req_id.to_bytes(4, byteorder='big')[0],  # req_id (对应设备的请求)
                req_id.to_bytes(4, byteorder='big')[1],
                req_id.to_bytes(4, byteorder='big')[2], 
                req_id.to_bytes(4, byteorder='big')[3],
                0x00, 0x0A,                    # timeout
                0x00, 0x00, 0x00, 0x06,        # content_len = 6
                0x00, 0x01,                    # func_tag (对应0x0001)
                0x00, 0x00, 0x00, 0x01         # 返回0x01表示注册成功
            ])
            
            # 发送回复
            client_socket.send(response)
            self.logger.info(f"已发送设备注册成功回复，req_id: {req_id}")
            
        except Exception as e:
            self.logger.error(f"发送注册回复失败: {e}")
    


    def start(self):
        if self.is_running:
            self.logger.warning(f"Socket server on port {self.port} is already running!")
            return
        self.logger.info("--------------------------------------whoami--------------------------------")
        self.is_running = True
        self.logger.info(self.injected_data)
        if self.injected_data is None:
            if self.device_sn is not None:
                self.device_thread = threading.Thread(target=self.inject_real_time_device_sn_data)
                self.device_thread.daemon = True
                self.device_thread.start()
                self.logger.info("Server started (device_sn real time data mode)")
                
            self.accept_thread = threading.Thread(target=self._accept_connections)
            self.accept_thread.daemon = True # 守护线程
            self.accept_thread.start()
            self.logger.info(f"Socket server started on port {self.port}")
        else:
            # 注入模式：启动数据处理线程
            self.logger.info("--------------------------------------whoami--------------------------------")
            self.inject_thread = threading.Thread(target=self._handle_injected_data)
            # self.inject_thread = threading.Thread(target=self.start_with_injected_data)
            self.inject_thread.daemon = True
            self.inject_thread.start()
            self.logger.info("Server started (injected data mode)")
        
        
    def start_with_injected_data(self):
        if not self.injected_data:
            return
        
        # 按设备分组
        device_groups = {}
        for data in self.injected_data:
            device_id = data[-1]
            if device_id not in device_groups:
                device_groups[device_id] = []
            device_groups[device_id].append(data)
        
        # 每个设备开一个线程发送
        def send_device_data(device_id, data_list):
            for data in data_list:
                self.data_callback(data)
                time.sleep(0.01)  # 发送间隔
        
        threads = []
        for device_id, data_list in device_groups.items():
            t = threading.Thread(target=send_device_data, args=(device_id, data_list))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
    
    
    def inject_real_time_device_sn_data(self):
        """处理注入实时编号的数据"""
        mock_addr = ('127.0.0.1', 0)
        
        def insert_at_position(original_dict, position, key, value):
            """在字典的指定位置插入键值对"""
            items = list(original_dict.items())
            items.insert(position, (key, value))
            return dict(items)
        
        try:
            for i in range(100000):
                if not self.is_running:
                    return None
                past_time = (datetime.now() - timedelta(seconds=2)).strftime("%Y-%m-%d %H:%M:%S")
                data = sql_provider_test.get_record_by_condition(
                    condition={"device_sn": self.device_sn, "create_time": past_time},  # 每次查一个设备
                    fields=["create_time", "breath_bpm", "breath_line", "heart_bpm", "heart_line", "distance", "signal_intensity", "state", "body_move_data", "device_sn"],
                )
                # 使用与_handle_client相同的处理逻辑
                self.logger.info(f"{past_time}, {data}")
                read_data = data[-1]
                parse_data = self.preprocess_data(read_data)
                parse_data = insert_at_position(parse_data, 9, "body_move_range", 1)
                parse_data = insert_at_position(parse_data, 10, "in_bed", 0)
                parse_data = tuple(parse_data.values())
                self.logger.info(f"parse_data: --------------------------------- {parse_data}")
                if parse_data:
                    self.data_callback(parse_data)
                    
                # 可选：添加延时模拟实时数据间隔
                time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error handling injected data: {str(e)}")
        finally:
            self.logger.info("Finished processing all injected data")
    
    
    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self.injected_data is None:
            try:
                self.server_socket.close()
            except Exception as e:
                self.logger.error(f"Error closing server socket on port {self.port}: {e}")
            
            if self.accept_thread and self.accept_thread.is_alive():
                self.accept_thread.join(timeout=2)
        else:
            # 注入模式
            if hasattr(self, 'inject_thread') and self.inject_thread.is_alive():
                self.inject_thread.join(timeout=2)
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
                parse_data = self._parse_data(data, addr, client_socket)
                if parse_data:
                    # self.logger.info(parse_data)
                    self.data_callback(parse_data)
        except Exception as e:
            self.logger.error(f"Error handling client {addr} on port {self.port}: {e}")
        finally:
            client_socket.close()
            self.logger.info(f"Connection closed with {addr} on port {self.port}")
    
    
    def _handle_injected_data(self):
        """处理注入的数据列表，模拟_handle_client的行为"""
        mock_addr = ('127.0.0.1', 0)
        try:
            for data in self.injected_data:
                if not self.is_running:
                    break
                # 使用与_handle_client相同的处理逻辑
                parse_data = data
                if parse_data:
                    self.data_callback(parse_data)
                    
                # 可选：添加延时模拟实时数据间隔
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error handling injected data: {e}")
        finally:
            self.logger.info("Finished processing all injected data")
    
    
    def _parse_device_id(self, data_bytes, addr):
        """Parse device ID data packet (function tag 0x0001)."""
        # 提取设备ID (从payload的5-18字节)
        
        payload = data_bytes[16:]
        
        # 协议文档指定，响应包含13字节的Radar ID
        if len(payload) >= 13:
            radar_id = payload[:13]
            
            # 将Radar ID格式化为十六进制字符串
            radar_id_hex = ''.join([f'{b:02x}' for b in radar_id])
            
        self.logger.info(f"radar_id_hex: -------------------------- {radar_id_hex}")
        # 存储这个addr对应的设备ID
        self.devices[addr] = radar_id_hex
        self.device_sn_call_back(radar_id_hex.upper())
        self.logger.info(f"Received device ID: {radar_id_hex} from {addr}")
        return {"device_id": radar_id_hex, "addr": addr}
    
    
    def _parse_data(self, data, addr, client_socket):
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
                self._send_registration_response(client_socket, data) # 发送回复
                return None  # Skip device ID packets
            elif func_tag == 0x040f:  # Body movement data
                return None  # Skip body movement packets for now
            elif func_tag == 0x0410:  # Device ID
                self._parse_device_id(data, addr)
                self._send_registration_response(client_socket, data)
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
            device_id.upper()
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
    
    