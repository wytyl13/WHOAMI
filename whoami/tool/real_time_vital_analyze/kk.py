#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/19 12:03
@Author  : weiyutao
@File    : kk.py
"""

import socket
import time 
from struct import unpack, pack
import os
import struct
import hashlib
import uuid

python_demo_version = "0.0.7"

print("Assure Python Demo Ver:", python_demo_version)
fileHandle = open('data.txt', 'a')


def litte2Big_short(data):
    convert_data = pack('>h', data)
    return convert_data


def log_info(logstr):
    fileHandle.write(logstr)
    fileHandle.flush()


def read_ota_file(fname):
    file_h = open(fname, "rb")
    bin_content = file_h.read()
    return bin_content

strLog = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) + "  Start Server"
log_info(strLog)

# 指定协议
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 让端口可以重复使用
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# 绑定ip和端口
server.bind(('0.0.0.0', 8000))
# 监听
server.listen(10)

def get_sign8(vx):
    if not vx or vx < 0x80:
        return vx
    return vx - 0x100


HeatMap_CNT = 0
PresenceDetection_CNT = 0


def parse_vital_data_packet(hex_data):
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


def parse_radar_id_packet(data_bytes):
    """
    解析雷达ID数据包(功能标签0x0001)
    根据协议，该数据包包含:
    - 雷达类型(1字节)
    - 硬件版本(4字节)
    - 唯一设备ID(13字节，即UUID)
    """
    # Add timestamp
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
    
    # 解析头部
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
    
    # 检查是否为雷达ID数据包
    if header["func_tag"] == 0x0001:
        payload = data_bytes[16:]
        
        # 协议中指定雷达ID数据包包含:
        # 1字节雷达类型 + 4字节硬件版本 + 13字节唯一设备ID
        if len(payload) >= 18:  # 雷达ID数据包应为18字节
            radar_data = {
                "timestamp": timestamp,  # Add timestamp to radar data
                "radar_type": payload[0],  # 1字节
                "hw_version": payload[1:5],  # 4字节
                "uuid": payload[5:18]  # 13字节
            }
            
            # 将硬件版本格式化为十六进制字符串
            hw_version_hex = ''.join([f'{b:02x}' for b in radar_data["hw_version"]])
            
            # 将UUID格式化为十六进制字符串
            uuid_hex = ''.join([f'{b:02x}' for b in radar_data["uuid"]])
            
            radar_data["hw_version_hex"] = hw_version_hex
            radar_data["uuid_hex"] = uuid_hex
            
            return {"header": header, "radar_data": radar_data, "timestamp": timestamp}
    
    return {"header": header, "data": data_bytes[16:], "timestamp": timestamp}


def parse_get_radar_id_response(data_bytes):
    """
    解析获取雷达ID响应数据包(功能标签0x0410)
    """
    # Add timestamp
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
    
    # 解析头部
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
    
    # 检查是否为获取雷达ID响应
    if header["func_tag"] == 0x0410:
        payload = data_bytes[16:]
        
        # 协议文档指定，响应包含13字节的Radar ID
        if len(payload) >= 13:
            radar_id = payload[:13]
            
            # 将Radar ID格式化为十六进制字符串
            radar_id_hex = ''.join([f'{b:02x}' for b in radar_id])
            
            return {"header": header, "radar_id": radar_id, "radar_id_hex": radar_id_hex, "timestamp": timestamp}
    
    return {"header": header, "data": data_bytes[16:], "timestamp": timestamp}


# 尝试多种格式的获取雷达ID请求
def send_get_radar_id_request(client_socket, request_type=0):
    """
    发送获取雷达ID请求
    功能标签：0x0410
    request_type: 0=默认请求，1=替代格式1，2=替代格式2
    """
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())  # Add timestamp
    
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
    log_info(timestamp + f"  已发送获取雷达ID请求(格式{request_type})\r\n")


def generate_device_fingerprint(address, first_packet):
    """
    根据设备地址和第一个数据包生成一个替代的设备标识符
    """
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())  # Add timestamp
    
    # 使用地址和数据包的前几个字节创建指纹
    fingerprint_data = f"{address[0]}:{address[1]}:".encode() + first_packet[:20]
    device_fingerprint = hashlib.md5(fingerprint_data).hexdigest()
    print(f"{timestamp} - 生成设备指纹: {device_fingerprint}")
    return device_fingerprint


def parse_body_movement_data(data_bytes):
    """
    解析身体运动累积能量数据包(功能标签0x040f)
    
    根据协议文档第11页:
    功能标签: 0x040f
    数据格式: 4字节浮点数(big endian)
    含义: "The data value is the accumulated energy for every body movement"
    """
    # Add timestamp
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
    
    # 解析数据包头部
    header = {
        "magic": data_bytes[0],        # 魔数(0x13)
        "version": data_bytes[1],      # 版本(通常是0x01)
        "type": data_bytes[2],         # 类型(0x02表示数据上报)
        "cmd": data_bytes[3],          # 命令
        "req_id": int.from_bytes(data_bytes[4:8], byteorder="big"),  # 请求ID
        "timeout": int.from_bytes(data_bytes[8:10], byteorder="big"),  # 超时
        "content_len": int.from_bytes(data_bytes[10:14], byteorder="big"),  # 内容长度
        "func_tag": int.from_bytes(data_bytes[14:16], byteorder="big")  # 功能标签(0x040f)
    }
    
    # 验证功能标签
    if header["func_tag"] != 0x040f:
        return {"header": header, "error": "非身体运动数据包", "timestamp": timestamp}
    
    # 解析有效载荷
    payload = data_bytes[16:]
    
    # 检查有效载荷长度是否足够
    if len(payload) < 4:
        return {"header": header, "error": f"数据长度不足: {len(payload)}字节，预期至少4字节", "timestamp": timestamp}
    
    try:
        # 将4字节解析为浮点数(IEEE 754格式，大端序)
        # 示例: 42 89 96 34 → 68.79
        movement_energy = struct.unpack('>f', payload[:4])[0]
        
        result = {
            "header": header,
            "movement_energy": movement_energy,
            "timestamp": timestamp
        }
        return result
        
    except struct.error as e:
        return {"header": header, "error": f"浮点数解析错误: {str(e)}", "timestamp": timestamp}


def get_report_interval(client_socket, req_id):
    """获取当前报告间隔设置并解析响应"""
    # req_id = int(time.time()) % 10000  # 使用当前时间生成唯一请求ID
    
    command = bytearray([
        0x13, 0x01,        # 魔数和版本
        0x01, 0x00,        # 类型和命令
        *(req_id.to_bytes(4, byteorder="big")),  # 请求ID
        0x00, 0x0A,        # 超时
        0x00, 0x00, 0x00, 0x06,  # 内容长度 (6字节)
        0x03, 0xEA,        # 功能标签 (0x03EA = Get Report Interval)
        0x00, 0x00, 0x00, 0x00   # 空数据
    ])
    
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
    print(f"{timestamp} - 发送获取报告间隔命令，请求ID={req_id}")
    
    client_socket.send(command)
    
    # 等待响应
    client_socket.settimeout(2.0)
    try:
        response = client_socket.recv(4096)
        
        # 检查响应是否有效
        if len(response) >= 20:  # 至少需要包含头部和间隔值
            response_func_tag = int.from_bytes(response[14:16], byteorder="big")
            
            # 检查是否是获取报告间隔的响应
            if response_func_tag == 0x03EA:
                interval = int.from_bytes(response[16:20], byteorder="big")
                print(f"{timestamp} - 当前报告间隔设置为: {interval} (约{interval*50}毫秒)")
            else:
                print(f"{timestamp} - 收到非预期响应，功能标签: 0x{response_func_tag:04x}")
                print(f"响应内容: {' '.join([f'{b:02x}' for b in response])}")
        else:
            print(f"{timestamp} - 响应太短或无效: {len(response)} 字节")
            if response:
                print(f"响应内容: {' '.join([f'{b:02x}' for b in response])}")
    
    except socket.timeout:
        print(f"{timestamp} - 等待响应超时，设备可能未接收命令")
    
    # 恢复原来的超时设置
    client_socket.settimeout(30.0)


def process_one_client(clientSocket, address):
    global HeatMap_CNT, PresenceDetection_CNT
    
    def hexdump(data):
        """将数据以十六进制格式打印，便于调试"""
        hex_str = ' '.join([f'{b:02x}' for b in data])
        ascii_str = ''.join([chr(b) if 32 <= b <= 126 else '.' for b in data])
        return f"HEX: {hex_str}\nASCII: {ascii_str}"

    # 初始化设备ID跟踪
    device_id = None
    first_packet = None
    device_fingerprint = None
    got_device_id = False
    req_id = None
    # 获取当前时间戳
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
    
    # 尝试获取设备ID
    print(f"{timestamp} - 尝试方法1: 等待设备发送ID数据包...")
    
    # 设置短暂等待时间，看看设备是否自己发送ID
    clientSocket.settimeout(2.0)
    try:
        # 接收可能的初始化数据包
        data = clientSocket.recv(4096)
        if data:
            timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
            first_packet = data
            # print(f"{timestamp} - 收到初始数据包:")
            # print(hexdump(data))
            
            # 检查是否是设备ID数据包
            if len(data) >= 16:
                func_tag = int.from_bytes(data[14:16], byteorder="big")
                if func_tag == 0x0001:
                    parse_data = parse_radar_id_packet(data)
                    if "radar_data" in parse_data:
                        radar_data = parse_data["radar_data"]
                        device_id = radar_data["uuid_hex"]
                        print(f"{parse_data['timestamp']} - 成功接收到设备ID: {device_id}")
                        got_device_id = True
                        
                        # 发送确认
                        response = bytearray([
                            0x13, 0x01,  # 魔数和版本
                            0x00, 0x01,  # 类型和命令
                            *(parse_data["header"]["req_id"].to_bytes(4, byteorder="big")),  # 请求ID
                            0x00, 0x0A,  # 超时
                            0x00, 0x00, 0x00, 0x06,  # 内容长度
                            0x00, 0x01,  # 功能标签
                            0x00, 0x00, 0x00, 0x00  # 数据
                        ])
                        clientSocket.send(response)
                        
                        
            
            # 如果没有收到设备ID，生成替代标识符
            if not got_device_id and first_packet:
                timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                device_fingerprint = generate_device_fingerprint(address, first_packet)
                print(f"{timestamp} - 生成替代设备标识符: {device_fingerprint}")
                log_info(timestamp + f"  生成设备指纹: {device_fingerprint}\r\n")
    except socket.timeout:
        timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
        print(f"{timestamp} - 初始等待超时，设备未主动发送ID")
    
    # 如果还没有获取到设备ID，尝试主动请求
    if not got_device_id:
        timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
        print(f"{timestamp} - 尝试方法2: 主动请求设备ID...")
        
        # 尝试多种格式的请求
        for request_type in range(3):
            send_get_radar_id_request(clientSocket, request_type)
            
            try:
                # 等待响应
                clientSocket.settimeout(2.0)
                data = clientSocket.recv(4096)
                if data:
                    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                    print(f"{timestamp} - 收到响应(请求类型{request_type}):")
                    print(hexdump(data))
                    
                    # 检查是否是设备ID响应
                    if len(data) >= 16:
                        func_tag = int.from_bytes(data[14:16], byteorder="big")
                        if func_tag == 0x0410:
                            radar_id_response = parse_get_radar_id_response(data)
                            req_id = radar_id_response["header"]["req_id"]
                            get_report_interval(clientSocket, req_id=req_id)
                            if "radar_id_hex" in radar_id_response:
                                device_id = radar_id_response["radar_id_hex"]
                                print(f"{radar_id_response['timestamp']} - 成功获取到设备ID: {device_id}")
                                got_device_id = True
                                break
            except socket.timeout:
                timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                print(f"{timestamp} - 请求类型{request_type}无响应")
    
    # 如果仍未获取到设备ID，使用替代标识符
    if not got_device_id:
        timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
        if not device_fingerprint and first_packet:
            device_fingerprint = generate_device_fingerprint(address, first_packet)
        elif not device_fingerprint:
            # 如果还没有收到任何数据包，只使用地址生成指纹
            device_fingerprint = hashlib.md5(f"{address[0]}:{address[1]}".encode()).hexdigest()
        
        print(f"{timestamp} - 无法获取真实设备ID，使用替代标识符: {device_fingerprint}")
        log_info(timestamp + f"  使用替代设备标识符: {device_fingerprint}\r\n")
        device_id = device_fingerprint
    
    # 重置超时为较长时间，继续接收数据
    clientSocket.settimeout(30.0)
    
    # 主数据接收循环
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
    print(f"{timestamp} - 设备 {device_id} 连接成功，开始接收数据...")
    log_info(timestamp + f"  设备 {device_id} 连接成功\r\n")
    
    while True:
        try:
            # 如果已经有第一个数据包但还没处理，先处理它
            if first_packet:
                data = first_packet
                first_packet = None  # 清除，避免重复处理
            else:
                # 接收新数据
                data = clientSocket.recv(4096)
            
            if not data:
                timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                print(f"{timestamp} - 设备 {device_id} 断开连接")
                log_info(timestamp + f"  设备 {device_id} 断开连接\r\n")
                break
            
            # 打印原始数据包
            timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
            # print(f"{timestamp} - 从设备 {device_id} 收到数据包:")
            # print(hexdump(data))
            
            # 检查数据长度
            if len(data) < 16:
                timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                print(f"{timestamp} - 数据包太短: {len(data)} 字节")
                continue
            
            # 检查魔数
            if data[0] != 0x13:
                timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                print(f"{timestamp} - 无效的魔数: 0x{data[0]:02x}")
                continue
            
            # 解析功能标签
            func_tag = int.from_bytes(data[14:16], byteorder="big")
            
            try:
                # 处理不同类型的数据包
                if func_tag == 0x03e8:  
                    # 生命体征数据
                    parse_data = parse_vital_data_packet(data)
                    print(f"{parse_data['timestamp']} - 设备 {device_id} 生命体征数据:")
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
                    print(vital)
                elif func_tag == 0x0001:  
                    # 设备ID数据包，忽略
                    continue
                    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                    print(f"{timestamp} - 收到设备ID数据包，发送确认响应")
                    
                    # 从数据包中提取请求ID
                    req_id = int.from_bytes(data[4:8], byteorder="big")
                    
                    # 创建确认响应
                    response = bytearray([
                        0x13, 0x01,  # 魔数和版本
                        0x00, 0x01,  # 类型和命令
                        *(req_id.to_bytes(4, byteorder="big")),  # 返回相同的请求ID
                        0x00, 0x0A,  # 超时
                        0x00, 0x00, 0x00, 0x06,  # 内容长度
                        0x00, 0x01,  # 功能标签 (与请求相同)
                        0x00, 0x00, 0x00, 0x00  # 空数据
                    ])
                    clientSocket.send(response)
                elif func_tag == 0x040f:
                    # 累积体动动量值
                    result = parse_body_movement_data(data)
                    print(f"{parse_data['timestamp']} - 设备 {device_id} 累积体动动量值:")
                    print(result)
                else:
                    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                    print(f"{timestamp} - 收到未处理的功能标签: 0x{func_tag:04x}")
            
            except Exception as e:
                timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
                print(f"{timestamp} - 处理数据包时出错: {e}")
                log_info(timestamp + f"  处理数据包错误: {e}\r\n")
        
        except socket.timeout:
            timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
            print(f"{timestamp} - 设备 {device_id} 连接超时")
            break
        
        except Exception as e:
            timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
            print(f"{timestamp} - 与设备 {device_id} 通信时发生错误: {e}")
            log_info(timestamp + f"  设备 {device_id} 错误: {e}\r\n")
            break


if __name__ == "__main__":
    timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
    print(f"{timestamp} - 启动服务器，监听端口8000...")
    print(f"{timestamp} - 等待设备连接...")
    while True:
        # 等待设备连接
        clientSocket, address = server.accept()
        timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
        print(f"{timestamp} - 设备已连接，地址: {address}")
        
        # 处理客户端连接
        try:
            process_one_client(clientSocket, address)
        except Exception as e:
            timestamp = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())
            print(f"{timestamp} - 处理客户端时发生错误: {e}")
        finally:
            try:
                clientSocket.close()
            except:
                pass