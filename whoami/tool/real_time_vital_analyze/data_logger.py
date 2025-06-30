#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/24 14:48
@Author  : weiyutao
@File    : data_logger.py
"""

import json
import csv
import os
from datetime import datetime
import threading

class DataLogger:
    def __init__(self, base_path="/work/ai/WHOAMI/whoami/out/data_logs"):
        self.base_path = base_path
        self.ensure_directory_exists()
        self.file_lock = threading.Lock()
    
    def ensure_directory_exists(self):
        """确保目录存在"""
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
    
    def write_to_json(self, data_point, filename=None):
        """写入JSON文件 - 每条记录一行"""
        if filename is None:
            filename = f"realtime_data_{datetime.now().strftime('%Y%m%d')}.json"
        
        filepath = os.path.join(self.base_path, filename)
        
        with self.file_lock:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_point, ensure_ascii=False) + '\n')
    
    def write_to_csv(self, data_point, filename=None):
        """写入CSV文件"""
        if filename is None:
            filename = f"realtime_data_{datetime.now().strftime('%Y%m%d')}.csv"
        
        filepath = os.path.join(self.base_path, filename)
        file_exists = os.path.exists(filepath)
        
        # 定义CSV字段顺序
        fieldnames = ['timestamp', 'time', 'device_id', 'heart_rate', 'breathing_rate', 
                     'anomaly_status', 'reconstruction_loss', 'breath_line', 'heart_line']
        
        with self.file_lock:
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 如果文件不存在，写入表头
                if not file_exists:
                    writer.writeheader()
                
                # 处理复杂数据类型（如列表）
                row_data = data_point.copy()
                if 'breath_line' in row_data and isinstance(row_data['breath_line'], list):
                    row_data['breath_line'] = json.dumps(row_data['breath_line'])
                if 'heart_line' in row_data and isinstance(row_data['heart_line'], list):
                    row_data['heart_line'] = json.dumps(row_data['heart_line'])
                
                writer.writerow(row_data)
    
    def write_combined_data(self, data_point, label_data=None, item_data=None):
        """将label和datapoint写入同一个文件"""
        date_str = datetime.now().strftime('%Y%m%d')
        device_id = data_point.get('device_id', 'unknown')
        
        # 合并所有数据到一个记录中
        combined_data = data_point.copy()
        
        # 添加原始label数据（如果存在）
        if label_data:
            combined_data['raw_label_data'] = label_data
        
        # 添加原始item数据（如果需要完整保存）
        if item_data:
            combined_data['raw_item_data'] = item_data
        
        # 写入合并后的数据到同一个文件
        filename = f"combined_data_{device_id}_{date_str}.json"
        self.write_to_json(combined_data, filename)
    
    def write_separate_files(self, data_point, label_data, item_data):
        """分别写入不同文件"""
        date_str = datetime.now().strftime('%Y%m%d')
        device_id = data_point.get('device_id', 'unknown')
        
        # 写入基础数据
        basic_data = {
            'timestamp': data_point['timestamp'],
            'time': data_point['time'],
            'device_id': device_id,
            'heart_rate': data_point['heart_rate'],
            'breathing_rate': data_point['breathing_rate']
        }
        self.write_to_json(basic_data, f"basic_data_{device_id}_{date_str}.json")
        
        # 写入标签数据
        if label_data:
            label_info = {
                'timestamp': data_point['timestamp'],
                'device_id': device_id,
                'anomaly_status': label_data[0],
                'reconstruction_loss': label_data[1]
            }
            self.write_to_json(label_info, f"label_data_{device_id}_{date_str}.json")
        
        # 写入原始波形数据
        waveform_data = {
            'timestamp': data_point['timestamp'],
            'device_id': device_id,
            'breath_line': item_data[2] if len(item_data) > 2 else None,
            'heart_line': item_data[4] if len(item_data) > 4 else None
        }
        self.write_to_json(waveform_data, f"waveform_data_{device_id}_{date_str}.json")
