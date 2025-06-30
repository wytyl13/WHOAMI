#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/24 16:48
@Author  : weiyutao
@File    : json_loader.py
"""


import json
import numpy as np

class SimpleJSONLoader:
    def __init__(self, filename):
        self.data = self.load_json(filename)
        
    def load_json(self, filename):
        """加载JSON文件"""
        data_list = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
        return data_list
    
    def get_array(self, field_name):
        """获取指定字段的numpy数组"""
        values = [item.get(field_name) for item in self.data]
        return np.array(values)
    
    def get_float_array(self, field_name):
        """获取数值字段的numpy数组"""
        values = [float(item.get(field_name, 0)) for item in self.data]
        return np.array(values, dtype=np.float64)
    
    def get_reconstruction_loss(self):
        """获取重构误差数组"""
        return self.get_float_array('reconstruction_loss')
    
    def get_timestamps(self):
        """获取时间戳数组"""
        return self.get_float_array('timestamp')
    
    def get_all_fields(self):
        """获取所有字段名"""
        if self.data:
            return list(self.data[0].keys())
        return []
    
    def get_all_data(self):
        """获取所有字段的numpy数组，返回字典"""
        all_fields = self.get_all_fields()
        result = {}
        
        for field in all_fields:
            if field in ['heart_rate', 'breathing_rate', 'reconstruction_loss', 'timestamp', 'breath_line', 'heart_line']:
                # 数值字段
                result[field] = self.get_float_array(field)
            elif field == 'anomaly_status':
                # 布尔字段
                values = [bool(item.get(field, False)) for item in self.data]
                result[field] = np.array(values, dtype=bool)
            else:
                # 字符串字段
                result[field] = self.get_array(field)
        
        return result
    
    def to_numpy_matrix(self, fields_order=None):
        """将所有数据转换为numpy矩阵格式(不带字段名)
        
        Args:
            fields_order: 指定字段顺序的列表，如果为None则使用默认顺序
            
        Returns:
            numpy数组: shape为(n_samples, n_features)
            字段顺序: 列对应的字段名列表
        """
        all_data = self.get_all_data()
        
        # 如果没有指定字段顺序，使用默认顺序(数值字段优先)
        if fields_order is None:
            numeric_fields = ['timestamp', 'reconstruction_loss', 'heart_rate', 'breathing_rate', 'breath_line', 'heart_line']
            bool_fields = ['anomaly_status']
            fields_order = [f for f in numeric_fields if f in all_data] + [f for f in bool_fields if f in all_data]
        
        # 按顺序堆叠数组
        arrays = []
        for field in fields_order:
            if field in all_data:
                arrays.append(all_data[field])
        
        # 转换为矩阵 (n_samples, n_features)
        matrix = np.column_stack(arrays)
        return matrix, fields_order
    
    def to_numpy_array(self, fields_order=None):
        """将所有数据转换为numpy数组格式(只返回数组，不返回字段名)"""
        matrix, _ = self.to_numpy_matrix(fields_order)
        return matrix
    
    
    