#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/24 16:41
@Author  : weiyutao
@File    : sx_data_provider.py
"""
from pydantic import BaseModel, model_validator, ValidationError
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
    Type
)
import numpy as np
import torch
import traceback

from whoami.provider.data_provider import DataProvider
from whoami.configs.sql_config import SqlConfig
from whoami.provider.sql_provider import SqlProvider
from whoami.provider.base_ import ModelType

class SxDataProvider(DataProvider):
    def __init__(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        data: Optional[np.ndarray] = None,
        sql_provider: Optional[SqlProvider] = None,
        sql_query: Optional[str] = "SELECT in_out_bed, signal_intensity, breath_line, heart_line, breath_bpm, heart_bpm, state, body_move_data, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log WHERE device_sn='13D7F349200080712111150807' AND create_time >= '2024-11-13 20:00:00' AND create_time < '2024-11-14 09:00:00'",
        # sql_query: Optional[str] = "SELECT in_out_bed, distance, breath_line, heart_line, breath_bpm, heart_bpm, state, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log_bxx_0103 WHERE device_sn='13D7F349200080712111150807' AND create_time >= '2024-11-13 20:00:00' AND create_time < '2024-11-14 09:00:00'",
        model: Type[ModelType] = None
    ) -> None:
        super().__init__(
            sql_config_path=sql_config_path, 
            sql_config=sql_config,
            data=data,
            sql_provider=sql_provider,
            sql_query=sql_query,
            model=model
        )
        
        
    
    def process_zeros_numpy(self, arr, threshold=30):
        """
        处理NumPy数组中的连续零元素：
        - 如果连续零的数量 >= threshold，保持为0
        - 否则，将其转换为1
        
        参数:
        arr -- 输入NumPy数组，包含0和1
        threshold -- 连续零的阈值，默认为30
        
        返回:
        处理后的NumPy数组
        """
        # 创建一个全1的结果数组
        result = np.ones_like(arr)
        
        # 找出所有为0的位置
        zero_positions = np.where(arr == 0)[0]
        
        if len(zero_positions) > 0:
            # 计算连续0的起始位置
            # 通过比较相邻位置的差值，找出不连续的点
            breaks = np.where(np.diff(zero_positions) > 1)[0]
            
            # 所有连续0序列的起始索引
            starts = np.concatenate(([0], breaks + 1))
            
            # 所有连续0序列的结束索引
            ends = np.concatenate((breaks, [len(zero_positions) - 1]))
            
            # 处理每个连续0的序列
            for start_idx, end_idx in zip(starts, ends + 1):
                # 获取实际数组中连续0的起始和结束位置
                start_pos = zero_positions[start_idx]
                end_pos = zero_positions[end_idx - 1] + 1  # +1是因为切片是左闭右开
                
                # 计算连续0的长度
                length = end_pos - start_pos
                
                # 如果连续0的长度大于等于阈值，设置为0
                if length >= threshold:
                    result[start_pos:end_pos] = 0
        
        return result
    
    
        
    def get_data(self, query: Optional[str] = None):
        data = self.sql_provider.exec_sql(self.sql_query)
        if data.size != 0:
            # 从数据库读取的none数据要先转换为np.nan否则不能直接转换为tensor
            data_with_npnan = np.where(data == None, np.nan, data).astype(np.float64)
            # 提取所有体动数据
            body_move_rows = data_with_npnan[~np.isnan(data_with_npnan[:, 7])]
            body_move_values_to_fill = body_move_rows[:, 7]
            match_col_8 = body_move_rows[:, 8]

            # 回填体动动量值，有动量的为原始数据，无动量的使用0填充
            # in_out_bed, signal_intensity, breath_line, heart_line, breath_bpm, heart_bpm, state, body_move_data, create_time
            for i in range(len(body_move_rows)):
                mask = (data_with_npnan[:, 8] == match_col_8[i])
                data_with_npnan[mask, 7] = body_move_values_to_fill[i]
            
            
            data_with_npnan[np.isnan(data_with_npnan[:, 7]), 7] = 0
            # 先过滤掉为none的字段，因为转换为torch.float64会报错
            # 后续需要根据这些字段去拿到体动值数据
            # 注意原始数据中，如果信号强度为0，心率呼吸率数据均为0，因此判断是否为空会保留这些为0的数据
            mask = ~(np.isnan(data_with_npnan[:, 4]) & np.isnan(data_with_npnan[:, 5]))
            original_data = data_with_npnan[mask]
            
            # 在离床判断条件 
            # signal_intensity !=0 or (signal_intensity == 0 and inout_bed == 1) 在床
            # signal_intensity == 0 and inout_bed != 1  离床 other
            in_out_bed = original_data[:, 0]
            try:
                signal_intensity = original_data[:, 1]
                result = np.full(in_out_bed.shape, -1)
                condition1 = (signal_intensity != 0) | ((signal_intensity == 0) & (in_out_bed == 1)) # 在床
                condition2 = (signal_intensity == 0) & (in_out_bed != 1) # 离床
                result[condition1] = 1   # 满足条件1的标记为1
                result[condition2] = 0
                result = self.process_zeros_numpy(result, 30)
                if np.any(result == -1):
                    raise ValueError('condition1 and condition2 not fill all data!')
                original_data[:, 0] = result
            except Exception as e:
                self.logger.error(traceback.format_exc())
                raise ValueError('fail to get in_out_bed data!') from e
            # 最后切记在转换为tensor之前一定要转换为numpy.float64或者其他数字格式，否则np.object格式是无法转换为tensor格式的
            return original_data.astype(np.float64)
        return data
    
    
    
    
    
    
    def get_item(self, index):
        # if you need not change any, call the get_item method in super class directly.
        # return super().get_item(index)
        try:
            data = self.data[index]
            data_tensor = torch.tensor(data, dtype=torch.float64)
        except Exception as e:
            self.logger.error(traceback.format_exc())
            raise ValueError('fail to init data!') from e
        return data_tensor
    
        """
        int_fields = data[2:]
        float_fields = data[:2] 
        int_tensor = torch.tensor(int_fields, dtype=torch.int32)
        float_tensor = torch.tensor(float_fields, dtype=torch.float32)
        float_tensor = torch.round(float_tensor * 100) / 100
        # return torch.tensor(self.get_item(index), dtype=torch.float32)
        return {
            "int_tensor": int_tensor,
            "float_tensor": float_tensor
        }
        """

    