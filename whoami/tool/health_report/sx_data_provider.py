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
    def get_data(self, query: Optional[str] = None):
        data = self.sql_provider.exec_sql(self.sql_query)
        if data.size != 0:
            # 从数据库读取的none数据要先转换为np.nan否则不能直接转换为tensor
            data_with_npnan = np.where(data == None, np.nan, data).astype(np.float64)
            
            # 提取所有体动数据
            body_move_rows = data_with_npnan[~np.isnan(data_with_npnan[:, 7])]
            body_move_values_to_fill = body_move_rows[:, 7]
            match_col_8 = body_move_rows[:, 8]

            # 回填体动动量值，有动量的为原始数据，无动量的实用0填充
            for i in range(len(body_move_rows)):
                mask = (data_with_npnan[:, 8] == match_col_8[i])
                data_with_npnan[mask, 7] = body_move_values_to_fill[i]
                data_with_npnan[~mask, 7] = 0
            # 先过滤掉为none的字段，因为转换为torch.float64会报错
            # 后续需要根据这些字段去拿到体动值数据
            mask = ~(np.isnan(data_with_npnan[:, 4]) & np.isnan(data_with_npnan[:, 5]))
            original_data = data[mask]
            
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

    