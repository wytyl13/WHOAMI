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
)
import numpy as np
import torch

from whoami.provider.data_provider import DataProvider
from whoami.configs.sql_config import SqlConfig
from whoami.provider.sql_provider import SqlProvider

class SxDataProvider(DataProvider):
    
    def __init__(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        data: Optional[np.ndarray] = None,
        sql_provider: Optional[SqlProvider] = None,
        sql_query: Optional[str] = "SELECT in_out_bed, distance, breath_line, heart_line, breath_bpm, heart_bpm, state, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log WHERE device_sn='13D7F349200080712111150807' AND create_time >= '2024-11-13 20:00:00' AND create_time < '2024-11-14 10:00:00'"
    ) -> None:
        super().__init__(
            sql_config_path=sql_config_path, 
            sql_config=sql_config,
            data=data,
            sql_provider=sql_provider,
            sql_query=sql_query
        )
    
    def get_data(self, query: Optional[str] = None):
        return self.sql_provider.exec_sql(self.sql_query)
    
    def get_item(self, index):
        # if you need not change any, call the get_item method in super class directly.
        # return super().get_item(index)
        
        data = self.data[index]
        data_tensor = torch.tensor(data, dtype=torch.float32)
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

    