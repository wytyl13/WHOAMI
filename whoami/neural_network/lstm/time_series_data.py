#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/28 09:25
@Author  : weiyutao
@File    : airbus.py
"""



import torch
import pandas as pd
import torch.utils
from torch.utils.data import Dataset
import numpy as np
import torch.utils.data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class TimeSeriesData(Dataset):
    def __init__(
        self,
        path_or_data,
        type,
        seq_len,
        stride=1,
        nrows=None
    ):
        if type == 'csv':
            self.data = pd.read_csv(path_or_data, delimiter=' ', nrows=nrows, header=None)
        elif type == 'pytorch':
            self.data = torch.load(path_or_data)
        elif type == 'numpy':
            if isinstance(path_or_data, str):
                self.data = np.load(path_or_data)
            elif isinstance(path_or_data, np.ndarray):
                self.data = path_or_data
        else:
            raise ValueError('type value is wrong: ', type)
        self.type = type
        self.seq_len = seq_len
        self.stride = stride
        self.scaler = MinMaxScaler()
        self.raw_data = self._handle_missing_values(self.data)
        print(np.isnan(self.raw_data).any())
        self.sequence_indices = self._create_sequence_indices()
        super().__init__()


    def _handle_missing_values(self, data):
        # 将None使用nan替换，然后置为0
        data_with_npnan = np.where(data == None, np.nan, data).astype(np.float64)
        data = np.nan_to_num(data_with_npnan, nan=0.0, posinf=0.0, neginf=0.0)
        data = self.scaler.fit_transform(data)
        return data.astype(np.float32)
    
    
    def _create_sequence_indices(self):
        indices = []
        max_start_idx = len(self.raw_data) - self.seq_len + 1
        for i in range(0, max_start_idx, self.stride):
            indices.append(i)
        return indices
    
    
    def __len__(self):
        return len(self.sequence_indices)


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        start_idx = self.sequence_indices[index]    
        end_idx = start_idx + self.seq_len
        sequence = self.raw_data[start_idx:end_idx]
        return torch.from_numpy(sequence)
    
    
def create_train_test_split(dataset, train_ratio):
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    test_size = dataset_size - train_size

    train_indices = list(range(train_size))
    test_indices = list(range(train_size, dataset_size))
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    return train_subset, test_subset


if __name__ == '__main__':
    import datetime
    from datetime import datetime, timedelta, timezone
    from torch.utils.data import DataLoader


    from whoami.tool.health_report.sx_device_wavve_vital_sign_log_20250522 import SxDeviceWavveVitalSignLog
    from whoami.provider.sql_provider import SqlProvider
    from whoami.configs.sql_config import SqlConfig


    SQL_CONFIG_PATH = "/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
    sql_provider = SqlProvider(model=SxDeviceWavveVitalSignLog, sql_config_path=SQL_CONFIG_PATH)

    batch_size = 32
    table_name = "sx_device_wavve_vital_sign_log_20250529"
    device_sn = "13331C9D100040711117950407"
    query_date = "2025-5-29"
    current_date = datetime.strptime(query_date, '%Y-%m-%d')
    pre_date_str = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
    sql_query = f"SELECT signal_intensity, breath_bpm, heart_bpm, state, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM {table_name} WHERE device_sn='{device_sn}' AND create_time >= '{pre_date_str} 22:15' AND create_time < '{query_date} 1:30'"
    results = sql_provider.exec_sql(sql_query)
    dataset = TimeSeriesData(results[:, :-1], 'numpy', 120, 30)
    train_dataset, test_dataset = create_train_test_split(dataset, train_ratio=0.8)
    print(f"train_loader size: {len(train_dataset)}")
    print(f"test_loader size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for i, batch in enumerate(train_loader):
        print(f"批次 {i}: {batch.shape}")
        if i >= 2:
            break
    
    