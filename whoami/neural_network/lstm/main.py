#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/28 09:21
@Author  : weiyutao
@File    : main.py
"""
import torch
import torch.optim as optim

from whoami.neural_network.lstm.lstm_auto_encoder import LSTMAutoEncoder
from whoami.neural_network.lstm.time_series_data import TimeSeriesData
from whoami.neural_network.lstm.time_series_data import create_train_test_split


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
seq_len, nb_feature = train_dataset[0].shape
for i, batch in enumerate(train_loader):
        print(f"批次 {i}: {batch.shape}")
        if i >= 2:
            break


device=torch.device('cpu')

model = LSTMAutoEncoder(num_layers=1, hidden_size=128, nb_feature=nb_feature, dropout=0, device=device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


def train(epoch):
    model.train()
    train_loss = 0
    for id_batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        print(data.shape) # torch.Size([32, 120, 4]) # batch_size, seq_len, input_size
        output = model.forward(data)
        print(f"output: ------------------------------------ {output}")
        loss = criterion(data, output.to(device))

        loss.backward()
        train_loss += loss.item()
        print(f"loss: -------------------------------------------- {loss.item()}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        
        print('\r', 'Training [{}/{} ({:.0f}%)] \tLoss: {:.6f})]'.format(
            id_batch + 1, len(train_loader),
            (id_batch + 1) * 100 / len(train_loader),
            loss.item()), sep='', end='', flush=True)
        
    avg_loss = train_loss / len(train_loader)
    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, avg_loss))


if __name__ == '__main__':
    for epoch in range(1 ,30):
        train(epoch)

