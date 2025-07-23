
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/20 10:07
@Author  : weiyutao
@File    : auto_encoders.py

original parameter update: w_new = w_old - lr * gradient
weight decay: w_new = (1 - lr * λ) * w_old - lr * gradient, 
    λ权重衰减系数，越大权重被压缩的越厉害
     (1 - lr * λ)，权重衰减因子，每次都让权重稍微缩小的系数，总是小于1
     weight decay直接在反向传播更新权重的时候起作用，在推理的时候间接起作用（因为推理的时候使用的是weight decay影响过的权重）
dropout：作用于前向传播的输出，一般作用于dense层
    output = w1×h1 + w2×h2 + w3×h3
    ∂output/∂w2 = h2
    
    [h1, h2, h3] = [2.0, 1.5, 3.0]
    如果dropout将前一层的输出h2置为0
    [h1, h2, h3] = [2.0, 0, 3.0]
    output = w1×h1 + w2×0 + w3×h3
    ∂output/∂w2 = 0    
    间接造成w2在本次反向传播中不被更新，因此dropout只会在训练的额时候有影响，不会间接作用于前向传播推理的时候
"""

import torch
import torch.nn as nn
import copy
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    定义一个编码器的子类，继承父类 nn.Module
    """
    def __init__(
        self, 
        seq_len, 
        n_features, 
        embedding_dim=64,
        dropout=0.2
    ):
        super(Encoder, self).__init__()
 
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.dropout = nn.Dropout(dropout) # 编码器使用较大的正则化方案
        # 使用双层LSTM
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True)
    
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=1,
          batch_first=True)

 
    def forward(self, x):
        # x (seq_len, n_features) - (60, 7)
        # x = x.reshape((1, self.seq_len, self.n_features)) # (batch, seq_len, n_features) - (1, 60, 7)
        x, (_, _) = self.rnn1(x) 
        # print(f"encoder rnn1 - x ----------- {x.shape}") # output_x - (batch, seq_len, hidden_size) - (1, 60, self.hidden_dim)
        x, (hidden_n, _) = self.rnn2(x) 
        # print(f"encoder rnn2 - x ----------- {x.shape}") # output_x - (batch, seq_len, hidden_size) - (1, 60, self.embedding_dim)  
        # print(f"encoder rnn2 - hidden_n ----------- {hidden_n.shape}") # hidden_n - (num_layers, batch_size, hidden_size) - (1, 1, self.embedding_dim)
        
        return hidden_n[-1] # 去最后一层隐藏层作为解码器的输入
    
    
    
class Decoder(nn.Module):
    """
    定义一个解码器的子类，继承父类 nn.Modul
    """
    def __init__(
        self, 
        seq_len, 
        input_dim=64, 
        n_features=1,
        dropout=0.1
    ):
        super(Decoder, self).__init__()
 
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.dropout = nn.Dropout(dropout) # 解码器使用更小的正则化方案
        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True)
 
        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True)
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

 
    def forward(self, x):
        # x in decode is the last layer of n_hidden in encoder. (batch_size, hidden_size) - (1, self.embedding_dim)  
        # print(f"the shape of x in decoder --------  {x.shape}")
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        # print(f"the changed x in decoder --------  {x.shape}")
        x, (hidden_n, cell_n) = self.rnn1(x)
        # print(f"decoder rnn1 x -------------- {x.shape}")
        
        x, (hidden_n, cell_n) = self.rnn2(x)
        # print(f"decoder rnn2 x -------------- {x.shape}")

        x = self.output_layer(x)
        # print(f"decoder output_layer x -------------- {x.shape}")
        return x


def preprocess_inference_data(data, seq_len, n_features, scaler, slide_window_flag=1):
    """
    预处理推理数据
    
    Args:
        data: 原始数据 (pandas DataFrame 或 numpy array)
        seq_len: 序列长度
        n_features: 特征数量
        scaler: 归一化器
        slide_window_flag: 是否使用滑动窗口
    
    Returns:
        tensor_data: 预处理后的张量数据
        scaler: 归一化器
        original_shape: 原始数据形状
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    original_shape = data.shape
    print(f"原始数据形状: {original_shape}")
    
    # 创建序列数据
    if slide_window_flag:
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i + seq_len])
        sequence_data = np.array(sequences)
    else:
        # 非滑动窗口，直接分割
        n_sequences = len(data) // seq_len
        sequence_data = data[:n_sequences * seq_len].reshape(n_sequences, seq_len, n_features)
    
    print(f"序列数据形状: {sequence_data.shape}")
    
    # 归一化
    seq_shape = sequence_data.shape
    reshaped_data = sequence_data.reshape(-1, seq_shape[-1])  # (n_samples * seq_len, n_features)
    
    if scaler is not None:
        normalized_data = scaler.transform(reshaped_data)
    else:
        # 如果没有提供scaler，创建新的
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(reshaped_data)
    
    # 重新reshape回序列形状
    normalized_sequences = normalized_data.reshape(seq_shape)
    
    # 转换为tensor
    tensor_data = torch.FloatTensor(normalized_sequences)
    
    return tensor_data, scaler, original_shape


def create_inference_dataloader(data, batch_size=32, shuffle=False):
    """
    创建推理用的DataLoader
    
    Args:
        data: 预处理后的tensor数据
        batch_size: 批次大小
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader
    """
    dataset = TensorDataset(data, data)  # 自动编码器输入=输出
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



def batch_inference(model, dataloader, device, return_errors=True):
    """
    批次推理
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        return_errors: 是否返回重构误差
    
    Returns:
        reconstructed_data: 重构数据
        reconstruction_errors: 重构误差（可选）
        raw_errors: 原始误差（每个时间步每个特征的误差）
    """
    model.eval()
    all_reconstructed = []
    all_errors = []
    all_raw_errors = []
    
    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device)
            
            # 前向传播
            reconstructed = model(batch_data)
            
            # 计算重构误差
            if return_errors:
                # 逐点误差：L1距离
                raw_error = torch.abs(batch_data - reconstructed)
                # 序列级误差：每个序列的平均误差
                sequence_error = raw_error.mean(dim=(1, 2))  # (batch_size,)
                
                all_raw_errors.append(raw_error.cpu().numpy())
                all_errors.append(sequence_error.cpu().numpy())
            
            all_reconstructed.append(reconstructed.cpu().numpy())
    
    # 合并所有批次的结果
    reconstructed_data = np.concatenate(all_reconstructed, axis=0)
    
    if return_errors:
        reconstruction_errors = np.concatenate(all_errors, axis=0)
        raw_errors = np.concatenate(all_raw_errors, axis=0)
        return reconstructed_data, reconstruction_errors, raw_errors
    else:
        return reconstructed_data


def create_data_loaders(train_data, val_data, batch_size=32):
    """
    创建DataLoader
    """
    # 创建TensorDataset（自动编码器的输入和目标是相同的）
    train_dataset = TensorDataset(train_data, train_data)
    val_dataset = TensorDataset(val_data, val_data)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader



def train_model(
    model, 
    train_loader, 
    val_loader, 
    n_epochs,
    lr=1e-4,
    patience=10,
    lr_scheduler=True,
    save_checkpoints=True
):
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-4
    )
    criterion = nn.L1Loss(reduction='mean').to(device)
    if lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                    patience=patience//2, verbose=True)
    
    
    history = dict(train=[], val=[], lr=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0
    print(f"开始训练: {n_epochs} epochs, 学习率: {lr}, 早停耐心度: {patience}")
    print("=" * 70)
  
    for epoch in range(1, n_epochs + 1):
        # =================== 训练阶段 ===================
        model = model.train()
        train_losses = []
        
        # 训练阶段时间控制
        last_update_time = 0
        
        for batch_idx, (batch_data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            # print(f"batch_data ------------- {batch_data.shape}")
            reconstructed = model(batch_data)
            # print(f"reconstructed ------------- {reconstructed.shape}")
 
            loss = criterion(reconstructed, batch_data)
            # 原地更新显示
            # 每秒更新一次显示
            current_time = time.time()
            if current_time - last_update_time >= 1.0:  # 1秒间隔
                print(f"\rEpoch {epoch} [{batch_idx+1}/{len(train_loader)}] Train Loss: {loss.item():.4f}", end='')
                last_update_time = current_time
                
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
 
        # 确保最后一个batch的损失也显示
        print(f"\rEpoch {epoch} [{len(train_loader)}/{len(train_loader)}] Train Loss: {train_losses[-1]:.4f}", end='', flush=True)
 
 
        # =================== 验证阶段 ===================
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for batch_idx, (batch_data, _) in enumerate(val_loader):
                batch_data = batch_data.to(device)
                reconstructed = model(batch_data)
 
                loss = criterion(reconstructed, batch_data)
                # 每秒更新一次验证损失显示
                current_time = time.time()
                if current_time - last_update_time >= 1.0:  # 1秒间隔
                    print(f"\rEpoch {epoch} - Validation [{batch_idx+1}/{len(val_loader)}] Val Loss: {loss.item():.4f}", end='', flush=True)
                    last_update_time = current_time
                val_losses.append(loss.item())
 
        # =================== 记录和调度 ===================
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['lr'].append(current_lr)
        
        # 学习率调度
        if lr_scheduler:
            scheduler.step(val_loss)
 
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improvement = " ⭐ NEW BEST!"
            
            # 保存检查点
            if save_checkpoints:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'history': history
                }, f'checkpoint_epoch_{epoch}.pth')
        else:
            patience_counter += 1
            improvement = ""

        print(f'\nEpoch {epoch}: train loss {train_loss} val loss {val_loss}')
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch} (patience: {patience})")
            break
        
        
    model.load_state_dict(best_model_wts)
    print(f"\n🎉 训练完成！最佳验证损失: {best_loss:.6f}")
    return model.eval(), history



class RecurrentAutoencoder(nn.Module):
    """
    定义一个自动编码器的子类，继承父类 nn.Module
    并且自动编码器通过编码器和解码器传递输入
    """
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def create_sequences(data, seq_length, n_features, slide_window_flag: int = 0):
    try:
        if slide_window_flag:
            sequences = []
            for i in range(len(data) - seq_length + 1):
                sequences.append(data[i:i + seq_length])
            return np.array(sequences)
        n_sequences = len(data) // seq_len
        n_samples_needed = n_sequences * seq_len 
        data = data[:n_samples_needed]
        data = data.reshape((n_sequences, seq_len, n_features))
        return data   
    except Exception as e:
        raise ValueError("Fail to exec create_sequences function!") from e


def time_series_split(data, train_ratio=0.8):
    """
    按时间顺序分割数据
    适用于时间序列异常检测
    """
    n_samples = len(data)
    split_idx = int(n_samples * train_ratio)
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data


def normalize_data(train_data, val_data, method='standard'):
    """
    对训练和验证数据进行归一化
    
    Args:
        train_data: 训练数据 (n_samples, seq_len, n_features)
        val_data: 验证数据 (n_samples, seq_len, n_features) 
        method: 归一化方法 ('standard', 'minmax', 'robust')
    
    Returns:
        normalized_train_data, normalized_val_data, scaler
    """
    # 获取原始形状
    train_shape = train_data.shape
    val_shape = val_data.shape
    
    # 重塑为2D: (n_samples * seq_len, n_features)
    train_reshaped = train_data.reshape(-1, train_shape[-1])
    val_reshaped = val_data.reshape(-1, val_shape[-1])
    
    # 选择归一化方法
    if method == 'standard':
        scaler = StandardScaler()  # 标准化: 均值0，标准差1
    elif method == 'minmax':
        scaler = MinMaxScaler()    # 最小-最大缩放: 范围[0,1]
    elif method == 'robust':
        scaler = RobustScaler()    # 鲁棒缩放: 使用中位数和四分位数
    else:
        raise ValueError("method must be 'standard', 'minmax', or 'robust'")
    
    # 在训练数据上拟合scaler，然后变换训练和验证数据
    train_normalized = scaler.fit_transform(train_reshaped)
    val_normalized = scaler.transform(val_reshaped)
    
    # 重塑回原始形状
    train_normalized = train_normalized.reshape(train_shape)
    val_normalized = val_normalized.reshape(val_shape)
    
    return train_normalized, val_normalized, scaler


if __name__ == '__main__':
    seq_len = 20
    n_features = 2
    batch_size = 128
    # =================== 1. 数据加载 ===================
    train_data = pd.read_csv("/work/ai/WHOAMI/device_info_13D2F34920008071211195A907_20250623_train_v1_200000_2dimension.csv")
    train_data = np.array(train_data)
    train_data = create_sequences(train_data, seq_len, n_features, slide_window_flag=1)
    print(train_data.shape)
    # =================== 1. 数据加载 ===================
    
    # =================== 2. 数据分割 ===================
    train_data, val_data = time_series_split(train_data)
    # =================== 2. 数据分割 ===================

    """
    # =================== 3. 归一化 ⭐ ===================
    # 归一化数据
    train_data, val_data, scaler = normalize_data(
        train_data, 
        val_data, 
        method='standard'  # 可选择: 'standard', 'minmax', 'robust'
    )
    import joblib
    # 🆕 添加这部分来保存scaler
    scaler_save_path = 'training_scaler.pkl'
    joblib.dump(scaler, scaler_save_path)
    print(f"✅ 归一化器已保存到: {scaler_save_path}")
    # =================== 3. 归一化 ⭐ ===================
    """
    
    
    # =================== 4. 转换为张量 ===================
    train_data = torch.tensor(train_data, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    # =================== 4. 转换为张量 ===================
    
    # =================== 5. 创建DataLoader ===================
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, batch_size=batch_size
    )
    # =================== 5. 创建DataLoader ===================
    
    
    model = RecurrentAutoencoder(
        seq_len=seq_len, 
        n_features=n_features, 
        embedding_dim=128
    )
    
    model = model.to(device)
    
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        n_epochs=150
    )
