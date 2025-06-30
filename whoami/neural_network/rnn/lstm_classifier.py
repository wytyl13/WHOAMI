#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/26 10:28
@Author  : weiyutao
@File    : lstm_classify.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class GRUClassifier(nn.Module):
    """
    - 轻量级GRU时序分类器
    - 单层GRU（替代双层LSTM）
    - embedding_dim=64（减少参数）
    - 简单分类头设计
    """
    def __init__(
        self, 
        seq_len, 
        n_features, 
        n_classes,
        embedding_dim=64,
        dropout=0.2
    ):
        super(GRUClassifier, self).__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        
        # 🔥 核心改进：单层GRU替代双层LSTM
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,  # 单层
            batch_first=True,
            dropout=0,  # 单层不需要dropout
            bidirectional=False  # 实时推理不用双向
        )
        
        # 🎯 分类头：Linear -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, n_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            output: (batch_size, n_classes)
        """
        # GRU特征提取
        gru_out, hidden = self.gru(x)  # gru_out: (batch, seq_len, embedding_dim)
        
        # 🔑 关键：使用最后一个时间步的输出
        last_output = gru_out[:, -1, :]  # (batch, embedding_dim)
        
        # 分类预测
        logits = self.classifier(last_output)  # (batch, n_classes)
        
        return logits
    
    def get_feature_embeddings(self, x):
        """获取特征嵌入（用于分析）"""
        with torch.no_grad():
            gru_out, _ = self.gru(x)
            return gru_out[:, -1, :].cpu().numpy()


def create_classification_sequences(data, labels, seq_length, slide_window_flag=1):
    """
    创建分类任务的序列数据
    
    Args:
        data: 原始数据 (n_samples, n_features)
        labels: 标签 (n_samples,)
        seq_length: 序列长度
        slide_window_flag: 是否使用滑动窗口
    
    Returns:
        sequences: (n_sequences, seq_length, n_features)
        sequence_labels: (n_sequences,)
    """
    data = np.array(data)
    labels = np.array(labels)
    
    if slide_window_flag:
        # 滑动窗口：每个窗口使用最后一个时间点的标签
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
            sequence_labels.append(labels[i + seq_length - 1])  # 使用窗口最后一个标签
            
        return np.array(sequences), np.array(sequence_labels)
    else:
        # 非滑动窗口：直接分割
        n_sequences = len(data) // seq_length
        n_samples_needed = n_sequences * seq_length
        
        data_reshaped = data[:n_samples_needed].reshape(n_sequences, seq_length, -1)
        # 对于标签，使用每个序列的最后一个标签
        labels_reshaped = labels[:n_samples_needed].reshape(n_sequences, seq_length)
        sequence_labels = labels_reshaped[:, -1]  # 每个序列的最后一个标签
        
        return data_reshaped, sequence_labels


def create_classification_dataloaders(
    train_data, train_labels, 
    val_data, val_labels, 
    batch_size=64
):
    """创建分类任务的DataLoader"""
    
    # 转换为tensor
    train_data = torch.FloatTensor(train_data)
    train_labels = torch.LongTensor(train_labels)
    val_data = torch.FloatTensor(val_data)
    val_labels = torch.LongTensor(val_labels)
    
    # 创建数据集
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader



def train_classifier(
    model, 
    train_loader, 
    val_loader, 
    n_epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    patience=15,
    save_checkpoints=True
):
    """
    训练分类器
    """
    # 🔧 优化器配置
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay  # 轻微权重衰减
    )
    
    # 🎯 分类损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 📉 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, 
        patience=patience//3, verbose=True
    )
    
    # 📊 记录训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    
    print(f"🚀 开始训练GRU分类器")
    print(f"📋 配置: epochs={n_epochs}, lr={lr}, patience={patience}")
    print("=" * 80)
    
    for epoch in range(1, n_epochs + 1):
        # =================== 训练阶段 ===================
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []
        
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录
            train_losses.append(loss.item())
            predictions = torch.argmax(logits, dim=1)
            train_predictions.extend(predictions.cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())
            
            # 进度显示
            if batch_idx % 50 == 0:
                print(f"\rEpoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}", end='')
        
        # =================== 验证阶段 ===================
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                logits = model(batch_data)
                loss = criterion(logits, batch_labels)
                
                val_losses.append(loss.item())
                predictions = torch.argmax(logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch_labels.cpu().numpy())
        
        # =================== 计算指标 ===================
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = accuracy_score(train_targets, train_predictions)
        val_acc = accuracy_score(val_targets, val_predictions)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 🏆 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improvement = " ⭐ NEW BEST!"
            
            if save_checkpoints:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'history': history
                }, f'best_gru_classifier_epoch_{epoch}.pth')
        else:
            patience_counter += 1
            improvement = ""
        
        # 打印进度
        print(f'\nEpoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} '
              f'train_acc={train_acc:.4f} val_acc={val_acc:.4f}{improvement}')
        
        # 🛑 早停
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch} (patience: {patience})")
            break
    
    # 加载最佳权重
    model.load_state_dict(best_model_wts)
    print(f"\n🎉 训练完成！最佳验证准确率: {best_acc:.4f}")
    
    return model.eval(), history


def evaluate_classifier(model, test_loader, class_names=None):
    """
    评估分类器性能
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_data)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    
    print(f"\n📊 测试集评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"\n详细分类报告:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    return accuracy, all_predictions, all_targets


def inference_single_sample(model, sample, scaler=None):
    """
    单样本推理
    
    Args:
        model: 训练好的模型
        sample: (seq_len, n_features) 单个序列
        scaler: 归一化器（可选）
    
    Returns:
        prediction: 预测类别
        confidence: 预测置信度
    """
    model.eval()
    try:
        # 预处理
        if scaler is not None:
            sample_reshaped = sample.reshape(-1, sample.shape[-1])
            sample_normalized = scaler.transform(sample_reshaped)
            sample = sample_normalized.reshape(sample.shape)
        
        # 转换为tensor并添加batch维度
        sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(sample_tensor)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probabilities).item()
            print(f"prediction: {prediction}")
            print(f"confidence: {confidence}")
    except Exception as e:
        raise ValueError("fail to exec lstm classifier model! {e}") from e
    return prediction, confidence



def write_data_to_file(sequences, labels, filename):
    """
    将序列数据和标签写入文件，格式为LSTM-FCN要求的格式
    
    参数:
    sequences: numpy array, 形状为 (n_samples, sequence_length) 或 (n_samples, sequence_length, n_features)
    labels: numpy array, 形状为 (n_samples,)，包含类别标签
    filename: str, 输出文件名
    """
    
    # 确保sequences是2D的
    if len(sequences.shape) == 3:
        # 如果是3D数组且最后一维是1，则压缩掉
        if sequences.shape[-1] == 1:
            sequences = sequences.squeeze(-1)
        else:
            # 如果有多个特征，需要展平或选择一个特征
            print(f"警告：检测到多特征数据 {sequences.shape}，将使用第一个特征")
            sequences = sequences[:, :, 0]
    
    # 打开文件进行写入
    with open(filename, 'w') as f:
        for i in range(len(sequences)):
            # 获取标签（确保是整数）
            label = int(labels[i])
            
            # 获取时间序列数据
            time_series = sequences[i]
            
            # 构建行：标签 + 时间序列值
            line_parts = [str(label)]
            line_parts.extend([str(value) for value in time_series])
            
            # 用空格连接并写入文件
            line = ' '.join(line_parts)
            f.write(line + '\n')
    
    print(f"已成功写入 {len(sequences)} 个样本到 {filename}")
    print(f"每个样本长度: {sequences.shape[1]}")
    print(f"标签范围: {np.min(labels)} - {np.max(labels)}")




# =================== 示例使用 ===================
if __name__ == '__main__':
    # 🔧 参数配置
    seq_len = 60
    n_features = 2
    n_classes = 3  # 假设3分类任务
    batch_size = 64
    
    print("🔥 轻量级GRU时序分类器 - 第一阶段实现")
    print(f"📋 配置: seq_len={seq_len}, n_features={n_features}, n_classes={n_classes}")
    
    # 🚀 创建模型
    model = GRUClassifier(
        seq_len=seq_len,
        n_features=n_features, 
        n_classes=n_classes,
        embedding_dim=64,  # 🎯 减少到64
        dropout=0.5
    ).to(device)
    
    # 📊 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 模型参数总数: {total_params:,}")
    print(f"📈 可训练参数: {trainable_params:,}")
    print(f"📏 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("\n✅ 模型创建完成！准备开始训练...")
    
    
    original_data = pd.read_csv("/work/ai/WHOAMI/device_info_13D2F34920008071211195A907_20250623_classifier_LABEL.csv", encoding='gbk')
    original_data = np.array(original_data)
    train_data = original_data[:, 1:-3]
    train_label = np.array([{'清醒': 0, '浅睡眠': 1, '深睡眠': 2}[label] for label in original_data[:, -1]])
    train_sequences, train_seq_labels = create_classification_sequences(
        train_data, train_label, seq_len, slide_window_flag=1
    )
    train_sequences = np.array(train_sequences, dtype=np.float32)
    train_labels = np.array(train_seq_labels, dtype=np.int64)
    print(len(original_data))
    print(len(train_sequences))
    print(len(train_seq_labels))
    print(train_sequences)
    print(train_seq_labels)
    
    # # 数据分割
    split_idx = int(len(train_sequences) * 0.8)
    val_sequences = train_sequences[split_idx:]
    val_seq_labels = train_seq_labels[split_idx:]
    train_sequences = train_sequences[:split_idx]
    train_seq_labels = train_seq_labels[:split_idx]
    
    
    # 写入训练数据
    write_data_to_file(train_sequences, train_seq_labels, 'train_data.txt')

    # 写入测试数据（使用验证集作为测试集）
    write_data_to_file(val_sequences, val_seq_labels, 'test.txt')
    
    
    # 创建DataLoader
    # train_loader, val_loader = create_classification_dataloaders(
    #     train_sequences, train_seq_labels,
    #     val_sequences, val_seq_labels,
    #     batch_size=batch_size
    # )
    
    # # 🚀 训练模型
    # model, history = train_classifier(
    #     model, train_loader, val_loader,
    #     n_epochs=50, lr=5e-4
    # )
    
    # # 📊 评估模型
    # accuracy, predictions, targets = evaluate_classifier(model, val_loader)
    
    
    # model = GRUClassifier(
    #     seq_len=seq_len,
    #     n_features=n_features, 
    #     n_classes=n_classes,
    #     embedding_dim=64,  # 🎯 减少到64
    #     dropout=0.2
    # ).to(device)
    # checkpoint = torch.load("/work/ai/WHOAMI/whoami/neural_network/best_gru_classifier_epoch_6_classifier_2dimensions.pth", map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # for i in range(5):
    #     print(train_sequences[i])
    #     print(type(train_sequences[i]))
    #     result, confidence = inference_single_sample(model, sample=train_sequences[i])
    #     print(result, confidence)