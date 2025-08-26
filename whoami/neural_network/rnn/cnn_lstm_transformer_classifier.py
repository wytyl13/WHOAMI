#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/07/29 09:48
@Author  : weiyutao
@File    : cnn_lstm_transformer_classifier.py

1、SimpleSleepDataset中初始化的step_size是5还是30
2、按照step_size作为滑动步伐，
"""


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import json
import pickle  # 用于保存scaler
import os  # 用于文件路径检查
import random
import matplotlib.animation as animation
from collections import deque
import threading
import time
import pickle  # 用于保存训练历史


# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class SimpleSleepDataset(Dataset):
    """简单的睡眠数据集 - 3个睡眠阶段"""
    
    def __init__(self, csv_file, window_size=60, step_size=30, max_samples=None, scaler_path=None):
        # 读取数据
        print("正在读取CSV文件...")
        self.df = pd.read_csv(csv_file, encoding='utf-8')
        
        # 如果指定了最大样本数，则只使用部分数据进行快速测试
        if max_samples and len(self.df) > max_samples:
            print(f"使用前 {max_samples} 行数据进行快速测试...")
            self.df = self.df.head(max_samples)
        
        self.window_size = window_size
        
        print(f"原始数据形状: {self.df.shape}")
        print(f"列名: {self.df.columns.tolist()}")
        
        # 处理中文标签 - 你的3个标签
        label_mapping = {
            '清醒': 0,    # Wake
            '浅睡眠': 1,  # Light Sleep 
            '深睡眠': 2,  # Deep Sleep
            # 如果有其他变体写法，也可以映射
            '清醒状态': 0,
            '浅度睡眠': 1,
            '深度睡眠': 2,
        }
        
        
        # 根据你的数据，label列包含中文
        if 'label' in self.df.columns:
            # 映射中文标签为数字
            self.df['label_num'] = self.df['label'].map(label_mapping)
            # 如果有未映射的标签，设为0（清醒）
            self.df['label_num'] = self.df['label_num'].fillna(0).astype(int)
            print(f"标签映射: {self.df['label'].value_counts()}")
        else:
            self.df['label_num'] = 0

        
        # 使用你的实际列名：breath_line, heart_line
        # 添加额外特征提升效果
        print("正在计算特征工程...")
        self.df['heart_rate'] = self.df['heart_line'].rolling(10, min_periods=1).std() * 100 + 70
        self.df['resp_rate'] = self.df['breath_line'].rolling(10, min_periods=1).std() * 50 + 15
        print("特征工程完成!")
        
        # 5个特征：呼吸率、心率、呼吸线、心线、信号质量
        # features = ['resp_rate', 'heart_rate', 'breath_line', 'heart_line', 'signal_intensity']
        # features = ['breath_bpm', 'heart_bpm', 'breath_line', 'heart_line', 'distance', 'signal_intensity', 'state']
        features = ['breath_bpm', 'heart_bpm', 'breath_line', 'heart_line', 'distance']
        # features = ['breath_line', 'heart_line']
        
        # 修复：处理缺失值 - 使用新的pandas语法
        print("处理缺失值...")
        self.df[features] = self.df[features].ffill().fillna(0)
        
        # 关键修复：标准化处理
        if scaler_path and os.path.exists(scaler_path):
            print(f"sleep_scaler.pkl: ------------------- {scaler_path}")
            # 预测时：加载训练时保存的scaler
            print(f"加载预训练的scaler: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            self.df[features] = scaler.transform(self.df[features])
            print("使用预训练scaler完成数据标准化!")
        else:
            # 训练时：创建新的scaler并保存
            print("创建新的scaler进行数据标准化...")
            scaler = StandardScaler()
            self.df[features] = scaler.fit_transform(self.df[features])
            # 保存scaler供预测时使用
            scaler_save_path = scaler_path if scaler_path else 'sleep_scaler.pkl'
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"数据标准化完成! Scaler已保存到: {scaler_save_path}")
        
        # 打印标准化后的数据统计
        print("标准化后的特征统计:")
        for feature in features:
            mean_val = self.df[feature].mean()
            std_val = self.df[feature].std()
            print(f"  {feature}: 均值={mean_val:.3f}, 标准差={std_val:.3f}")
        
        # 创建窗口 - 添加进度显示
        print("开始创建滑动窗口样本...")
        self.windows = []
        self.labels = []
        
        total_windows = (len(self.df) - window_size) // step_size + 1
        print(f"预计创建 {total_windows} 个窗口样本...")
        
        for i in range(0, len(self.df) - window_size + 1, step_size):
            window = self.df[features].iloc[i:i+window_size].values
            label = self.df['label_num'].iloc[i+step_size-1]
            
            self.windows.append(window)
            self.labels.append(label)
            
            # 每1000个窗口打印一次进度
            if len(self.windows) % 1000 == 0:
                progress = len(self.windows) / total_windows * 100
                print(f"进度: {len(self.windows)}/{total_windows} ({progress:.1f}%)")
        
        self.windows = np.array(self.windows)
        self.labels = np.array(self.labels)
        
        print(f"创建了 {len(self.windows)} 个样本")
        print(f"数据形状: {self.windows.shape}")
        print(f"标签分布: {np.bincount(self.labels)}")
        
        # 显示标签分布详情
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        label_names = ['清醒', '浅睡眠', '深睡眠']  # 3个标签
        for label, count in zip(unique_labels, counts):
            label_name = label_names[label] if label < len(label_names) else f'标签{label}'
            print(f"  {label_name}: {count}个样本 ({count/len(self.labels)*100:.1f}%)")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        # 修复：直接返回标量标签，避免维度不匹配
        return torch.FloatTensor(self.windows[idx]), torch.LongTensor([self.labels[idx]]).squeeze()

class SimpleSleepNet(nn.Module):
    """超简单的睡眠分期网络 - 3个睡眠阶段"""
    
    def __init__(
        self, 
        input_size=5, 
        seq_length=30, 
        num_classes=3, 
        bidirectional_flag=False, 
        lstm_hidden_size=64, 
        transformer_flag=False, 
        transformer_num_heads=2,
        dropout=0.3
    ):  # 3个类别：清醒、浅睡眠、深睡眠
        super().__init__()
        
        # 简单的CNN+LSTM
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 双向LSTM提升效果 - 修复dropout警告
        self.lstm = nn.LSTM(128, lstm_hidden_size, num_layers=2, batch_first=True, dropout=dropout, bidirectional=bidirectional_flag)
        classifier_input_size = lstm_hidden_size * 2 if bidirectional_flag else lstm_hidden_size
        self.transformer_flag = transformer_flag
        self.pos_encoder = PositionalEncoding(classifier_input_size, max_len=seq_length)
        self.transformer = nn.ModuleList([
            TransformerBlock(classifier_input_size, num_heads=transformer_num_heads, dropout=dropout)
        ])
        print(f"classifier_input_size: -------------------------------- {classifier_input_size}")
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size // 2),  # 双向LSTM输出128维
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size // 2, classifier_input_size // 4),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size // 4, num_classes)  # 3个输出
        )

        
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        # 更深的CNN提取特征
        # x dimension (batch_size, seq_length, feature_size) = (128, 60, 5)
        # kernel_size (32, 3, 5) = 480
        # output_dimension (128, 32, 60)
        # kernel_size (64, 3, 5) = 960
        # output_dimension (128, 64, 60)
        # kernel_size (128, 3, 5) = 1920
        # output_dimension (128, 128, 60)

        x = torch.relu(self.conv1(x)) #  dimension (batch_size, 32, 60) 
        x = torch.relu(self.conv2(x)) # dimension (128, 64, 60)
        x = torch.relu(self.conv3(x)) # dimension (128, 128, 60)
        x = x.transpose(1, 2) # (128, 60, 128)
        
        # num = 2 f bidirections else 1
        lstm_out, _ = self.lstm(x) # (128, 60, lstm_hidden_size * num)

        # transformer
        if self.transformer_flag:
            transformer_in = self.pos_encoder(lstm_out) # (128, 60, lstm_hidden_size * num)
            for layer in self.transformer:
                transformer_in = layer(transformer_in)
            # transformer_in dimension (128, 60, 128)
            transformer_out = torch.mean(transformer_in, dim=1) # dimension (128, 128)
            lstm_out = transformer_out
        else:
            lstm_out = torch.mean(lstm_out, dim=1)
            print(f"lstm_out.shape: --------------------------------- {lstm_out.shape}")
        # 分类
        output = self.classifier(lstm_out)
        return output

class PositionalEncoding(nn.Module):
    """位置编码 - 为Transformer提供序列位置信息"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [0, 1, 2, ...]
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # e^([0, 2, 4, 6, ...] * -ln(10000 / 128))
        # 1/e^(2i * ln(10000)/128))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        return x + self.pe[:x.size(1), :]


class TransformerBlock(nn.Module):
    """Transformer模块 - 包含自注意力和前馈网络"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # 自注意力
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class LSTMTransformerSleepNet(nn.Module):
    """LSTM+Transformer混合模型 - 用于睡眠分期"""
    
    def __init__(self, input_size=5, seq_length=60, num_classes=3, 
                 lstm_hidden_size=64, num_transformer_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        
        # 1. CNN部分 - 提取局部特征
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # 2. LSTM部分 - 处理时序动态
        self.lstm = nn.LSTM(128, lstm_hidden_size, num_layers=2, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        
        # 3. Transformer部分 - 捕获长距离依赖
        transformer_dim = lstm_hidden_size * 2  # 双向LSTM输出维度
        self.pos_encoder = PositionalEncoding(transformer_dim, max_len=seq_length)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(transformer_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(32, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features] -> [batch, features, seq_len]
        batch_size = x.size(0)
        x_cnn = x.transpose(1, 2)
        
        # 1. CNN特征提取
        cnn_out = torch.relu(self.conv1(x_cnn))
        cnn_out = torch.relu(self.conv2(cnn_out))
        cnn_out = torch.relu(self.conv3(cnn_out))  # [batch, 128, seq_len]
        
        # 2. LSTM处理
        # 调整维度以适应LSTM输入 [batch, seq_len, features]
        lstm_in = cnn_out.transpose(1, 2)  # [batch, seq_len, 128]
        lstm_out, _ = self.lstm(lstm_in)  # [batch, seq_len, 128]
        
        # 3. Transformer处理
        transformer_in = self.pos_encoder(lstm_out)  # 添加位置编码
        
        for layer in self.transformer_layers:
            transformer_in = layer(transformer_in)
        
        # 全局池化获取序列表示
        transformer_out = torch.mean(transformer_in, dim=1)  # [batch, 128]
        
        # 4. 分类
        output = self.classifier(transformer_out)
        
        return output



def train_simple_model(
    epochs=50, 
    max_samples=None, 
    patience=10, 
    min_delta=0.001, 
    param_name=None, 
    dataset=None,
    input_size=None,
    bidirectional_flag=None,
    transformer_flag=None
):
    import shutil
    """
    训练函数
    Args:
        csv_file: 数据文件路径
        epochs: 最大训练轮数
        max_samples: 最大样本数
        patience: 早停耐心值，连续多少个epoch验证损失不改善就停止
        min_delta: 认为是改善的最小变化量
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    print("开始训练简单睡眠分期模型...")
    
    # 创建保存目录
    save_dir = f"/work/ai/WHOAMI/train_data/train/{param_name}"
    # 如果目录存在，先清空它
    if os.path.exists(save_dir):
        print(f"目录 {save_dir} 已存在，正在清空...")
        shutil.rmtree(save_dir)
        print("目录已清空")
    os.makedirs(save_dir, exist_ok=True)
    print(f"模型和图片将保存到: {save_dir}")
    
    
    
    # 2. 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    # 修复：添加drop_last=True避免不完整的batch导致错误
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True, generator=torch.Generator().manual_seed(42))
    
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
    
    
    
    # 3. 创建模型 - 3个睡眠阶段
    model = SimpleSleepNet(input_size=input_size, seq_length=60, num_classes=3, bidirectional_flag=bidirectional_flag, lstm_hidden_size=64, transformer_flag=transformer_flag).to(device)
    
    # 处理类别不平衡 - 为少数类别加权
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / (class_counts + 1e-6)  # 避免除零
    class_weights = torch.FloatTensor(class_weights / class_weights.sum() * len(class_weights))
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 早停机制相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    print(f"早停设置: patience={patience}, min_delta={min_delta}")
    
    # 4. 设置绘图
    plt.ion()  # 开启交互模式
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('real time loss', fontsize=16)
    
    # 数据存储
    train_losses_history = []
    val_losses_history = []
    val_accs_history = []
    batch_train_losses = deque(maxlen=100)  # 只保留最近100个batch的损失
    epochs_completed = []
    
    # 子图设置
    ax1 = axes[0, 0]  # 每个epoch的损失
    ax2 = axes[0, 1]  # 验证准确率
    ax3 = axes[1, 0]  # 实时batch损失
    ax4 = axes[1, 1]  # 训练和验证损失对比
    
    def update_plots():
        """更新所有子图"""
        # 清除之前的图
        for ax in axes.flat:
            ax.clear()
        
        # 1. Epoch损失曲线
        if train_losses_history and val_losses_history:
            ax1.plot(range(1, len(train_losses_history) + 1), train_losses_history, 'b-', label='Train Loss', linewidth=2)
            ax1.plot(range(1, len(val_losses_history) + 1), val_losses_history, 'r-', label='Val Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Epoch Loss Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 验证准确率
        if val_accs_history:
            ax2.plot(range(1, len(val_accs_history) + 1), val_accs_history, 'g-', linewidth=2, marker='o', markersize=4)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Val Accuracy')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        # 3. 实时Batch损失（最近100个batch）
        if batch_train_losses:
            ax3.plot(list(batch_train_losses), 'orange', linewidth=1, alpha=0.7)
            ax3.set_xlabel('Recent Batches')
            ax3.set_ylabel('Batch Loss')
            ax3.set_title('Real Time Batch Loss (Recent 100)')
            ax3.grid(True, alpha=0.3)
        
        # 4. 训练与验证损失对比
        if train_losses_history and val_losses_history:
            epochs_range = range(1, len(train_losses_history) + 1)
            ax4.plot(epochs_range, train_losses_history, 'b-', label='Train', linewidth=2, marker='o', markersize=3)
            ax4.plot(epochs_range, val_losses_history, 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Train VS Val Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)  # 短暂暂停以更新显示
    
    def save_checkpoint_and_plot(epoch, model, optimizer, train_loss, val_loss, val_acc, is_best=False):
        """保存检查点和对应的训练图表"""
        # 保存模型检查点
        suffix = "_best" if is_best else f"_epoch_{epoch}"
        checkpoint_path = os.path.join(save_dir, f'checkpoint{suffix}.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }, checkpoint_path)
        
        # 保存对应的训练图表
        plot_path = os.path.join(save_dir, f'training_curves{suffix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f'检查点已保存: {checkpoint_path}')
        print(f'训练图表已保存: {plot_path}')
    
    # 5. 训练循环
    try:
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            batch_count = 0
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*50}")
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                if batch_y.dim() > 1:
                    batch_y = batch_y.squeeze()
                
                if batch_x.size(0) == 0 or batch_y.size(0) == 0:
                    continue
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                # batch_y dimension (128, 3) one-hot [0, 1] 
                # outputs dimension (128, 3) 
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                train_loss += current_loss
                batch_count += 1
                
                # 添加到batch损失历史
                batch_train_losses.append(current_loss)
                
                # 实时打印训练进度
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"\rEpoch {epoch+1} [{batch_idx+1:4d}/{len(train_loader):4d}] ({progress:5.1f}%) "
                      f"Batch Loss: {current_loss:.4f} | Avg Loss: {train_loss/(batch_idx+1):.4f}", end='', flush=True)
            
            avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
            train_losses_history.append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            val_batch_count = 0
            
            print(f"\n{'-'*30} 验证阶段 {'-'*30}")
            
            with torch.no_grad():
                for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    if batch_y.dim() > 1:
                        batch_y = batch_y.squeeze()
                    
                    if batch_x.size(0) == 0 or batch_y.size(0) == 0:
                        continue
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_batch_count += 1
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    
                    # 实时打印验证进度
                    val_progress = (batch_idx + 1) / len(val_loader) * 100
                    current_acc = val_correct / val_total if val_total > 0 else 0
                    print(f"\r验证进度 [{batch_idx+1:3d}/{len(val_loader):3d}] ({val_progress:5.1f}%) "
                          f"Val Loss: {loss.item():.4f} | Acc: {current_acc:.4f}", end='', flush=True)
            
            # 计算epoch统计
            val_acc = val_correct / val_total if val_total > 0 else 0
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
            
            val_losses_history.append(avg_val_loss)
            val_accs_history.append(val_acc)
            epochs_completed.append(epoch + 1)
            
            # 更新图表
            update_plots()
            
            # 早停检查
            is_best_model = False
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                is_best_model = True
                print(f"\n*** 发现更好的模型! 验证损失: {avg_val_loss:.6f} ***")
            else:
                patience_counter += 1
                print(f"\n早停计数器: {patience_counter}/{patience}")
            
            # Epoch总结
            print(f'\n{"="*60}')
            print(f'Epoch {epoch+1}/{epochs} 完成!')
            print(f'训练损失: {avg_train_loss:.6f}')
            print(f'验证损失: {avg_val_loss:.6f}')
            print(f'验证准确率: {val_acc:.6f} ({val_acc*100:.2f}%)')
            print(f'最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})')
            print(f'{"="*60}')
            
            # 保存检查点和图表（每10个epoch或最佳模型）
            if (epoch + 1) % 10 == 0 or is_best_model:
                save_checkpoint_and_plot(epoch + 1, model, optimizer, avg_train_loss, avg_val_loss, val_acc, is_best_model)
            
            # 早停检查
            if patience_counter >= patience:
                print(f"\n早停触发! 验证损失连续 {patience} 个epoch没有改善")
                print(f"最佳模型在第 {best_epoch} epoch, 验证损失: {best_val_loss:.6f}")
                break
    
    except KeyboardInterrupt:
        print("\n\n训练被用户中断!")
        save_interrupted = input("是否保存当前模型? (y/n): ")
        if save_interrupted.lower() == 'y':
            interrupted_path = os.path.join(save_dir, 'interrupted_model.pth')
            torch.save(model.state_dict(), interrupted_path)
            print(f"模型已保存为: {interrupted_path}")
    
    finally:
        # 6. 保存最终模型和图表
        # 如果有最佳模型，使用最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n使用最佳模型状态 (Epoch {best_epoch})")
        
        final_model_path = os.path.join(save_dir, 'transformer_sleep_model_final.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"\n最终模型已保存为: {final_model_path}")
        
        # 保存最终的训练曲线图
        final_plot_path = os.path.join(save_dir, 'final_training_curves.png')
        plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        print(f"最终训练曲线图已保存为: {final_plot_path}")
        
        # 保存训练历史数据
        training_history = {
            'train_losses': train_losses_history,
            'val_losses': val_losses_history,
            'val_accuracies': val_accs_history,
            'epochs': epochs_completed,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'early_stopped': patience_counter >= patience
        }
        
        history_path = os.path.join(save_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(training_history, f)
        print(f"训练历史已保存为: {history_path}")
        
        # 显示最终统计
        if train_losses_history:
            print(f"\n{'='*60}")
            print("训练完成统计:")
            print(f"总训练轮数: {len(train_losses_history)}")
            print(f"最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")
            print(f"最佳验证准确率: {max(val_accs_history):.6f} (Epoch {val_accs_history.index(max(val_accs_history))+1})")
            print(f"最终训练损失: {train_losses_history[-1]:.6f}")
            print(f"最终验证损失: {val_losses_history[-1]:.6f}")
            print(f"最终验证准确率: {val_accs_history[-1]:.6f}")
            if patience_counter >= patience:
                print(f"早停触发: 是")
            else:
                print(f"早停触发: 否")
            print(f"所有文件保存位置: {save_dir}")
            print(f"{'='*60}")
        
        plt.ioff()  # 关闭交互模式
        plt.show()  # 保持图表显示
    
    return model


def predict_realtime(model_path, csv_file, max_predict_samples=None, print_interval=1):
    """实时预测函数 - 3个睡眠阶段
    
    Args:
        model_path: 模型文件路径
        csv_file: 数据文件路径
        max_predict_samples: 最大预测样本数，None表示预测所有
        print_interval: 打印间隔，1表示每秒打印，10表示每10秒打印
    """
    
    # 加载模型
    model = LSTMTransformerSleepNet(input_size=5, num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 你的3个标签对应关系
    stage_names = ['清醒', '浅睡眠', '深睡眠']
    
    # 实时预测示例 - 使用训练时保存的scaler
    dataset = SimpleSleepDataset(csv_file, window_size=60, step_size=1, max_samples=1000, scaler_path='sleep_scaler.pkl')
    
    predictions = []
    confidences = []
    
    # 确定预测数量
    if max_predict_samples is None:
        predict_count = len(dataset)
    else:
        predict_count = min(max_predict_samples, len(dataset))
    
    print(f"开始预测，总共{predict_count}个样本，每{print_interval}秒打印一次结果")
    print("-" * 60)
    
    with torch.no_grad():
        for i in range(predict_count):
            sample, _ = dataset[i]
            sample = sample.unsqueeze(0)  # 添加batch维度
            
            output = model(sample)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, predicted_class].item()
            
            predictions.append({
                'second': i,
                'stage': stage_names[predicted_class],
                'stage_code': predicted_class,
                'confidence': confidence,
                'timestamp': f"第{i}秒"
            })
            
            confidences.append(confidence)
            
            # 根据设定间隔打印结果
            if i % print_interval == 0:
                print(f"第{i}秒: {stage_names[predicted_class]} (置信度: {confidence:.3f})")
            
            # 每100个样本显示一次进度
            if (i + 1) % 100 == 0:
                progress = (i + 1) / predict_count * 100
                print(f"进度: {i+1}/{predict_count} ({progress:.1f}%)")
    
    # 统计预测结果
    stage_counts = {}
    for pred in predictions:
        stage = pred['stage']
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    print(f"\n预测总结:")
    print(f"预测样本数: {len(predictions)}")
    print(f"平均置信度: {np.mean(confidences):.3f}")
    print(f"睡眠阶段分布:")
    for stage, count in stage_counts.items():
        percentage = count/len(predictions)*100
        print(f"  {stage}: {count}次 ({percentage:.1f}%)")
    
    return predictions


def inference_single_sample_(sample_data, model, scaler, use_raw_features):
    """
    预测单个样本的睡眠阶段 - 支持原始特征输入
    
    Args:
        sample_data: numpy array
                    - 如果 use_raw_features=True: [30, 3] (breath_line, heart_line, signal_intensity)
                    - 如果 use_raw_features=False: [30, 5] (已处理的5个特征)
        model: 已加载的模型实例 (SimpleSleepNet)
        scaler: 已加载的标准化器实例 (StandardScaler)
        use_raw_features: 是否使用原始3个特征输入
    
    Returns:
        dict: 预测结果
    """
    
    try:
        # 1. 输入验证和特征工程
        if not isinstance(sample_data, np.ndarray):
            raise TypeError(f"输入数据必须是numpy数组，实际类型: {type(sample_data)}")
        
        if use_raw_features:
            # 原始3个特征输入 [30, 3]
            if sample_data.shape != (30, 3):
                raise ValueError(f"原始特征输入形状必须为 (30, 3)，实际为 {sample_data.shape}")
            
            # 进行特征工程
            processed_data = _perform_feature_engineering(sample_data)
        else:
            # 已处理的5个特征输入 [30, 5]
            if sample_data.shape != (30, 5):
                raise ValueError(f"处理后特征输入形状必须为 (30, 5)，实际为 {sample_data.shape}")
            processed_data = sample_data
        
        # 2. 数据预处理（标准化）
        normalized_data = scaler.transform(processed_data)
        tensor_data = torch.FloatTensor(normalized_data).unsqueeze(0).to("cuda:0")  # [1, 30, 5]
        
        # 3. 模型预测
        model.eval()
        with torch.no_grad():
            output = model(tensor_data)  # [1, 3]
            probabilities = torch.softmax(output, dim=1)  # [1, 3]
            predicted_class = torch.argmax(output, dim=1).item()  # 0, 1, or 2
            confidence = probabilities[0, predicted_class].item()  # 置信度
        
        
        return predicted_class, confidence
        
    except Exception as e:
        raise ValueError(f"fail to exec deep learning model! {str(e)}") from e


def load_sleep_scaler(scaler_path='sleep_scaler.pkl'):
    """
    加载标准化器
    
    Args:
        scaler_path: 标准化器文件路径
    
    Returns:
        StandardScaler: 已加载的标准化器实例
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"找不到标准化器文件: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ 标准化器已从 {scaler_path} 加载")
    return scaler


def _perform_feature_engineering(raw_data):
    """
    对原始3个特征进行特征工程，生成5个特征
    
    Args:
        raw_data: numpy array [30, 3] - (breath_line, heart_line, signal_intensity)
    
    Returns:
        numpy array [30, 5] - 特征工程后的5个特征
    """
    import pandas as pd
    
    # 转换为DataFrame方便处理
    df = pd.DataFrame(raw_data, columns=['breath_line', 'heart_line', 'signal_intensity'])
    
    # 特征工程 (与训练时保持一致)
    df['heart_rate'] = df['heart_line'].rolling(10, min_periods=1).std() * 100 + 70
    df['resp_rate'] = df['breath_line'].rolling(10, min_periods=1).std() * 50 + 15
    df['signal_quality'] = df['signal_intensity'] / 50.0
    
    # 5个特征的顺序（与训练时一致）
    features = ['resp_rate', 'heart_rate', 'breath_line', 'heart_line', 'signal_quality']
    
    # 处理缺失值
    df[features] = df[features].ffill().fillna(0)
    
    return df[features].values


def set_seed(seed=42):
    """固定所有可能的随机种子，确保实验可复现"""
    # Python内置随机库
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU（单卡）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多卡环境
    # CuDNN（确保卷积操作确定性）
    torch.backends.cudnn.deterministic = True  # 固定卷积算法
    torch.backends.cudnn.benchmark = False     # 禁用自动选择最快算法（可能引入随机性）


def process_data(label_studio_json_file: str):
    def read_json_file(file_path):
        """读取JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    json_data = read_json_file(label_studio_json_file)
    return json_data


def parse_datetime(date_str):
    """解析日期时间字符串"""
    try:
        return pd.to_datetime(date_str, format='%Y/%m/%d %H:%M:%S')
    except:
        try:
            return pd.to_datetime(date_str)
        except:
            return None


def process_csv_with_time_labels(config_data, base_path="/work/ai/WHOAMI/train_data/vital_sleep_classifier", output_path="merged_time_labeled_data.csv"):
    """
    修复版本：根据时间区间给CSV数据分配标签
    """
    
    def parse_datetime(date_str):
        """解析日期时间字符串"""
        try:
            return pd.to_datetime(date_str, format='%Y/%m/%d %H:%M:%S')
        except:
            try:
                return pd.to_datetime(date_str)
            except:
                return None
    
    all_processed_data = []
    
    # 直接处理每个配置项
    for item in config_data:
        csv_path = item['csv']
        csv_filename = csv_path.split('-')[-1]  # 提取文件名
        full_path = os.path.join(base_path, csv_filename)
        
        print(f"\n📁 处理文件: {csv_filename}")
        
        if not os.path.exists(full_path):
            print(f"✗ 文件不存在: {full_path}")
            continue
        # 读取CSV文件
        df = pd.read_csv(full_path)
        print(f"  📊 文件行数: {len(df)}")

        df = df.dropna()  # 删除NaN行
        df = df[~((df['breath_line'] == 0) & (df['heart_line'] == 0))]  # 删除同时为0的行
        
        # 获取时间列（假设是第一列）
        time_column = df.columns[0]
        print(f"  📅 时间列: {time_column}")
        
        # 转换时间列
        df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        
        # 添加新列 - 分别处理，避免混淆
        df_copy = df.copy()
        df_copy['label'] = 'unlabeled'  # 默认标签
        df_copy['source_file'] = csv_filename  # 文件来源
        
        print(f"  🔧 初始化后的列数: {len(df_copy.columns)}")
        
        # 处理标签区间
        label_data = item.get('label', [])
        if not isinstance(label_data, list):
            label_data = [label_data]
        
        for idx, label_info in enumerate(label_data):
            start_time = parse_datetime(label_info['start'])
            end_time = parse_datetime(label_info['end'])
            labels = label_info.get('timeserieslabels', [])
            
            if not labels:
                continue
                
            label_text = str(labels[0]).strip()  # 确保是干净的字符串
            
            print(f"  🏷️  区间 {idx+1}: {label_info['start']} - {label_info['end']}")
            print(f"      标签: '{label_text}'")
            
            if start_time is None or end_time is None:
                print(f"      ⚠️ 时间解析失败")
                continue
            
            # 找到时间范围内的数据
            time_mask = (df_copy[time_column] >= start_time) & (df_copy[time_column] <= end_time)
            matched_count = time_mask.sum()
            
            if matched_count > 0:
                # 重要：确保只设置label列，不影响其他列
                df_copy.loc[time_mask, 'label'] = label_text
                print(f"      ✅ 匹配 {matched_count} 行数据")
            else:
                print(f"      ⚠️ 未找到匹配的时间数据")
        
        # 验证结果
        print(f"  🔍 标签统计: {df_copy['label'].value_counts().to_dict()}")
        print(f"  📂 文件来源: {df_copy['source_file'].iloc[0] if len(df_copy) > 0 else 'None'}")
        df_copy = df_copy[df_copy['label'] != "异常"]
        all_processed_data.append(df_copy)
    # 合并所有数据
    if all_processed_data:
        print(f"\n🔄 合并 {len(all_processed_data)} 个文件...")
        final_df = pd.concat(all_processed_data, ignore_index=True)
        
        # 最终验证
        print(f"📊 合并后总行数: {len(final_df)}")
        print(f"📋 最终标签分布:")
        for label, count in final_df['label'].value_counts().items():
            print(f"  '{label}': {count} 行")
        
        print(f"📂 文件来源分布:")
        for source, count in final_df['source_file'].value_counts().items():
            print(f"  '{source}': {count} 行")
        
        # 保存文件
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✅ 保存完成: {output_path}")
        
        return final_df
    else:
        print("❌ 没有处理任何数据")
        return None



def verify_label(csv_file, config_data, sample_count):
    from datetime import datetime
    def parse_flexible_time(time_str):
        """灵活解析不同格式的时间字符串"""
        formats = [
            "%Y-%m-%d %H:%M:%S",  # 2025-06-23 06:51:23
            "%Y/%m/%d %H:%M:%S",  # 2025/06/23 06:51:23
            "%Y-%m-%d %H:%M",     # 2025-06-23 06:51
            "%Y/%m/%d %H:%M"      # 2025/06/23 06:51
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"无法解析时间格式: {time_str}")
    result = []
    array_data = np.array(pd.read_csv(csv_file, encoding='utf-8'))
    sample_data = array_data[np.random.choice(a=len(array_data), size=sample_count, replace=False)]
    verify_data = sample_data[:, [0, -2, -1]]
    for item in verify_data:
        for item_config in config_data:
            if item_config["csv"].split("-")[-1] == item[-1]:
                for label_item in item_config["label"]:
                    target_time = parse_flexible_time(item[0])
                    start_time = parse_datetime(label_item["start"])
                    end_time = parse_datetime(label_item["end"])
                    if start_time <= target_time <= end_time:
                        if item[1] == label_item["timeserieslabels"][0]:
                            result.append(1)
                        else:
                            result.append(0)
                break
    print(sum(result))
    return sum(result) / len(result) if len(result) > 0 else 0, array_data


# 使用示例 - 适配你的数据
if __name__ == "__main__":

    # 读取数据
    json_data = process_data("/work/ai/WHOAMI/train_data/eee/project-4-at-2025-07-28-07-16-c32546d7.json")
    print(json_data)
    output_path="/work/ai/WHOAMI/train_data/vital_sleep_classifier/out.csv"
    process_csv_with_time_labels(config_data=json_data, output_path=output_path)

    # 验证数据标注准确率
    precision, array_data = verify_label(csv_file=output_path, config_data=json_data, sample_count=1000)
    print(precision)
    
    print(len(array_data))
    
    set_seed(42)
    # 你的CSV文件路径
    csv_file = output_path
    
    print("🚀 开始训练3类睡眠分期模型...")
    print("="*50)
    
    # # 快速测试选项 - 如果数据太大，可以先用小样本测试
    USE_QUICK_TEST = False  # 改为False使用全部数据，True使用部分数据测试
    max_samples = 50000 if USE_QUICK_TEST else None  # 快速测试使用5万行数据
    
    if USE_QUICK_TEST:
        print("⚡ 快速测试模式 - 使用部分数据")
    else:
        print("🐌 完整训练模式 - 使用全部数据")
    
    # 1. 加载数据
    dataset = SimpleSleepDataset(csv_file, window_size=60, step_size=30, max_samples=max_samples, scaler_path='sleep_scaler.pkl')
    
    train_param_json = [
        {"name": "7_bi_transformer", "input_size": 5, "bidirectional_flag": True, "transformer_flag": True},
        {"name": "7_bi", "input_size": 5, "bidirectional_flag": True, "transformer_flag": False},
        {"name": "7_transformer", "input_size": 5, "bidirectional_flag": False, "transformer_flag": True},
        {"name": "7", "input_size": 5, "bidirectional_flag": False, "transformer_flag": False},
    ]
    
    for item in train_param_json:
        try:
            # 训练模型
            model = train_simple_model(
                dataset=dataset, 
                epochs=150, 
                max_samples=max_samples, 
                param_name=item["name"],
                input_size=item["input_size"],
                bidirectional_flag=item["bidirectional_flag"],
                transformer_flag=item["transformer_flag"],
            )
            print("✅ 训练完成！")
            
            print("\n🔮 开始实时预测...")
            print("="*50)
            
            # 预测选项
            print("选择预测模式:")
            print("1. 每秒预测 - 预测100秒")
            print("2. 每10秒预测 - 预测1000秒") 
            print("3. 快速预测 - 预测所有可能的样本")
            
            # 不同预测模式
            # 模式1: 每秒显示，预测100秒
            # print("\n📊 模式1: 每秒预测结果")
            # predictions_1s = predict_realtime('transformer_sleep_model.pth', csv_file, 
            #                                 max_predict_samples=100, print_interval=1)
            
            # print("\n📊 模式2: 每10秒预测结果")  
            # predictions_10s = predict_realtime('transformer_sleep_model.pth', csv_file,
            #                                 max_predict_samples=1000, print_interval=60)
            
            # print(f"✅ 预测完成")
            
            # # 保存预测结果
            # with open('transformer_sleep_predictions_1s.json', 'w', encoding='utf-8') as f:
            #     json.dump(predictions_1s, f, ensure_ascii=False, indent=2)
            
            # with open('transformer_sleep_predictions_10s.json', 'w', encoding='utf-8') as f:
            #     json.dump(predictions_10s, f, ensure_ascii=False, indent=2)
                
            # print("📄 预测结果已保存到: sleep_predictions_1s.json 和 sleep_predictions_10s.json")
            
        except FileNotFoundError:
            print("❌ 找不到CSV文件，请检查文件路径！")
            print("当前查找文件:", csv_file)
            print("请将你的CSV文件放在脚本同目录下，或修改csv_file变量")

        except Exception as e:
            print(f"❌ 出错了: {e}")
            import traceback
            traceback.print_exc()
            print("请检查数据格式是否正确")
            print("确保CSV包含列: create_time, breath_line, heart_line, distance, signal_intensity, label")


    """
    # predict
    import numpy as np

    # 生成30秒 × 3个特征的模拟传感器数据
    sample_data = np.random.randn(30, 3)

    # 或者生成更真实的传感器数据范围
    sample_data = np.array([
        np.random.normal(0, 1, 30),      # breath_line: 呼吸信号 (均值0, 标准差1)
        np.random.normal(0, 1, 30),      # heart_line: 心电信号 (均值0, 标准差1)  
        np.random.uniform(20, 80, 30)    # signal_intensity: 信号强度 (20-80范围)
    ]).T  # 转置为 (30, 3)
    scaler = load_sleep_scaler(scaler_path="/work/ai/WHOAMI/sleep_scaler.pkl")
    model = SimpleSleepNet(
        input_size=5,
        seq_length=30, 
        num_classes=3,
    )
    
    # checkpoint = torch.load(self.model_path, map_location=self.device)
    model.load_state_dict(torch.load("/work/ai/WHOAMI/simple_sleep_model.pth"))
    print(sample_data)
    result, confidence = inference_single_sample_(sample_data=sample_data, model=model, scaler=scaler, use_raw_features=True)
    print(result, confidence)
    """