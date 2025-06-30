#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/20 15:47
@Author  : weiyutao
@File    : predict.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

from auto_encoders import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_anomalies(reconstruction_errors, threshold_method='percentile', threshold_value=95):
    """
    基于重构误差检测异常
    
    Args:
        reconstruction_errors: 重构误差数组
        threshold_method: 阈值方法 ('percentile', 'std', 'fixed')
        threshold_value: 阈值参数
    
    Returns:
        anomalies: 异常标记 (bool array)
        threshold: 使用的阈值
    """
    if threshold_method == 'percentile':
        threshold = np.percentile(reconstruction_errors, threshold_value)
    elif threshold_method == 'std':
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        threshold = mean_error + threshold_value * std_error
    elif threshold_method == 'fixed':
        threshold = threshold_value
    else:
        raise ValueError("threshold_method must be 'percentile', 'std', or 'fixed'")
    
    anomalies = reconstruction_errors > threshold
    
    return anomalies, threshold


def visualize_results(reconstruction_errors, anomalies, threshold, raw_data=None, feature_names=None):
    """
    可视化异常检测结果
    
    Args:
        reconstruction_errors: 重构误差
        anomalies: 异常标记
        threshold: 阈值
        raw_data: 原始数据（可选）
        feature_names: 特征名称（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 重构误差时间序列
    axes[0, 0].plot(reconstruction_errors, alpha=0.7, label='重构误差')
    axes[0, 0].axhline(y=threshold, color='r', linestyle='--', label=f'阈值: {threshold:.4f}')
    axes[0, 0].scatter(np.where(anomalies)[0], reconstruction_errors[anomalies], 
                      color='red', s=20, label=f'异常点 ({np.sum(anomalies)}个)')
    axes[0, 0].set_title('重构误差时间序列')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('重构误差')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 重构误差分布
    axes[0, 1].hist(reconstruction_errors, bins=50, alpha=0.7, color='skyblue', density=True)
    axes[0, 1].axvline(x=threshold, color='r', linestyle='--', label=f'阈值: {threshold:.4f}')
    axes[0, 1].set_title('重构误差分布')
    axes[0, 1].set_xlabel('重构误差')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 异常点在时间轴上的分布
    anomaly_indices = np.where(anomalies)[0]
    axes[1, 0].scatter(anomaly_indices, np.ones_like(anomaly_indices), 
                      color='red', alpha=0.6, s=30)
    axes[1, 0].set_title(f'异常点时间分布 (总计: {len(anomaly_indices)}个)')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_yticks([])
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计信息
    stats_text = f"""
    数据统计:
    • 总序列数: {len(reconstruction_errors)}
    • 异常序列数: {np.sum(anomalies)}
    • 异常率: {np.sum(anomalies)/len(reconstruction_errors)*100:.2f}%
    
    误差统计:
    • 平均误差: {np.mean(reconstruction_errors):.4f}
    • 误差标准差: {np.std(reconstruction_errors):.4f}
    • 最大误差: {np.max(reconstruction_errors):.4f}
    • 最小误差: {np.min(reconstruction_errors):.4f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def save_results(reconstruction_errors, anomalies, threshold, output_path='anomaly_results.csv'):
    """
    保存异常检测结果
    
    Args:
        reconstruction_errors: 重构误差
        anomalies: 异常标记
        threshold: 阈值
        output_path: 输出文件路径
    """
    results_df = pd.DataFrame({
        'sequence_id': range(len(reconstruction_errors)),
        'reconstruction_error': reconstruction_errors,
        'is_anomaly': anomalies,
        'threshold': threshold
    })
    
    results_df.to_csv(output_path, index=False)
    print(f"✅ 结果已保存到: {output_path}")
    
    return results_df


def main_inference_pipeline(
    model_path,
    data_path,
    scaler_path=None,
    seq_len=60,
    n_features=6,
    batch_size=32,
    threshold_method='std',
    threshold_value=2,
    output_path='anomaly_results.csv',
    visualize=True
):
    """
    完整的推理流水线
    
    Args:
        model_path: 模型文件路径
        data_path: 测试数据路径
        scaler_path: 归一化器路径（可选）
        seq_len: 序列长度
        n_features: 特征数量
        batch_size: 批次大小
        threshold_method: 阈值方法
        threshold_value: 阈值参数
        output_path: 输出文件路径
        visualize: 是否可视化
    
    Returns:
        results_df: 结果DataFrame
    """
    print("🚀 开始异常检测推理流水线...")
    print("=" * 50)
    
    # 1. 加载模型和归一化器
    print("📥 加载模型...")
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    # 2. 加载测试数据
    print("📥 加载测试数据...")
    if data_path.endswith('.csv'):
        test_data = pd.read_csv(data_path)
    else:
        test_data = np.load(data_path)
    print(f"测试数据形状: {test_data.shape}")
    
    # 3. 数据预处理
    print("🔧 数据预处理...")
    tensor_data, scaler, original_shape = preprocess_inference_data(
        test_data, seq_len, n_features, scaler, slide_window_flag=1
    )
    
    # 4. 创建DataLoader
    dataset = TensorDataset(tensor_data, tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 5. 批次推理
    print("🔮 执行推理...")
    reconstructed_data, reconstruction_errors, raw_errors = batch_inference(
        model, dataloader, device, return_errors=True
    )
    
    # 6. 异常检测
    print("🔍 检测异常...")
    anomalies, threshold = detect_anomalies(
        reconstruction_errors, threshold_method, threshold_value
    )
    
    print(f"检测到 {np.sum(anomalies)} 个异常序列 (异常率: {np.sum(anomalies)/len(anomalies)*100:.2f}%)")
    
    # 7. 可视化结果
    if visualize:
        print("📊 生成可视化...")
        visualize_results(reconstruction_errors, anomalies, threshold)
    
    # 8. 保存结果
    print("💾 保存结果...")
    results_df = save_results(reconstruction_errors, anomalies, threshold, output_path)
    
    print("✅ 推理完成!")
    return results_df


def create_training_scaler():
    """
    重新创建训练时使用的归一化器
    使用与训练时完全相同的数据和步骤
    """
    print("🔧 创建训练时的归一化器...")
    
    # 1. 加载训练数据（与训练时相同）
    train_data = pd.read_csv("/work/soft/LSTM-Autoencoders/kdd_data/device_info_20250616_nomaly.csv")
    train_data = np.array(train_data)
    
    # 2. 创建序列（与训练时相同）
    seq_len = 60
    n_features = 7
    slide_window_flag = 1
    
    if slide_window_flag:
        sequences = []
        for i in range(len(train_data) - seq_len + 1):
            sequences.append(train_data[i:i + seq_len])
        train_data = np.array(sequences)
    
    # 3. 分割数据（与训练时相同）
    train_ratio = 0.8
    n_samples = len(train_data)
    split_idx = int(n_samples * train_ratio)
    train_data = train_data[:split_idx]  # 只要训练部分
    
    # 4. 创建归一化器（与训练时相同）
    train_shape = train_data.shape
    train_reshaped = train_data.reshape(-1, train_shape[-1])  # (n_samples * seq_len, n_features)
    
    scaler = StandardScaler()  # 与训练时相同的方法
    scaler.fit(train_reshaped)
    
    print(f"✅ 归一化器创建成功!")
    print(f"   特征数量: {scaler.n_features_in_}")
    print(f"   特征均值: {scaler.mean_}")
    print(f"   特征标准差: {scaler.scale_}")
    
    return scaler


def load_model_and_scaler(model_path, scaler_path=None):
    """
    加载训练好的模型和归一化器
    
    Args:
        model_path: 模型文件路径
        scaler_path: 归一化器文件路径（如果有的话）
    
    Returns:
        model, scaler
    """
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 重建模型（需要与训练时参数一致）
    seq_len = 60  # 根据你的训练参数调整
    n_features = 6
    embedding_dim = 128
    
    model = RecurrentAutoencoder(seq_len, n_features, embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功! 最佳损失: {checkpoint['best_loss']:.6f}")
    
    # 如果有单独的scaler文件，加载它
    scaler = None
    if scaler_path:
        import joblib
        scaler = joblib.load(scaler_path)
        print("✅ 归一化器加载成功!")
    
    return model, scaler

# 使用示例
if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "/work/ai/WHOAMI/whoami/neural_network/rnn/checkpoint_epoch_147.pth"  # 替换为你的模型路径
    # DATA_PATH = "/work/soft/LSTM-Autoencoders/kdd_data/device_info_20250616_nomaly.csv"        # 替换为你的测试数据路径
    DATA_PATH = "/work/ai/WHOAMI/whoami/neural_network/rnn/device_info_20250616.csv"        # 替换为你的测试数据路径
    SCALER_PATH = None                      # 如果有单独的scaler文件
    
    """
    如果预测数据很少且预测数据和训练数据分布基本一致，可以使用训练数据的scaler去进行归一化
    因为训练数据和预测数据的统计特征基本相似，但是如果分布不同需要使用预测数据的scaler
    # 第一步：创建训练时的归一化器
    training_scaler = create_training_scaler()
    
    # 第二步：保存归一化器（可选）
    import joblib
    joblib.dump(training_scaler, 'training_scaler.pkl')
    print("📁 归一化器已保存到: training_scaler.pkl")
    """
    
    
    # 执行推理
    # scaler_path=None即可在初始化数据的时候使用预测数据的统计特征去创建归一化scaler
    results = main_inference_pipeline(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        scaler_path='/work/ai/WHOAMI/whoami/neural_network/rnn/training_scaler.pkl',
        seq_len=60,
        n_features=6,
        batch_size=32,
        threshold_method='fixed',  # 'percentile', 'std', 'fixed'
        threshold_value=0.5,             # 对于percentile是百分位数，对于std是标准差倍数
        output_path='anomaly_results.csv',
        visualize=True
    )
    
    # 查看结果摘要
    print("\n📋 结果摘要:")
    print(results.describe())
    
    # 查看异常样本
    anomaly_samples = results[results['is_anomaly'] == True]
    print(f"\n🚨 异常样本 (前10个):")
    print(anomaly_samples.head(10))