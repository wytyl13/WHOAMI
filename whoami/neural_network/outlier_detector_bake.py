import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import time

# 定义固定的列名
feature_columns = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']

# 通用参数配置 - 适合各种数据分布
UNIVERSAL_CONFIG = {
    'contamination': 0.02,     # 固定2% - 保守估计，适合大多数场景
    'n_neighbors': 50,         # 增加邻居数，提高稳定性
    'algorithm': 'auto',       # 自动选择最优算法
    'leaf_size': 30,           # 默认值，平衡性能和内存
    'metric': 'minkowski',     # 标准欧几里得距离
    'p': 2,                    # 欧几里得距离参数
    'n_jobs': -1              # 使用所有CPU核心加速
}

print(f"使用通用参数配置:")
for key, value in UNIVERSAL_CONFIG.items():
    print(f"  {key}: {value}")

# 读取10万条数据
full_data = pd.read_csv('/work/soft/LSTM-Autoencoders/kdd_data/device_info_20250616_nomaly_5dimension.csv')
print(f"原始数据: {full_data.shape}")

# 分层采样
def stratified_time_sampling(data, n_samples=10000):
    data['time_group'] = pd.cut(range(len(data)), bins=10, labels=False)
    sampled_list = []
    for group in range(10):
        group_data = data[data['time_group'] == group]
        if len(group_data) >= 1000:
            sampled_list.append(group_data.sample(n=1000, random_state=42))
        else:
            sampled_list.append(group_data)
    result = pd.concat(sampled_list).drop('time_group', axis=1)
    return result.sample(frac=1).reset_index(drop=True)

# 准备训练数据
training_data = full_data.iloc[:, :5].copy()
training_data.columns = feature_columns

# 使用分层采样
sampled_data = stratified_time_sampling(training_data)
print(f"采样后数据: {sampled_data.shape}")

# 添加数据标准化
scaler = StandardScaler()
sampled_data_scaled = scaler.fit_transform(sampled_data)

# 训练检测器 - 使用通用参数
detector = LocalOutlierFactor(**UNIVERSAL_CONFIG, novelty=True)
detector.fit(sampled_data_scaled)
print("训练完成!")

def detect_anomaly(sensor_values):
    """通用异常检测函数 - 适配任意维度的传感器数据"""
    # 确保输入数据维度正确
    if len(sensor_values) != len(feature_columns):
        raise ValueError(f"输入数据维度错误: 期望{len(feature_columns)}维，实际{len(sensor_values)}维")
    
    # 标准化输入数据
    sensor_values_scaled = scaler.transform([sensor_values])
    
    # 获取异常分数和预测结果
    score = detector.decision_function(sensor_values_scaled)[0]
    prediction = detector.predict(sensor_values_scaled)[0]
    
    # 使用预测结果，但可以通过分数进行调整
    result = 'normal' if prediction == 1 else 'anomaly'
    
    # 可选：如果需要更保守的判断，可以调整阈值
    # conservative_threshold = detector.offset_ * 1.5  # 更严格的阈值
    # result = 'normal' if score > conservative_threshold else 'anomaly'
    
    return result

# 读取测试数据
test_data = pd.read_csv('/work/soft/LSTM-Autoencoders/kdd_data/device_info_20250616_5dimension.csv', header=None)
print(f"测试数据形状: {test_data.shape}")

# 检查数据分布（可选信息）
if len(test_data.columns) > len(feature_columns):
    status_counts = test_data.iloc[:, len(feature_columns)].value_counts()
    print(f"数据分布: {dict(status_counts)}")

# 开始检测
print("开始异常检测...")
start_time = time.time()

with open('detection_log.txt', 'w', encoding='utf-8') as f:
    f.write("序号,传感器1,传感器2,传感器3,传感器4,传感器5,原始状态,检测状态\n")
    correct = 0
    total = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    for idx in range(len(test_data)):
        sensor_values = test_data.iloc[idx, :5].values
        original_status = test_data.iloc[idx, 5] if len(test_data.columns) > 5 else 'unknown'

        detected_status = detect_anomaly(sensor_values)

        # 详细统计（只有当有真实标签时）
        if original_status != 'unknown':
            if original_status == 'anomaly' and detected_status == 'anomaly':
                tp += 1
            elif original_status == 'normal' and detected_status == 'normal':
                tn += 1
            elif original_status == 'normal' and detected_status == 'anomaly':
                fp += 1
            elif original_status == 'anomaly' and detected_status == 'normal':
                fn += 1

            if str(original_status) == detected_status:
                correct += 1

        sensor_str = ','.join([f"{val:.6f}" for val in sensor_values])
        f.write(f"{idx+1},{sensor_str},{original_status},{detected_status}\n")

        total += 1

        if (idx + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            speed = (idx + 1) / elapsed_time
            
            if original_status != 'unknown':
                accuracy = correct / total * 100
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                print(f"已处理 {idx+1} 条，准确率: {accuracy:.2f}%, 精确率: {precision:.3f}, 召回率: {recall:.3f}, 速度: {speed:.1f}条/秒")
            else:
                print(f"已处理 {idx+1} 条，速度: {speed:.1f}条/秒")

# 最终统计
total_time = time.time() - start_time

if tp + tn + fp + fn > 0:  # 有真实标签的情况
    accuracy = correct / total * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n=== 最终结果 ===")
    print(f"总准确率: {accuracy:.2f}%")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"误报率: {false_positive_rate:.4f}")
    print(f"真正例(TP): {tp}, 真负例(TN): {tn}")
    print(f"假正例(FP): {fp}, 假负例(FN): {fn}")
    
    # 检测结果分布
    detected_anomaly_count = tp + fp
    total_anomaly_count = tp + fn
    print(f"检测为异常: {detected_anomaly_count} 条")
    print(f"实际异常: {total_anomaly_count} 条")
else:
    print(f"\n=== 检测完成 ===")
    print("无真实标签，无法计算准确率指标")

print(f"总处理时间: {total_time:.2f}秒")
print(f"平均速度: {total/total_time:.1f}条/秒")
print(f"结果已保存到: detection_log.txt")

# 保存配置信息
with open('detection_config.txt', 'w', encoding='utf-8') as f:
    f.write("异常检测配置信息\n")
    f.write("=" * 30 + "\n")
    f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据维度: {len(feature_columns)}\n")
    f.write(f"训练样本: {len(sampled_data)} 条\n")
    f.write(f"测试样本: {len(test_data)} 条\n")
    f.write("\n参数配置:\n")
    for key, value in UNIVERSAL_CONFIG.items():
        f.write(f"  {key}: {value}\n")

print(f"配置信息已保存到: detection_config.txt")
