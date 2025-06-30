import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
import time

# 定义固定的列名
feature_columns = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']

# 读取训练数据
full_data = pd.read_csv('/work/soft/LSTM-Autoencoders/kdd_data/device_info_20250616_nomaly_5dimension.csv')
print(f"原始训练数据: {full_data.shape}")

# 数据质量分析
def analyze_data_quality(data):
    """分析数据质量"""
    print("\n=== 数据质量分析 ===")
    
    # 基本统计
    print("数据统计:")
    print(data.describe())
    
    # 检查重复数据
    duplicates = data.duplicated().sum()
    print(f"\n重复数据: {duplicates} 条 ({duplicates/len(data)*100:.2f}%)")
    
    # 检查极值
    print("\n可能的极值:")
    for col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        print(f"  {col}: {outliers} 个极值 ({outliers/len(data)*100:.2f}%)")
    
    return data

# 改进的数据预处理
def improved_data_preprocessing(data, use_robust_scaling=True, remove_outliers=True):
    """改进的数据预处理"""
    print("\n=== 数据预处理 ===")
    
    # 1. 移除重复数据
    original_size = len(data)
    data = data.drop_duplicates()
    removed_duplicates = original_size - len(data)
    print(f"移除重复数据: {removed_duplicates} 条")
    
    # 2. 可选：移除训练数据中的极端异常值
    if remove_outliers:
        print("使用DBSCAN移除训练数据中的极端异常值...")
        
        # 先进行初步标准化
        temp_scaler = StandardScaler()
        data_scaled = temp_scaler.fit_transform(data)
        
        # 使用DBSCAN找出极端异常点
        dbscan = DBSCAN(eps=2.0, min_samples=10)
        clusters = dbscan.fit_predict(data_scaled)
        
        # 保留主要聚类，移除噪声点（标记为-1）
        main_cluster_mask = clusters != -1
        removed_outliers = (~main_cluster_mask).sum()
        data = data[main_cluster_mask]
        
        print(f"移除极端异常值: {removed_outliers} 条 ({removed_outliers/original_size*100:.2f}%)")
    
    # 3. 使用RobustScaler而不是StandardScaler（对异常值更鲁棒）
    if use_robust_scaling:
        scaler = RobustScaler()  # 使用中位数和四分位距，对异常值更鲁棒
        print("使用RobustScaler进行标准化")
    else:
        scaler = StandardScaler()
        print("使用StandardScaler进行标准化")
    
    print(f"最终训练数据: {len(data)} 条")
    return data, scaler

# 改进的采样策略
def improved_sampling(data, target_samples=20000):  # 增加到20000
    """改进的采样策略 - 增加样本量并保证多样性"""
    print(f"\n=== 改进采样策略 ===")
    
    if len(data) <= target_samples:
        print("数据量不足，使用全部数据")
        return data
    
    # 方法1: 增加时间分层的粒度
    n_time_groups = 20  # 增加到20个时间段
    samples_per_group = target_samples // n_time_groups
    
    data_copy = data.copy()
    data_copy['time_group'] = pd.cut(range(len(data_copy)), bins=n_time_groups, labels=False)
    
    sampled_list = []
    for group in range(n_time_groups):
        group_data = data_copy[data_copy['time_group'] == group]
        if len(group_data) >= samples_per_group:
            sampled_list.append(group_data.sample(n=samples_per_group, random_state=42))
        else:
            sampled_list.append(group_data)
    
    # 方法2: 添加随机采样补充
    result = pd.concat(sampled_list).drop('time_group', axis=1)
    if len(result) < target_samples:
        remaining_needed = target_samples - len(result)
        remaining_data = data.drop(result.index)
        if len(remaining_data) > 0:
            additional = remaining_data.sample(n=min(remaining_needed, len(remaining_data)), random_state=42)
            result = pd.concat([result, additional])
    
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"采样结果: {len(result)} 条")
    return result

# 执行改进的数据预处理
training_data = full_data.iloc[:, :5].copy()
training_data.columns = feature_columns

# 分析数据质量
training_data = analyze_data_quality(training_data)

# 改进的预处理
training_data, scaler = improved_data_preprocessing(training_data)

# 改进的采样
sampled_data = improved_sampling(training_data, target_samples=20000)

# 标准化
sampled_data_scaled = scaler.fit_transform(sampled_data)

# 更保守的参数配置
CONSERVATIVE_CONFIG = {
    'n_neighbors': 50,         # 增加邻居数，提高稳定性
    'algorithm': 'kd_tree',    
    'leaf_size': 40,           
    'metric': 'euclidean',     
    'n_jobs': -1              
}

print(f"\n使用保守参数配置:")
for key, value in CONSERVATIVE_CONFIG.items():
    print(f"  {key}: {value}")

# 训练检测器
detector = LocalOutlierFactor(**CONSERVATIVE_CONFIG, novelty=True)
detector.fit(sampled_data_scaled)
print("训练完成!")

# 更保守的阈值设置
train_scores = detector.decision_function(sampled_data_scaled)
score_mean = np.mean(train_scores)
score_std = np.std(train_scores)
score_percentiles = np.percentile(train_scores, [1, 5, 10, 15, 20])

print(f"\n训练数据异常分数统计:")
print(f"  均值: {score_mean:.6f}")
print(f"  标准差: {score_std:.6f}")
print(f"  1%分位数: {score_percentiles[0]:.6f}")
print(f"  5%分位数: {score_percentiles[1]:.6f}")
print(f"  10%分位数: {score_percentiles[2]:.6f}")

# 使用百分位数作为阈值（更稳定）
PERCENTILE_THRESHOLDS = {
    'ultra_strict': score_percentiles[0],    # 1%分位数
    'very_strict': score_percentiles[1],     # 5%分位数
    'strict': score_percentiles[2],          # 10%分位数
    'moderate': score_percentiles[3],        # 15%分位数
    'lenient': score_percentiles[4]          # 20%分位数
}

# 默认使用更严格的阈值
selected_strictness = 'very_strict'  # 5%分位数
threshold = PERCENTILE_THRESHOLDS[selected_strictness]

print(f"\n选择严格程度: {selected_strictness}")
print(f"异常阈值: {threshold:.6f}")
print(f"预期异常率: ~{selected_strictness.replace('_', ' ')}")

def detect_anomaly(sensor_values):
    """保守的异常检测"""
    if len(sensor_values) != len(feature_columns):
        raise ValueError(f"输入数据维度错误: 期望{len(feature_columns)}维，实际{len(sensor_values)}维")
    
    sensor_values_scaled = scaler.transform([sensor_values])
    score = detector.decision_function(sensor_values_scaled)[0]
    
    if score < threshold:
        return 'anomaly', score
    else:
        return 'normal', score

# 读取测试数据
test_data = pd.read_csv('/work/soft/LSTM-Autoencoders/kdd_data/device_info_20250616_5dimension.csv', header=None)
print(f"\n测试数据形状: {test_data.shape}")

# 开始检测
print("开始异常检测...")
start_time = time.time()

with open('detection_log_improved.txt', 'w', encoding='utf-8') as f:
    f.write("序号,传感器1,传感器2,传感器3,传感器4,传感器5,原始状态,检测状态,异常分数\n")
    correct = 0
    total = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    
    all_scores = []
    detected_anomaly_count = 0

    for idx in range(len(test_data)):
        sensor_values = test_data.iloc[idx, :5].values
        original_status = test_data.iloc[idx, 5] if len(test_data.columns) > 5 else 'unknown'

        detected_status, score = detect_anomaly(sensor_values)
        all_scores.append(score)
        
        if detected_status == 'anomaly':
            detected_anomaly_count += 1

        # 统计
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
        f.write(f"{idx+1},{sensor_str},{original_status},{detected_status},{score:.8f}\n")

        total += 1

        if (idx + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            speed = (idx + 1) / elapsed_time
            current_anomaly_ratio = detected_anomaly_count / (idx + 1)
            
            if original_status != 'unknown':
                accuracy = correct / total * 100
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                print(f"已处理 {idx+1} 条，准确率: {accuracy:.2f}%, 精确率: {precision:.3f}, 召回率: {recall:.3f}, 检出异常率: {current_anomaly_ratio:.4f}, 速度: {speed:.1f}条/秒")
            else:
                print(f"已处理 {idx+1} 条，检出异常率: {current_anomaly_ratio:.4f}, 速度: {speed:.1f}条/秒")

# 最终统计
total_time = time.time() - start_time
final_anomaly_ratio = detected_anomaly_count / total

if tp + tn + fp + fn > 0:
    accuracy = correct / total * 100
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n=== 最终结果 ===")
    print(f"使用阈值: {threshold:.6f} ({selected_strictness})")
    print(f"总准确率: {accuracy:.2f}%")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"误报率: {false_positive_rate:.4f}")
    print(f"检出异常率: {final_anomaly_ratio:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
else:
    print(f"\n=== 检测完成 ===")
    print(f"使用阈值: {threshold:.6f} ({selected_strictness})")
    print(f"检出异常率: {final_anomaly_ratio:.4f}")

print(f"总处理时间: {total_time:.2f}秒")
print(f"结果已保存到: detection_log_improved.txt")

# 提供调整建议
print(f"\n=== 调整建议 ===")
if final_anomaly_ratio > 0.1:  # 如果检出超过10%
    print("检出异常率较高，建议:")
    print("1. 使用 'ultra_strict' 模式")
    print("2. 增加训练数据量")
    print("3. 检查训练数据质量")
elif final_anomaly_ratio < 0.001:  # 如果检出少于0.1%
    print("检出异常率很低，可能:")
    print("1. 阈值过于严格，可尝试 'strict' 或 'moderate'")
    print("2. 训练数据覆盖了测试场景")

print(f"\n可用严格程度及对应阈值:")
for level, thresh in PERCENTILE_THRESHOLDS.items():
    print(f"  {level}: {thresh:.6f}")
