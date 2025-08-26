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

class SimpleSleepDataset(Dataset):
    """简单的睡眠数据集 - 3个睡眠阶段"""
    
    def __init__(self, csv_file, window_size=30, step_size=1, max_samples=None, scaler_path=None):
        # 读取数据
        print("正在读取CSV文件...")
        self.df = pd.read_csv(csv_file, encoding='gbk')
        
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
        self.df['signal_quality'] = self.df['signal_intensity'] / 50.0  # 归一化信号强度
        print("特征工程完成!")
        
        # 5个特征：呼吸率、心率、呼吸线、心线、信号质量
        features = ['resp_rate', 'heart_rate', 'breath_line', 'heart_line', 'signal_quality']
        
        # 修复：处理缺失值 - 使用新的pandas语法
        print("处理缺失值...")
        self.df[features] = self.df[features].ffill().fillna(0)
        
        # 关键修复：标准化处理
        if scaler_path and os.path.exists(scaler_path):
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
            label = self.df['label_num'].iloc[i+window_size-1]
            
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
    
    def __init__(self, input_size=5, seq_length=30, num_classes=3):  # 3个类别：清醒、浅睡眠、深睡眠
        super().__init__()
        
        # 简单的CNN+LSTM
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 双向LSTM提升效果 - 修复dropout警告
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 双向LSTM输出128维
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)  # 3个输出
        )
    
    def forward(self, x):
        # x: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # 更深的CNN提取特征
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # [batch, 128]
        
        # 为LSTM添加序列维度
        x = x.unsqueeze(1)  # [batch, 1, 128]
        
        # 双向LSTM
        lstm_out, _ = self.lstm(x)
        
        # 分类
        output = self.classifier(lstm_out[:, -1, :])
        return output


def train_simple_model(csv_file, epochs=50, max_samples=None):
    """训练函数"""
    
    print("开始训练简单睡眠分期模型...")
    
    # 1. 加载数据
    dataset = SimpleSleepDataset(csv_file, window_size=30, step_size=5, max_samples=max_samples, scaler_path='sleep_scaler.pkl')
    
    # 2. 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 修复：添加drop_last=True避免不完整的batch导致错误
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
    
    # 3. 创建模型 - 3个睡眠阶段
    model = SimpleSleepNet(input_size=5, num_classes=3)  # 5个特征，3个睡眠类别
    
    # 处理类别不平衡 - 为少数类别加权
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / (class_counts + 1e-6)  # 避免除零
    class_weights = torch.FloatTensor(class_weights / class_weights.sum() * len(class_weights))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 4. 训练循环
    train_losses = []
    val_accs = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # 修复：确保标签维度正确
            if batch_y.dim() > 1:
                batch_y = batch_y.squeeze()
            
            # 检查batch是否为空
            if batch_x.size(0) == 0 or batch_y.size(0) == 0:
                continue
                
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 实时打印训练损失
            print(f"\rEpoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] Train Loss: {loss.item():.4f}", end='')
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                # 修复：确保标签维度正确
                if batch_y.dim() > 1:
                    batch_y = batch_y.squeeze()
                
                # 检查batch是否为空
                if batch_x.size(0) == 0 or batch_y.size(0) == 0:
                    continue
                    
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
                # 实时打印验证损失
                print(f"\rEpoch {epoch+1} - Validation [{batch_idx+1}/{len(val_loader)}] Val Loss: {loss.item():.4f}", end='', flush=True)
        
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accs.append(val_acc)
        val_losses.append(avg_val_loss)
        
        # 每个epoch结束后打印总结
        print(f'\nEpoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}')
    
    # 5. 保存模型
    torch.save(model.state_dict(), 'simple_sleep_model.pth')
    print("模型已保存为: simple_sleep_model.pth")
    
    # 6. 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot([x*max(train_losses+val_losses) for x in val_accs], label='Val Acc (scaled)')
    plt.title('All Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
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
    model = SimpleSleepNet(input_size=5, num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 你的3个标签对应关系
    stage_names = ['清醒', '浅睡眠', '深睡眠']
    
    # 实时预测示例 - 使用训练时保存的scaler
    dataset = SimpleSleepDataset(csv_file, window_size=30, step_size=1, max_samples=1000, scaler_path='sleep_scaler.pkl')
    
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


# 使用示例 - 适配你的数据
if __name__ == "__main__":
    # 你的CSV文件路径
    csv_file = "/work/ai/WHOAMI/device_info_13D2F34920008071211195A907_20250623_classifier_LABEL.csv"
    
    print("🚀 开始训练3类睡眠分期模型...")
    print("="*50)
    
    # # 快速测试选项 - 如果数据太大，可以先用小样本测试
    USE_QUICK_TEST = False  # 改为False使用全部数据，True使用部分数据测试
    max_samples = 50000 if USE_QUICK_TEST else None  # 快速测试使用5万行数据
    
    if USE_QUICK_TEST:
        print("⚡ 快速测试模式 - 使用部分数据")
    else:
        print("🐌 完整训练模式 - 使用全部数据")
    
    try:
        # 训练模型
        model = train_simple_model(csv_file, epochs=150, max_samples=max_samples)
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
        print("\n📊 模式1: 每秒预测结果")
        predictions_1s = predict_realtime('simple_sleep_model.pth', csv_file, 
                                        max_predict_samples=100, print_interval=1)
        
        print("\n📊 模式2: 每10秒预测结果")  
        predictions_10s = predict_realtime('simple_sleep_model.pth', csv_file,
                                         max_predict_samples=1000, print_interval=10)
        
        print(f"✅ 预测完成")
        
        # 保存预测结果
        with open('sleep_predictions_1s.json', 'w', encoding='utf-8') as f:
            json.dump(predictions_1s, f, ensure_ascii=False, indent=2)
        
        with open('sleep_predictions_10s.json', 'w', encoding='utf-8') as f:
            json.dump(predictions_10s, f, ensure_ascii=False, indent=2)
            
        print("📄 预测结果已保存到: sleep_predictions_1s.json 和 sleep_predictions_10s.json")
        
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