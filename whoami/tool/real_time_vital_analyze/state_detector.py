#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/24 16:44
@Author  : weiyutao
@File    : state_detector.py
"""
import time
import numpy as np
import matplotlib.pyplot as plt


class WindowBasedStateDetector:
    def __init__(self):
        # 分析参数
        self.analysis_window = 20  # 分析最近20秒
        self.min_duration = 20     # 最小持续时间
        
        # 状态记录
        self.current_state = "离床"
        self.state_start_time = None
        self.last_analysis_time = None
        
    def analyze_current_state(self, data_cache):
        """基于缓存数据分析当前状态"""
        if len(data_cache) < 10:  # 数据太少
            return self.current_state
            
        # 1. 获取最近20秒的重构误差
        recent_errors = self.get_recent_errors(data_cache, self.analysis_window)
        
        if len(recent_errors) < 5:  # 数据不够
            return self.current_state
            
        # 2. 计算特征
        features = self.extract_features(recent_errors)
        
        # 3. 分类状态
        detected_state = self.classify_state(features)
        
        # 4. 时间过滤
        filtered_state = self.apply_time_filter(detected_state)
        
        return filtered_state
    
    def get_recent_errors(self, data_cache, seconds):
        """从缓存中获取最近N秒的重构误差"""
        if not data_cache:
            return []
            
        current_time = data_cache[-1]['timestamp']
        cutoff_time = current_time - seconds
        
        recent_errors = []
        for data_point in reversed(data_cache):
            if data_point['timestamp'] >= cutoff_time:
                recent_errors.append(data_point['reconstruction_loss'])
            else:
                break
                
        return list(reversed(recent_errors))
    
    def extract_features(self, errors):
        """提取误差特征"""
        if len(errors) < 3:
            return {}
            
        return {
            'mean': np.mean(errors),
            'max': np.max(errors),
            'std': np.std(errors),
            'recent_trend': self.calculate_trend(errors[-10:]),  # 最近趋势
            'peak_count': self.count_peaks(errors),
            'stability': np.std(errors) < 0.05,
            'sudden_change': self.detect_sudden_change(errors)
        }
    
    def classify_state(self, features):
        """基于特征分类状态"""
        if not features:
            return self.current_state
            
        mean_err = features['mean']
        max_err = features['max']
        is_stable = features['stability']
        
        # 离床：误差小且稳定
        if max_err < 0.1 and is_stable:
            return "离床"
            
        # 呼吸急促：大峰值
        if max_err > 100:
            return "呼吸急促"
            
        # 呼吸暂停：在床但误差很小
        if max_err < 0.1 and is_stable and self.current_state != "离床":
            return "呼吸暂停"
            
        # 体动：中等峰值
        if 1 < max_err < 50 and features['peak_count'] > 0:
            return "体动"
            
        return "在床正常"
    
    def apply_time_filter(self, detected_state):
        """应用时间过滤规则"""
        current_time = time.time()
        
        if detected_state == self.current_state:
            return self.current_state
            
        # 如果还没有状态开始时间，初始化
        if self.state_start_time is None:
            self.state_start_time = current_time
            
        duration = current_time - self.state_start_time
        
        # 短时间状态变化的过滤逻辑
        if duration < self.min_duration:
            # 短暂离床后在床 -> 继续在床
            if (self.current_state == "离床" and 
                detected_state in ["在床正常", "体动", "呼吸暂停", "呼吸急促"]):
                self.current_state = detected_state
                return detected_state
                
            # 短暂在床后离床 -> 继续离床
            if (self.current_state in ["在床正常", "体动", "呼吸暂停", "呼吸急促"] and
                detected_state == "离床"):
                self.current_state = "离床"
                return "离床"
                
            # 其他短时间变化，忽略
            return self.current_state
        
        # 正常状态转换
        print(f"状态转换: {self.current_state} -> {detected_state} (持续{duration:.1f}秒)")
        self.current_state = detected_state
        self.state_start_time = current_time
        return detected_state
    
    def count_peaks(self, errors):
        """简单峰值计数"""
        if len(errors) < 3:
            return 0
            
        peaks = 0
        for i in range(1, len(errors) - 1):
            if errors[i] > errors[i-1] and errors[i] > errors[i+1] and errors[i] > 1:
                peaks += 1
        return peaks
    
    def calculate_trend(self, errors):
        """计算趋势"""
        if len(errors) < 3:
            return 'stable'
            
        first_half = np.mean(errors[:len(errors)//2])
        second_half = np.mean(errors[len(errors)//2:])
        
        if second_half > first_half * 1.2:
            return 'increasing'
        elif second_half < first_half * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def detect_sudden_change(self, errors):
        """检测突然变化"""
        if len(errors) < 5:
            return False
            
        recent_avg = np.mean(errors[-3:])
        previous_avg = np.mean(errors[-10:-3])
        
        return abs(recent_avg - previous_avg) > 10



# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



def plot_six_lines(all_data, timestamps, states):
    """
    绘制六个折线图 - 包含损失值
    
    Args:
        all_data: numpy数组，形状为 (n_samples, 6)
                 字段顺序：['timestamp', 'reconstruction_loss', 'heart_rate', 'breathing_rate', 'breath_line', 'heart_line']
        timestamps: 时间戳数组
        states: 状态检测结果列表
    """
    # 转换为相对时间（秒）
    time_relative = (timestamps - timestamps[0])
    
    # 提取各个信号数据（根据正确的字段顺序）
    # ['timestamp', 'reconstruction_loss', 'heart_rate', 'breathing_rate', 'breath_line', 'heart_line']
    reconstruction_loss = all_data[:, 1]  # reconstruction_loss
    heart_rate = all_data[:, 2]           # heart_rate
    breathing_rate = all_data[:, 3]       # breathing_rate
    breath_line = all_data[:, 4]          # breath_line
    heart_line = all_data[:, 5]           # heart_line
    
    # 创建6个子图
    fig, axes = plt.subplots(6, 1, figsize=(6, 6))
    
    # 1. 呼吸率
    axes[0].plot(time_relative, breathing_rate, 'b-', linewidth=1)
    axes[0].set_title('Breathing Rate')
    axes[0].set_ylabel('Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(np.arange(0, time_relative[-1], 500))  # 每500秒一个刻度
    
    # 2. 心率  
    axes[1].plot(time_relative, heart_rate, 'r-', linewidth=1)
    axes[1].set_title('Heart Rate')
    axes[1].set_ylabel('Rate')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(np.arange(0, time_relative[-1], 500))
    
    # 3. 呼吸线
    axes[2].plot(time_relative, breath_line, 'g-', linewidth=1)
    axes[2].set_title('Breath Signal')
    axes[2].set_ylabel('Signal')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(np.arange(0, time_relative[-1], 500))
    
    # 4. 心线
    axes[3].plot(time_relative, heart_line, 'm-', linewidth=1)
    axes[3].set_title('Heart Signal')
    axes[3].set_ylabel('Signal')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xticks(np.arange(0, time_relative[-1], 500))
    
    # 5. 损失值
    axes[4].plot(time_relative, reconstruction_loss, 'orange', linewidth=1)
    axes[4].set_title('Reconstruction Loss')
    axes[4].set_ylabel('Loss')
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xticks(np.arange(0, time_relative[-1], 500))
    
    # 6. 状态值
    if states:
        state_mapping = {'离床': 0, '在床正常': 1, '呼吸暂停': 2, '体动': 3, '呼吸急促': 4}
        state_values = [state_mapping.get(s, 0) for s in states]
        state_x = np.linspace(0, time_relative[-1], len(states))
        
        axes[5].plot(state_x, state_values, 'ko-', linewidth=1, markersize=2)
        axes[5].set_title('Status')
        axes[5].set_ylabel('State')
        axes[5].set_xlabel('Time (s)')
        axes[5].set_yticks([0, 1, 2, 3, 4])
        axes[5].set_yticklabels(['Off', 'Normal', 'Apnea', 'Move', 'Fast'])
        axes[5].grid(True, alpha=0.3)
        axes[5].set_xticks(np.arange(0, time_relative[-1], 500))
    else:
        axes[5].text(0.5, 0.5, 'No Status Data', transform=axes[5].transAxes, ha='center')
        axes[5].set_title('Status')
        axes[5].set_xlabel('Time (s)')
        axes[5].set_xticks(np.arange(0, max(time_relative) if len(time_relative) > 0 else 10000, 500))
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes




if __name__ == '__main__':
    from whoami.tool.real_time_vital_analyze.json_loader import SimpleJSONLoader
    from whoami.utils.utils import Utils

    utils = Utils()
    loader = SimpleJSONLoader("/work/ai/WHOAMI/whoami/out/data_logs/realtime_data_20250624.json")

    fields = loader.get_all_fields()
    all_data = loader.to_numpy_array()
    timestamps = loader.get_timestamps()
    print(fields)
    print(all_data[19:, :][0])
    slide_error_data = utils.create_sliding_windows(data=all_data, field_index=1)
    print(len(slide_error_data))

    detector = WindowBasedStateDetector()
    states = []
    for i, window in enumerate(slide_error_data):
        # 将窗口转换为data_cache格式
        data_cache = []
        for j, error_value in enumerate(window):
            real_timestamp = timestamps[i + j]
            data_point = {
                'timestamp': real_timestamp,  # 需要时间戳
                'reconstruction_loss': error_value,
            }
            data_cache.append(data_point)
        
        # 使用detector分析
        state = detector.analyze_current_state(data_cache)
        states.append(state)
    
    plot_six_lines(all_data=all_data[19:, :], timestamps=timestamps[19:], states=states)