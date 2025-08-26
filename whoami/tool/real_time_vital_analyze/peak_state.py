#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/25 09:52
@Author  : weiyutao
@File    : peak_state.py
实时状态监测 - 每个时间点都输出状态
结合基础状态判断和峰值状态判断
"""

import math
import time
import numpy as np
import torch
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List
from whoami.neural_network.rnn.lstm_classifier import GRUClassifier, inference_single_sample
from whoami.neural_network.rnn.cnn_lstm_classifier import SimpleSleepNet, inference_single_sample_, load_sleep_scaler

class PeakState(Enum):
    """峰值检测状态"""
    INACTIVE = "inactive"    # 非激活状态（数据太小）
    BASELINE = "baseline"    # 基线状态
    RISING = "rising"        # 上升状态  
    PEAK = "peak"           # 峰值状态
    FALLING = "falling"     # 下降状态
    WARMING_UP = "warming_up"  # 冷启动状态

@dataclass
class PeakEvent:
    """峰值事件"""
    event_type: str          # "peak_start", "peak_end", "peak_completed"
    peak_value: float        # 峰值大小
    start_time: float        # 峰值开始时间
    end_time: float          # 峰值结束时间
    duration: float          # 峰值持续时间
    start_index: int         # 峰值开始索引
    end_index: int           # 峰值结束索引
    state_value: int = 0     # 添加state值记录

class RealTimeStateMonitor:
    """
    实时状态监测器
    每个时间点都判断并输出状态
    """
    
    def __init__(self,
        # 基础状态阈值
        off_bed_threshold: float = 0.05,        # 离床阈值
        apnea_threshold: float = 0.1,          # 呼吸暂停上限阈值
        
        # 状态持续时间要求
        off_bed_duration: float = 20.0,        # 离床状态需要持续20秒
        apnea_duration: float = 8.0,          # 呼吸暂停需要持续20秒
        normal_duration: float = 45.0,          # 在床正常需要持续45秒
        
        # 峰值检测参数
        activation_threshold: float = 1.0,     # 峰值检测激活阈值
        deactivation_threshold: float = 0.5,   # 峰值检测去激活阈值
        min_baseline: float = 0.1,             # 最小基线值
        
        # 添加这个参数
        use_fixed_baseline: bool = False,      # 是否使用固定基线
        fixed_baseline_value: float = 10.0,     # 固定基线值

        baseline_alpha: float = 0.1,           # 基线适应速度
        variance_beta: float = 0.2,            # 方差适应速度
        
        # rise_factor: float = 1.5,              # 上升倍数（更宽松）
        # peak_factor: float = 2.0,              # 峰值倍数（宽更松）
        # fall_factor: float = 1.3,              # 下降倍数（更宽松）

        rise_factor: float = 2.0,              # 更严格上升倍数（更严格）
        peak_factor: float = 2.5,              #更严格峰值倍数（更严格）
        fall_factor: float = 1.5,              # 下降倍数（更严格）
        
        min_peak_duration: float = 2.0,        # 最小峰值持续时间(秒)
        min_peak_height: float = 5.0,          # 最小绝对峰值高度
        warmup_samples: int = 20,             # 冷启动样本数
        
        # 深度学习模型参数
        # model_path: str = "/work/ai/WHOAMI/whoami/neural_network/simple_sleep_model_v1_12W_2dimension_60epoch.pth",
        model_path: str = "/work/ai/WHOAMI/whoami/neural_network/simple_sleep_model_20W_150epochs_2dimensions_classifier.pth",
        seq_len: int = 30,                     # 序列长度（60秒）
        n_features: int = 5,                   # 特征数量
        n_classes: int = 3,                    # 分类数量
        embedding_dim: int = 64,               # 嵌入维度
        dropout: float = 0.2,                   # dropout比例
        # scaler_path: str = "/work/ai/WHOAMI/whoami/neural_network/sleep_scaler_v1_12W.pkl"
        scaler_path: str = "/work/ai/WHOAMI/whoami/neural_network/sleep_scaler_20W_2dimensions.pkl",
        
        
        device_sn: str = None
    ):             
        
        # 基础状态参数
        self.off_bed_threshold = off_bed_threshold
        self.apnea_threshold = apnea_threshold
        
        # 状态持续时间要求
        self.off_bed_duration = off_bed_duration
        self.apnea_duration = apnea_duration
        self.normal_duration = normal_duration
        
        # 峰值检测参数
        self.activation_threshold = activation_threshold
        self.deactivation_threshold = deactivation_threshold
        self.min_baseline = min_baseline
        
        # 添加这两行
        self.use_fixed_baseline = use_fixed_baseline
        self.fixed_baseline_value = fixed_baseline_value
        
        self.baseline_alpha = baseline_alpha
        self.variance_beta = variance_beta
        self.rise_factor = rise_factor
        self.peak_factor = peak_factor
        self.fall_factor = fall_factor
        self.min_peak_duration = min_peak_duration
        self.min_peak_height = min_peak_height
        self.warmup_samples = warmup_samples
        
        # 深度学习模型参数
        self.model_path = model_path
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        
        # 统计状态（只基于激活数据）
        self.active_baseline = min_baseline
        self.active_variance = 1.0
        self.active_sample_count = 0
        
        # 全局状态
        self.total_sample_count = 0
        self.is_active = False
        
        # 状态机状态
        self.current_peak_state = PeakState.WARMING_UP
        self.state_start_time = None
        self.state_start_index = None
        
        # 当前峰值信息
        self.current_peak_max = 0.0
        self.current_peak_start_time = None
        self.current_peak_start_index = None
        self.current_peak_max_time = None
        self.current_peak_max_index = None
        
        # 历史记录
        self.peak_history = []
        self.last_update_time = None
        
        # 状态缓冲和验证
        self.confirmed_state = "在床正常"          # 当前确认的状态
        self.candidate_state = None               # 候选状态
        self.candidate_start_time = None          # 候选状态开始时间
        self.candidate_start_index = None         # 候选状态开始索引
        
        # 添加当前state值存储
        self.current_body_move_energy = 0              # 当前传入的state值
        
        # state历史缓冲区（用于20秒窗口统计）
        self.state_history = []                   # [(timestamp, state_value), ...]
        self.body_move_energy_window_duration = 20.0         # 20秒窗口
        self.mean_body_move_energy_threshold = 15.0       # 30%阈值
        
        # 原始数值历史缓冲区（用于8秒稳定性判断）
        self.value_history = []                   # [(timestamp, value), ...]
        self.value_window_duration = 8.0          # 8秒窗口
        self.stability_threshold = 0.1            # 稳定性阈值（前后变动不超过0.1）
        
        # 新增：breath_line_heart_line历史缓冲区（用于60秒深度学习推理）
        self.breath_line_heart_line_history = []  # [(timestamp, breath_line_heart_line), ...]
        self.deep_learning_window_duration = 30.0  # 60秒窗口
        
        # 状态历史（用于检查60秒内是否有离床状态）
        self.overall_state_history = []           # [(timestamp, overall_state), ...]
        
        # 深度学习模型相关
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_initialized = False
        self.scaler_path = scaler_path
        # 类别映射（根据你的模型输出调整）
        self.class_mapping = {
            0: "清醒",
            1: "浅睡眠", 
            2: "深睡眠",
        }
        self.scaler = load_sleep_scaler(self.scaler_path)
        from whoami.tool.real_time_vital_analyze.dual_memory_threshold import HighPeakBiasedThreshold
        self.dynamic_threshold = 10.0
        self.threshold_calculator = HighPeakBiasedThreshold(
            high_peak_percentile=0.85,    # 更严格：只有前15%才算高峰值
            min_threshold_ratio=0.6,      # 更保守：最小阈值为高峰值均值的60%
            peak_height_bias=0.9,         # 更偏向：90%权重给高峰值
            downward_sensitivity=0.05,    # 更保守：对下降极其不敏感
        )
        self.device_sn = device_sn

    def _initialize_model(self):
        """延迟初始化深度学习模型"""
        if self.model_initialized:
            return
            
        # try:
        #     self.model = GRUClassifier(
        #         seq_len=self.seq_len,
        #         n_features=self.n_features, 
        #         n_classes=self.n_classes,
        #         embedding_dim=self.embedding_dim,
        #         dropout=self.dropout
        #     ).to(self.device)
            
        #     checkpoint = torch.load(self.model_path, map_location=self.device)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.model.eval()
            
        #     self.model_initialized = True
        #     print(f"深度学习模型已加载: {self.model_path}")
            
        # except Exception as e:
        #     print(f"深度学习模型加载失败: {e}")
        #     self.model_initialized = False
        
        
        try:
            self.model = SimpleSleepNet(
                input_size=self.n_features,
                seq_length=self.seq_len, 
                num_classes=self.n_classes,
            ).to(self.device)
            
            # checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            
            self.model_initialized = True
            print(f"深度学习模型已加载: {self.model_path}")
            
        except Exception as e:
            print(f"深度学习模型加载失败: {e}")
            self.model_initialized = False


    def update(
        self, 
        value: float, 
        timestamp: Optional[float] = None, 
        body_move_energy: int = 0, 
        breath_line_heart_line: Optional[np.ndarray] = None
    ) -> Tuple[str, str, Optional[PeakEvent]]:
        """
        更新状态监测器
        
        Args:
            value: 传感器数值
            timestamp: 时间戳
            body_move_energy: 当前时间点的状态值，用于峰值状态判断
            breath_line_heart_line: 1*2的numpy array，包含呼吸线和心率线数据
        
        Returns:
            (总体状态, 峰值状态, 峰值事件或None)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # 验证breath_line_heart_line参数
        if breath_line_heart_line is not None:
            if not isinstance(breath_line_heart_line, np.ndarray):
                raise ValueError("breath_line_heart_line必须是numpy array")
            # if breath_line_heart_line.shape != (1, 2):
            #     raise ValueError(f"breath_line_heart_line形状必须是(1, 2)，当前为{breath_line_heart_line.shape}")
        
        self.total_sample_count += 1
        self.current_body_move_energy = body_move_energy  # 保存当前state值
        
        # 更新各种历史缓冲区
        self._update_state_history(timestamp, body_move_energy)
        self._update_value_history(timestamp, value)
        
        # 更新breath_line_heart_line历史缓冲区
        if breath_line_heart_line is not None:
            self._update_breath_line_heart_line_history(timestamp, breath_line_heart_line)
        
        # 1. 基础状态判断（基于当前值和历史稳定性）
        raw_base_state = self._classify_base_state(value, timestamp)
        
        # 2. 峰值状态检测
        peak_state, peak_event = self._update_peak_detection(value, timestamp, body_move_energy)
        
        # 3. 状态持续时间验证
        validated_base_state = self._validate_state_duration(raw_base_state, timestamp)
        
        # 4. 综合状态判断
        overall_state = self._classify_overall_state(value, validated_base_state, peak_state, peak_event)
        
        # 5. 深度学习状态细分（仅在特定条件下）
        if self._check_deep_learning_conditions(overall_state, timestamp):
            deep_learning_state = self._get_deep_learning_state(timestamp)
            if deep_learning_state:
                overall_state = deep_learning_state
        else:
            if overall_state == "在床正常":
                overall_state = "清醒"
        # 6. 更新状态历史
        self._update_overall_state_history(timestamp, overall_state)
        
        # 7. 更新缓存
        self.last_update_time = timestamp
        
        print(f"当前动态阈值: ----------------------------------------------------------------------------- {self.dynamic_threshold:.2f}")
        return overall_state, peak_state.value, peak_event
    
    
    def _update_breath_line_heart_line_history(self, timestamp: float, breath_line_heart_line: np.ndarray):
        """更新breath_line_heart_line历史缓冲区，维护60秒窗口"""
        # 添加当前数据到历史
        self.breath_line_heart_line_history.append((timestamp, breath_line_heart_line.copy()))
        
        # 清理超过60秒的历史数据
        cutoff_time = timestamp - self.deep_learning_window_duration
        self.breath_line_heart_line_history = [
            (t, data) for t, data in self.breath_line_heart_line_history 
            if t >= cutoff_time
        ]
    
    
    def _update_overall_state_history(self, timestamp: float, overall_state: str):
        """更新总体状态历史缓冲区，维护60秒窗口"""
        # 添加当前状态到历史
        self.overall_state_history.append((timestamp, overall_state))
        
        # 清理超过60秒的历史数据
        cutoff_time = timestamp - self.deep_learning_window_duration
        self.overall_state_history = [
            (t, state) for t, state in self.overall_state_history 
            if t >= cutoff_time
        ]
    
    
    def _check_deep_learning_conditions(self, current_overall_state: str, timestamp: float) -> bool:
        """
        检查是否满足调用深度学习模型的条件
        
        条件：
        1. 当前状态为"在床正常"
        2. 最近60秒不存在"离床"状态
        3. 有足够的breath_line_heart_line数据（60秒）
        
        Returns:
            bool: 是否满足条件
        """
        # 条件1：当前状态必须为"在床正常"
        if current_overall_state != "在床正常":
            return False
        
        # 条件2：检查最近60秒内是否有"离床"状态
        cutoff_time = timestamp - self.deep_learning_window_duration
        recent_states = [
            state for t, state in self.overall_state_history 
            if t >= cutoff_time and t <= timestamp
        ]
        
        if "离床" in recent_states:
            return False
        
        # 条件3：检查是否有足够的breath_line_heart_line数据
        recent_breath_data = [
            data for t, data in self.breath_line_heart_line_history 
            if t >= cutoff_time and t <= timestamp
        ]
        
        # 需要至少有50个数据点（可以调整这个阈值）
        if len(recent_breath_data) < self.seq_len:
            return False
        
        return True
    
    
    def _get_deep_learning_state(self, timestamp: float) -> Optional[str]:
        """
        使用深度学习模型进行状态推理
        
        Returns:
            str or None: 推理得到的状态，如果推理失败则返回None
        """
        try:
            # 延迟初始化模型
            if not self.model_initialized:
                self._initialize_model()
            
            if not self.model_initialized or self.model is None:
                return None
            
            # 获取最近60秒的数据
            cutoff_time = timestamp - self.deep_learning_window_duration
            recent_data = [
                data for t, data in self.breath_line_heart_line_history 
                if t >= cutoff_time and t <= timestamp
            ]
            
            if len(recent_data) < self.seq_len:  # 数据不足
                return None
            
            # 将数据转换为模型输入格式
            # 如果数据点数超过60，取最近的60个点
            # 如果数据点数不足60，进行插值或填充
            if len(recent_data) >= self.seq_len:
                # 取最近的60个点
                selected_data = recent_data[-self.seq_len:]
            else:
                # 数据不足60个点，可以选择填充或跳过
                return None
            
            # 构建输入数组 (seq_len, n_features)
            breath_line_heart_line_array = np.array([data for data in selected_data])
            breath_line_heart_line_array = breath_line_heart_line_array.reshape(self.seq_len, 3)
            
            # print(breath_line_heart_line_array)
            # 调用推理函数
            result, confidence = inference_single_sample_(model=self.model, scaler=self.scaler, sample_data=breath_line_heart_line_array, use_raw_features=True)
            # if confidence < 0.6:
            #     result = 0
            # 映射结果到状态名称
            if result in self.class_mapping:
                predicted_state = self.class_mapping[result]
                print(f"深度学习推理结果: {predicted_state} (置信度: {confidence:.3f})")
                return predicted_state
            else:
                print(f"未知的预测结果: {result}")
                return None
                
        except Exception as e:
            print(f"深度学习推理失败: {e}")
            return None
    
    
    def get_debug_info(self) -> dict:
        """获取调试信息，包括状态验证详情和深度学习相关信息"""
        debug_info = {
            'confirmed_state': self.confirmed_state,
            'candidate_state': self.candidate_state,
            'candidate_duration': 0.0,
            'required_duration': 0.0,
            'current_body_move_energy': self.current_body_move_energy,
            'state_history_length': len(self.state_history),
            'value_history_length': len(self.value_history),
            'breath_line_heart_line_history_length': len(self.breath_line_heart_line_history),
            'overall_state_history_length': len(self.overall_state_history),
            'recent_state_stats': self._get_recent_state_stats(),
            'apnea_stability_check': self._is_stable_in_apnea_range(self.last_update_time) if self.last_update_time else False,
            'deep_learning_conditions': self._check_deep_learning_conditions("在床正常", self.last_update_time) if self.last_update_time else False,
            'model_initialized': self.model_initialized
        }
        
        if self.candidate_state and self.candidate_start_time:
            duration = self.last_update_time - self.candidate_start_time
            required = self._get_required_duration(self.candidate_state)
            debug_info.update({
                'candidate_duration': duration,
                'required_duration': required,
                'progress': f"{duration:.1f}/{required:.1f}秒"
            })
        
        return debug_info


    # 以下方法保持不变...
    def _update_state_history(self, timestamp: float, state: int):
        """更新state历史缓冲区，维护20秒窗口"""
        # 添加当前state到历史
        self.state_history.append((timestamp, state))
        
        # 清理超过20秒的历史数据
        cutoff_time = timestamp - self.body_move_energy_window_duration
        self.state_history = [(t, s) for t, s in self.state_history if t >= cutoff_time]
    
    
    def _update_value_history(self, timestamp: float, value: float):
        """更新value历史缓冲区，维护8秒窗口（新增）"""
        # 添加当前value到历史
        self.value_history.append((timestamp, value))
        
        # 清理超过8秒的历史数据
        cutoff_time = timestamp - self.value_window_duration
        self.value_history = [(t, v) for t, v in self.value_history if t >= cutoff_time]
    
    
    
    def _is_stable_in_apnea_range(self, timestamp: float) -> bool:
        """判断最近8秒内数据是否稳定在呼吸暂停范围内（新增）"""
        cutoff_time = timestamp - self.value_window_duration
        
        # 获取8秒窗口内的数据
        window_values = [(t, v) for t, v in self.value_history 
                         if t >= cutoff_time and t <= timestamp]
        
        # 数据不足时返回False（至少需要5个数据点）
        if len(window_values) < 5:
            return False
        
        # 提取数值列表
        values = [v for _, v in window_values]
        
        # 检查所有值是否在 [0.1, 2] 范围内
        if not all(self.off_bed_threshold <= v <= self.apnea_threshold for v in values):
            return False
        
        # 检查稳定性：相邻值变化不超过stability_threshold
        for i in range(1, len(values)):
            if abs(values[i] - values[i-1]) > self.stability_threshold:
                return False
        
        # 可选：额外检查整体变化范围
        value_range = max(values) - min(values)
        if value_range > self.stability_threshold * 2:  # 整体变化范围不超过阈值的2倍
            return False
        
        return True
    
    
    def _calculate_state_statistics(self, end_timestamp: float) -> int:
        """
        计算指定时间点前20秒内的state统计
        返回: 1表示呼吸异常（平均值>30%），0表示正常
        """
        cutoff_time = end_timestamp - self.body_move_energy_window_duration
        
        # 获取20秒窗口内的state值
        window_body_move_energy = [state for timestamp, state in self.state_history 
                        if timestamp >= cutoff_time and timestamp <= end_timestamp]
        
        if not window_body_move_energy:
            return 0  # 没有数据时默认为正常
        
        # 计算state值的平均值
        body_move_energy_mean = sum(window_body_move_energy) / len(window_body_move_energy)
        
        # 根据阈值判断
        return 1 if body_move_energy_mean < self.mean_body_move_energy_threshold else 0
    
    
    def _validate_state_duration(self, raw_state: str, timestamp: float) -> str:
        """验证状态持续时间，避免误判"""
        
        # 如果候选状态与原始状态相同，继续累积时间
        if self.candidate_state == raw_state:
            duration = timestamp - self.candidate_start_time
            
            # 检查是否达到持续时间要求
            required_duration = self._get_required_duration(raw_state)
            
            if duration >= required_duration:
                # 达到要求，确认状态切换
                old_state = self.confirmed_state
                self.confirmed_state = raw_state
                self.candidate_state = None
                self.candidate_start_time = None
                self.candidate_start_index = None
                
                if old_state != self.confirmed_state:
                    print(f"       ☆ 状态确认: {old_state} -> {self.confirmed_state} (持续{duration:.1f}秒)")
                
                return self.confirmed_state
            else:
                # 还未达到要求，保持原状态
                return self.confirmed_state
        
        # 候选状态发生变化
        else:
            if raw_state != self.confirmed_state:
                # 开始新的候选状态
                self.candidate_state = raw_state
                self.candidate_start_time = timestamp
                self.candidate_start_index = self.total_sample_count
                return self.confirmed_state  # 保持当前确认状态
            else:
                # 回到确认状态，取消候选
                self.candidate_state = None
                self.candidate_start_time = None
                self.candidate_start_index = None
                return self.confirmed_state
    
    
    def _get_required_duration(self, state: str) -> float:
        """获取状态的最小持续时间要求"""
        duration_map = {
            "离床": self.off_bed_duration,
            "呼吸暂停": self.apnea_duration,
            "在床正常": self.normal_duration
        }
        return duration_map.get(state, 1.0)  # 默认1秒
    
    
    def _classify_base_state(self, value: float, timestamp: float) -> str:
        """基础状态分类 - 基于稳定性判断（修改版）"""
        # 离床判断保持原样（瞬时判断）
        if value < self.off_bed_threshold:
            return "离床"
        
        # 呼吸暂停判断 - 基于8秒稳定性
        if self._is_stable_in_apnea_range(timestamp):
            return "呼吸暂停"
        
        # 默认为在床正常
        return "在床正常"
    
    
    def _update_peak_detection(self, value: float, timestamp: float, state: int) -> Tuple[PeakState, Optional[PeakEvent]]:
        """峰值检测更新（加入state参数）"""
        # 判断是否应该激活峰值检测
        should_activate = self._should_activate(value)
        
        if not should_activate:
            # 数据被过滤，强制返回非激活状态
            self.is_active = False
            
            if self.current_peak_state in [PeakState.RISING, PeakState.PEAK, PeakState.FALLING]:
                event = self._force_complete_peak(timestamp, state)
                self.current_peak_state = PeakState.INACTIVE
                return self.current_peak_state, event
            else:
                self.current_peak_state = PeakState.INACTIVE
                return self.current_peak_state, None
        
        # 数据激活，进行正常检测
        self.is_active = True
        self.active_sample_count += 1
        
        # 更新激活数据的统计量
        self._update_active_statistics(value)
        
        # 状态转移
        new_state, event = self._update_peak_state(value, timestamp, state)
        
        return new_state, event
    
    
    def _classify_overall_state(self, value: float, validated_base_state: str, 
                              peak_state: PeakState, peak_event: Optional[PeakEvent]) -> str:
        """综合状态判断 - 使用验证后的基础状态和state值"""
        
        # 1. 已验证的基础状态优先（离床、呼吸暂停）
        if validated_base_state in ["离床", "呼吸暂停"]:
            return validated_base_state
        
        
        # 2. 峰值进行中的状态判断 - 新增
        if peak_state in [PeakState.RISING, PeakState.PEAK, PeakState.FALLING]:
            if self.current_peak_max > self.dynamic_threshold:  # 使用当前峰值的最大值判断
                # 根据当前的state统计值判断
                current_state_value = self._calculate_state_statistics(self.last_update_time)
                if current_state_value != 0:
                    return "呼吸急促"
                else:
                    return "体动"
        
        
        # 2. 只有峰值完成事件才能改变状态
        if peak_event and peak_event.event_type == "peak_completed":
            if peak_event.peak_value > self.dynamic_threshold:  # 修改：峰值大于20就先判断为体动
                # 根据峰值事件中保存的state值判断
                if peak_event.state_value != 0:
                    return "呼吸急促"
                else:
                    return "体动"
        
        # 3. 查看最近完成的峰值（缩短时间窗口）
        recent_peaks = self.get_recent_peaks(time_window=10.0)  # 从30秒缩短到10秒
        if recent_peaks:
            # 只考虑最近的一个峰值
            latest_peak = max(recent_peaks, key=lambda p: p.end_time)
            time_since_peak = self.last_update_time - latest_peak.end_time
            
            # 峰值完成后只保持5秒状态
            if time_since_peak <= 5.0:
                if latest_peak.peak_value > self.dynamic_threshold:  # 修改：峰值大于20就先判断为体动
                    # 根据峰值事件中保存的state值判断
                    if latest_peak.state_value != 0:
                        return "呼吸急促"
                    else:
                        return "体动"
        
        # 4. 默认使用验证后的基础状态
        return validated_base_state

    
    def _get_recent_state_stats(self) -> dict:
        """获取最近20秒的state统计信息"""
        if not self.state_history or not self.last_update_time:
            return {'total': 0, 'mean': 0.0, 'is_abnormal': False}
        
        cutoff_time = self.last_update_time - self.body_move_energy_window_duration
        recent_states = [state for timestamp, state in self.state_history 
                        if timestamp >= cutoff_time]
        
        if not recent_states:
            return {'total': 0, 'mean': 0.0, 'is_abnormal': False}
        
        total_count = len(recent_states)
        state_mean = sum(recent_states) / total_count
        
        return {
            'total': total_count,
            'mean': state_mean,
            'is_abnormal': state_mean < self.mean_body_move_energy_threshold
        }

    
    # 以下是峰值检测的内部方法（与之前的FilteredPeakDetector相同，但加入state参数）
    def _should_activate(self, value: float) -> bool:
        if not self.is_active:
            return value >= self.activation_threshold
        else:
            return value >= self.deactivation_threshold

    
    def _update_active_statistics(self, value: float):
        adjusted_value = max(value, self.min_baseline)
        # 基线更新逻辑
        if self.use_fixed_baseline:
            # 使用固定基线，不更新
            self.active_baseline = self.fixed_baseline_value
        else:
            if self.active_sample_count == 1:
                self.active_baseline = adjusted_value
            else:
                alpha = 0.05 if self.active_sample_count <= self.warmup_samples else self.baseline_alpha
                
                """限制基线的更新幅度"""
                if self.current_peak_state in [PeakState.RISING, PeakState.PEAK, PeakState.FALLING]:
                    # 峰值状态下有幅度更新基线
                    new_baseline = alpha * adjusted_value + (1 - alpha) * self.active_baseline
                    
                    # 🔑 关键：限制基线变化幅度
                    max_increase = self.active_baseline * 0.01  # 单次最多增加5%
                    max_decrease = self.active_baseline * 0.01  # 单次最多减少5%
                    
                    if new_baseline > self.active_baseline + max_increase:
                        new_baseline = self.active_baseline + max_increase
                    elif new_baseline < self.active_baseline - max_decrease:
                        new_baseline = self.active_baseline - max_decrease
                        
                    self.active_baseline = new_baseline
                else:
                    """非峰值状态正常更新"""
                    self.active_baseline = alpha * adjusted_value + (1 - alpha) * self.active_baseline
        
        error = adjusted_value - self.active_baseline
        if self.active_sample_count == 1:
            self.active_variance = 1.0
        else:
            beta = 0.4 if self.active_sample_count <= self.warmup_samples else self.variance_beta
            self.active_variance = beta * error**2 + (1 - beta) * self.active_variance
            
        self.active_variance = max(0.1, self.active_variance)

    
    def _get_thresholds(self):
        rise_threshold = self.active_baseline * self.rise_factor
        peak_threshold = self.active_baseline * self.peak_factor
        fall_threshold = self.active_baseline * self.fall_factor
        return rise_threshold, peak_threshold, fall_threshold

    
    def _update_peak_state(self, value: float, timestamp: float, state: int) -> Tuple[PeakState, Optional[PeakEvent]]:
        """更新峰值状态（加入state参数）"""
        if self.active_sample_count <= self.warmup_samples:
            return PeakState.WARMING_UP, None
        
        rise_threshold, peak_threshold, fall_threshold = self._get_thresholds()
        current_state = self.current_peak_state
        event = None
        
        if current_state == PeakState.WARMING_UP or current_state == PeakState.INACTIVE:
            self.current_peak_state = PeakState.BASELINE
            self.state_start_time = timestamp
            self.state_start_index = self.total_sample_count
            
        elif current_state == PeakState.BASELINE:
            if value > rise_threshold:
                self.current_peak_state = PeakState.RISING
                self.state_start_time = timestamp
                self.state_start_index = self.total_sample_count
                self.current_peak_max = value
                self.current_peak_start_time = timestamp
                self.current_peak_start_index = self.total_sample_count
                
        elif current_state == PeakState.RISING:
            self.current_peak_max = max(self.current_peak_max, value)
            
            if value > peak_threshold:
                self.current_peak_state = PeakState.PEAK
                self.state_start_time = timestamp
                self.state_start_index = self.total_sample_count
                self.current_peak_max_time = timestamp
                self.current_peak_max_index = self.total_sample_count
                
                event = PeakEvent(
                    event_type="peak_start",
                    peak_value=self.current_peak_max,
                    start_time=self.current_peak_start_time,
                    end_time=timestamp,
                    duration=timestamp - self.current_peak_start_time,
                    start_index=self.current_peak_start_index,
                    end_index=self.total_sample_count,
                    state_value=self._calculate_state_statistics(timestamp)  # 使用统计结果
                )
                
            elif value < fall_threshold:
                self.current_peak_state = PeakState.BASELINE
                self.state_start_time = timestamp
                self.state_start_index = self.total_sample_count
                self._reset_peak_info()
                
        elif current_state == PeakState.PEAK:
            if value > self.current_peak_max:
                self.current_peak_max = value
                self.current_peak_max_time = timestamp
                self.current_peak_max_index = self.total_sample_count
                
            if value < fall_threshold:
                self.current_peak_state = PeakState.FALLING
                self.state_start_time = timestamp
                self.state_start_index = self.total_sample_count
                
        elif current_state == PeakState.FALLING:
            if value < self.active_baseline * 1.2:
                peak_duration = timestamp - self.current_peak_start_time
                
                if (peak_duration >= self.min_peak_duration and 
                    self.current_peak_max >= self.min_peak_height):
                    
                    event = PeakEvent(
                        event_type="peak_completed",
                        peak_value=self.current_peak_max,
                        start_time=self.current_peak_start_time,
                        end_time=timestamp,
                        duration=peak_duration,
                        start_index=self.current_peak_start_index,
                        end_index=self.total_sample_count,
                        state_value=self._calculate_state_statistics(timestamp)  # 使用统计结果
                    )
                    
                    self.threshold_calculator.add_peak(self.current_peak_max)
                    if self.device_sn not in ["13D7F349200080712111150807", "13301C9D100040711117953507"]:
                        self.dynamic_threshold = self.threshold_calculator.calculate_threshold()
                    
                    self.peak_history.append(event)
                
                self.current_peak_state = PeakState.BASELINE
                self.state_start_time = timestamp
                self.state_start_index = self.total_sample_count
                self._reset_peak_info()
                
            elif value > peak_threshold:
                self.current_peak_state = PeakState.PEAK
                self.state_start_time = timestamp
                self.state_start_index = self.total_sample_count
                
                if value > self.current_peak_max:
                    self.current_peak_max = value
                    self.current_peak_max_time = timestamp
                    self.current_peak_max_index = self.total_sample_count
        
        return self.current_peak_state, event

    
    def _force_complete_peak(self, timestamp: float, state: int) -> Optional[PeakEvent]:
        """强制完成峰值检测（加入state参数）"""
        if (self.current_peak_start_time is not None and 
            self.current_peak_max >= self.min_peak_height):
            
            peak_duration = timestamp - self.current_peak_start_time
            
            event = PeakEvent(
                event_type="peak_completed",
                peak_value=self.current_peak_max,
                start_time=self.current_peak_start_time,
                end_time=timestamp,
                duration=peak_duration,
                start_index=self.current_peak_start_index,
                end_index=self.total_sample_count,
                state_value=self._calculate_state_statistics(timestamp)  # 使用统计结果
            )
            # 将峰值添加到阈值计算器
            self.threshold_calculator.add_peak(self.current_peak_max)
        
            # 更新动态阈值
            if self.device_sn not in ["13D7F349200080712111150807", "13301C9D100040711117953507"]:
                self.dynamic_threshold = self.threshold_calculator.calculate_threshold()
            self.peak_history.append(event)
            self._reset_peak_info()
            return event
        
        self._reset_peak_info()
        return None

    
    def _reset_peak_info(self):
        self.current_peak_max = 0.0
        self.current_peak_start_time = None
        self.current_peak_start_index = None
        self.current_peak_max_time = None
        self.current_peak_max_index = None

    
    def get_recent_peaks(self, time_window: float = 30.0) -> List[PeakEvent]:
        if not self.peak_history or self.last_update_time is None:
            return []
        
        cutoff_time = self.last_update_time - time_window
        return [peak for peak in self.peak_history if peak.end_time >= cutoff_time]

    
    @property
    def sample_count(self):
        return self.total_sample_count