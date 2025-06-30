#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/25 09:52
@Author  : weiyutao
@File    : peak_state.py
实时状态监测 - 每个时间点都输出状态
结合基础状态判断和峰值状态判断
"""


import math
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List





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
        off_bed_threshold: float = 0.1,        # 离床阈值
        apnea_threshold: float = 0.5,          # 呼吸暂停上限阈值
        
        # 状态持续时间要求
        off_bed_duration: float = 5.0,        # 离床状态需要持续20秒
        apnea_duration: float = 20.0,          # 呼吸暂停需要持续20秒
        normal_duration: float = 45.0,          # 在床正常需要持续45秒
        
        # 峰值检测参数
        activation_threshold: float = 1.0,     # 峰值检测激活阈值
        deactivation_threshold: float = 0.5,   # 峰值检测去激活阈值
        min_baseline: float = 0.1,             # 最小基线值
        baseline_alpha: float = 0.1,           # 基线适应速度
        variance_beta: float = 0.2,            # 方差适应速度
        rise_factor: float = 1.5,              # 上升倍数
        peak_factor: float = 2.0,              # 峰值倍数
        fall_factor: float = 1.3,              # 下降倍数
        min_peak_duration: float = 2.0,        # 最小峰值持续时间(秒)
        min_peak_height: float = 5.0,          # 最小绝对峰值高度
        warmup_samples: int = 20
    ):             # 冷启动样本数
        
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
        self.baseline_alpha = baseline_alpha
        self.variance_beta = variance_beta
        self.rise_factor = rise_factor
        self.peak_factor = peak_factor
        self.fall_factor = fall_factor
        self.min_peak_duration = min_peak_duration
        self.min_peak_height = min_peak_height
        self.warmup_samples = warmup_samples
        
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
        self.current_state_value = 0              # 当前传入的state值
        
        # state历史缓冲区（用于20秒窗口统计）
        self.state_history = []                   # [(timestamp, state_value), ...]
        self.state_window_duration = 20.0         # 20秒窗口
        self.abnormal_state_threshold = 0.05       # 30%阈值
        
    def update(self, value: float, timestamp: Optional[float] = None, state: int = 0) -> Tuple[str, str, Optional[PeakEvent]]:
        """
        更新状态监测器
        
        Args:
            value: 传感器数值
            timestamp: 时间戳
            state: 当前时间点的状态值，用于峰值状态判断
        
        Returns:
            (总体状态, 峰值状态, 峰值事件或None)
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.total_sample_count += 1
        self.current_state_value = state  # 保存当前state值
        
        # 更新state历史缓冲区
        self._update_state_history(timestamp, state)
        
        # 1. 基础状态判断（基于当前值）
        raw_base_state = self._classify_base_state(value)
        
        # 2. 峰值状态检测
        peak_state, peak_event = self._update_peak_detection(value, timestamp, state)
        
        # 3. 状态持续时间验证
        validated_base_state = self._validate_state_duration(raw_base_state, timestamp)
        
        # 4. 综合状态判断
        overall_state = self._classify_overall_state(value, validated_base_state, peak_state, peak_event)
        
        # 5. 更新缓存
        self.last_update_time = timestamp
        
        return overall_state, peak_state.value, peak_event
    
    def _update_state_history(self, timestamp: float, state: int):
        """更新state历史缓冲区，维护20秒窗口"""
        # 添加当前state到历史
        self.state_history.append((timestamp, state))
        
        # 清理超过20秒的历史数据
        cutoff_time = timestamp - self.state_window_duration
        self.state_history = [(t, s) for t, s in self.state_history if t >= cutoff_time]
    
    def _calculate_state_statistics(self, end_timestamp: float) -> int:
        """
        计算指定时间点前20秒内的state统计
        返回: 1表示呼吸异常（非0概率>30%），0表示正常
        """
        cutoff_time = end_timestamp - self.state_window_duration
        
        # 获取20秒窗口内的state值
        window_states = [state for timestamp, state in self.state_history 
                        if timestamp >= cutoff_time and timestamp <= end_timestamp]
        
        if not window_states:
            return 0  # 没有数据时默认为正常
        
        # 计算非0 state的比例
        non_zero_count = sum(1 for state in window_states if state != 0)
        total_count = len(window_states)
        non_zero_ratio = non_zero_count / total_count
        
        # 根据阈值判断
        return 1 if non_zero_ratio > self.abnormal_state_threshold else 0
    
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
    
    def _classify_base_state(self, value: float) -> str:
        """基础状态分类"""
        if value < self.off_bed_threshold:
            return "离床"
        elif self.off_bed_threshold <= value < self.apnea_threshold:
            return "呼吸暂停"
        else:
            return "在床正常"  # 基础在床状态
    
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
        
        # 2. 只有峰值完成事件才能改变状态
        if peak_event and peak_event.event_type == "peak_completed":
            if peak_event.peak_value > 20:  # 修改：峰值大于20就先判断为体动
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
                if latest_peak.peak_value > 20:  # 修改：峰值大于20就先判断为体动
                    # 根据峰值事件中保存的state值判断
                    if latest_peak.state_value != 0:
                        return "呼吸急促"
                    else:
                        return "体动"
        
        # 4. 默认使用验证后的基础状态
        return validated_base_state
    
    def get_debug_info(self) -> dict:
        """获取调试信息，包括状态验证详情"""
        debug_info = {
            'confirmed_state': self.confirmed_state,
            'candidate_state': self.candidate_state,
            'candidate_duration': 0.0,
            'required_duration': 0.0,
            'current_state_value': self.current_state_value,  # 添加当前state值到调试信息
            'state_history_length': len(self.state_history),   # state历史长度
            'recent_state_stats': self._get_recent_state_stats()  # 最近的state统计
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
    
    def _get_recent_state_stats(self) -> dict:
        """获取最近20秒的state统计信息"""
        if not self.state_history or not self.last_update_time:
            return {'total': 0, 'non_zero': 0, 'ratio': 0.0}
        
        cutoff_time = self.last_update_time - self.state_window_duration
        recent_states = [state for timestamp, state in self.state_history 
                        if timestamp >= cutoff_time]
        
        if not recent_states:
            return {'total': 0, 'non_zero': 0, 'ratio': 0.0}
        
        total_count = len(recent_states)
        non_zero_count = sum(1 for state in recent_states if state != 0)
        ratio = non_zero_count / total_count
        
        return {
            'total': total_count,
            'non_zero': non_zero_count, 
            'ratio': ratio,
            'is_abnormal': ratio > self.abnormal_state_threshold
        }
    
    # 以下是峰值检测的内部方法（与之前的FilteredPeakDetector相同，但加入state参数）
    def _should_activate(self, value: float) -> bool:
        if not self.is_active:
            return value >= self.activation_threshold
        else:
            return value >= self.deactivation_threshold
    
    def _update_active_statistics(self, value: float):
        adjusted_value = max(value, self.min_baseline)
        
        if self.active_sample_count == 1:
            self.active_baseline = adjusted_value
        else:
            alpha = 0.3 if self.active_sample_count <= self.warmup_samples else self.baseline_alpha
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






    
    
    
    
    
# 使用示例
if __name__ == "__main__":
    # 创建实时状态监测器
    monitor = RealTimeStateMonitor(
        off_bed_threshold=0.1,          # 离床阈值
        apnea_threshold=0.5,            # 呼吸暂停阈值
        activation_threshold=1.0,       # 峰值检测激活阈值
        rise_factor=1.5,
        peak_factor=2.0,
        min_peak_duration=1.0,
        min_peak_height=5.0
    )
    
    from whoami.tool.real_time_vital_analyze.json_loader import SimpleJSONLoader
    from whoami.utils.utils import Utils

    utils = Utils()
    loader = SimpleJSONLoader("/work/ai/WHOAMI/whoami/out/data_logs/realtime_data_20250624.json")

    fields = loader.get_all_fields()
    all_data = loader.to_numpy_array()
    timestamps = loader.get_timestamps()

    test_data = all_data[:, 1]
    
    print("开始实时状态监测...")
    print("时间\t误差值\t\t总体状态\t\t峰值状态")
    print("-" * 80)
    
    overall_states = []
    for i, value in enumerate(test_data):
        timestamp = i * 1.0
        
        # 更新状态
        overall_state, peak_state, peak_event = monitor.update(value, timestamp)
        overall_states.append((value, overall_state))
        # 每10秒打印一次状态（或者你可以改为每1秒）
        # if i % 10 == 0 or peak_event:
        #     print(f"{timestamp:6.0f}\t{value:8.4f}\t{overall_state:15s}\t{peak_state}")
            
        #     # 如果有峰值事件，额外打印详细信息
        #     if peak_event and peak_event.event_type == "peak_completed":
        #         print(f"       ★ 峰值完成: {peak_event.peak_value:.2f} "
        #               f"({peak_event.start_time:.0f}-{peak_event.end_time:.0f}秒, "
        #               f"持续{peak_event.duration:.1f}秒)")
    
    print(f"\n监测完成:")
    print(f"总样本数: {monitor.sample_count}")
    print(f"检测到峰值数: {len(monitor.peak_history)}")
    print(f"当前总体状态: {monitor.current_peak_state}")
    print(f"当前峰值状态: {monitor.current_peak_state.value}")  
    print(overall_states)