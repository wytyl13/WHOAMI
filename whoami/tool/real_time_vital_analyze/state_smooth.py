#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/07/15 14:38
@Author  : weiyutao
@File    : state_smooth.py
"""

"""
改进的睡眠状态平滑器 - 解决清醒/浅睡眠频繁切换问题
"""

import time
from collections import deque, Counter
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class StateTransition:
    """状态转换规则"""
    from_state: str
    to_state: str
    min_duration: float
    confidence_threshold: float = 0.7

class EnhancedStateSmoother:
    """增强版状态平滑处理器 - 专门优化清醒/浅睡眠切换"""
    
    def __init__(self, 
                 window_size: int = 10,           # 增大窗口以获得更稳定的判断
                 min_state_duration: float = 60.0,  # 增加到60秒
                 anomaly_states: set = None,
                 normal_states: set = None,
                 confidence_threshold: float = 0.75,  # 提高置信度阈值
                 hysteresis_margin: float = 15.0):    # 添加滞后时间
        
        self.window_size = window_size
        self.min_state_duration = min_state_duration
        self.confidence_threshold = confidence_threshold
        self.hysteresis_margin = hysteresis_margin  # 滞后时间，防止快速切换
        
        self.anomaly_states = anomaly_states or {"呼吸暂停", "呼吸急促", "体动", "离床"}
        self.normal_states = normal_states or {"清醒", "浅睡眠", "深睡眠"}
        
        # 状态历史缓冲区
        self.normal_state_history: deque = deque(maxlen=window_size)
        self.all_state_history: deque = deque(maxlen=window_size * 2)  # 包含异常状态的完整历史
        
        # 当前状态管理
        self.current_confirmed_normal_state: Optional[str] = None
        self.current_normal_state_start_time: Optional[float] = None
        self.current_output_state: Optional[str] = None
        
        # 候选状态管理
        self.candidate_normal_state: Optional[str] = None
        self.candidate_normal_start_time: Optional[float] = None
        self.candidate_confidence: float = 0.0
        
        # 状态转换规则（特别针对清醒/浅睡眠）
        self.transition_rules = {
            ("清醒", "浅睡眠"): StateTransition("清醒", "浅睡眠", 90.0, 0.8),  # 更严格
            ("浅睡眠", "清醒"): StateTransition("浅睡眠", "清醒", 60.0, 0.7),   # 适中
            ("浅睡眠", "深睡眠"): StateTransition("浅睡眠", "深睡眠", 120.0, 0.8), # 更严格
            ("深睡眠", "浅睡眠"): StateTransition("深睡眠", "浅睡眠", 60.0, 0.7),
            ("深睡眠", "清醒"): StateTransition("深睡眠", "清醒", 45.0, 0.8),
            ("清醒", "深睡眠"): StateTransition("清醒", "深睡眠", 180.0, 0.9),   # 非常严格
        }
        
        # 最近切换时间跟踪（用于滞后控制）
        self.last_transition_time: Optional[float] = None
        
        print(f"增强版状态平滑器初始化:")
        print(f"  - 滑动窗口大小: {window_size}")
        print(f"  - 最小持续时间: {min_state_duration}秒")
        print(f"  - 置信度阈值: {confidence_threshold}")
        print(f"  - 滞后时间: {hysteresis_margin}秒")
        print(f"  - 状态转换规则: {len(self.transition_rules)}个")
    
    def _preprocess_state(self, raw_state: str) -> str:
        """预处理原始状态"""
        if raw_state == "在床正常":
            return "清醒"
        return raw_state
    
    def _get_transition_rule(self, from_state: str, to_state: str) -> Optional[StateTransition]:
        """获取状态转换规则"""
        return self.transition_rules.get((from_state, to_state))
    
    def _calculate_state_confidence(self, target_state: str) -> float:
        """计算状态置信度"""
        if len(self.normal_state_history) < 3:
            return 0.0
        
        recent_states = [state for state, _ in self.normal_state_history]
        state_counts = Counter(recent_states)
        
        # 基础置信度：目标状态在窗口中的比例
        base_confidence = state_counts[target_state] / len(recent_states)
        
        # 趋势置信度：检查最近的状态是否趋向目标状态
        recent_half = recent_states[len(recent_states)//2:]
        trend_confidence = recent_half.count(target_state) / len(recent_half)
        
        # 连续性置信度：检查目标状态的连续性
        continuity_score = self._calculate_continuity_score(recent_states, target_state)
        
        # 综合置信度
        final_confidence = (base_confidence * 0.4 + 
                          trend_confidence * 0.4 + 
                          continuity_score * 0.2)
        
        return final_confidence
    
    def _calculate_continuity_score(self, states: List[str], target_state: str) -> float:
        """计算状态连续性得分"""
        if not states:
            return 0.0
        
        # 找到目标状态的连续片段
        max_continuous = 0
        current_continuous = 0
        
        for state in reversed(states):  # 从最新状态往回看
            if state == target_state:
                current_continuous += 1
                max_continuous = max(max_continuous, current_continuous)
            else:
                current_continuous = 0
        
        return min(max_continuous / len(states), 1.0)
    
    def _is_in_hysteresis_period(self, timestamp: float) -> bool:
        """检查是否在滞后期内"""
        if self.last_transition_time is None:
            return False
        return (timestamp - self.last_transition_time) < self.hysteresis_margin
    
    def smooth_state(self, raw_state: str, timestamp: float) -> str:
        """增强版状态平滑处理"""
        
        # 预处理
        processed_state = self._preprocess_state(raw_state)
        
        # 记录所有状态历史
        self.all_state_history.append((processed_state, timestamp))
        
        # 异常状态立即响应
        if processed_state in self.anomaly_states:
            self.current_output_state = processed_state
            return processed_state
        
        # 检查是否是支持的正常状态
        if processed_state not in self.normal_states:
            output_state = self.current_confirmed_normal_state or processed_state
            self.current_output_state = output_state
            return output_state
        
        # 添加到正常状态历史
        self.normal_state_history.append((processed_state, timestamp))
        
        # 首次正常状态
        if self.current_confirmed_normal_state is None:
            self._confirm_normal_state(processed_state, timestamp)
            self.current_output_state = processed_state
            return processed_state
        
        # 状态没有变化
        if processed_state == self.current_confirmed_normal_state:
            # 重置候选状态
            if self.candidate_normal_state != processed_state:
                self._reset_candidate_state()
            self.current_output_state = processed_state
            return processed_state
        
        # 检查是否在滞后期内
        if self._is_in_hysteresis_period(timestamp):
            # 在滞后期内，保持当前状态
            self.current_output_state = self.current_confirmed_normal_state
            return self.current_confirmed_normal_state
        
        # 处理状态变化
        return self._handle_enhanced_state_change(processed_state, timestamp)
    
    def _handle_enhanced_state_change(self, new_state: str, timestamp: float) -> str:
        """增强版状态变化处理"""
        
        # 获取转换规则
        transition_rule = self._get_transition_rule(self.current_confirmed_normal_state, new_state)
        
        # 计算置信度
        confidence = self._calculate_state_confidence(new_state)
        
        # 更新候选状态
        if new_state != self.candidate_normal_state:
            self.candidate_normal_state = new_state
            self.candidate_normal_start_time = timestamp
            self.candidate_confidence = confidence
        else:
            # 更新置信度
            self.candidate_confidence = max(self.candidate_confidence, confidence)
        
        # 检查是否满足转换条件
        duration = timestamp - self.candidate_normal_start_time
        
        # 使用转换规则或默认规则
        if transition_rule:
            required_duration = transition_rule.min_duration
            required_confidence = transition_rule.confidence_threshold
        else:
            required_duration = self.min_state_duration
            required_confidence = self.confidence_threshold
        
        # 判断是否可以切换
        duration_ok = duration >= required_duration
        confidence_ok = self.candidate_confidence >= required_confidence
        
        if duration_ok and confidence_ok:
            # 确认状态切换
            self._confirm_normal_state(new_state, timestamp)
            self.last_transition_time = timestamp
            return new_state
        
        # 显示进度信息
        progress = min(duration / required_duration * 100, 100)
        confidence_progress = min(self.candidate_confidence / required_confidence * 100, 100)
        
        print(f"候选状态 '{new_state}': 时间{duration:.1f}s/{required_duration}s ({progress:.1f}%), "
              f"置信度{self.candidate_confidence:.2f}/{required_confidence:.2f} ({confidence_progress:.1f}%)")
        
        # 保持当前状态
        return self.current_confirmed_normal_state
    
    def _confirm_normal_state(self, normal_state: str, timestamp: float):
        """确认正常状态切换"""
        old_state = self.current_confirmed_normal_state
        self.current_confirmed_normal_state = normal_state
        self.current_normal_state_start_time = timestamp
        
        # 重置候选状态
        self._reset_candidate_state()
        
        if old_state != normal_state:
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            print(f"✅ [状态切换] {old_state} -> {normal_state} ({time_str})")
    
    def _reset_candidate_state(self):
        """重置候选状态"""
        self.candidate_normal_state = None
        self.candidate_normal_start_time = None
        self.candidate_confidence = 0.0
    
    def get_detailed_state_info(self) -> Dict:
        """获取详细状态信息"""
        current_time = time.time()
        
        info = {
            "current_confirmed_normal_state": self.current_confirmed_normal_state,
            "candidate_normal_state": self.candidate_normal_state,
            "candidate_confidence": self.candidate_confidence,
            "current_output_state": self.current_output_state,
            "normal_state_history": list(self.normal_state_history),
            "all_state_history": list(self.all_state_history),
            "is_in_hysteresis": self._is_in_hysteresis_period(current_time),
            "transition_rules": {str(k): v.__dict__ for k, v in self.transition_rules.items()},
        }
        
        if self.current_normal_state_start_time:
            info["current_normal_state_duration"] = current_time - self.current_normal_state_start_time
        
        if self.candidate_normal_start_time:
            info["candidate_normal_duration"] = current_time - self.candidate_normal_start_time
        
        if self.last_transition_time:
            info["time_since_last_transition"] = current_time - self.last_transition_time
        
        return info
    
    def analyze_state_stability(self, time_window: float = 300.0) -> Dict:
        """分析状态稳定性"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # 过滤最近的状态
        recent_states = [(state, ts) for state, ts in self.all_state_history if ts > cutoff_time]
        
        if not recent_states:
            return {"stability_score": 0.0, "transition_count": 0}
        
        # 计算转换次数
        transitions = 0
        prev_state = None
        
        for state, _ in recent_states:
            if prev_state and state != prev_state and state in self.normal_states:
                transitions += 1
            prev_state = state
        
        # 计算稳定性得分
        max_transitions = len(recent_states) // 2  # 理论最大转换次数
        stability_score = 1.0 - (transitions / max(max_transitions, 1))
        
        return {
            "stability_score": stability_score,
            "transition_count": transitions,
            "time_window": time_window,
            "recent_states_count": len(recent_states)
        }


# 使用示例
if __name__ == "__main__":
    # 创建增强版平滑器
    smoother = EnhancedStateSmoother(
        window_size=12,
        min_state_duration=60.0,
        confidence_threshold=0.75,
        hysteresis_margin=20.0
    )
    
    # 模拟频繁切换的场景
    test_states = [
        # 模拟清醒/浅睡眠频繁切换
        (0, "清醒"),
        (10, "浅睡眠"),      # 短暂跳变
        (20, "清醒"),
        (30, "浅睡眠"),      # 又一次短暂跳变
        (40, "清醒"),
        (50, "清醒"),
        (60, "清醒"),
        (70, "浅睡眠"),      # 开始真正的转换
        (80, "浅睡眠"),
        (90, "浅睡眠"),
        (100, "浅睡眠"),
        (110, "浅睡眠"),
        (120, "清醒"),       # 短暂跳变，应该被过滤
        (130, "浅睡眠"),
        (140, "浅睡眠"),
        (150, "浅睡眠"),
        (160, "浅睡眠"),     # 应该确认为浅睡眠
        (170, "深睡眠"),     # 开始向深睡眠转换
        (180, "深睡眠"),
        (190, "深睡眠"),
        (200, "深睡眠"),
    ]
    
    base_time = time.time()
    
    print("测试增强版状态平滑器 - 解决频繁切换问题")
    print("="*80)
    
    for i, (offset, raw_state) in enumerate(test_states):
        timestamp = base_time + offset
        
        print(f"\n🔸 测试 #{i+1} (时间+{offset}s)")
        smoothed = smoother.smooth_state(raw_state, timestamp)
        
        # 分析稳定性
        stability = smoother.analyze_state_stability(180.0)
        
        print(f"   输入: {raw_state} -> 输出: {smoothed}")
        print(f"   稳定性得分: {stability['stability_score']:.2f}")
        print(f"   转换次数: {stability['transition_count']}")
    
    print("\n" + "="*80)
    print("测试完成")
    
    # 最终分析
    final_info = smoother.get_detailed_state_info()
    stability_analysis = smoother.analyze_state_stability(300.0)
    
    print(f"\n📊 最终状态分析:")
    print(f"当前确认状态: {final_info['current_confirmed_normal_state']}")
    print(f"候选状态: {final_info['candidate_normal_state']}")
    print(f"候选置信度: {final_info['candidate_confidence']:.2f}")
    print(f"整体稳定性: {stability_analysis['stability_score']:.2f}")
    print(f"总转换次数: {stability_analysis['transition_count']}")
    
    print(f"\n🎯 关键改进:")
    print(f"✅ 针对清醒/浅睡眠转换使用更严格的时间要求(90s)")
    print(f"✅ 引入置信度机制，基于状态连续性和趋势")
    print(f"✅ 添加滞后时间机制，防止快速回跳")
    print(f"✅ 不同状态转换使用不同的验证标准")
    print(f"✅ 提供详细的稳定性分析")