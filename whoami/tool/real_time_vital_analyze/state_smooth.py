#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/28 14:03
@Author  : weiyutao
@File    : state_smooth.py
"""

import time
from collections import deque, Counter
from typing import Optional, List, Dict
from dataclasses import dataclass

from whoami.tool.real_time_vital_analyze.sleep_data_state import SleepDataState

@dataclass
class StateTransition:
    """状态转换规则"""
    from_state: str
    to_state: str
    min_duration: float  # 最小持续时间(秒)

class StateSmoother:
    """状态平滑处理器 - 睡眠监测专用版本"""
    
    def __init__(self, 
                 window_size: int = 10,           # 滑动窗口大小（增大以适应60秒要求）
                 min_state_duration: float = 60.0,  # 正常状态最小持续时间60秒
                 anomaly_states: set = None,      # 异常状态集合
                 normal_states: set = None):      # 正常状态集合
        
        self.window_size = window_size
        self.min_state_duration = min_state_duration
        
        # 更新异常状态集合，包含离床
        self.anomaly_states = anomaly_states or {"呼吸暂停", "呼吸急促", "体动", "离床"}
        
        # 正常状态集合（只有这些状态会被平滑处理）
        self.normal_states = normal_states or {"清醒", "浅睡眠", "深睡眠"}
        
        # 正常状态历史缓冲区（只存储正常状态，用于平滑）
        self.normal_state_history: deque = deque(maxlen=window_size)
        
        # 当前确认的正常状态（异常状态不会改变这个值）
        self.current_confirmed_normal_state: Optional[str] = None
        self.current_normal_state_start_time: Optional[float] = None
        
        # 候选正常状态（等待确认的正常状态）
        self.candidate_normal_state: Optional[str] = None
        self.candidate_normal_start_time: Optional[float] = None
        
        # 当前实际输出状态（可能是正常状态或异常状态）
        self.current_output_state: Optional[str] = None
        
        print(f"状态平滑器初始化:")
        print(f"  - 滑动窗口大小: {window_size}")
        print(f"  - 正常状态最小持续时间: {min_state_duration}秒")
        print(f"  - 异常状态 (立即响应): {self.anomaly_states}")
        print(f"  - 正常状态 (需要平滑): {self.normal_states}")
        print(f"  - 预处理规则: '在床正常' -> '清醒'")
        print(f"  - 关键特性: 异常状态不影响正常状态平滑")
    
    def _preprocess_state(self, raw_state: str) -> str:
        """
        预处理原始状态
        将"在床正常"转换为"清醒"
        """
        if raw_state == "在床正常":
            print(f"    🔄 预处理: '{raw_state}' -> '清醒'")
            return "清醒"
        return raw_state
    
    def smooth_state(self, raw_state: str, timestamp: float) -> str:
        """
        对原始状态进行平滑处理
        
        Args:
            raw_state: 原始检测到的状态
            timestamp: 时间戳
            
        Returns:
            平滑后的状态
        """
        
        # 打印输入信息
        time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
        # print(f"\n📥 输入: 原始状态='{raw_state}', 时间={time_str}")
        
        # 1. 预处理：将"在床正常"改为"清醒"
        processed_state = self._preprocess_state(raw_state)
        
        # print(f"    当前确认正常状态='{self.current_confirmed_normal_state}', 候选正常状态='{self.candidate_normal_state}'")
        # print(f"    当前输出状态='{self.current_output_state}'")
        
        # 2. 异常状态立即响应（不影响正常状态平滑逻辑）
        if processed_state in self.anomaly_states:
            # print(f"    🚨 判断: 这是异常状态，立即响应且不影响正常状态平滑")
            self.current_output_state = processed_state
            # print(f"📤 输出: 平滑后状态='{processed_state}' (异常立即响应)")
            return processed_state
        
        # 3. 检查是否是支持的正常状态
        if processed_state not in self.normal_states:
            # print(f"    ⚠️  警告: 未知状态'{processed_state}'，保持当前正常状态")
            output_state = self.current_confirmed_normal_state or processed_state
            self.current_output_state = output_state
            # print(f"📤 输出: 平滑后状态='{output_state}' (未知状态)")
            return output_state
        
        # 4. 处理正常状态（添加到正常状态历史）
        self.normal_state_history.append((processed_state, timestamp))
        
        # 5. 首次正常状态
        if self.current_confirmed_normal_state is None:
            # print(f"    ✅ 判断: 这是首次正常状态，直接确认")
            self._confirm_normal_state(processed_state, timestamp)
            self.current_output_state = processed_state
            # print(f"📤 输出: 平滑后状态='{processed_state}' (首次正常状态)")
            return processed_state
        
        # 6. 正常状态没有变化
        if processed_state == self.current_confirmed_normal_state:
            # print(f"    ✅ 判断: 正常状态与当前确认状态相同，保持不变")
            # 重置候选状态，因为又回到了当前确认状态
            if self.candidate_normal_state != processed_state:
                # print(f"    🔄 重置候选正常状态 (从'{self.candidate_normal_state}'重置为None)")
                self.candidate_normal_state = None
                self.candidate_normal_start_time = None
            self.current_output_state = processed_state
            # print(f"📤 输出: 平滑后状态='{processed_state}' (正常状态未变)")
            return processed_state
        
        # 7. 检测到新的正常状态变化，进行平滑处理
        if processed_state != self.current_confirmed_normal_state:
            # print(f"    🔄 判断: 检测到新的正常状态，开始平滑处理")
            result = self._handle_normal_state_change(processed_state, timestamp)
            self.current_output_state = result
            # print(f"📤 输出: 平滑后状态='{result}' (经过正常状态平滑处理)")
            return result
        
        # 默认保持当前确认的正常状态
        self.current_output_state = self.current_confirmed_normal_state
        # print(f"📤 输出: 平滑后状态='{self.current_confirmed_normal_state}' (默认保持)")
        return self.current_confirmed_normal_state
    
    def _handle_normal_state_change(self, new_normal_state: str, timestamp: float) -> str:
        """处理正常状态变化（需要平滑）"""
        
        # print(f"      🔍 开始正常状态平滑分析: '{self.current_confirmed_normal_state}' -> '{new_normal_state}'")
        
        # 方法1: 滑动窗口投票（只使用正常状态历史）
        smoothed_by_voting = self._smooth_by_voting()
        # print(f"      📊 投票法结果: '{smoothed_by_voting}'")
        
        # 方法2: 持续时间验证
        smoothed_by_duration = self._smooth_by_duration(new_normal_state, timestamp)
        # print(f"      ⏱️  持续时间验证结果: '{smoothed_by_duration}'")
        
        # 综合决策：如果两种方法都支持新状态，则切换
        if smoothed_by_voting == new_normal_state and smoothed_by_duration == new_normal_state:
            # print(f"      ✅ 综合决策: 投票通过 + 持续时间达标 -> 确认切换")
            self._confirm_normal_state(new_normal_state, timestamp)
            return new_normal_state
        
        # 显示当前候选状态的进度
        if self.candidate_normal_state == new_normal_state and self.candidate_normal_start_time:
            duration = timestamp - self.candidate_normal_start_time
            progress = duration / self.min_state_duration * 100
            # print(f"      ⏳ 候选正常状态进度: '{new_normal_state}' 已持续 {duration:.1f}s / {self.min_state_duration}s ({progress:.1f}%)")
        
        # print(f"      ❌ 综合决策: 条件未满足 -> 保持当前正常状态 '{self.current_confirmed_normal_state}'")
        return self.current_confirmed_normal_state
    
    def _smooth_by_voting(self) -> str:
        """滑动窗口投票法（只对正常状态进行投票）"""
        if len(self.normal_state_history) < 2:
            # print(f"        📊 投票法: 正常状态历史数据不足({len(self.normal_state_history)}个) -> 保持当前状态")
            return self.current_confirmed_normal_state
        
        # 统计正常状态的出现次数
        recent_normal_states = [state for state, _ in self.normal_state_history]
        state_counts = Counter(recent_normal_states)
        
        # print(f"        📊 投票法: 最近{len(recent_normal_states)}个正常状态 = {recent_normal_states}")
        # print(f"        📊 投票统计: {dict(state_counts)}")
        
        # 获取出现最多的正常状态
        most_common_state = state_counts.most_common(1)[0][0]
        
        # 需要达到一定比例才切换（提高到70%以配合60秒要求）
        total_count = len(recent_normal_states)
        most_common_count = state_counts[most_common_state]
        percentage = most_common_count / total_count * 100
        
        # print(f"        📊 最多状态: '{most_common_state}' 出现{most_common_count}/{total_count}次 ({percentage:.1f}%)")
        
        if most_common_count / total_count >= 0.7:
            # print(f"        📊 投票结果: 达到70%阈值 -> 支持'{most_common_state}'")
            return most_common_state
        
        # print(f"        📊 投票结果: 未达到70%阈值 -> 保持'{self.current_confirmed_normal_state}'")
        return self.current_confirmed_normal_state
    
    def _smooth_by_duration(self, new_normal_state: str, timestamp: float) -> str:
        """持续时间验证法（只对正常状态验证）"""
        
        # 如果候选正常状态发生变化，重新开始计时
        if new_normal_state != self.candidate_normal_state:
            if self.candidate_normal_state is not None:
                old_duration = timestamp - self.candidate_normal_start_time if self.candidate_normal_start_time else 0
                # print(f"        ⏱️  持续时间: 候选正常状态变更 '{self.candidate_normal_state}'({old_duration:.1f}s) -> '{new_normal_state}'(重新计时)")
            else:
                # print(f"        ⏱️  持续时间: 设置新候选正常状态 '{new_normal_state}' 开始计时")
                pass
            
            self.candidate_normal_state = new_normal_state
            self.candidate_normal_start_time = timestamp
            # print(f"        ⏱️  验证结果: 重新计时 -> 保持'{self.current_confirmed_normal_state}'")
            return self.current_confirmed_normal_state
        
        # 检查候选正常状态是否持续足够长时间
        if self.candidate_normal_start_time is not None:
            duration = timestamp - self.candidate_normal_start_time
            # print(f"        ⏱️  持续时间: 候选正常状态'{new_normal_state}' 已持续 {duration:.1f}s / {self.min_state_duration}s")
            
            if duration >= self.min_state_duration:
                # print(f"        ⏱️  验证结果: 持续时间达标 -> 支持'{new_normal_state}'")
                return new_normal_state
            else:
                # print(f"        ⏱️  验证结果: 持续时间不足 -> 保持'{self.current_confirmed_normal_state}'")
                pass
        return self.current_confirmed_normal_state
    
    def _confirm_normal_state(self, normal_state: str, timestamp: float):
        """确认正常状态切换"""
        old_normal_state = self.current_confirmed_normal_state
        self.current_confirmed_normal_state = normal_state
        self.current_normal_state_start_time = timestamp
        
        # 重置候选正常状态
        self.candidate_normal_state = None
        self.candidate_normal_start_time = None
        
        if old_normal_state != normal_state:
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            # print(f"      ✅ [正常状态切换] {old_normal_state} -> {normal_state} ({time_str})")
    
    def get_state_info(self) -> Dict:
        """获取当前状态信息"""
        current_time = time.time()
        
        info = {
            "current_confirmed_normal_state": self.current_confirmed_normal_state,
            "candidate_normal_state": self.candidate_normal_state,
            "current_output_state": self.current_output_state,
            "normal_state_history": list(self.normal_state_history),
            "anomaly_states": list(self.anomaly_states),
            "normal_states": list(self.normal_states),
        }
        
        if self.current_normal_state_start_time:
            info["current_normal_state_duration"] = current_time - self.current_normal_state_start_time
        
        if self.candidate_normal_start_time:
            info["candidate_normal_duration"] = current_time - self.candidate_normal_start_time
        
        return info
    
    def get_progress_info(self) -> str:
        """获取当前正常状态切换进度信息"""
        if self.candidate_normal_state and self.candidate_normal_start_time:
            current_time = time.time()
            duration = current_time - self.candidate_normal_start_time
            progress = min(duration / self.min_state_duration * 100, 100)
            return f"候选正常状态 '{self.candidate_normal_state}': {duration:.1f}s / {self.min_state_duration}s ({progress:.1f}%)"
        return "无候选正常状态"


# 修改后的添加数据点方法
def add_data_point_with_smoothing(self, 
                                 device_id: str,
                                 timestamp: float,
                                 breath_bpm: float,
                                 breath_line: float,
                                 heart_bpm: float,
                                 heart_line: float,
                                 reconstruction_error: float,
                                 state: str):
    """添加新的数据点 - 包含状态平滑（60秒最小持续时间）"""
    
    # 如果还没有状态平滑器，创建一个（使用60秒配置）
    if not hasattr(self, 'state_smoother'):
        self.state_smoother = StateSmoother(
            window_size=10,          # 增大窗口
            min_state_duration=60.0, # 60秒最小持续时间
            anomaly_states={"呼吸暂停", "呼吸急促", "体动", "离床"},  # 包含离床
            normal_states={"清醒", "浅睡眠", "深睡眠"}  # 只有这些状态需要平滑
        )
    
    # 对原始状态进行平滑处理
    smoothed_state = self.state_smoother.smooth_state(state, timestamp)
    
    if state != smoothed_state:
        # print(f"🔄 状态平滑: {state} -> {smoothed_state}")
        pass
    
    # 创建数据点（使用平滑后的状态）
    data_point = SleepDataState(
        device_id=device_id,
        timestamp=timestamp,
        breath_bpm=breath_bpm,
        breath_line=breath_line,
        heart_bpm=heart_bpm,
        heart_line=heart_line,
        reconstruction_error=reconstruction_error,
        state=smoothed_state  # 使用平滑后的状态
    )
    
    # 1. 添加到缓冲区
    self.data_buffer.append(data_point)
    self._clean_buffer(timestamp)
    
    # 2. 检查是否需要存储
    should_store, reason = self._should_store(data_point)
    
    if should_store:
        if reason == "首次异常":
            self._store_anomaly_context(data_point)
            self.context_stored = True
        elif reason == "持续异常":
            self._store_single_data(data_point, reason)
        elif reason == "异常结束":
            self._store_single_data(data_point, reason)
            self.context_stored = False
        else:
            self._store_single_data(data_point, reason)
        
        self.last_stored_data = data_point
        self.last_storage_time = timestamp
    
    # 3. 异常状态跟踪
    self._track_anomaly_state(data_point)


# 使用示例和测试
if __name__ == "__main__":
    # 测试修改后的状态平滑器
    print("初始化修改后的状态平滑器...")
    smoother = StateSmoother(
        window_size=10, 
        min_state_duration=60.0,
        anomaly_states={"呼吸暂停", "呼吸急促", "体动", "离床"},
        normal_states={"清醒", "浅睡眠", "深睡眠"}
    )
    
    # 模拟真实的睡眠状态序列（测试异常状态不影响正常状态平滑）
    test_states = [
        # 前30秒：在床正常，应该被转换为清醒
        (0, "在床正常"),    # 应该被转换为"清醒"
        (10, "在床正常"),   # 应该被转换为"清醒"
        (20, "浅睡眠"),     # 短暂跳变到浅睡眠（不够60秒）
        (30, "在床正常"),   # 应该被转换为"清醒"
        
        # 40秒：异常状态
        (40, "呼吸暂停"),   # 异常状态，立即响应
        (50, "在床正常"),   # 回到"清醒"，应该继续之前的清醒状态
        
        # 60-120秒：持续切换到浅睡眠
        (60, "浅睡眠"),      
        (70, "浅睡眠"),
        (80, "体动"),       # 异常状态插入，不应该影响浅睡眠的平滑
        (90, "浅睡眠"),
        (100, "浅睡眠"),
        (110, "浅睡眠"),
        (120, "浅睡眠"),    # 持续60秒（除去异常），应该确认切换
        
        # 130秒：另一个异常
        (130, "离床"),      # 异常状态，立即响应
        (140, "浅睡眠"),    # 回到浅睡眠，应该保持浅睡眠状态
        
        # 150-210秒：切换到深睡眠
        (150, "深睡眠"),
        (160, "深睡眠"),
        (170, "呼吸急促"),  # 异常状态插入
        (180, "深睡眠"),
        (190, "深睡眠"),
        (200, "深睡眠"),
        (210, "深睡眠"),    # 持续60秒（除去异常），应确认切换到深睡眠
        
        # 220秒：回到清醒
        (220, "在床正常"),  # 应该转换为清醒，开始新的候选状态
        (230, "清醒"),
        (240, "清醒"),
    ]
    
    base_time = time.time()
    
    print("\n" + "="*80)
    print("开始测试 - 异常状态不影响正常状态平滑")
    print("关键测试点：")
    print("1. '在床正常' -> '清醒' 预处理")
    print("2. 异常状态立即响应：呼吸暂停、呼吸急促、体动、离床")
    print("3. 异常状态不影响正常状态的平滑进度")
    print("4. 从异常状态回到正常状态时，基于之前的正常状态进行平滑")
    print("="*80)
    
    for i, (offset, raw_state) in enumerate(test_states):
        timestamp = base_time + offset
        
        print(f"\n🔸 测试 #{i+1} (时间+{offset}s)")
        print("-"*50)
        
        # 调用状态平滑
        smoothed = smoother.smooth_state(raw_state, timestamp)
        
        # 获取当前状态信息
        state_info = smoother.get_state_info()
        
        # 总结这次处理的结果
        if raw_state != smoothed:
            if raw_state == "在床正常":
                result_type = "预处理转换"
            else:
                result_type = "状态被平滑过滤"
        elif raw_state in smoother.anomaly_states:
            result_type = "异常立即响应"
        elif smoother.current_confirmed_normal_state == smoothed:
            result_type = "状态保持不变"
        else:
            result_type = "状态成功切换"
        
        print(f"🎯 本次处理结果: {result_type}")
        print(f"   当前确认正常状态: {state_info['current_confirmed_normal_state']}")
        print(f"   当前输出状态: {state_info['current_output_state']}")
        if state_info['candidate_normal_state']:
            print(f"   候选正常状态: {state_info['candidate_normal_state']}")
        print("-"*50)
    
    print(f"\n" + "="*80)
    print("测试完成")
    print("="*80)
    
    # 最终状态
    final_info = smoother.get_state_info()
    print(f"最终确认正常状态: {final_info['current_confirmed_normal_state']}")
    print(f"最终输出状态: {final_info['current_output_state']}")
    print(f"最终候选正常状态: {final_info['candidate_normal_state']}")
    print(f"异常状态集合: {final_info['anomaly_states']}")
    print(f"正常状态集合: {final_info['normal_states']}")
    
    print("\n📋 关键改进总结:")
    print("✅ '在床正常' 自动转换为 '清醒'")
    print("✅ 异常状态立即响应且不影响正常状态平滑")
    print("✅ 维护独立的正常状态历史记录用于平滑")
    print("✅ 从异常状态回到正常状态时基于之前的正常状态进行判断")
    print("🔍 短暂的正常状态跳变被有效过滤")