import numpy as np
from collections import deque
from typing import Optional, Tuple, List
import math

class HighPeakBiasedThreshold:
    """
    偏向高峰值的阈值计算类
    
    核心改进：
    1. 强化对历史最高峰值的记忆和偏向
    2. 阈值更多地锚定在高峰值区间
    3. 低峰值只能缓慢温和地降低阈值
    4. 增加"峰值高度分层"概念
    """
    
    def __init__(self, 
                 short_memory_size: int = 10,
                 high_peak_memory_size: int = 15,      # 专门记忆高峰值
                 high_peak_percentile: float = 0.8,    # 高峰值判定阈值（更严格）
                 threshold_percentile: float = 0.6,    # 提高基础阈值百分位
                 peak_height_bias: float = 0.7,        # 高峰值偏向权重
                 upward_sensitivity: float = 0.9,      # 向上调整敏感度
                 downward_sensitivity: float = 0.15,   # 大幅降低向下调整敏感度
                 threshold_decay_rate: float = 0.98,   # 提高衰减率（更慢衰减）
                 min_threshold_ratio: float = 0.5,     # 提高最小阈值保护
                 max_threshold_ratio: float = 0.8,     # 添加最大阈值限制
                 patience_periods: int = 5):           # 增加耐心期
        """
        初始化偏向高峰值的阈值计算器
        """
        self.short_memory_size = short_memory_size
        self.high_peak_memory_size = high_peak_memory_size
        self.high_peak_percentile = high_peak_percentile
        self.threshold_percentile = threshold_percentile
        self.peak_height_bias = peak_height_bias
        self.upward_sensitivity = upward_sensitivity
        self.downward_sensitivity = downward_sensitivity
        self.threshold_decay_rate = threshold_decay_rate
        self.min_threshold_ratio = min_threshold_ratio
        self.max_threshold_ratio = max_threshold_ratio
        self.patience_periods = patience_periods
        
        # 短期记忆：最近的峰值
        self.short_memory = deque(maxlen=short_memory_size)
        
        # 高峰值专用记忆：只保留真正的高峰值
        self.high_peak_memory = deque(maxlen=high_peak_memory_size)
        
        # 所有峰值历史
        self.all_peaks = []
        
        # 阈值历史记录
        self.threshold_history = deque(maxlen=20)
        
        # 历史最高峰值（永久记忆）
        self.historical_max_peak = 0.0
        
        # 分层峰值统计
        self.peak_layers = {
            'ultra_high': [],  # 超高峰值
            'high': [],        # 高峰值
            'medium': [],      # 中等峰值
            'low': []          # 低峰值
        }
        
        # 小峰值连续计数器
        self.low_peak_counter = 0
        
        # 当前阈值缓存
        self.current_threshold = None
        
    def add_peak(self, peak_value: float, timestamp: Optional[float] = None) -> None:
        """添加新的峰值数据"""
        if peak_value <= 0:
            raise ValueError("峰值必须为正数")
            
        # 更新短期记忆
        self.short_memory.append(peak_value)
        
        # 更新所有峰值历史
        self.all_peaks.append(peak_value)
        
        # 更新历史最高峰值
        self.historical_max_peak = max(self.historical_max_peak, peak_value)
        
        # 更新高峰值记忆
        self._update_high_peak_memory(peak_value)
        
        # 更新峰值分层
        self._update_peak_layers()
        
        # 更新小峰值计数器
        self._update_low_peak_counter(peak_value)
    
    def _update_high_peak_memory(self, peak_value: float) -> None:
        """更新高峰值专用记忆"""
        if len(self.all_peaks) < 5:
            # 初期阶段，所有峰值都加入
            self.high_peak_memory.append(peak_value)
            return
        
        # 动态计算高峰值判定阈值
        historical_high_threshold = np.percentile(self.all_peaks, 
                                                self.high_peak_percentile * 100)
        
        # 只有真正的高峰值才能进入高峰值记忆
        if peak_value >= historical_high_threshold:
            self.high_peak_memory.append(peak_value)
            
        # 如果高峰值记忆为空，至少保留一些较高的峰值
        if len(self.high_peak_memory) == 0 and len(self.all_peaks) >= 5:
            top_peaks = sorted(self.all_peaks[-20:], reverse=True)[:5]
            for peak in top_peaks:
                if len(self.high_peak_memory) < self.high_peak_memory_size:
                    self.high_peak_memory.append(peak)
    
    def _update_peak_layers(self) -> None:
        """更新峰值分层"""
        if len(self.all_peaks) < 10:
            return
            
        recent_peaks = self.all_peaks[-30:]  # 最近30个峰值用于分层
        
        # 计算分层阈值
        p25 = np.percentile(recent_peaks, 25)
        p50 = np.percentile(recent_peaks, 50)
        p75 = np.percentile(recent_peaks, 75)
        p90 = np.percentile(recent_peaks, 90)
        
        # 清空旧分层
        for layer in self.peak_layers.values():
            layer.clear()
        
        # 重新分层
        for peak in recent_peaks[-10:]:  # 只保留最近10个峰值的分层
            if peak >= p90:
                self.peak_layers['ultra_high'].append(peak)
            elif peak >= p75:
                self.peak_layers['high'].append(peak)
            elif peak >= p50:
                self.peak_layers['medium'].append(peak)
            else:
                self.peak_layers['low'].append(peak)
    
    def _update_low_peak_counter(self, peak_value: float) -> None:
        """更新小峰值计数器"""
        if not self.high_peak_memory:
            self.low_peak_counter = 0
            return
            
        # 使用更严格的小峰值判定标准
        high_peak_mean = np.mean(self.high_peak_memory)
        is_low_peak = peak_value < high_peak_mean * 0.5  # 小于高峰值均值的50%
        
        if is_low_peak:
            self.low_peak_counter += 1
        else:
            self.low_peak_counter = max(0, self.low_peak_counter - 1)  # 缓慢重置
    
    def calculate_threshold(self) -> float:
        """计算当前时点的动态阈值"""
        # 高峰值偏向的基础阈值
        base_threshold = self._calculate_high_biased_base_threshold()
        
        # 高度感知调整
        height_awareness_adjustment = self._calculate_height_awareness_adjustment()
        
        # 防护性调整（更保守）
        protective_adjustment = self._calculate_conservative_protective_adjustment()
        
        # 计算候选阈值
        candidate_threshold = base_threshold * (1 + height_awareness_adjustment + protective_adjustment)
        
        # 应用强化的阈值保护机制
        final_threshold = self._apply_enhanced_threshold_protection(candidate_threshold)
        
        # 更新阈值历史
        self.threshold_history.append(final_threshold)
        self.current_threshold = final_threshold
        
        return final_threshold
    
    def _calculate_high_biased_base_threshold(self) -> float:
        """计算偏向高峰值的基础阈值"""
        if len(self.high_peak_memory) == 0:
            if len(self.all_peaks) >= 3:
                return np.percentile(self.all_peaks, 50)  # 提高到中位数
            else:
                return max(self.all_peaks) * 0.7 if self.all_peaks else 0.0
        
        # 使用高峰值记忆计算基础阈值
        high_peak_base = np.percentile(list(self.high_peak_memory), 
                                     self.threshold_percentile * 100)
        
        # 添加历史最高峰值的影响
        max_peak_influence = self.historical_max_peak * 0.3
        
        # 加权组合
        base_threshold = (high_peak_base * self.peak_height_bias + 
                         max_peak_influence * (1 - self.peak_height_bias))
        
        return base_threshold
    
    def _calculate_height_awareness_adjustment(self) -> float:
        """计算高度感知调整"""
        if len(self.short_memory) < 2 or not self.high_peak_memory:
            return 0.0
        
        recent_peak = list(self.short_memory)[-1]
        high_peak_mean = np.mean(self.high_peak_memory)
        
        # 计算相对高度
        relative_height = recent_peak / high_peak_mean if high_peak_mean > 0 else 0.0
        
        if relative_height > 0.8:
            # 接近高峰值，积极向上调整
            return 0.2 * (relative_height - 0.8) / 0.2
        elif relative_height < 0.3:
            # 明显低峰值，但调整要很保守
            return -0.05 * (0.3 - relative_height) / 0.3
        else:
            # 中等峰值，保持稳定
            return 0.0
    
    def _calculate_conservative_protective_adjustment(self) -> float:
        """计算保守的防护性调整"""
        if len(self.short_memory) < 2:
            return 0.0
        
        short_list = list(self.short_memory)
        short_mean = np.mean(short_list)
        
        if not self.high_peak_memory:
            return 0.0
        
        high_mean = np.mean(self.high_peak_memory)
        relative_level = (short_mean - high_mean) / high_mean if high_mean > 0 else 0.0
        
        # 更保守的调整策略
        if relative_level > 0:
            # 短期峰值较高，积极向上调整
            adjustment = relative_level * self.upward_sensitivity
            return np.clip(adjustment, 0, 0.5)
        else:
            # 短期峰值较低，极其保守的向下调整
            if self.low_peak_counter >= self.patience_periods:
                # 需要更多连续小峰值才开始降低
                patience_factor = min(0.5, (self.low_peak_counter - self.patience_periods) / 10.0)
                adjustment = relative_level * self.downward_sensitivity * patience_factor
                return np.clip(adjustment, -0.15, 0)  # 进一步限制下降幅度
            else:
                return 0.0
    
    def _apply_enhanced_threshold_protection(self, candidate_threshold: float) -> float:
        """应用增强的阈值保护机制"""
        # 1. 基于高峰值记忆的最小阈值保护
        if self.high_peak_memory:
            high_peak_mean = np.mean(self.high_peak_memory)
            min_threshold = high_peak_mean * self.min_threshold_ratio
            candidate_threshold = max(candidate_threshold, min_threshold)
        
        # 2. 基于历史最高峰值的最小阈值保护
        if self.historical_max_peak > 0:
            absolute_min = self.historical_max_peak * 0.3  # 不能低于历史最高的30%
            candidate_threshold = max(candidate_threshold, absolute_min)
        
        # 3. 最大阈值限制（避免过高）
        if self.high_peak_memory:
            high_peak_mean = np.mean(self.high_peak_memory)
            max_threshold = high_peak_mean * self.max_threshold_ratio
            candidate_threshold = min(candidate_threshold, max_threshold)
        
        # 4. 阈值下降速度限制（更严格）
        if self.current_threshold is not None:
            max_decrease_ratio = 0.05  # 单次最大下降5%
            min_allowed = self.current_threshold * (1 - max_decrease_ratio)
            candidate_threshold = max(candidate_threshold, min_allowed)
        
        # 5. 自然衰减（非常缓慢）
        if (self.current_threshold is not None and 
            candidate_threshold < self.current_threshold and
            self.low_peak_counter < self.patience_periods):
            natural_decay = self.current_threshold * self.threshold_decay_rate
            candidate_threshold = max(candidate_threshold, natural_decay)
        
        return candidate_threshold
    
    def get_detailed_statistics(self) -> dict:
        """获取详细统计信息"""
        stats = {
            'short_memory_count': len(self.short_memory),
            'short_memory_values': list(self.short_memory),
            'high_peak_memory_count': len(self.high_peak_memory),
            'high_peak_memory_values': list(self.high_peak_memory),
            'total_peaks_count': len(self.all_peaks),
            'historical_max_peak': self.historical_max_peak,
            'current_threshold': self.current_threshold or 0.0,
            'base_threshold': self._calculate_high_biased_base_threshold(),
            'low_peak_counter': self.low_peak_counter,
            'height_awareness_adj': self._calculate_height_awareness_adjustment(),
            'protective_adjustment': self._calculate_conservative_protective_adjustment(),
            'peak_layers': {k: len(v) for k, v in self.peak_layers.items()},
        }
        
        if self.short_memory:
            stats['short_term_mean'] = np.mean(self.short_memory)
            
        if self.high_peak_memory:
            stats['high_peak_mean'] = np.mean(self.high_peak_memory)
            stats['min_threshold_protection'] = np.mean(self.high_peak_memory) * self.min_threshold_ratio
            
        return stats
    
    def reset(self) -> None:
        """重置所有数据"""
        self.short_memory.clear()
        self.high_peak_memory.clear()
        self.all_peaks.clear()
        self.threshold_history.clear()
        self.historical_max_peak = 0.0
        for layer in self.peak_layers.values():
            layer.clear()
        self.low_peak_counter = 0
        self.current_threshold = None


# 测试对比
if __name__ == "__main__":
    print("高峰值偏向阈值计算器测试")
    print("=" * 70)
    
    # 创建偏向高峰值的阈值计算器
    high_biased_calculator = HighPeakBiasedThreshold(
        short_memory_size=8,
        high_peak_memory_size=12,
        high_peak_percentile=0.8,    # 只有前20%的峰值才算高峰值
        threshold_percentile=0.6,    # 基础阈值提高到60%分位
        peak_height_bias=0.8,        # 80%权重给高峰值
        upward_sensitivity=0.9,      # 对上升敏感
        downward_sensitivity=0.1,    # 对下降不敏感
        patience_periods=5,          # 需要5个连续小峰值
        min_threshold_ratio=0.5,     # 最小阈值为高峰值均值的50%
        max_threshold_ratio=0.8      # 最大阈值为高峰值均值的80%
    )
    
    # 模拟数据：包含大峰值后的连续小峰值
    sample_peaks = [
        200, 180, 160, 140, 350, 480, 520, 280, 300,  # 初始大峰值，包含一个超高峰值520
        120, 110, 100, 90, 80, 70, 60, 50,            # 连续小峰值
        40, 35, 30, 25, 20, 25, 30, 35,               # 继续小峰值
        45, 55, 65, 280, 320                          # 逐渐回升，最后是中高峰值
    ]
    
    print("峰值序列测试：观察阈值如何保持在高峰值区间")
    print("-" * 70)
    
    for i, peak in enumerate(sample_peaks):
        high_biased_calculator.add_peak(peak)
        current_threshold = high_biased_calculator.calculate_threshold()
        
        detection_status = "🔴 检测" if peak > current_threshold else "⚪ 正常"
        
        print(f"峰值 #{i+1:2d}: {peak:6.1f} -> 阈值: {current_threshold:6.1f} [{detection_status}]")
        
        # 每6个峰值显示详细信息
        if (i + 1) % 6 == 0:
            stats = high_biased_calculator.get_detailed_statistics()
            print(f"  📊 统计信息:")
            print(f"     历史最高峰值: {stats['historical_max_peak']:6.1f}")
            print(f"     高峰值记忆均值: {stats.get('high_peak_mean', 0):6.1f}")
            print(f"     最小阈值保护: {stats.get('min_threshold_protection', 0):6.1f}")
            print(f"     小峰值计数: {stats['low_peak_counter']}")
            print(f"     高度感知调整: {stats['height_awareness_adj']:+6.3f}")
            print(f"     防护性调整: {stats['protective_adjustment']:+6.3f}")
            print(f"     峰值分层: {stats['peak_layers']}")
            print()
    
    print("\n" + "="*70)
    print("测试完成！观察结果：")
    print("1. 阈值应该锚定在高峰值区间（接近350-520的范围）")
    print("2. 连续小峰值不应该显著拉低阈值")
    print("3. 阈值下降应该非常缓慢和保守")
    print("4. 当出现新的高峰值时，阈值应该快速向上调整")