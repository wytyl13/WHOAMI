import numpy as np
from collections import deque
from typing import Optional, Tuple, List
import math

class RobustDualMemoryThreshold:
    """
    健壮的双记忆阈值计算类
    
    核心改进：
    1. 防止小峰值严重拉低阈值
    2. 阈值具有下降惯性，只能缓慢降低
    3. 区分上升调整和下降调整的敏感度
    4. 增加阈值最小值保护
    """
    
    def __init__(self, 
                 short_memory_size: int = 10,
                 long_memory_percentile: float = 0.7,
                 threshold_percentile: float = 0.3,
                 upward_sensitivity: float = 0.8,      # 向上调整敏感度（对大峰值敏感）
                 downward_sensitivity: float = 0.2,    # 向下调整敏感度（对小峰值不敏感）
                 threshold_decay_rate: float = 0.95,   # 阈值自然衰减率
                 min_threshold_ratio: float = 0.3,     # 阈值最小值为长期均值的比例
                 min_long_samples: int = 5,
                 patience_periods: int = 3):           # 连续小峰值才开始缓慢降低阈值
        """
        初始化健壮的双记忆阈值计算器
        """
        self.short_memory_size = short_memory_size
        self.long_memory_percentile = long_memory_percentile
        self.threshold_percentile = threshold_percentile
        self.upward_sensitivity = upward_sensitivity
        self.downward_sensitivity = downward_sensitivity
        self.threshold_decay_rate = threshold_decay_rate
        self.min_threshold_ratio = min_threshold_ratio
        self.min_long_samples = min_long_samples
        self.patience_periods = patience_periods
        
        # 短期记忆：最近的峰值
        self.short_memory = deque(maxlen=short_memory_size)
        
        # 长期记忆：只保留显著峰值
        self.significant_peaks = []
        
        # 所有峰值历史
        self.all_peaks = []
        
        # 阈值历史记录
        self.threshold_history = deque(maxlen=20)
        
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
        
        # 更新长期显著峰值记忆
        self._update_significant_peaks()
        
        # 更新小峰值计数器
        self._update_low_peak_counter(peak_value)
    
    def _update_significant_peaks(self) -> None:
        """更新长期显著峰值记忆"""
        if len(self.all_peaks) < 5:
            self.significant_peaks = self.all_peaks.copy()
            return
        
        historical_threshold = np.percentile(self.all_peaks, 
                                           self.long_memory_percentile * 100)
        
        self.significant_peaks = [peak for peak in self.all_peaks 
                                if peak >= historical_threshold]
        
        if len(self.significant_peaks) > self.min_long_samples * 2:
            decay_length = len(self.significant_peaks) // 2
            self.significant_peaks = self.significant_peaks[-decay_length:]
    
    def _update_low_peak_counter(self, peak_value: float) -> None:
        """更新小峰值计数器"""
        if not self.significant_peaks:
            self.low_peak_counter = 0
            return
            
        # 判断当前峰值是否为"小峰值"
        significant_mean = np.mean(self.significant_peaks)
        is_low_peak = peak_value < significant_mean * 0.6  # 小于显著峰值均值的60%
        
        if is_low_peak:
            self.low_peak_counter += 1
        else:
            self.low_peak_counter = 0  # 重置计数器
    
    def calculate_threshold(self) -> float:
        """计算当前时点的动态阈值"""
        # 基础阈值
        base_threshold = self._calculate_base_threshold()
        
        # 防护性调整（重点改进）
        protective_adjustment = self._calculate_protective_adjustment()
        
        # 趋势调整
        trend_adjustment = self._calculate_trend_adjustment()
        
        # 计算候选阈值
        candidate_threshold = base_threshold * (1 + protective_adjustment + trend_adjustment)
        
        # 应用阈值保护机制
        final_threshold = self._apply_threshold_protection(candidate_threshold)
        
        # 更新阈值历史
        self.threshold_history.append(final_threshold)
        self.current_threshold = final_threshold
        
        return final_threshold
    
    def _calculate_base_threshold(self) -> float:
        """计算基础阈值"""
        if len(self.significant_peaks) < self.min_long_samples:
            if len(self.all_peaks) >= 3:
                return np.percentile(self.all_peaks, 25)
            else:
                return min(self.all_peaks) * 0.5 if self.all_peaks else 0.0
        
        return np.percentile(self.significant_peaks, 
                           self.threshold_percentile * 100)
    
    def _calculate_protective_adjustment(self) -> float:
        """计算防护性调整（关键改进）"""
        if len(self.short_memory) < 2:
            return 0.0
        
        short_list = list(self.short_memory)
        short_mean = np.mean(short_list)
        
        if not self.significant_peaks:
            return 0.0
        
        long_mean = np.mean(self.significant_peaks)
        relative_level = (short_mean - long_mean) / long_mean if long_mean > 0 else 0.0
        
        # 关键改进：区分上升和下降调整
        if relative_level > 0:
            # 短期峰值较高，积极向上调整
            adjustment = relative_level * self.upward_sensitivity
            return np.clip(adjustment, 0, 1.0)
        else:
            # 短期峰值较低，消极向下调整
            # 只有连续多个小峰值才开始缓慢降低阈值
            if self.low_peak_counter >= self.patience_periods:
                # 计算耐心衰减：连续小峰值越多，衰减越明显
                patience_factor = min(1.0, (self.low_peak_counter - self.patience_periods) / 5.0)
                adjustment = relative_level * self.downward_sensitivity * patience_factor
                return np.clip(adjustment, -0.3, 0)  # 限制下降幅度
            else:
                # 不足够的连续小峰值，不进行下降调整
                return 0.0
    
    def _calculate_trend_adjustment(self) -> float:
        """计算趋势调整"""
        if len(self.threshold_history) < 3:
            return 0.0
        
        recent_thresholds = list(self.threshold_history)[-5:]
        if len(recent_thresholds) < 3:
            return 0.0
        
        threshold_trend = self._calculate_sequence_trend(recent_thresholds)
        
        # 更保守的趋势调整
        if len(self.short_memory) > 0:
            current_peak = list(self.short_memory)[-1]
            if self.significant_peaks:
                peak_significance = current_peak / np.mean(self.significant_peaks)
                if threshold_trend < -0.1 and peak_significance > 0.8:
                    return 0.15  # 适度向上调整
                elif threshold_trend > 0.1 and peak_significance < 0.2:
                    return -0.05  # 轻微向下调整
        
        return 0.0
    
    def _apply_threshold_protection(self, candidate_threshold: float) -> float:
        """应用阈值保护机制"""
        # 1. 最小阈值保护
        if self.significant_peaks:
            min_threshold = np.mean(self.significant_peaks) * self.min_threshold_ratio
            candidate_threshold = max(candidate_threshold, min_threshold)
        
        # 2. 阈值下降速度限制
        if self.current_threshold is not None:
            max_decrease_ratio = 0.1  # 单次最大下降10%
            min_allowed = self.current_threshold * (1 - max_decrease_ratio)
            candidate_threshold = max(candidate_threshold, min_allowed)
        
        # 3. 自然衰减（缓慢下降）
        if (self.current_threshold is not None and 
            candidate_threshold < self.current_threshold and
            self.low_peak_counter < self.patience_periods):
            # 在没有足够小峰值时，应用自然衰减而非急剧下降
            natural_decay = self.current_threshold * self.threshold_decay_rate
            candidate_threshold = max(candidate_threshold, natural_decay)
        
        return candidate_threshold
    
    def _calculate_sequence_trend(self, values: List[float]) -> float:
        """计算序列趋势"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        if n * np.sum(x**2) - np.sum(x)**2 == 0:
            return 0.0
            
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        mean_value = np.mean(values)
        if mean_value > 0:
            return slope / mean_value
        else:
            return 0.0
    
    def get_detailed_statistics(self) -> dict:
        """获取详细统计信息"""
        stats = {
            'short_memory_count': len(self.short_memory),
            'short_memory_values': list(self.short_memory),
            'significant_peaks_count': len(self.significant_peaks),
            'significant_peaks_values': self.significant_peaks[-10:] if self.significant_peaks else [],
            'total_peaks_count': len(self.all_peaks),
            'current_threshold': self.current_threshold or 0.0,
            'base_threshold': self._calculate_base_threshold(),
            'low_peak_counter': self.low_peak_counter,
            'protective_adjustment': self._calculate_protective_adjustment(),
        }
        
        if self.short_memory:
            stats['short_term_mean'] = np.mean(self.short_memory)
            
        if self.significant_peaks:
            stats['significant_peaks_mean'] = np.mean(self.significant_peaks)
            stats['min_threshold_protection'] = np.mean(self.significant_peaks) * self.min_threshold_ratio
            
        return stats
    
    def reset(self) -> None:
        """重置所有数据"""
        self.short_memory.clear()
        self.significant_peaks.clear()
        self.all_peaks.clear()
        self.threshold_history.clear()
        self.low_peak_counter = 0
        self.current_threshold = None


# 测试对比
if __name__ == "__main__":
    print("健壮阈值计算器测试 - 防止小峰值拉低阈值")
    print("=" * 70)
    
    # 创建健壮的阈值计算器
    robust_calculator = RobustDualMemoryThreshold(
        short_memory_size=8,
        long_memory_percentile=0.6,
        threshold_percentile=0.4,
        upward_sensitivity=0.8,     # 对大峰值敏感
        downward_sensitivity=0.2,   # 对小峰值不敏感
        patience_periods=3,         # 需要3个连续小峰值才开始降低阈值
        min_threshold_ratio=0.3     # 阈值最小值保护
    )
    
    # 模拟数据：包含大峰值后的连续小峰值
    sample_peaks = [
        180, 170, 160, 140, 340, 450, 220, 280,  # 初始大峰值
        120, 110, 100, 90, 80, 70, 60, 50,       # 连续小峰值（测试重点）
        40, 35, 30, 25, 20, 25, 30, 250          # 继续小峰值，最后一个大峰值
    ]
    
    for i, peak in enumerate(sample_peaks):
        robust_calculator.add_peak(peak)
        current_threshold = robust_calculator.calculate_threshold()
        
        detection_status = "🔴 检测" if peak > current_threshold else "⚪ 正常"
        
        print(f"峰值 #{i+1:2d}: {peak:6.1f} -> 阈值: {current_threshold:6.1f} [{detection_status}]")
        
        # 显示关键信息
        if (i + 1) % 6 == 0:
            stats = robust_calculator.get_detailed_statistics()
            print(f"  📊 统计:")
            print(f"     小峰值计数: {stats['low_peak_counter']}, "
                  f"保护性调整: {stats['protective_adjustment']:+6.3f}")
            print(f"     短期均值: {stats.get('short_term_mean', 0):6.1f}, "
                  f"显著峰值均值: {stats.get('significant_peaks_mean', 0):6.1f}")
            print(f"     最小阈值保护: {stats.get('min_threshold_protection', 0):6.1f}")
            print()