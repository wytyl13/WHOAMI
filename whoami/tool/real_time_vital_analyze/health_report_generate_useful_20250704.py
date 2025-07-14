#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/28 14:35
@Author  : weiyutao
@File    : health_report_generate.py
"""
from datetime import datetime
from typing import (
    Dict,
    List,
    Any,
    Tuple
)
import statistics
import json
import pytz


from whoami.provider.sql_provider import SqlProvider
from whoami.tool.real_time_vital_analyze.sleep_data_state import SleepDataState
from whoami.tool.real_time_vital_analyze.sleep_statistics_model import SleepStatistics

sql_provider = SqlProvider(
    model=SleepDataState, 
    sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml",
)
sql_provider_sleep_statistic = SqlProvider(
    model=SleepStatistics, 
    sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml",
)

class HealthReportGenerate:
    def __init__(self, start_date, end_date, device_sn):
        self.valid_states = ['清醒', '浅睡眠', '深睡眠', '离床', '呼吸急促', '呼吸暂停', '体动']
        self.device_sn = device_sn
        self.start_date = start_date
        self.end_date = end_date
        
        # 方法1: 尝试传入时间戳字符串
        start_timestamp = self._date_to_timestamp(start_date)
        end_timestamp = self._date_to_timestamp(end_date)
        print(start_timestamp, end_timestamp)

        try:
            self.sql_data = sql_provider.get_record_by_condition(
                condition={
                    "device_id": self.device_sn,
                    "timestamp": {"min": start_timestamp, "max": end_timestamp}
                },
                fields=["timestamp", "breath_bpm", "breath_line", "heart_bpm", "heart_line", "state"]
            )
        except Exception as e:
            print(f"   错误: {e}")
    
    
    def _date_to_timestamp(self, date_str):
        """将日期字符串转换为Unix时间戳（上海时区）"""
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        
        # 明确指定为上海时区
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        dt_with_tz = shanghai_tz.localize(dt)
        
        return int(dt_with_tz.timestamp())
    
    
    # def _date_to_timestamp(self, date_str):
    #     """将日期字符串转换为Unix时间戳"""
    #     dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    #     return int(dt.timestamp())
    
    def _seconds_to_time_format(self, seconds: float) -> str:
        """将秒数转换为 X小时Y分Z秒 格式"""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60
        
        return f"{hours}小时{minutes}分{remaining_seconds}秒"
    
    def validate_data(self) -> bool:
        """验证数据完整性"""
        if not self.sql_data:
            print("❌ 没有数据需要分析")
            return False
            
        required_fields = ['timestamp', 'breath_bpm', 'heart_bpm', 'state']
        
        for i, record in enumerate(self.sql_data):
            for field in required_fields:
                if field not in record:
                    print(f"❌ 第{i+1}条记录缺少字段: {field}")
                    return False
                    
            if record['state'] not in self.valid_states:
                print(f"⚠️  第{i+1}条记录包含未知状态: {record['state']}")
                
        print("✅ 数据验证通过")
        return True
    
    def calculate_basic_metrics(self) -> Dict:
        """计算基础生理指标"""
        breath_rates = [record['breath_bpm'] for record in self.sql_data]
        heart_rates = [record['heart_bpm'] for record in self.sql_data]
        
        # 计算心率变异性系数 (标准差/平均值)
        heart_rate_mean = statistics.mean(heart_rates)
        heart_rate_std = statistics.stdev(heart_rates) if len(heart_rates) > 1 else 0
        heart_rate_cv = (heart_rate_std / heart_rate_mean) if heart_rate_mean != 0 else 0
        
        return {
            'avg_breath_bpm': round(statistics.mean(breath_rates)),  # 整数
            'avg_heart_bpm': round(statistics.mean(heart_rates)),    # 整数
            'heart_rate_variability': round(heart_rate_cv, 4)        # 保留4位小数的变异系数
        }
    
    def calculate_state_statistics(self) -> Dict:
        """计算状态统计信息"""
        state_counts = {state: 0 for state in self.valid_states}
        state_durations_seconds = {state: 0.0 for state in self.valid_states}
        state_changes = {state: 0 for state in self.valid_states}
        
        # 计算状态出现次数
        for record in self.sql_data:
            state_counts[record['state']] += 1
            
        # 计算状态变化次数
        previous_state = None
        for record in self.sql_data:
            current_state = record['state']
            if previous_state is not None and previous_state != current_state:
                state_changes[current_state] += 1
            elif previous_state is None:
                state_changes[current_state] += 1
            previous_state = current_state
            
        # 计算各状态持续时长（秒）
        for i in range(len(self.sql_data) - 1):
            current_state = self.sql_data[i]['state']
            current_time = self.sql_data[i]['timestamp']
            next_time = self.sql_data[i + 1]['timestamp']
            duration = next_time - current_time
            state_durations_seconds[current_state] += duration
            
        # 处理最后一条记录（假设持续30秒）
        if self.sql_data:
            last_state = self.sql_data[-1]['state']
            state_durations_seconds[last_state] += 30
            
        return {
            'state_durations_seconds': state_durations_seconds,
            'state_changes': state_changes
        }
    
    def calculate_time_metrics(self) -> Dict:
        """计算时间相关指标"""
        if len(self.sql_data) < 2:
            return {}
            
        start_time = self.sql_data[0]['timestamp']
        end_time = self.sql_data[-1]['timestamp']
        total_duration_seconds = end_time - start_time + 30  # 加上最后一个状态的假设持续时间
        
        state_stats = self.calculate_state_statistics()
        state_durations_seconds = state_stats['state_durations_seconds']
        
        # 计算在床和离床时长（秒）
        on_bed_duration_seconds = sum(duration for state, duration in state_durations_seconds.items() if state != '离床')
        off_bed_duration_seconds = state_durations_seconds['离床']
        
        return {
            'total_duration_seconds': total_duration_seconds,
            'on_bed_duration_seconds': on_bed_duration_seconds,
            'off_bed_duration_seconds': off_bed_duration_seconds,
            'awake_duration_seconds': state_durations_seconds['清醒'],
            'light_sleep_duration_seconds': state_durations_seconds['浅睡眠'],
            'deep_sleep_duration_seconds': state_durations_seconds['深睡眠']
        }
    
    def generate_comprehensive_report(self) -> Dict:
        """生成简化的分析报告，只返回用户需要的指标"""
        if not self.validate_data():
            return {}
            
        basic_metrics = self.calculate_basic_metrics()
        state_stats = self.calculate_state_statistics()
        time_metrics = self.calculate_time_metrics()
        
        # 获取状态变化次数
        state_changes = state_stats['state_changes']
        
        
            
        # id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
        # device_sn = Column(String(64), nullable=False, comment='设备序列号')
        # sleep_start_time = Column(DateTime, nullable=False, comment='睡眠区间起始时间')
        # sleep_end_time = Column(DateTime, nullable=False, comment='睡眠区间终止时间')
        
        # # 生理指标
        # avg_breath_rate = Column(Float, nullable=True, comment='平均呼吸率')
        # avg_heart_rate = Column(Float, nullable=True, comment='平均心率')
        # heart_rate_variability = Column(Float, nullable=True, comment='心率变异性系数')
        
        # # 行为统计
        # body_movement_count = Column(Integer, nullable=True, comment='体动次数')
        # apnea_count = Column(Integer, nullable=True, comment='呼吸暂停次数')
        # rapid_breathing_count = Column(Integer, nullable=True, comment='呼吸急促次数')
        # leave_bed_count = Column(Integer, nullable=True, comment='离床次数')
        
        # # 时长统计 (以秒为单位存储)
        # total_duration = Column(Integer, nullable=True, comment='统计总时长(秒)')
        # in_bed_duration = Column(Integer, nullable=True, comment='在床时长(秒)')
        # out_bed_duration = Column(Integer, nullable=True, comment='离床时长(秒)')
        # deep_sleep_duration = Column(Integer, nullable=True, comment='深睡眠时长(秒)')
        # light_sleep_duration = Column(Integer, nullable=True, comment='浅睡眠时长(秒)')
        # awake_duration = Column(Integer, nullable=True, comment='清醒时长(秒)')
        
        # # 系统字段
        # creator = Column(String(64), nullable=True, comment='创建者')
        # create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
        # updater = Column(String(64), nullable=True, comment='更新者')
        # update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')
        # deleted = Column(BINARY(1), default=b'0', nullable=True, comment='是否删除')
        # tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')
        
        
        
        # 构建最终的简化报告
        report = {
            # 基础生理指标（整数）
            'avg_breath_rate': basic_metrics['avg_breath_bpm'],
            'avg_heart_rate': basic_metrics['avg_heart_bpm'],
            'heart_rate_variability': basic_metrics['heart_rate_variability'],
            
            # 状态变化次数（整数）
            'body_movement_count': state_changes.get('体动', 0),
            'apnea_count': state_changes.get('呼吸暂停', 0),
            'rapid_breathing_count': state_changes.get('呼吸急促', 0),
            'leave_bed_count': state_changes.get('离床', 0),
            
            # 时长指标（X小时Y分Z秒格式）
            'total_duration': self._seconds_to_time_format(time_metrics.get('total_duration_seconds', 0)),
            'in_bed_duration': self._seconds_to_time_format(time_metrics.get('on_bed_duration_seconds', 0)),
            'out_bed_duration': self._seconds_to_time_format(time_metrics.get('off_bed_duration_seconds', 0)),
            'deep_sleep_duration': self._seconds_to_time_format(time_metrics.get('deep_sleep_duration_seconds', 0)),
            'light_sleep_duration': self._seconds_to_time_format(time_metrics.get('light_sleep_duration_seconds', 0)),
            'awake_duration': self._seconds_to_time_format(time_metrics.get('awake_duration_seconds', 0))
        }
        report["device_sn"] = self.device_sn
        report["sleep_start_time"] = self.start_date
        report["sleep_end_time"] = self.end_date
        sql_provider_sleep_statistic.add_record(
            data = report
        )
        return report

if __name__ == '__main__':
    health_reprot_generate = HealthReportGenerate(
        start_date="2025-7-3 21:00:00", 
        end_date="2025-7-4 07:00:00", 
        device_sn="13D2F34920008071211195A907"
    )
    # health_reprot_generate = HealthReportGenerate(
    #     start_date="2025-7-3 12:00:00", 
    #     end_date="2025-7-3 14:00:00", 
    #     device_sn="13d7f349200080712111150807"
    # )
    
    # 生成简化报告
    report = health_reprot_generate.generate_comprehensive_report()
    print("简化报告数据:")
    print(report)
    