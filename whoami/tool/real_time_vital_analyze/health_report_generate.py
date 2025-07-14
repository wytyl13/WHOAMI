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
import asyncio

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

from whoami.tool.agent.tool.real_time_health_report_advice import RealTimeHealthReportAdvice
from whoami.configs.llm_config import LLMConfig
from whoami.llm_api.ollama_llm import OllamaLLM
from pathlib import Path
from whoami.tool.agent.tool.enhance_retrieval import EnhanceRetrieval

llm_qwen = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))
enhance_qwen = EnhanceRetrieval(llm=llm_qwen)
health_report_tool = RealTimeHealthReportAdvice(enhance_llm=enhance_qwen)


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
    
    
    def calculate_time_points(self) -> Dict:
        """计算关键时间点：上床时间、入睡时间、醒来时间、离床时间"""
        if not self.sql_data:
            return {}
        
        # 获取统计开始和结束时间戳
        start_timestamp = self._date_to_timestamp(self.start_date)
        end_timestamp = self._date_to_timestamp(self.end_date)
        
        # 默认时间：统计结束时间+1分钟
        default_time = datetime.fromtimestamp(end_timestamp + 60).strftime("%Y-%m-%d %H:%M:%S")
        
        # 在床状态：除了'离床'以外的所有状态
        in_bed_states = ['清醒', '浅睡眠', '深睡眠', '呼吸急促', '呼吸暂停', '体动']
        sleep_states = ['浅睡眠', '深睡眠']
        
        # 初始化结果
        bed_time = default_time        # 上床时间
        sleep_time = default_time      # 入睡时间
        wake_time = default_time       # 醒来时间
        leave_bed_time = default_time  # 离床时间
        
        # 1. 计算上床时间：统计起始时间之后的首次在床时间
        for record in self.sql_data:
            if record['state'] in in_bed_states:
                bed_time = datetime.fromtimestamp(record['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                break
        
        # 2. 计算入睡时间：统计起始时间之后的第一次浅睡眠或深睡眠时间
        for record in self.sql_data:
            if record['state'] in sleep_states:
                sleep_time = datetime.fromtimestamp(record['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                break
        
        # 3. 计算醒来时间：最后一次从睡眠状态转换到非睡眠状态的时间点
        wake_time = default_time  # 默认值
        last_wake_timestamp = None
        
        # 从前往后遍历，记录所有醒来时间点，最后取最后一个
        for i in range(len(self.sql_data) - 1):
            current_record = self.sql_data[i]
            next_record = self.sql_data[i + 1]
            
            current_state = current_record['state']
            next_state = next_record['state']
            
            # 如果当前状态是睡眠状态，下一个状态不是睡眠状态，说明这是一次醒来
            if current_state in sleep_states and next_state not in sleep_states:
                last_wake_timestamp = next_record['timestamp']
        
        # 如果找到了醒来时间点，使用它
        if last_wake_timestamp:
            wake_time = datetime.fromtimestamp(last_wake_timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # 特殊情况：如果统计结束时间正处于睡眠状态，则醒来时间为统计结束时间+1分
        if self.sql_data and self.sql_data[-1]['state'] in sleep_states:
            wake_time = default_time
        
        # 4. 修正后的离床时间计算：找到最后一次从非离床状态转换到离床状态的时间点
        last_leave_bed_start_timestamp = None
        
        # 特殊情况：如果第一条记录就是离床状态，记录这个时间点
        if self.sql_data and self.sql_data[0]['state'] == '离床':
            last_leave_bed_start_timestamp = self.sql_data[0]['timestamp']
        
        # 从前往后遍历，找到所有从非离床状态转换到离床状态的时间点
        for i in range(len(self.sql_data) - 1):
            current_record = self.sql_data[i]
            next_record = self.sql_data[i + 1]
            
            current_state = current_record['state']
            next_state = next_record['state']
            
            # 如果当前状态不是离床，下一个状态是离床，说明这是一次离床开始
            if current_state != '离床' and next_state == '离床':
                last_leave_bed_start_timestamp = next_record['timestamp']
        
        # 如果找到了离床开始时间点，使用它
        if last_leave_bed_start_timestamp:
            leave_bed_time = datetime.fromtimestamp(last_leave_bed_start_timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # 特殊情况：如果统计结束时间正处于在床状态，则离床时间为统计结束时间+1分
        if self.sql_data and self.sql_data[-1]['state'] in in_bed_states:
            leave_bed_time = default_time
        
        return {
            'bed_time': bed_time,
            'sleep_time': sleep_time,
            'wake_time': wake_time,
            'leave_bed_time': leave_bed_time
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
        
        # 特殊处理离床次数：排除边界情况
        if self.sql_data:
            # 排除第一条记录是离床状态的情况
            if self.sql_data[0]['state'] == '离床':
                state_changes['离床'] -= 1
                
            # 排除最后一条记录是离床状态的情况
            if self.sql_data[-1]['state'] == '离床':
                state_changes['离床'] -= 1
                
            # 确保离床次数不为负数
            state_changes['离床'] = max(0, state_changes['离床'])
            
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
    
    
    async def health_report_generate_tool(self, report):
            result = await health_report_tool.execute(
                health_report_statistics=report
            )
            return result
    
    
    def generate_comprehensive_report(self) -> Dict:
        """生成简化的分析报告，只返回用户需要的指标"""
        if not self.validate_data():
            return {}
            
        basic_metrics = self.calculate_basic_metrics()
        state_stats = self.calculate_state_statistics()
        time_metrics = self.calculate_time_metrics()
        time_points = self.calculate_time_points()  # 新添加的时间点计算
        
        # 获取状态变化次数
        state_changes = state_stats['state_changes']
        
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
            'awake_duration': self._seconds_to_time_format(time_metrics.get('awake_duration_seconds', 0)),
            
            # 新添加的时间点指标
            'bed_time': time_points.get('bed_time', ''),           # 上床时间
            'sleep_time': time_points.get('sleep_time', ''),       # 入睡时间
            'wake_time': time_points.get('wake_time', ''),         # 醒来时间
            'leave_bed_time': time_points.get('leave_bed_time', '') # 离床时间
        }
        
        report["device_sn"] = self.device_sn
        report["sleep_start_time"] = self.start_date
        report["sleep_end_time"] = self.end_date
        report["health_report"] = asyncio.run(self.health_report_generate_tool(str(report)))
        
        sql_provider_sleep_statistic.add_record(data=report)
        return report
    
    
    def generate_comprehensive_report_bake(self) -> Dict:
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
    # health_reprot_generate = HealthReportGenerate(
    #     start_date="2025-7-3 21:00:00", 
    #     end_date="2025-7-4 07:00:00", 
    #     device_sn="13D2F34920008071211195A907"
    # )
    health_reprot_generate = HealthReportGenerate(
        start_date="2025-7-11 21:00:00", 
        end_date="2025-7-12 07:00:00", 
        device_sn="132A1C9D100040711117953007"
    )
    
    # 生成简化报告
    report = health_reprot_generate.generate_comprehensive_report()
    print("简化报告数据:")
    print(report)
    