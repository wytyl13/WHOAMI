#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/28
@Author  : weiyutao
@File    : sleep_statistics_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import re
from typing import List, Dict, Any, Optional


# 项目路径配置
project_root = "/work/ai/WHOAMI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 页面配置
st.set_page_config(
    page_title="睡眠统计数据分析",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 样式配置
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .stApp > header {
        height: 0;
    }
    .main-header {
        margin-bottom: 2rem;
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
    }
    .metric-card h4 {
        margin-bottom: 0.5rem;
        font-size: 1rem;
        opacity: 0.9;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .sleep-stage-card {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.3rem;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .sleep-stage-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }
    .sleep-stage-card h4 {
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .sleep-stage-card h3 {
        margin: 0;
        font-size: 1.3rem;
    }
    .deep-sleep { 
        background: linear-gradient(135deg, #2E8B57, #228B22); 
        color: white; 
    }
    .light-sleep { 
        background: linear-gradient(135deg, #4682B4, #1E90FF); 
        color: white; 
    }
    .awake { 
        background: linear-gradient(135deg, #FF8C00, #FF6347); 
        color: white; 
    }
    .out-bed { 
        background: linear-gradient(135deg, #DC143C, #B22222); 
        color: white; 
    }
    
    .section-header {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        text-align: center;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e8f4fd;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #1f77b4;
        box-shadow: 0 0 10px rgba(31, 119, 180, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">😴 睡眠统计数据分析仪表板</h1>', unsafe_allow_html=True)

# 初始化数据库连接 - 统计数据库
@st.cache_resource
def init_database_connection():
    """初始化统计数据库连接 (SleepStatistics)"""
    try:
        from whoami.provider.sql_provider import SqlProvider
        from whoami.tool.real_time_vital_analyze.sleep_statistics_model import SleepStatistics
        
        sql_provider = SqlProvider(
            model=SleepStatistics,
            sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
        )
        return sql_provider
    except Exception as e:
        st.error(f"统计数据库连接失败: {str(e)}")
        return None

# 初始化详细数据连接 - 详细数据库
@st.cache_resource
def init_sleep_data_connection():
    """初始化睡眠详细数据连接 (SleepDataState)"""
    try:
        from whoami.provider.sql_provider import SqlProvider
        from whoami.tool.real_time_vital_analyze.sleep_data_state import SleepDataState
        
        sleep_data_provider = SqlProvider(
            model=SleepDataState,
            sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
        )
        return sleep_data_provider
    except Exception as e:
        st.error(f"睡眠详细数据连接失败: {str(e)}")
        return None

# 获取所有设备序列号 - 使用统计数据库 (sql_provider)
@st.cache_data(ttl=300)  # 5分钟缓存
def get_all_device_sns(_sql_provider) -> List[str]:
    """获取所有存在的设备序列号"""
    if _sql_provider is None:
        return []
    
    try:
        # 优先使用无条件查询，避免条件查询可能的字段不匹配问题
        try:
            # 先尝试只查询device_sn字段
            records = _sql_provider.get_record_by_condition(
                condition=None,  # 不使用条件
                fields=["device_sn"],
            )
        except Exception:
            # 如果字段查询失败，尝试获取所有字段
            records = _sql_provider.get_record_by_condition(
                condition=None,
                fields=None,  # 获取所有字段
            )
        
        if not records:
            return []
        
        # 提取device_sn字段，处理可能的字段名称差异
        device_sns = set()
        for record in records:
            # 尝试多种可能的设备序列号字段名（优先使用device_id）
            device_sn = None
            for field_name in ["device_id", "device_sn", "deviceSn", "deviceId", "sn", "serial_number"]:
                if field_name in record and record[field_name]:
                    device_sn = record[field_name]
                    break
            
            if device_sn:
                device_sns.add(str(device_sn))
        
        device_list = sorted(list(device_sns))
        return device_list
        
    except Exception:
        return []

# 获取指定设备的时间区间 - 使用统计数据库 (sql_provider)
@st.cache_data(ttl=300)
def get_time_ranges_for_device(_sql_provider, device_sn: str) -> List[Dict]:
    """获取指定设备的所有时间区间"""
    if _sql_provider is None:
        return []
    
    try:
        # 直接获取所有记录，避免条件查询的字段匹配问题
        all_records = _sql_provider.get_record_by_condition(
            condition=None,  # 不使用条件，避免字段不匹配
            fields=None,     # 获取所有字段
        )
        
        # 在客户端过滤指定设备的记录
        device_records = []
        for record in all_records:
            # 尝试多种可能的设备字段名（优先使用device_id）
            record_device = None
            for field_name in ["device_id", "device_sn", "deviceSn", "deviceId", "sn", "serial_number"]:
                if field_name in record and record[field_name]:
                    record_device = str(record[field_name])
                    break
            
            if record_device == device_sn:
                device_records.append(record)
        
        if not device_records:
            return []
        
        time_ranges = []
        for record in device_records:
            try:
                # 尝试多种可能的时间字段名
                start_time = None
                end_time = None
                record_id = record.get("id")
                total_duration = None
                
                # 查找开始时间字段
                for field_name in ["sleep_start_time", "start_time", "begin_time", "startTime"]:
                    if field_name in record and record[field_name]:
                        start_time = record[field_name]
                        break
                
                # 查找结束时间字段
                for field_name in ["sleep_end_time", "end_time", "finish_time", "endTime"]:
                    if field_name in record and record[field_name]:
                        end_time = record[field_name]
                        break
                
                # 查找时长字段
                for field_name in ["total_duration", "duration", "total_time", "time_duration"]:
                    if field_name in record and record[field_name] is not None:
                        total_duration = record[field_name]
                        break
                
                # 只有当有开始和结束时间时才添加
                if start_time and end_time:
                    duration_display = format_duration_display(total_duration)
                    
                    time_ranges.append({
                        "id": record_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": total_duration,
                        "display": f"{start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%Y-%m-%d %H:%M')} ({duration_display})"
                    })
                    
            except Exception:
                continue
        
        # 按开始时间排序，最新的在前
        if time_ranges:
            time_ranges.sort(key=lambda x: x["start_time"], reverse=True)
        
        return time_ranges
        
    except Exception:
        return []

# 获取具体的睡眠数据 - 使用统计数据库 (sql_provider)
@st.cache_data(ttl=60)
def get_sleep_data_by_id(_sql_provider, record_id: int) -> Optional[Dict]:
    """通过ID获取具体的睡眠统计数据"""
    if _sql_provider is None:
        return None
    
    try:
        # 直接获取所有记录，避免ID条件查询可能的问题
        all_records = _sql_provider.get_record_by_condition(
            condition=None,  # 不使用条件
            fields=None,     # 获取所有字段
        )
        
        # 在客户端查找匹配的ID
        for record in all_records:
            # 尝试多种可能的ID字段名
            current_id = None
            for id_field in ["id", "record_id", "pk", "primary_key"]:
                if id_field in record and record[id_field] is not None:
                    current_id = record[id_field]
                    break
            
            if current_id == record_id:
                return record
        
        return None
        
    except Exception:
        return None

# 备用方法：使用时间范围查询 - 使用统计数据库 (sql_provider)
@st.cache_data(ttl=60)
def get_sleep_data_by_time_range(_sql_provider, device_sn: str, sleep_start_time: datetime, sleep_end_time: datetime) -> Optional[Dict]:
    """使用时间范围获取睡眠统计数据"""
    if _sql_provider is None:
        return None
    
    try:
        # 获取所有记录，在客户端过滤
        all_records = _sql_provider.get_record_by_condition(
            condition=None,
            fields=None,
        )
        
        # 在客户端查找匹配的记录
        for record in all_records:
            # 检查设备匹配
            record_device = None
            for field_name in ["device_id", "device_sn", "deviceSn", "deviceId", "sn", "serial_number"]:
                if field_name in record and record[field_name]:
                    record_device = str(record[field_name])
                    break
            
            if record_device != device_sn:
                continue
            
            # 检查时间匹配
            record_start = None
            record_end = None
            
            for field_name in ["sleep_start_time", "start_time", "begin_time", "startTime"]:
                if field_name in record and record[field_name]:
                    record_start = record[field_name]
                    break
            
            for field_name in ["sleep_end_time", "end_time", "finish_time", "endTime"]:
                if field_name in record and record[field_name]:
                    record_end = record[field_name]
                    break
            
            if (record_start == sleep_start_time and record_end == sleep_end_time):
                return record
        
        return None
        
    except Exception:
        return None

# 获取睡眠详细数据 - 使用详细数据库 (sleep_data_provider)
@st.cache_data(ttl=60)
def get_sleep_detail_data(_sleep_data_provider, device_sn: str, start_time: datetime, end_time: datetime) -> List[Dict]:
    """获取睡眠期间的详细数据（呼吸、心率、状态）"""
    if _sleep_data_provider is None:
        return []
    
    try:
        # 将datetime转换为时间戳
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()
        
        # 查询详细数据
        sql_data = _sleep_data_provider.get_record_by_condition(
            condition={
                "device_id": device_sn,
                "timestamp": {"min": start_timestamp, "max": end_timestamp}
            },
            fields=["timestamp", "breath_bpm", "breath_line", "heart_bpm", "heart_line", "state"]
        )
        
        if not sql_data:
            return []
        
        # 转换时间戳为datetime并排序
        processed_data = []
        for record in sql_data:
            try:
                record_time = datetime.fromtimestamp(record.get("timestamp", 0))
                processed_data.append({
                    "timestamp": record.get("timestamp"),
                    "datetime": record_time,
                    "breath_bpm": record.get("breath_bpm"),
                    "breath_line": record.get("breath_line"),
                    "heart_bpm": record.get("heart_bpm"),
                    "heart_line": record.get("heart_line"),
                    "state": record.get("state", "unknown")
                })
            except Exception:
                continue
        
        # 按时间排序
        processed_data.sort(key=lambda x: x["timestamp"])
        
        return processed_data
        
    except Exception as e:
        st.error(f"获取详细睡眠数据失败: {str(e)}")
        return []

# 获取所有记录进行诊断 - 使用统计数据库 (sql_provider)
@st.cache_data(ttl=300)
def get_all_records_for_diagnosis(_sql_provider) -> List[Dict]:
    """获取所有记录用于诊断"""
    if _sql_provider is None:
        return []
    
    try:
        # 使用最简单的查询方式
        records = _sql_provider.get_record_by_condition(
            condition=None,
            fields=None,
        )
        return records
    except Exception as e:
        st.error(f"获取所有记录失败: {str(e)}")
        return []

# 改进的时长格式化函数
def format_duration_display(duration_value) -> str:
    """格式化时长显示，支持多种数据类型"""
    if duration_value is None:
        return "未知"
    
    # 如果是字符串且已经格式化好了
    if isinstance(duration_value, str):
        if any(unit in duration_value for unit in ['小时', '分钟', '秒', 'h', 'm', 's']):
            return duration_value
        # 尝试转换为数字
        try:
            duration_value = float(duration_value)
        except:
            return duration_value
    
    # 如果是数字，假设单位是小时
    if isinstance(duration_value, (int, float)):
        hours = int(duration_value)
        minutes = int((duration_value - hours) * 60)
        
        if hours > 0 and minutes > 0:
            return f"{hours}小时{minutes}分钟"
        elif hours > 0:
            return f"{hours}小时"
        elif minutes > 0:
            return f"{minutes}分钟"
        else:
            return f"{duration_value:.1f}小时"
    
    return str(duration_value)

# 改进的时长解析函数
def parse_duration_to_seconds(duration_value) -> int:
    """解析各种格式的时长为秒数"""
    if duration_value is None:
        return 0
    
    # 如果是数字，假设单位是小时
    if isinstance(duration_value, (int, float)):
        return int(duration_value * 3600)  # 小时转秒
    
    # 如果是字符串
    if isinstance(duration_value, str):
        duration_str = str(duration_value).strip()
        if not duration_str or duration_str == "未知":
            return 0
        
        try:
            import re
            # 先尝试解析"X小时Y分Z秒"格式
            pattern = r'(?:(\d+(?:\.\d+)?)小时)?(?:(\d+)分)?(?:(\d+)秒)?'
            match = re.search(pattern, duration_str)
            
            if match and any(match.groups()):
                hours = float(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                return int(hours * 3600 + minutes * 60 + seconds)
            
            # 尝试解析纯数字（假设是小时）
            try:
                hours = float(duration_str)
                return int(hours * 3600)
            except:
                pass
                
        except:
            pass
    
    return 0

# 秒数转换为易读格式
def seconds_to_readable(seconds: int) -> str:
    """将秒数转换为易读的时间格式"""
    if seconds == 0:
        return "0分钟"
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0:
        parts.append(f"{minutes}分钟")
    if secs > 0 and hours == 0:  # 只有在小时为0时才显示秒
        parts.append(f"{secs}秒")
    
    return "".join(parts) if parts else "0分钟"

# 安全获取数值的辅助函数
def safe_get_numeric(data: Dict, key: str, default: float = 0.0) -> float:
    """安全获取数值，处理None和各种类型"""
    value = data.get(key, default)
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_get_integer(data: Dict, key: str, default: int = 0) -> int:
    """安全获取整数，处理None和各种类型"""
    value = data.get(key, default)
    if value is None:
        return default
    
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

# 创建睡眠阶段分布图 - 使用统计数据
def create_sleep_stages_chart(sleep_data: Dict) -> go.Figure:
    """创建美化的睡眠阶段分布饼图"""
    # 获取各阶段时长（秒） - 使用改进的解析函数
    deep_sleep_sec = parse_duration_to_seconds(sleep_data.get("deep_sleep_duration"))
    light_sleep_sec = parse_duration_to_seconds(sleep_data.get("light_sleep_duration"))
    awake_sec = parse_duration_to_seconds(sleep_data.get("awake_duration"))
    out_bed_sec = parse_duration_to_seconds(sleep_data.get("out_bed_duration"))
    
    # 数据准备 - 使用美化的颜色
    labels = ["深睡眠", "浅睡眠", "清醒", "离床"]
    values = [deep_sleep_sec, light_sleep_sec, awake_sec, out_bed_sec]
    colors = ["#2E8B57", "#4682B4", "#FF8C00", "#DC143C"]  # 更柔和的颜色
    
    # 过滤掉值为0的项
    non_zero_data = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    
    if not non_zero_data:
        # 如果没有数据，显示提示
        fig = go.Figure()
        fig.add_annotation(
            text="暂无睡眠阶段数据",
            x=0.5, y=0.5,
            font=dict(size=20, color="gray"),
            showarrow=False
        )
        fig.update_layout(
            title="睡眠阶段分布",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=450
        )
        return fig
    
    labels, values, colors = zip(*non_zero_data)
    
    # 创建美化的饼图
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=colors,
            line=dict(color='white', width=3)  # 白色边框
        ),
        textinfo='label+percent',
        textposition='auto',
        textfont=dict(size=14, color='white', family="Arial"),
        hovertemplate='<b>%{label}</b><br>' +
                     '时长: %{customdata}<br>' +
                     '占比: %{percent}<br>' +
                     '<extra></extra>',
        customdata=[seconds_to_readable(v) for v in values],
        hole=0.4,  # 环形图
    )])
    
    fig.update_layout(
        title={
            'text': "睡眠阶段分布",
            'x': 0.5,
            'font': {'size': 20, 'color': '#1f77b4', 'family': "Arial"}
        },
        font=dict(size=12, family="Arial"),
        height=450,
        margin=dict(t=60, b=50, l=50, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=12)
        )
    )
    
    return fig

# 创建行为统计图表 - 使用统计数据
def create_behavior_chart(sleep_data: Dict) -> go.Figure:
    """创建美化的行为统计图"""
    behaviors = ["体动次数", "呼吸暂停", "呼吸急促", "离床次数"]
    counts = [
        safe_get_integer(sleep_data, "body_movement_count"),
        safe_get_integer(sleep_data, "apnea_count"),
        safe_get_integer(sleep_data, "rapid_breathing_count"),
        safe_get_integer(sleep_data, "leave_bed_count")
    ]
    colors = ["#20B2AA", "#FFD700", "#FF6347", "#9370DB"]  # 更鲜艳的颜色
    
    # 确保 counts 都是有效的数值
    counts = [max(0, int(c)) if c is not None else 0 for c in counts]
    
    fig = go.Figure(data=[
        go.Bar(
            x=behaviors,
            y=counts,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2),
                opacity=0.8
            ),
            text=[f"{c}次" for c in counts],
            textposition='auto',
            textfont=dict(size=14, color='white', family="Arial"),
            hovertemplate='<b>%{x}</b><br>' +
                         '次数: %{y}<br>' +
                         '<extra></extra>',
        )
    ])
    
    # 修复 update_layout - 使用新版本 Plotly 语法
    fig.update_layout(
        title={
            'text': "睡眠行为统计",
            'x': 0.5,
            'font': {'size': 20, 'color': '#1f77b4', 'family': "Arial"}
        },
        xaxis={
            'title': {
                'text': "行为类型",
                'font': {'size': 14, 'color': '#333'}
            },
            'tickfont': {'size': 12, 'color': '#333'},
            'showgrid': False,
            'showline': True,
            'linewidth': 1,
            'linecolor': 'lightgray'
        },
        yaxis={
            'title': {
                'text': "次数",
                'font': {'size': 14, 'color': '#333'}
            },
            'tickfont': {'size': 12, 'color': '#333'},
            'showgrid': True,
            'gridcolor': 'lightgray',
            'gridwidth': 0.5,
            'showline': True,
            'linewidth': 1,
            'linecolor': 'lightgray'
        },
        height=450,
        margin=dict(t=60, b=80, l=80, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(size=12, family="Arial")
    )
    
    return fig

# 创建呼吸线图表 - 使用详细数据库
def create_breath_line_chart(detail_data: List[Dict]) -> go.Figure:
    """创建呼吸线折线图"""
    if not detail_data:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无呼吸数据",
            x=0.5, y=0.5,
            font=dict(size=20, color="gray"),
            showarrow=False
        )
        fig.update_layout(
            title="呼吸波形",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    # 提取数据
    times = [d["datetime"] for d in detail_data]
    breath_lines = [d["breath_line"] for d in detail_data if d["breath_line"] is not None]
    breath_bpms = [d["breath_bpm"] for d in detail_data if d["breath_bpm"] is not None]
    
    if not breath_lines and not breath_bpms:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无有效呼吸数据",
            x=0.5, y=0.5,
            font=dict(size=20, color="gray"),
            showarrow=False
        )
        return fig
    
    fig = go.Figure()
    
    # 呼吸波形线
    if breath_lines:
        breath_times = [d["datetime"] for d in detail_data if d["breath_line"] is not None]
        fig.add_trace(go.Scatter(
            x=breath_times,
            y=breath_lines,
            mode='lines',
            name='呼吸波形',
            line=dict(color='#4CAF50', width=2),
            hovertemplate='<b>呼吸波形</b><br>' +
                         '时间: %{x}<br>' +
                         '数值: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        # title={
        #     'text': "呼吸波形图",
        #     'x': 0.5,
        #     'font': {'size': 18, 'color': '#1f77b4', 'family': "Arial"}
        # },
        xaxis={
            'title': {'text': "时间", 'font': {'size': 12}},
            'tickfont': {'size': 10},
            'showgrid': True,
            'gridcolor': 'lightgray'
        },
        yaxis={
            'title': {'text': "呼吸强度", 'font': {'size': 12}},
            'tickfont': {'size': 10},
            'showgrid': True,
            'gridcolor': 'lightgray'
        },
        height=400,
        margin=dict(t=50, b=50, l=60, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# 创建心率线图表 - 使用详细数据库 (仅显示heart_line)
def create_heart_line_chart(detail_data: List[Dict]) -> go.Figure:
    """创建心率线折线图 (仅显示heart_line数据)"""
    if not detail_data:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无心率数据",
            x=0.5, y=0.5,
            font=dict(size=20, color="gray"),
            showarrow=False
        )
        fig.update_layout(
            title="心率波形",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    # 提取heart_line数据
    heart_lines = [d["heart_line"] for d in detail_data if d["heart_line"] is not None]
    
    if not heart_lines:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无有效心率波形数据",
            x=0.5, y=0.5,
            font=dict(size=20, color="gray"),
            showarrow=False
        )
        fig.update_layout(
            title="心率波形图",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    fig = go.Figure()
    
    # 仅显示心率波形线
    heart_times = [d["datetime"] for d in detail_data if d["heart_line"] is not None]
    fig.add_trace(go.Scatter(
        x=heart_times,
        y=heart_lines,
        mode='lines',
        name='心率波形',
        line=dict(color='#F44336', width=2),
        hovertemplate='<b>心率波形</b><br>' +
                     '时间: %{x}<br>' +
                     '数值: %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        # title={
        #     'text': "心率波形图",
        #     'x': 0.5,
        #     'font': {'size': 18, 'color': '#1f77b4', 'family': "Arial"}
        # },
        xaxis={
            'title': {'text': "时间", 'font': {'size': 12}},
            'tickfont': {'size': 10},
            'showgrid': True,
            'gridcolor': 'lightgray'
        },
        yaxis={
            'title': {'text': "心率强度", 'font': {'size': 12}},
            'tickfont': {'size': 10},
            'showgrid': True,
            'gridcolor': 'lightgray'
        },
        height=400,
        margin=dict(t=50, b=50, l=60, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False  # 只有一条线，不需要图例
    )
    
    return fig




# 与监控系统完全一致的睡眠状态图表函数
def create_state_chart_unified(detail_data: List[Dict]) -> go.Figure:
    """创建与监控系统颜色统一的睡眠状态图表"""
    if not detail_data:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无状态数据",
            x=0.5, y=0.5,
            font=dict(size=20, color="gray"),
            showarrow=False
        )
        fig.update_layout(
            title="睡眠状态",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300
        )
        return fig
    
    # 🎨 与监控系统完全一致的状态颜色映射
    STATUS_COLORS = {
        '在床正常': '#4CAF50',  # 绿色
        '离床': '#FF9800',      # 橙色
        '呼吸急促': '#F44336',  # 红色
        '体动': '#2196F3',      # 蓝色
        '呼吸暂停': '#9C27B0',  # 紫色
        '清醒': '#FFC107',      # 黄色
        '浅睡眠': '#00BCD4',    # 青色
        '深睡眠': '#3F51B5',    # 深蓝色
        
        # 兼容其他可能的状态值
        '深睡眠': '#3F51B5',
        '浅睡眠': '#00BCD4',
        'deep_sleep': '#3F51B5',
        'light_sleep': '#00BCD4',
        'awake': '#FFC107',
        'out_bed': '#FF9800',
        'unknown': '#808080',    # 灰色 - 未知状态
    }

    # 🎯 与监控系统一致的状态图标
    STATUS_ICONS = {
        '在床正常': '🟢',
        '离床': '🟠',
        '呼吸急促': '🔴',
        '体动': '🔵',
        '呼吸暂停': '🟣',
        '清醒': '😊',
        '浅睡眠': '😴',
        '深睡眠': '😴',
        # 兼容
        'deep_sleep': '😴',
        'light_sleep': '😴',
        'awake': '😊',
        'out_bed': '🟠',
        'unknown': '❓',
    }
    
    # 显示名称映射（带图标）
    state_display_names = {}
    for state in STATUS_COLORS.keys():
        icon = STATUS_ICONS.get(state, '⚪')
        state_display_names[state] = f"{icon} {state}"
    
    # 处理状态数据，将连续的相同状态合并为段
    segments = []
    if detail_data:
        detail_data_sorted = sorted(detail_data, key=lambda x: x["datetime"])
        current_state = detail_data_sorted[0]["state"]
        start_time = detail_data_sorted[0]["datetime"]
        
        for i, record in enumerate(detail_data_sorted[1:], 1):
            if record["state"] != current_state or i == len(detail_data_sorted) - 1:
                # 状态改变或到达最后一条记录
                end_time = record["datetime"] if record["state"] != current_state else record["datetime"]
                segments.append({
                    'state': current_state,
                    'start': start_time,
                    'end': end_time,
                    'duration': (end_time - start_time).total_seconds() / 60  # 转换为分钟
                })
                current_state = record["state"]
                start_time = record["datetime"]
    
    if not segments:
        fig = go.Figure()
        fig.add_annotation(text="无有效状态数据", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # 按状态分组
    state_segments = {}
    for segment in segments:
        state = segment['state']
        if state not in state_segments:
            state_segments[state] = []
        state_segments[state].append(segment)
    
    # 为每个状态创建trace
    for state, state_segs in state_segments.items():
        # 🎨 使用统一的颜色映射
        color = STATUS_COLORS.get(state, '#808080')  # 默认灰色
        display_name = state_display_names.get(state, f"⚪ {state}")
        
        # 创建填充区域的坐标
        x_coords = []
        y_coords = []
        
        for segment in state_segs:
            x_coords.extend([
                segment['start'], segment['end'], segment['end'], 
                segment['start'], segment['start'], None
            ])
            y_coords.extend([0, 0, 1, 1, 0, None])
        
        # 添加填充trace
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color=color, width=0),
            fill='toself',
            fillcolor=color,
            opacity=0.8,
            name=display_name,
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # 添加hover信息trace
        for segment in state_segs:
            fig.add_trace(go.Scatter(
                x=[segment['start'], segment['end']],
                y=[0.5, 0.5],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=20),
                showlegend=False,
                hovertemplate=f'<b>{display_name}</b><br>' +
                             f'开始: {segment["start"].strftime("%H:%M:%S")}<br>' +
                             f'结束: {segment["end"].strftime("%H:%M:%S")}<br>' +
                             f'时长: {segment["duration"]:.1f}分钟<br>' +
                             '<extra></extra>',
            ))
    
    fig.update_layout(
        # title={
        #     'text': "睡眠状态时间线",
        #     'x': 0.5,
        #     'font': {'size': 18, 'color': '#1f77b4', 'family': "Arial"}
        # },
        xaxis={
            'title': {'text': "时间", 'font': {'size': 12}},
            'tickfont': {'size': 10},
            'showgrid': True,
            'gridcolor': 'lightgray'
        },
        yaxis={
            'title': {'text': "状态", 'font': {'size': 12}},
            'showticklabels': False,
            'range': [-0.1, 1.1]
        },
        height=300,
        margin=dict(t=50, b=50, l=80, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )
    
    return fig



def create_state_chart(detail_data: List[Dict]) -> go.Figure:
    """创建JSON安全的睡眠状态图表"""
    if not detail_data:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无状态数据",
            x=0.5, y=0.5,
            font=dict(size=20, color="gray"),
            showarrow=False
        )
        fig.update_layout(
            title="睡眠状态",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300
        )
        return fig
    
    # 使用简单的ASCII字符，避免Unicode问题
    STATUS_COLORS = {
        '在床正常': '#4CAF50',
        '离床': '#FF9800',
        '呼吸急促': '#F44336',
        '体动': '#2196F3',
        '呼吸暂停': '#9C27B0',
        '清醒': '#FFC107',
        '浅睡眠': '#00BCD4',
        '深睡眠': '#3F51B5',
        'deep_sleep': '#3F51B5',
        'light_sleep': '#00BCD4',
        'awake': '#FFC107',
        'out_bed': '#FF9800',
        'unknown': '#808080',
    }
    
    # 使用简单的文本标记，避免emoji导致的编码问题
    # STATUS_ICONS = {
    #     '在床正常': '[正常]',
    #     '离床': '[离床]',
    #     '呼吸急促': '[急促]',
    #     '体动': '[体动]',
    #     '呼吸暂停': '[暂停]',
    #     '清醒': '[清醒]',
    #     '浅睡眠': '[浅睡]',
    #     '深睡眠': '[深睡]',
    #     'deep_sleep': '[深睡]',
    #     'light_sleep': '[浅睡]',
    #     'awake': '[清醒]',
    #     'out_bed': '[离床]',
    #     'unknown': '[未知]',
    # }
    
    
    STATUS_ICONS = {
        '在床正常': '正常',
        '离床': '离床',
        '呼吸急促': '急促',
        '体动': '体动',
        '呼吸暂停': '暂停',
        '清醒': '清醒',
        '浅睡眠': '浅睡',
        '深睡眠': '深睡',
        'deep_sleep': '深睡',
        'light_sleep': '浅睡',
        'awake': '清醒',
        'out_bed': '离床',
        'unknown': '未知',
    }
    
    def safe_string(text):
        """确保字符串对JSON安全"""
        if text is None:
            return ""
        # 移除或替换可能导致JSON问题的字符
        text = str(text)
        # 替换反斜杠
        text = text.replace('\\', '/')
        # 移除控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return text
    
    # 处理状态数据，将连续的相同状态合并为段
    segments = []
    if detail_data:
        detail_data_sorted = sorted(detail_data, key=lambda x: x["datetime"])
        current_state = safe_string(detail_data_sorted[0]["state"])
        start_time = detail_data_sorted[0]["datetime"]
        
        for i, record in enumerate(detail_data_sorted[1:], 1):
            record_state = safe_string(record["state"])
            if record_state != current_state or i == len(detail_data_sorted) - 1:
                end_time = record["datetime"] if record_state != current_state else record["datetime"]
                segments.append({
                    'state': current_state,
                    'start': start_time,
                    'end': end_time,
                    'duration': (end_time - start_time).total_seconds() / 60
                })
                current_state = record_state
                start_time = record["datetime"]
    
    if not segments:
        fig = go.Figure()
        fig.add_annotation(text="无有效状态数据", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # 按状态分组
    state_segments = {}
    for segment in segments:
        state = segment['state']
        if state not in state_segments:
            state_segments[state] = []
        state_segments[state].append(segment)
    
    # 为每个状态创建trace
    for state, state_segs in state_segments.items():
        color = STATUS_COLORS.get(state, '#808080')
        icon = STATUS_ICONS.get(state, '[?]')
        # display_name = f"{icon} {safe_string(state)}"
        display_name = safe_string(state)
        
        # 创建填充区域的坐标
        x_coords = []
        y_coords = []
        
        for segment in state_segs:
            x_coords.extend([
                segment['start'], segment['end'], segment['end'], 
                segment['start'], segment['start'], None
            ])
            y_coords.extend([0, 0, 1, 1, 0, None])
        
        # 添加填充trace
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color=color, width=0),
            fill='toself',
            fillcolor=color,
            opacity=0.8,
            name=display_name,
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # 添加hover信息trace - 使用安全的字符串格式
        for segment in state_segs:
            # 创建安全的hover文本
            hover_text = (
                f"<b>{safe_string(display_name)}</b><br>"
                f"开始: {segment['start'].strftime('%H:%M:%S')}<br>"
                f"结束: {segment['end'].strftime('%H:%M:%S')}<br>"
                f"时长: {segment['duration']:.1f}分钟<br>"
                "<extra></extra>"
            )
            
            fig.add_trace(go.Scatter(
                x=[segment['start'], segment['end']],
                y=[0.5, 0.5],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=20),
                showlegend=False,
                hovertemplate=hover_text,
            ))
    
    fig.update_layout(
        # title={
        #     'text': "睡眠状态时间线",
        #     'x': 0.5,
        #     'font': {'size': 18, 'color': '#1f77b4', 'family': "Arial"}
        # },
        xaxis={
            'title': {'text': "时间", 'font': {'size': 12}},
            'tickfont': {'size': 10},
            'showgrid': True,
            'gridcolor': 'lightgray'
        },
        yaxis={
            'title': {'text': "状态", 'font': {'size': 12}},
            'showticklabels': False,
            'range': [-0.1, 1.1]
        },
        height=300,
        margin=dict(t=50, b=50, l=80, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )
    
    return fig








def main():
    # 初始化数据库连接
    sql_provider = init_database_connection()  # 统计数据库
    sleep_data_provider = init_sleep_data_connection()  # 详细数据库 - 修复：正确初始化
    
    if sql_provider is None:
        st.error("无法连接到统计数据库，请检查配置")
        return
    
    # 添加诊断模式
    st.sidebar.markdown("## 🔧 诊断工具")
    
    if st.sidebar.button("🔍 数据库诊断"):
        st.markdown("## 🔧 数据库诊断模式")
        
        with st.spinner("正在诊断数据库..."):
            try:
                # 测试基本连接
                st.markdown("### 1. 统计数据库连接测试")
                all_records = get_all_records_for_diagnosis(sql_provider)
                
                if all_records:
                    st.success(f"✅ 统计数据库连接正常，共有 {len(all_records)} 条记录")
                else:
                    st.error("❌ 统计数据库中没有记录")
                
                # 测试详细数据连接
                st.markdown("### 2. 详细数据库连接测试")
                if sleep_data_provider is not None:
                    st.success("✅ 详细数据库连接正常")
                    
                    # 测试查询一条记录
                    try:
                        test_records = sleep_data_provider.get_record_by_condition(
                            condition=None,
                            fields=["timestamp", "breath_bpm", "heart_bpm", "state"],
                            limit=1
                        )
                        if test_records:
                            st.success(f"✅ 详细数据查询正常，测试记录: {len(test_records)} 条")
                            st.json(test_records[0])
                        else:
                            st.warning("⚠️ 详细数据表为空")
                    except Exception as e:
                        st.error(f"❌ 详细数据查询失败: {str(e)}")
                else:
                    st.error("❌ 详细数据库连接失败")
                
                if all_records:
                    # 显示数据结构
                    st.markdown("### 3. 统计数据结构分析")
                    first_record = all_records[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**字段列表:**")
                        fields = list(first_record.keys())
                        for i, field in enumerate(fields, 1):
                            st.write(f"{i:2d}. `{field}`")
                    
                    with col2:
                        st.markdown("**字段类型和示例值:**")
                        for field, value in first_record.items():
                            value_type = type(value).__name__
                            value_str = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
                            st.write(f"`{field}`: {value_type} = {value_str}")
                    
                    # 分析设备字段
                    st.markdown("### 4. 设备字段分析")
                    device_field_candidates = ["device_id", "device_sn", "deviceSn", "deviceId", "sn", "serial_number"]
                    found_device_fields = []
                    
                    for field in device_field_candidates:
                        if field in first_record:
                            found_device_fields.append(f"✅ `{field}`: {first_record[field]}")
                        else:
                            found_device_fields.append(f"❌ `{field}`: 不存在")
                    
                    for field_info in found_device_fields:
                        st.write(field_info)
                    
                    # 分析时间字段
                    st.markdown("### 5. 时间字段分析")
                    time_field_candidates = [
                        "sleep_start_time", "start_time", "begin_time", "startTime",
                        "sleep_end_time", "end_time", "finish_time", "endTime",
                        "total_duration", "duration", "total_time", "time_duration"
                    ]
                    found_time_fields = []
                    
                    for field in time_field_candidates:
                        if field in first_record:
                            found_time_fields.append(f"✅ `{field}`: {first_record[field]}")
                        else:
                            found_time_fields.append(f"❌ `{field}`: 不存在")
                    
                    for field_info in found_time_fields:
                        st.write(field_info)
                    
                    # 显示设备分布
                    st.markdown("### 6. 设备分布统计")
                    device_counts = {}
                    
                    # 找到正确的设备字段
                    device_field = None
                    for field_name in ["device_id", "device_sn", "deviceSn", "deviceId", "sn", "serial_number"]:
                        if field_name in first_record:
                            device_field = field_name
                            break
                    
                    if device_field:
                        st.info(f"使用字段: `{device_field}`")
                        for record in all_records:
                            device_value = record.get(device_field, "Unknown")
                            device_counts[str(device_value)] = device_counts.get(str(device_value), 0) + 1
                        
                        device_df = pd.DataFrame([
                            {"设备标识": k, "记录数量": v} for k, v in device_counts.items()
                        ])
                        st.dataframe(device_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("未找到设备标识字段")
                    
                    # 显示最近的几条记录
                    st.markdown("### 7. 数据样本（前3条）")
                    for i, record in enumerate(all_records[:3], 1):
                        with st.expander(f"记录 {i} (ID: {record.get('id', 'N/A')})"):
                            # 创建格式化的显示
                            formatted_record = {}
                            for key, value in record.items():
                                if isinstance(value, datetime):
                                    formatted_record[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    formatted_record[key] = value
                            st.json(formatted_record)
                    
                    # SqlProvider 特殊提示
                    st.markdown("### 8. SqlProvider 适配信息")
                    st.info("""
                    **检测到的 SqlProvider 特性:**
                    - ✅ 支持 `condition=None` 查询所有记录
                    - ✅ 支持 `fields=None` 获取所有字段
                    - ✅ 自动过滤 `deleted=False` 的记录
                    - ✅ 使用客户端过滤避免条件查询问题
                    - ✅ 正确分离统计数据库和详细数据库连接
                    """)
                    
            except Exception as e:
                st.error(f"❌ 诊断过程中出错: {str(e)}")
                st.info("这可能是由于 SqlProvider 的 `deleted` 字段过滤或其他模型特定的问题")
        
        return  # 诊断模式下不继续执行正常流程
    
    # 第一步：设备选择
    st.markdown("## 📱 步骤1: 选择设备")
    
    device_sns = get_all_device_sns(sql_provider)  # 使用统计数据库
    
    if not device_sns:
        st.warning("暂无设备数据")
        return
    
    selected_device = st.selectbox(
        "请选择要查看的设备序列号:",
        options=device_sns,
        help=f"当前共有 {len(device_sns)} 个设备"
    )
    
    if not selected_device:
        return
    
    # 显示设备的基本信息
    with st.expander("📋 设备信息预览"):
        try:
            # 获取该设备的记录 - 使用统计数据库
            sample_records = get_time_ranges_for_device(sql_provider, selected_device)
            if sample_records:
                st.success(f"设备 {selected_device} 共有 {len(sample_records)} 条记录")
                
                # 尝试获取第一条完整记录 - 使用统计数据库
                first_record_id = sample_records[0]["id"]
                first_full_record = get_sleep_data_by_id(sql_provider, first_record_id)
                
                if first_full_record:
                    st.markdown("**数据字段预览（基于第一条记录）:**")
                    fields_info = []
                    for field, value in first_full_record.items():
                        value_type = type(value).__name__
                        value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                        fields_info.append({
                            "字段名": field,
                            "类型": value_type,
                            "示例值": value_str
                        })
                    
                    st.dataframe(pd.DataFrame(fields_info), use_container_width=True, hide_index=True)
                else:
                    st.warning("⚠️ 无法获取完整记录数据，但时间范围列表正常")
                    st.info("这通常意味着字段查询有限制，数据仍然可以正常显示")
            else:
                st.warning("该设备暂无记录")
        except Exception as e:
            st.error(f"获取设备信息失败: {str(e)}")
            st.info("💡 建议：使用侧边栏的'数据库诊断'功能检查数据库状态")
    
    # 第二步：时间区间选择
    st.markdown("## ⏰ 步骤2: 选择时间区间")
    
    time_ranges = get_time_ranges_for_device(sql_provider, selected_device)  # 使用统计数据库
    
    if not time_ranges:
        st.warning(f"设备 {selected_device} 暂无睡眠记录")
        return
    
    st.markdown(f"""
    <div class="info-box">
        <strong>设备 {selected_device}</strong> 共有 <strong>{len(time_ranges)}</strong> 条睡眠记录
    </div>
    """, unsafe_allow_html=True)
    
    # 时间区间选择器
    time_range_options = [tr["display"] for tr in time_ranges]
    selected_range_display = st.selectbox(
        "请选择要查看的睡眠时间区间:",
        options=time_range_options,
        help="选择具体的睡眠时间段进行详细分析"
    )
    
    if not selected_range_display:
        return
    
    # 找到对应的时间区间数据
    selected_range = next(tr for tr in time_ranges if tr["display"] == selected_range_display)
    
    # 第三步：数据展示
    st.markdown("## 📊 步骤3: 睡眠数据分析")
    
    # 获取详细数据 - 优先使用ID查询 - 使用统计数据库
    sleep_data = get_sleep_data_by_id(sql_provider, selected_range["id"])
    
    # 如果ID查询失败，尝试时间范围查询 - 使用统计数据库
    if not sleep_data:
        st.warning("ID查询失败，尝试时间范围查询...")
        sleep_data = get_sleep_data_by_time_range(
            sql_provider, 
            selected_device, 
            selected_range["start_time"], 
            selected_range["end_time"]
        )
    
    if not sleep_data:
        st.error("无法获取睡眠数据")
        
        # 调试信息
        st.markdown("### 🔧 调试信息")
        st.write("选择的记录ID:", selected_range["id"])
        st.write("设备序列号:", selected_device)
        st.write("睡眠开始时间:", selected_range["start_time"])
        st.write("睡眠结束时间:", selected_range["end_time"])
        
        # 尝试获取该设备的所有记录用于调试 - 使用统计数据库
        try:
            debug_records = get_time_ranges_for_device(sql_provider, selected_device)
            st.write("该设备的所有记录数:", len(debug_records))
            if debug_records:
                st.write("第一条记录示例:", debug_records[0])
        except Exception as e:
            st.write("调试查询失败:", str(e))
        
        return
    
    # 基本信息展示 - 使用统计数据
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_duration_display = format_duration_display(sleep_data.get('total_duration'))
        st.markdown(f"""
        <div class="metric-card">
            <h4>🕐 总时长</h4>
            <h2>{total_duration_display}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        in_bed_duration_display = format_duration_display(sleep_data.get('in_bed_duration'))
        st.markdown(f"""
        <div class="metric-card">
            <h4>🛏️ 在床时长</h4>
            <h2>{in_bed_duration_display}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_hr = safe_get_numeric(sleep_data, 'avg_heart_rate')
        st.markdown(f"""
        <div class="metric-card">
            <h4>💓 平均心率</h4>
            <h2>{avg_hr:.1f} BPM</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_br = safe_get_numeric(sleep_data, 'avg_breath_rate')
        st.markdown(f"""
        <div class="metric-card">
            <h4>🫁 平均呼吸率</h4>
            <h2>{avg_br:.1f} /min</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 睡眠阶段卡片 - 使用统计数据
    st.markdown("### 睡眠阶段概览")
    stage_col1, stage_col2, stage_col3, stage_col4 = st.columns(4)
    
    with stage_col1:
        deep_sleep_display = format_duration_display(sleep_data.get('deep_sleep_duration'))
        st.markdown(f"""
        <div class="sleep-stage-card deep-sleep">
            <h4>😴 深睡眠</h4>
            <h3>{deep_sleep_display}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with stage_col2:
        light_sleep_display = format_duration_display(sleep_data.get('light_sleep_duration'))
        st.markdown(f"""
        <div class="sleep-stage-card light-sleep">
            <h4>🌙 浅睡眠</h4>
            <h3>{light_sleep_display}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with stage_col3:
        awake_display = format_duration_display(sleep_data.get('awake_duration'))
        st.markdown(f"""
        <div class="sleep-stage-card awake">
            <h4>😊 清醒</h4>
            <h3>{awake_display}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with stage_col4:
        out_bed_display = format_duration_display(sleep_data.get('out_bed_duration'))
        st.markdown(f"""
        <div class="sleep-stage-card out-bed">
            <h4>🚶 离床</h4>
            <h3>{out_bed_display}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # 详细分析图表 - 使用统计数据
    st.markdown('<h2 class="section-header">📈 详细分析图表</h2>', unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # 睡眠阶段分布图 - 使用统计数据
        try:
            sleep_chart = create_sleep_stages_chart(sleep_data)
            st.plotly_chart(sleep_chart, use_container_width=True)
        except Exception as e:
            st.error(f"睡眠阶段图表生成失败: {str(e)}")
    
    with chart_col2:
        # 行为统计图 - 使用统计数据
        try:
            behavior_chart = create_behavior_chart(sleep_data)
            st.plotly_chart(behavior_chart, use_container_width=True)
        except Exception as e:
            st.error(f"行为统计图表生成失败: {str(e)}")
    
    # 新增：详细波形和状态分析 - 使用详细数据库
    st.markdown('<h2 class="section-header">🌊 波形和状态分析</h2>', unsafe_allow_html=True)
    
    # 检查详细数据库连接状态
    if sleep_data_provider is not None:
        with st.spinner("正在加载详细数据..."):
            detail_data = get_sleep_detail_data(
                sleep_data_provider,  # 使用详细数据库
                selected_device,
                selected_range["start_time"], 
                selected_range["end_time"]
            )
        
        if detail_data:
            st.success(f"成功加载 {len(detail_data)} 条详细数据记录")
            
            # 状态时间线（单独一行，因为高度较小） - 使用详细数据库
            st.markdown("### 睡眠状态时间线")
            try:
                state_chart = create_state_chart(detail_data)
                st.plotly_chart(state_chart, use_container_width=True)
            except Exception as e:
                st.error(f"状态图表生成失败: {str(e)}")
            
            # 呼吸和心率波形（并排显示） - 使用详细数据库
            wave_col1, wave_col2 = st.columns(2)
            
            with wave_col1:
                st.markdown("### 呼吸波形")
                try:
                    breath_chart = create_breath_line_chart(detail_data)
                    st.plotly_chart(breath_chart, use_container_width=True)
                except Exception as e:
                    st.error(f"呼吸波形图表生成失败: {str(e)}")
            
            with wave_col2:
                st.markdown("### 心率波形")
                try:
                    heart_chart = create_heart_line_chart(detail_data)
                    st.plotly_chart(heart_chart, use_container_width=True)
                except Exception as e:
                    st.error(f"心率波形图表生成失败: {str(e)}")
            
            # 数据统计信息
            with st.expander("📊 详细数据统计"):
                col1, col2, col3 = st.columns(3)
                
                # 统计各类数据的有效数量
                breath_count = len([d for d in detail_data if d["breath_line"] is not None])
                heart_count = len([d for d in detail_data if d["heart_line"] is not None])
                state_count = len([d for d in detail_data if d["state"] and d["state"] != "unknown"])
                
                with col1:
                    st.metric("有效呼吸数据", f"{breath_count} 条")
                
                with col2:
                    st.metric("有效心率数据", f"{heart_count} 条")
                
                with col3:
                    st.metric("有效状态数据", f"{state_count} 条")
                
                # 状态分布统计
                if detail_data:
                    state_stats = {}
                    for d in detail_data:
                        state = d["state"]
                        if state:
                            state_stats[state] = state_stats.get(state, 0) + 1
                    
                    if state_stats:
                        st.markdown("**状态分布统计:**")
                        state_df = pd.DataFrame([
                            {"状态": k, "记录数": v, "占比": f"{v/len(detail_data)*100:.1f}%"} 
                            for k, v in state_stats.items()
                        ])
                        st.dataframe(state_df, use_container_width=True, hide_index=True)
        
        else:
            st.warning("该时间段内暂无详细数据记录")
            st.info("💡 可能原因：数据还未同步到详细数据表，或者时间范围内没有实时监测数据")
    else:
        st.error("无法连接到详细数据库，请检查配置")
        st.info("详细数据包括呼吸波形、心率波形和状态信息，需要单独的数据库连接")

if __name__ == "__main__":
    main()