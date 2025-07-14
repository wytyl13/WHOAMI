#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/21 11:46
@Author  : weiyutao
@File    : test_.py
"""

# 需要安装: pip install streamlit-elements plotly

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import random
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_elements import elements, mui, html
import queue
import os
import sys
from typing import (
    List,
    Dict,
    Any
)
import threading
import pytz

beijing_tz = pytz.timezone('Asia/Shanghai')
# 直接指定项目根目录
project_root = "/work/ai/WHOAMI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 获取URL参数中的device_sn
def get_device_sn_from_url():
    """从URL参数中获取device_sn"""
    try:
        # 使用新版本的query_params API
        if hasattr(st, 'query_params'):
            query_params = st.query_params
            device_sn = query_params.get('device_sn', None)
        else:
            # 兼容旧版本
            query_params = st.experimental_get_query_params()
            device_sn = query_params.get('device_sn', [None])[0]
        
        return device_sn if device_sn else "未知设备"
    except:
        return "未知设备"

# 获取设备编号
device_sn = get_device_sn_from_url()

# 页面配置
st.set_page_config(
    page_title=f"💓 {device_sn} 无闪动实时监控",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="collapsed"  # 收起侧边栏减少空白
)

# 缩小顶部空白并优化整体紧凑度
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
        margin-bottom: 0.5rem;
        font-size: 1.8rem;
    }
    /* 修复下拉菜单显示不全问题 */
    .stSelectbox > div > div > div {
        min-height: 3rem !important;
        padding: 0.5rem 0.75rem !important;
        line-height: 1.4 !important;
    }
    .stSelectbox > div > div > div > div {
        font-size: 0.9rem !important;
        overflow: visible !important;
        white-space: nowrap !important;
    }
    .stSelectbox label {
        font-size: 0.85rem !important;
        margin-bottom: 0.3rem !important;
    }
    .stButton > button {
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
    }
    .stExpander {
        margin: 0.5rem 0;
    }
    .stDataFrame {
        font-size: 0.85rem;
    }
    .report-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .report-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .report-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .report-processing {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)

# 动态标题显示
st.markdown(f'<h1 class="main-header">💓 {device_sn} 无闪动实时监控</h1>', unsafe_allow_html=True)

# 定义所有状态颜色映射（原始5个状态 + 新增3个状态）
STATUS_COLORS = {
    '在床正常': '#4CAF50',  # 绿色
    '离床': '#FF9800',      # 橙色
    '呼吸急促': '#F44336',  # 红色
    '体动': '#2196F3',      # 蓝色
    '呼吸暂停': '#9C27B0',  # 紫色
    '清醒': '#FFC107',      # 黄色
    '浅睡眠': '#00BCD4',    # 青色
    '深睡眠': '#3F51B5'     # 深蓝色
}

STATUS_ICONS = {
    '在床正常': '🟢',
    '离床': '🟠',
    '呼吸急促': '🔴',
    '体动': '🔵',
    '呼吸暂停': '🟣',
    '清醒': '😊',
    '浅睡眠': '😴',
    '深睡眠': '😴'
}

# 初始化状态
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
    st.session_state.data = []
    st.session_state.last_update = time.time()
    st.session_state.max_points = 200  # 可配置的最大数据点数
    st.session_state.show_data_modal = False  # 数据展开状态
    st.session_state.show_chart_modal = False  # 图表展开状态
    st.session_state.selected_device_id = device_sn  # 直接使用URL传入的device_sn
    
    # 新增监控时间相关状态
    st.session_state.monitor_start_time = None  # 监控开始时间
    st.session_state.monitor_end_time = None    # 监控结束时间
    st.session_state.monitor_start_datetime = None  # 监控开始日期时间字符串
    st.session_state.monitor_end_datetime = None    # 监控结束日期时间字符串
    
    # 新增报告生成相关状态
    st.session_state.report_generating = False  # 是否正在生成报告
    st.session_state.report_status = None       # 报告生成状态 ('success', 'error', 'processing')
    st.session_state.report_message = ""        # 报告生成消息
    st.session_state.report_available = False   # 是否有可用报告

    st.session_state.report_result = None       # 存储报告生成结果

@st.cache_resource
def get_shared_socket_manager():
    """
    全局共享的 SocketServerManager 实例
    所有用户共用同一个实例，避免端口冲突
    """
    print("🔧 正在初始化共享的 SocketServerManager...")
    
    # 导入你的管理器
    from whoami.tool.base.rnn_model_info import RNNModelInfo
    from whoami.configs.detector_config import DetectorConfig
    from whoami.neural_network.rnn.model import LSTM
    from whoami.tool.base.consumer_tool_pool import ConsumerToolPool
    from whoami.tool.real_time_vital_analyze.socket_server_manager import SocketServerManager
    CONFIG_PATH = '/work/ai/WHOAMI/whoami/scripts/detect/detect_config.yaml'
    CONFIG = DetectorConfig.from_file(CONFIG_PATH).__dict__
    TOPIC_DICT = CONFIG['topics']
    conf_dict = CONFIG["conf"]
    model_path_dict = CONFIG["model_path"]
    class_list_dict = CONFIG["class_list"]
    topic_list = TOPIC_DICT
    model_paths = {}
    for conf_key, conf_value in conf_dict.items():
        for topic_name in topic_list:
            topic_key = conf_key + topic_name
            model_paths[topic_key] = RNNModelInfo(
                model_path="/work/ai/WHOAMI/whoami/"+model_path_dict[topic_name],
                model_type_class=LSTM,
                classes=class_list_dict[topic_name],
                conf=conf_value[topic_name]
            )
    print(f"model_paths: --------------------------------------\n {model_paths}")
    consumer_tool_pool = ConsumerToolPool(model_paths=model_paths)
    
    from whoami.provider.sql_provider import SqlProvider
    from whoami.tool.health_report.sx_device_wavve_vital_sign_log_20250522 import SxDeviceWavveVitalSignLog
    sql_provider_test = SqlProvider(
        model=SxDeviceWavveVitalSignLog, 
        sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml",
    )
    result = sql_provider_test.get_record_by_condition(
        condition={"device_sn": "13D2F34920008071211195A907"},
        fields=["create_time", "breath_bpm", "breath_line", "heart_bpm", "heart_line", "distance", "signal_intensity", "state", "body_move_data", "device_sn"],
        date_range={"date_field": "create_time", "start_date": "2025-6-29 21:00:00", "end_date": "2025-6-30 07:00:00"}
    )
    # print(result[0])
    
    from datetime import datetime, timedelta
    
    def preprocess_query_results_safe(records: List[Dict[str, Any]], 
                                source_timezone: str = 'Asia/Shanghai') -> List[Dict[str, Any]]:
        """更安全的版本 - 明确指定源时区，并处理body_move_data填充和breath_bpm过滤"""
        if not records:
            return []
        
        import pytz
        
        # 使用pytz处理时区（更准确，考虑夏令时等）
        source_tz = pytz.timezone(source_timezone)
        utc_tz = pytz.UTC
        
        # 第一步：构建body_move_data的映射关系
        # 找到所有不为None的body_move_data及其对应的create_time
        body_move_mapping = {}
        for record in records:
            if (record.get('body_move_data') is not None and 
                'create_time' in record):
                create_time_key = record['create_time']
                body_move_mapping[create_time_key] = record['body_move_data']
        
        # 第二步：更新所有相同timestamp的body_move_data
        for record in records:
            if 'create_time' in record and record['create_time'] in body_move_mapping:
                record['body_move_data'] = body_move_mapping[record['create_time']]
        
        # 第三步：过滤掉breath_bpm为None的记录
        filtered_records = []
        for record in records:
            if record.get('breath_bpm') is not None:
                filtered_records.append(record)
        
        # 第四步：按原逻辑处理剩余记录
        processed_records = []
        for record in filtered_records:
            # 保持原来的heart_bpm过滤逻辑
            if record.get('heart_bpm') is None:
                continue
            
            processed_record = record.copy()
            
            # 时间戳转换逻辑
            if 'create_time' in processed_record and isinstance(processed_record['create_time'], datetime):
                dt = processed_record['create_time']
                
                if dt.tzinfo is None:
                    # 明确指定原始数据的时区
                    dt_localized = source_tz.localize(dt)
                else:
                    dt_localized = dt
                
                # 转换为UTC时间戳
                processed_record['create_time'] = dt_localized.astimezone(utc_tz).timestamp()
            
            # body_move_data处理（如果仍为None则设为0）
            if 'body_move_data' in processed_record and processed_record['body_move_data'] is None:
                processed_record['body_move_data'] = 0
            
            processed_records.append(processed_record)
        
        return processed_records
    
    result = preprocess_query_results_safe(result)

    injected_data = []
    for item in result:
        tuple_data = (
            item.get('create_time', 0),           # 位置0: timestamp
            item.get('breath_bpm', 0),            # 位置1: breath_bpm
            item.get('breath_line', 0),           # 位置2: breath_line
            item.get('heart_bpm', 0),             # 位置3: heart_bpm
            item.get('heart_line', 0),            # 位置4: heart_line
            item.get('distance', 0),              # 位置5: target_distance
            item.get('signal_intensity', 0),      # 位置6: signal_strength
            item.get('state', 0),                 # 位置7: state
            item.get('body_move_data', 0),        # 位置8: body_move_energy (缺失字段)
            0,                                    # 位置9: body_move_range (缺失字段)
            0,                                    # 位置10: in_bed 
            item.get('device_sn', '00000001')     # 位置11: device_id
        )
        injected_data.append(tuple_data)
    
    # 创建 SocketServerManager 实例
    socket_manager = SocketServerManager(
        max_producers=10,      # 最大生产者数量
        max_consumers=15,      # 最大消费者数量
        production_queue_size=500,  # 生产队列大小
        consumer_tool_pool=consumer_tool_pool,
        sliding_window_size=20,
        # injected_data=injected_data
    )
    
    # 尝试启动服务器，如果端口被占用就跳过（说明已经有实例在运行）
    try:
        socket_manager.start_socket_server(port=8888, backlog=5)
        print("✅ SocketServerManager 启动成功")
    except OSError as e:
        if "Address already in use" in str(e):
            print("⚠️ 端口 8888 已被占用，服务器可能已经在运行中")
        else:
            print(f"❌ 启动失败: {e}")
            # 不要抛出异常，继续返回 socket_manager
    
    return socket_manager

# 获取共享的 socket_manager 实例
st.session_state.socket_manager = get_shared_socket_manager()

from whoami.tool.real_time_vital_analyze.data_logger import DataLogger
logger = DataLogger()

def get_available_devices():
    """获取所有可用的设备ID"""
    socket_manager = st.session_state.socket_manager
    if hasattr(socket_manager, 'latest_real_time_data'):
        return list(socket_manager.latest_real_time_data.keys())
    return []

def get_current_port():
    """获取当前streamlit应用的端口号"""
    try:
        # 从环境变量或配置中获取端口
        import streamlit.web.bootstrap as bootstrap
        # 默认端口
        port = 8501
        # 尝试从配置中获取实际端口
        if hasattr(st, 'get_option'):
            port = st.get_option('server.port') or 8501
        return port
    except:
        return 8501



def generate_health_report_async(start_datetime, end_datetime, device_id):
    """异步生成健康报告"""
    try:
        st.session_state.report_generating = True
        st.session_state.report_status = 'processing'
        st.session_state.report_message = "正在生成健康报告，请稍候..."
        
        # 添加详细日志
        print(f"📋 开始生成报告...")
        print(f"📋 设备ID: {device_id}")
        print(f"📋 开始时间: {start_datetime}")
        print(f"📋 结束时间: {end_datetime}")
        
        # 导入报告生成模块
        from whoami.tool.real_time_vital_analyze.health_report_generate import HealthReportGenerate
        
        print(f"📋 模块导入成功，创建报告生成实例...")
        
        # 使用传入的参数
        health_report_generate = HealthReportGenerate(
            start_date=start_datetime,
            end_date=end_datetime,
            device_sn=device_id
        )
        
        print(f"📋 实例创建成功，开始生成报告...")
        
        # 生成简化报告
        report = health_report_generate.generate_comprehensive_report()
        
        print(f"📋 报告生成完成！")
        print(f"📋 报告内容长度: {len(str(report)) if report else 0}")
        
        # 更新状态为成功，并显示报告摘要
        st.session_state.report_generating = False
        st.session_state.report_status = 'success'
        st.session_state.report_message = f"""健康报告生成成功！
时间范围：{start_datetime} 至 {end_datetime}
设备编号：{device_id}
报告数据量：{len(str(report)) if report else 0} 字符
生成时间：{datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')}"""
        st.session_state.report_available = True
        st.session_state.report_result = report  # 存储报告结果
        
    except ImportError as e:
        error_msg = f"模块导入失败：{str(e)}"
        print(f"❌ {error_msg}")
        st.session_state.report_generating = False
        st.session_state.report_status = 'error'
        st.session_state.report_message = error_msg
        st.session_state.report_available = False
        
    except Exception as e:
        error_msg = f"报告生成失败：{str(e)}"
        print(f"❌ {error_msg}")
        print(f"❌ 错误类型：{type(e).__name__}")
        import traceback
        print(f"❌ 详细错误：{traceback.format_exc()}")
        
        st.session_state.report_generating = False
        st.session_state.report_status = 'error'
        st.session_state.report_message = f"""{error_msg}
错误类型：{type(e).__name__}
设备ID：{device_id}
时间范围：{start_datetime} ~ {end_datetime}"""
        st.session_state.report_available = False




def generate_data():
    """生成新数据点"""
    current_time = time.time()
    
    # 固定1秒更新一次
    if current_time - st.session_state.last_update < 1.0:
        return
    
    socket_manager = st.session_state.socket_manager
        
    selected_device_id = st.session_state.selected_device_id
    
    # 如果device_sn为"未知设备"或者在socket_manager中找不到，使用模拟数据
    if (selected_device_id == "未知设备" or 
        not hasattr(socket_manager, 'latest_real_time_data') or 
        selected_device_id not in socket_manager.latest_real_time_data):
        
        # 伪造数据
        hr = 75 + np.sin(current_time * 0.1) * 8 + random.uniform(-3, 3)
        br = 16 + np.sin(current_time * 0.08) * 2 + random.uniform(-1, 1)
        
        current_datetime = datetime.now(beijing_tz)
        
        data_point = {
            'time': current_datetime.strftime('%H:%M:%S'),
            'timestamp': current_time,
            'datetime': current_datetime.strftime('%Y-%m-%d %H:%M:%S'),  # 新增完整日期时间
            'heart_rate': max(60, min(100, round(hr, 1))),
            'breathing_rate': max(12, min(20, round(br, 1))),
            # 修改为8种状态之一（原始5个状态 + 新增3个状态）
            'status': random.choice(['在床正常', '离床', '呼吸急促', '体动', '呼吸暂停', '清醒', '浅睡眠', '深睡眠']),
            'reconstruction_loss': random.uniform(0.1, 2.0),  # 伪造重构损失值
            # 添加原始线条数据
            'breath_line': random.uniform(0.8, 1.2),  # 伪造呼吸线数据
            'heart_line': random.uniform(0.9, 1.1),   # 伪造心线数据
            'device_id': selected_device_id
        }
        
        st.session_state.data.append(data_point)
    else:
        try:
            item = socket_manager.latest_real_time_data[selected_device_id]
            
            # 获取label数据
            label_data = None
            if hasattr(socket_manager, 'latest_real_time_label') and selected_device_id in socket_manager.latest_real_time_label:
                label_data = socket_manager.latest_real_time_label[selected_device_id]
            
            # 从实时数据中获取时间戳并转换为北京时间
            real_timestamp = item[0]  # 实时数据中的时间戳
            real_datetime = datetime.fromtimestamp(real_timestamp, tz=beijing_tz)
            
            # 解析item数据（根据你的实际数据结构调整）
            data_point = {
                'time': real_datetime.strftime('%H:%M:%S'),
                'timestamp': current_time,
                'datetime': real_datetime.strftime('%Y-%m-%d %H:%M:%S'),  # 使用实时数据时间
                'heart_rate': item[3],  # 根据你的item结构调整
                'breathing_rate': item[1],  # 根据你的item结构调整
                'device_id': selected_device_id,
                # 修改状态定义，使用label_data[2]，可以是8种状态之一
                'status': label_data[2] if label_data and len(label_data) > 2 else random.choice(['在床正常', '离床', '呼吸急促', '体动', '呼吸暂停', '清醒', '浅睡眠', '深睡眠']),
                'reconstruction_loss': label_data[1] if label_data else 0.0,  # float值
                # 添加原始线条数据
                'breath_line': item[2],  # 呼吸线数据
                'heart_line': item[4]    # 心线数据
            }
            
            logger.write_to_json(data_point)
            st.session_state.data.append(data_point)
            
        except queue.Empty:
            st.error(f"读取设备 {selected_device_id} 队列为空")
            return
        except Exception as e:
            st.error(f"读取设备 {selected_device_id} 队列出错: {e}")
    
    # 根据设置限制数据点数量
    if len(st.session_state.data) > st.session_state.max_points:
        st.session_state.data = st.session_state.data[-st.session_state.max_points:]
    
    st.session_state.last_update = current_time



def create_reconstruction_loss_chart(title, rl_values, statuses, times):
    """创建重构损失值和状态的复合图表"""
    svg_width = 700
    svg_height = 280  # 增加高度以容纳更多状态图例
    padding_left = 50
    padding_right = 30
    padding_top = 25
    padding_bottom = 80  # 增加底部空间给8个状态的图例
    chart_width = svg_width - padding_left - padding_right
    chart_height = svg_height - padding_top - padding_bottom
    
    # 动态计算Y轴范围
    if not rl_values:
        y_min, y_max = 0, 3
    else:
        min_val = min(rl_values)
        max_val = max(rl_values)
        if min_val == max_val:
            if min_val == 0:
                y_min, y_max = 0, 3
            else:
                center = min_val
                range_size = max(abs(center * 0.2), 1.5)
                y_min, y_max = max(0, center - range_size), center + range_size
        else:
            range_size = max_val - min_val
            margin = range_size * 0.1
            y_min = max(0, min_val - margin)
            y_max = max_val + margin
            if y_max - y_min < 1.5:
                center = (y_min + y_max) / 2
                y_min = max(0, center - 0.75)
                y_max = center + 0.75
    
    # 计算数据点
    points = []
    for i, value in enumerate(rl_values):
        x = padding_left + (i / max(1, len(rl_values) - 1)) * chart_width
        if y_max != y_min:
            y = padding_top + chart_height - ((value - y_min) / (y_max - y_min)) * chart_height
        else:
            y = padding_top + chart_height / 2
        points.append((x, y, statuses[i]))
    
    # Y轴刻度
    y_ticks = []
    for i in range(6):
        tick_value = y_min + (y_max - y_min) * i / 5
        tick_y = padding_top + chart_height - (i / 5) * chart_height
        y_ticks.append((tick_y, tick_value))
    
    # X轴刻度
    x_ticks = []
    if len(times) <= 10:
        tick_count = len(times)
    elif len(times) <= 50:
        tick_count = 8
    else:
        tick_count = 6
        
    for i in range(tick_count):
        if tick_count == 1:
            idx = 0
        else:
            idx = int(i * (len(times) - 1) / (tick_count - 1))
        tick_x = padding_left + (i / max(1, tick_count - 1)) * chart_width
        x_ticks.append((tick_x, times[idx]))
    
    # 生成SVG
    svg_content = f"""
    <div style="margin: 2px 0;">
        <h4 style="color: #9C27B0; margin-bottom: 3px; text-align: center; font-size: 1rem;">{title}</h4>
        <svg width="{svg_width}" height="{svg_height}" style="border: 1px solid #ddd; background: white;">
            <!-- 网格线 -->
            <defs>
                <pattern id="grid_rl" width="8" height="8" patternUnits="userSpaceOnUse">
                    <path d="M 8 0 L 0 0 0 8" fill="none" stroke="#f5f5f5" stroke-width="1"/>
                </pattern>
            </defs>
            <rect x="{padding_left}" y="{padding_top}" width="{chart_width}" height="{chart_height}" fill="url(#grid_rl)"/>
            
            <!-- 状态区域背景 -->"""
    
    # 添加状态区域的背景高亮
    for i in range(len(points)):
        status = statuses[i]
        if status != '在床正常':  # 只有非正常状态才显示背景
            x_start = points[i][0] - (chart_width / max(1, len(points) - 1)) / 2 if len(points) > 1 else padding_left
            x_end = points[i][0] + (chart_width / max(1, len(points) - 1)) / 2 if len(points) > 1 else padding_left + chart_width
            # 使用状态对应的颜色
            color = STATUS_COLORS.get(status, '#FFE6E6')
            svg_content += f"""
            <rect x="{max(padding_left, x_start)}" y="{padding_top}" width="{min(chart_width, x_end - x_start)}" height="{chart_height}" fill="{color}" opacity="0.2"/>"""
    
    svg_content += f"""
            <!-- Y轴 -->
            <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_height}" stroke="#333" stroke-width="1.5"/>
            
            <!-- X轴 -->
            <line x1="{padding_left}" y1="{padding_top + chart_height}" x2="{padding_left + chart_width}" y2="{padding_top + chart_height}" stroke="#333" stroke-width="1.5"/>
            
            <!-- Y轴刻度和标签 -->"""
    
    for tick_y, tick_value in y_ticks:
        svg_content += f"""
            <line x1="{padding_left - 3}" y1="{tick_y}" x2="{padding_left}" y2="{tick_y}" stroke="#333" stroke-width="1"/>
            <text x="{padding_left - 8}" y="{tick_y + 3}" fill="#666" font-size="9" text-anchor="end">{tick_value:.2f}</text>"""
    
    # X轴刻度和标签
    for tick_x, tick_time in x_ticks:
        svg_content += f"""
            <line x1="{tick_x}" y1="{padding_top + chart_height}" x2="{tick_x}" y2="{padding_top + chart_height + 3}" stroke="#333" stroke-width="1"/>
            <text x="{tick_x}" y="{padding_top + chart_height + 15}" fill="#666" font-size="9" text-anchor="middle">{tick_time}</text>"""
    
    # 数据折线（根据状态分段着色）
    if len(points) > 1:
        for i in range(len(points) - 1):
            x1, y1, status1 = points[i]
            x2, y2, status2 = points[i + 1]
            # 使用第一个点的状态颜色
            line_color = STATUS_COLORS.get(status1, '#4CAF50')
            svg_content += f"""
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{line_color}" stroke-width="2" stroke-linecap="round"/>"""
    
    # 数据点（根据状态着色）
    svg_content += f"""
            <!-- 数据点 -->"""
    
    for x, y, status in points:
        point_color = STATUS_COLORS.get(status, '#4CAF50')
        point_size = "5" if status != '在床正常' else "3"  # 异常状态点稍大
        svg_content += f"""
            <circle cx="{x}" cy="{y}" r="{point_size}" fill="{point_color}" stroke="white" stroke-width="1"/>"""
    
    # 图例和标签
    svg_content += f"""
            <!-- 图例 -->"""
    
    # 动态生成图例 - 显示所有8个状态
    legend_x = padding_left + 10
    legend_y = padding_top + chart_height + 25
    
    # 分两行显示8个状态图例
    for i, (status, color) in enumerate(STATUS_COLORS.items()):
        x_pos = legend_x + (i % 4) * 100  # 每行4个状态
        y_pos = legend_y + (i // 4) * 15  # 分两行
        svg_content += f"""
            <circle cx="{x_pos}" cy="{y_pos}" r="3" fill="{color}" stroke="white" stroke-width="1"/>
            <text x="{x_pos + 8}" y="{y_pos + 3}" fill="{color}" font-size="8">{status}</text>"""
    
    # 当前状态显示
    latest_status = statuses[-1]
    svg_content += f"""
            <!-- 当前值显示 -->
            <text x="{svg_width - 15}" y="20" fill="#9C27B0" font-size="12" font-weight="bold" text-anchor="end">当前: {rl_values[-1]:.3f}</text>
            
            <!-- 当前状态显示 -->
            <text x="{svg_width - 15}" y="35" fill="{STATUS_COLORS.get(latest_status, '#4CAF50')}" font-size="10" font-weight="bold" text-anchor="end">{latest_status}</text>
            
            <!-- 范围显示 -->
            <text x="{svg_width - 15}" y="50" fill="#999" font-size="9" text-anchor="end">范围: {y_min:.2f} - {y_max:.2f}</text>"""
    
    svg_content += f"""
            <!-- Y轴标签 -->
            <text x="15" y="{padding_top + chart_height/2}" fill="#333" font-size="10" text-anchor="middle" transform="rotate(-90, 15, {padding_top + chart_height/2})">Loss</text>
            
            <!-- X轴标签 -->
            <text x="{padding_left + chart_width/2}" y="{svg_height - 8}" fill="#333" font-size="10" text-anchor="middle">时间</text>
            
        </svg>
    </div>"""
    
    return svg_content

def create_chart_html(chart_data):
    """创建带坐标轴的紧凑SVG图表"""
    times = [point['time'] for point in chart_data]
    hr_values = [point['heart_rate'] for point in chart_data]
    br_values = [point['breathing_rate'] for point in chart_data]
    rl_values = [point['reconstruction_loss'] for point in chart_data]
    statuses = [point['status'] for point in chart_data]
    # 添加原始线条数据
    breath_line_values = [point['breath_line'] for point in chart_data]
    heart_line_values = [point['heart_line'] for point in chart_data]
    
    # 紧凑的图表尺寸
    svg_width = 350  # 减小宽度以容纳更多图表
    svg_height = 200  # 减小高度
    padding_left = 40
    padding_right = 25
    padding_top = 20
    padding_bottom = 35
    
    # 可用绘图区域
    chart_width = svg_width - padding_left - padding_right
    chart_height = svg_height - padding_top - padding_bottom
    
    def calculate_dynamic_range(values, default_min, default_max):
        """动态计算坐标轴范围"""
        if not values:
            return default_min, default_max
        
        min_val = min(values)
        max_val = max(values)
        
        # 如果最大值和最小值相同（比如都是0），使用默认范围
        if min_val == max_val:
            if min_val == 0:
                return default_min, default_max
            else:
                # 以当前值为中心，创建合理范围
                center = min_val
                range_size = max(abs(center * 0.2), (default_max - default_min) * 0.1)
                return center - range_size, center + range_size
        
        # 添加10%的边距使图表更美观
        range_size = max_val - min_val
        margin = range_size * 0.1
        y_min = max(0, min_val - margin) if default_min >= 0 else min_val - margin
        y_max = max_val + margin
        
        # 确保范围不会太小
        if y_max - y_min < (default_max - default_min) * 0.3:
            center = (y_min + y_max) / 2
            half_range = (default_max - default_min) * 0.15
            y_min = max(0, center - half_range) if default_min >= 0 else center - half_range
            y_max = center + half_range
        
        return y_min, y_max
    
    def create_chart_svg(title, values, color, default_min, default_max, unit):
        # 动态计算Y轴范围
        y_min, y_max = calculate_dynamic_range(values, default_min, default_max)
        
        # 计算数据点
        points = []
        for i, value in enumerate(values):
            x = padding_left + (i / max(1, len(values) - 1)) * chart_width
            if y_max != y_min:  # 避免除零错误
                y = padding_top + chart_height - ((value - y_min) / (y_max - y_min)) * chart_height
            else:
                y = padding_top + chart_height / 2  # 如果范围为0，放在中间
            points.append(f"{x},{y}")
        
        # Y轴刻度
        y_ticks = []
        for i in range(6):  # 5个刻度间隔
            tick_value = y_min + (y_max - y_min) * i / 5
            tick_y = padding_top + chart_height - (i / 5) * chart_height
            y_ticks.append((tick_y, tick_value))
        
        # X轴刻度（显示时间）
        x_ticks = []
        if len(times) <= 10:
            tick_count = len(times)
        elif len(times) <= 30:
            tick_count = 5
        else:
            tick_count = 4
            
        for i in range(tick_count):
            if tick_count == 1:
                idx = 0
            else:
                idx = int(i * (len(times) - 1) / (tick_count - 1))
            tick_x = padding_left + (i / max(1, tick_count - 1)) * chart_width
            x_ticks.append((tick_x, times[idx]))
        
        # 生成SVG
        svg_content = f"""
        <div style="margin: 2px; display: inline-block; vertical-align: top;">
            <h5 style="color: {color}; margin-bottom: 3px; text-align: center; font-size: 0.9rem;">{title}</h5>
            <svg width="{svg_width}" height="{svg_height}" style="border: 1px solid #ddd; background: white;">
                <!-- 网格线 -->
                <defs>
                    <pattern id="grid_{title.replace(' ', '_')}" width="8" height="8" patternUnits="userSpaceOnUse">
                        <path d="M 8 0 L 0 0 0 8" fill="none" stroke="#f5f5f5" stroke-width="1"/>
                    </pattern>
                </defs>
                <rect x="{padding_left}" y="{padding_top}" width="{chart_width}" height="{chart_height}" fill="url(#grid_{title.replace(' ', '_')})"/>
                
                <!-- Y轴 -->
                <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_height}" stroke="#333" stroke-width="1.5"/>
                
                <!-- X轴 -->
                <line x1="{padding_left}" y1="{padding_top + chart_height}" x2="{padding_left + chart_width}" y2="{padding_top + chart_height}" stroke="#333" stroke-width="1.5"/>
                
                <!-- Y轴刻度和标签 -->"""
        
        for tick_y, tick_value in y_ticks:
            svg_content += f"""
                <line x1="{padding_left - 3}" y1="{tick_y}" x2="{padding_left}" y2="{tick_y}" stroke="#333" stroke-width="1"/>
                <text x="{padding_left - 6}" y="{tick_y + 3}" fill="#666" font-size="8" text-anchor="end">{tick_value:.1f}</text>"""
        
        # X轴刻度和标签
        for tick_x, tick_time in x_ticks:
            svg_content += f"""
                <line x1="{tick_x}" y1="{padding_top + chart_height}" x2="{tick_x}" y2="{padding_top + chart_height + 3}" stroke="#333" stroke-width="1"/>
                <text x="{tick_x}" y="{padding_top + chart_height + 12}" fill="#666" font-size="7" text-anchor="middle">{tick_time}</text>"""
        
        # 数据折线（只有多个点时才绘制）
        if len(points) > 1:
            svg_content += f"""
                <!-- 数据折线 -->
                <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round"/>"""
        
        # 数据点
        svg_content += f"""
                <!-- 数据点 -->"""
        
        for point in points:
            x, y = point.split(',')
            svg_content += f"""
                <circle cx="{x}" cy="{y}" r="2" fill="{color}" stroke="white" stroke-width="1"/>"""
        
        # 轴标签
        svg_content += f"""
                <!-- Y轴标签 -->
                <text x="12" y="{padding_top + chart_height/2}" fill="#333" font-size="8" text-anchor="middle" transform="rotate(-90, 12, {padding_top + chart_height/2})">{unit}</text>
                
                <!-- 当前值显示 -->
                <text x="{svg_width - 10}" y="15" fill="{color}" font-size="10" font-weight="bold" text-anchor="end">{values[-1]:.1f}</text>
                
            </svg>
        </div>"""
        
        return svg_content
    
    # 生成五个图表，分为两行显示
    hr_svg = create_chart_svg("心率", hr_values, "#FF4B4B", 50, 110, "BPM")
    heart_line_svg = create_chart_svg("心线原始", heart_line_values, "#FF8080", -2, 2, "")
    br_svg = create_chart_svg("呼吸率", br_values, "#00CC88", 10, 25, "/min")
    breath_line_svg = create_chart_svg("呼吸线原始", breath_line_values, "#80CCAA", -2, 2, "")
    
    # 重构损失图表保持原来的大宽度，使用8种状态
    rl_svg = create_reconstruction_loss_chart("重构损失值 & 状态检测", rl_values, statuses, times)
    
    # 返回分行布局的HTML
    return f"""
    <div style="text-align: center;">
        <div style="margin-bottom: 10px;">
            {hr_svg}
            {heart_line_svg}
        </div>
        <div style="margin-bottom: 10px;">
            {br_svg}
            {breath_line_svg}
        </div>
        <div>
            {rl_svg}
        </div>
    </div>
    """

# 显示当前监控设备信息
available_devices = get_available_devices()
device_count = len(available_devices)

# 显示设备信息
st.markdown(f"""
<div style="background-color: #f0f2f6; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.85rem;">
🖥️ <strong>监控设备:</strong> {device_sn} | 
📡 <strong>可用设备数:</strong> {device_count} | 
🔗 <strong>数据源:</strong> {'实时数据' if device_sn in available_devices else '模拟数据'}
</div>
""", unsafe_allow_html=True)

# 显示监控时间信息
if st.session_state.monitor_start_datetime or st.session_state.monitor_end_datetime:
    start_info = st.session_state.monitor_start_datetime or "未记录"
    end_info = st.session_state.monitor_end_datetime or "监控中..." if st.session_state.monitoring else "未记录"
    st.markdown(f"""
    <div style="background-color: #e8f5e8; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.85rem;">
    ⏰ <strong>监控时段:</strong> {start_info} ~ {end_info}
    </div>
    """, unsafe_allow_html=True)

# 控制按钮和设置
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    if st.button("开始监控", type="primary"):
        st.session_state.monitoring = True
        # 记录开始时间 - 从当前实时数据中获取或使用当前时间
        if st.session_state.data:
            # 如果已有数据，使用最新数据的时间
            latest_data = st.session_state.data[-1]
            st.session_state.monitor_start_datetime = latest_data['datetime']
        else:
            # 如果没有数据，使用当前时间
            current_datetime = datetime.now(beijing_tz)
            st.session_state.monitor_start_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        
        st.session_state.monitor_start_time = time.time()
        
        # 重置报告状态
        st.session_state.report_status = None
        st.session_state.report_message = ""
        st.session_state.report_available = False

with col2:
    if st.button("停止监控"):
        st.session_state.monitoring = False
        # 记录结束时间 - 从当前实时数据中获取
        if st.session_state.data:
            latest_data = st.session_state.data[-1]
            st.session_state.monitor_end_datetime = latest_data['datetime']
        else:
            current_datetime = datetime.now(beijing_tz)
            st.session_state.monitor_end_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        
        st.session_state.monitor_end_time = time.time()
        
        # 停止监控后自动开始生成报告
        if st.session_state.monitor_start_datetime and st.session_state.monitor_end_datetime:
            # 传递参数给线程，避免在线程中访问session_state
            report_thread = threading.Thread(
                target=generate_health_report_async,
                args=(
                    st.session_state.monitor_start_datetime,
                    st.session_state.monitor_end_datetime,
                    st.session_state.selected_device_id
                )
            )
            report_thread.daemon = True
            report_thread.start()

with col3:
    if st.button("清除数据"):
        st.session_state.data = []
        # 清除监控时间记录
        st.session_state.monitor_start_time = None
        st.session_state.monitor_end_time = None
        st.session_state.monitor_start_datetime = None
        st.session_state.monitor_end_datetime = None
        # 清除报告状态
        st.session_state.report_status = None
        st.session_state.report_message = ""
        st.session_state.report_available = False

with col4:
    # 数据点数量设置
    new_max = st.selectbox(
        "最大数据点", 
        [50, 100, 200, 500, 1000], 
        index=2,  # 默认200
        key="max_points_selector"
    )
    if new_max != st.session_state.max_points:
        st.session_state.max_points = new_max
        # 如果当前数据超过新限制，裁剪数据
        if len(st.session_state.data) > new_max:
            st.session_state.data = st.session_state.data[-new_max:]

with col5:
    if st.button("📊 数据详情"):
        st.session_state.show_data_modal = not st.session_state.show_data_modal

with col6:
    if st.button("📈 高级图表"):
        st.session_state.show_chart_modal = not st.session_state.show_chart_modal

with col7:
    # 报告查看按钮 - 无条件可以点击
    current_port = get_current_port()
    current_device_sn = st.session_state.selected_device_id
    report_url = f"http://1.71.15.121:{current_port}/report?device_sn={current_device_sn}"
    
    if st.button("📋 查看报告"):
        # 使用st.link_button或直接跳转
        st.markdown(f'<meta http-equiv="refresh" content="0; url={report_url}">', unsafe_allow_html=True)

# 显示报告生成状态
# 显示报告生成状态
if st.session_state.report_status:
    if st.session_state.report_status == 'processing':
        st.markdown(f"""
        <div class="report-status report-processing">
        🔄 {st.session_state.report_message}
        </div>
        """, unsafe_allow_html=True)
        
        # 显示进度条
        with st.spinner('正在生成健康报告...'):
            time.sleep(1)  # 给线程时间执行
    
    elif st.session_state.report_status == 'success':
        st.markdown(f"""
        <div class="report-status report-success">
        ✅ {st.session_state.report_message}
        </div>
        """, unsafe_allow_html=True)
        
        # 显示报告结果摘要
        if hasattr(st.session_state, 'report_result') and st.session_state.report_result:
            with st.expander("📋 报告生成结果详情", expanded=False):
                st.json(st.session_state.report_result if isinstance(st.session_state.report_result, dict) else {"报告内容": str(st.session_state.report_result)[:500] + "..."})
        
    elif st.session_state.report_status == 'error':
        st.markdown(f"""
        <div class="report-status report-error">
        ❌ {st.session_state.report_message}
        </div>
        """, unsafe_allow_html=True)
        
        # 显示调试信息
        with st.expander("🔍 错误调试信息", expanded=True):
            st.text(f"当前状态变量：")
            st.text(f"- monitoring: {st.session_state.monitoring}")
            st.text(f"- monitor_start_datetime: {st.session_state.monitor_start_datetime}")
            st.text(f"- monitor_end_datetime: {st.session_state.monitor_end_datetime}")
            st.text(f"- selected_device_id: {st.session_state.selected_device_id}")
            st.text(f"- data points: {len(st.session_state.data)}")
            st.text(f"- device_sn (URL): {device_sn}")
            st.text(f"- report_generating: {st.session_state.report_generating}")
            st.text(f"- report_available: {st.session_state.report_available}")

# 显示当前状态 - 紧凑样式
st.markdown(f"""
<div style="background-color: #f0f2f6; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.85rem;">
📊 <strong>当前数据点:</strong> {len(st.session_state.data)}/{st.session_state.max_points} | 
⏱️ <strong>更新频率:</strong> 1秒/次 |
📋 <strong>报告状态:</strong> {'可查看' if st.session_state.report_available else '等待生成'}
</div>
""", unsafe_allow_html=True)

# 如果正在监控，生成新数据
if st.session_state.monitoring:
    generate_data()

# 使用所有数据进行显示（不再限制显示数量）
chart_data = st.session_state.data if st.session_state.data else []

# 使用 Elements 创建无闪动的实时界面
with elements("realtime_monitor"):
    
    if chart_data:
        latest = chart_data[-1]
        current_device_display = latest.get('device_id', device_sn)
        
        # 顶部指标卡片 - 修改状态显示以支持双状态
        with mui.Grid(container=True, spacing=1.5):
            # 心率卡片
            with mui.Grid(item=True, xs=2.4):
                with mui.Card(elevation=2, sx={"height": "120px"}):
                    with mui.CardContent(sx={"padding": "12px", "&:last-child": {"paddingBottom": "12px"}}):
                        mui.Typography("心率", variant="subtitle2", color="textSecondary", sx={"fontSize": "0.8rem"})
                        mui.Typography(
                            f"{latest['heart_rate']} BPM", 
                            variant="h6", 
                            color="error",
                            sx={"fontWeight": "bold", "margin": "4px 0"}
                        )
                        if len(chart_data) > 1:
                            delta = latest['heart_rate'] - chart_data[-2]['heart_rate']
                            mui.Typography(
                                f"{'↑' if delta > 0 else '↓'} {abs(delta):.1f}",
                                variant="caption",
                                color="success" if 60 <= latest['heart_rate'] <= 100 else "warning",
                                sx={"fontSize": "0.75rem"}
                            )
            
            # 心线卡片
            with mui.Grid(item=True, xs=2.4):
                with mui.Card(elevation=2, sx={"height": "120px"}):
                    with mui.CardContent(sx={"padding": "12px", "&:last-child": {"paddingBottom": "12px"}}):
                        mui.Typography("心线原始", variant="subtitle2", color="textSecondary", sx={"fontSize": "0.8rem"})
                        mui.Typography(
                            f"{latest['heart_line']:.3f}", 
                            variant="h6", 
                            color="error",
                            sx={"fontWeight": "bold", "margin": "4px 0", "opacity": 0.8}
                        )
                        mui.Typography(
                            "原始信号",
                            variant="caption",
                            color="textSecondary",
                            sx={"fontSize": "0.7rem"}
                        )
            
            # 呼吸率卡片
            with mui.Grid(item=True, xs=2.4):
                with mui.Card(elevation=2, sx={"height": "120px"}):
                    with mui.CardContent(sx={"padding": "12px", "&:last-child": {"paddingBottom": "12px"}}):
                        mui.Typography("呼吸率", variant="subtitle2", color="textSecondary", sx={"fontSize": "0.8rem"})
                        mui.Typography(
                            f"{latest['breathing_rate']} /min", 
                            variant="h6", 
                            color="primary",
                            sx={"fontWeight": "bold", "margin": "4px 0"}
                        )
                        if len(chart_data) > 1:
                            delta = latest['breathing_rate'] - chart_data[-2]['breathing_rate']
                            mui.Typography(
                                f"{'↑' if delta > 0 else '↓'} {abs(delta):.1f}",
                                variant="caption",
                                color="success" if 12 <= latest['breathing_rate'] <= 20 else "warning",
                                sx={"fontSize": "0.75rem"}
                            )
            
            # 呼吸线卡片
            with mui.Grid(item=True, xs=2.4):
                with mui.Card(elevation=2, sx={"height": "120px"}):
                    with mui.CardContent(sx={"padding": "12px", "&:last-child": {"paddingBottom": "12px"}}):
                        mui.Typography("呼吸线原始", variant="subtitle2", color="textSecondary", sx={"fontSize": "0.8rem"})
                        mui.Typography(
                            f"{latest['breath_line']:.3f}", 
                            variant="h6", 
                            color="primary",
                            sx={"fontWeight": "bold", "margin": "4px 0", "opacity": 0.8}
                        )
                        mui.Typography(
                            "原始信号",
                            variant="caption",
                            color="textSecondary",
                            sx={"fontSize": "0.7rem"}
                        )
            
            # 状态卡片 - 修改为显示8种状态
            with mui.Grid(item=True, xs=2.4):
                with mui.Card(elevation=2, sx={"height": "120px"}):
                    with mui.CardContent(sx={"padding": "12px", "&:last-child": {"paddingBottom": "12px"}}):
                        mui.Typography("状态监测", variant="subtitle2", color="textSecondary", sx={"fontSize": "0.8rem"})
                        
                        # 获取当前状态的颜色
                        current_status = latest['status']
                        status_color_mapping = {
                            '在床正常': 'success',
                            '离床': 'warning', 
                            '呼吸急促': 'error',
                            '体动': 'info',
                            '呼吸暂停': 'secondary',
                            '清醒': 'warning',
                            '浅睡眠': 'info', 
                            '深睡眠': 'primary'
                        }
                        
                        mui.Typography(
                            f"{STATUS_ICONS.get(current_status, '⚪')} {current_status}", 
                            variant="h6", 
                            color=status_color_mapping.get(current_status, 'textPrimary'),
                            sx={"fontWeight": "bold", "margin": "4px 0", "fontSize": "0.9rem"}
                        )
                        
                        # 重构损失值显示
                        mui.Typography(
                            f"Loss: {latest['reconstruction_loss']:.3f}",
                            variant="caption",
                            color="textSecondary",
                            sx={"fontSize": "0.75rem"}
                        )
    
    # 分隔线 - 紧凑间距
    mui.Divider(sx={"margin": "12px 0"})
    
    # 图表区域 - 使用SVG替代Canvas
    if len(chart_data) > 1:
        chart_html = create_chart_html(chart_data)
        html.div(
            dangerouslySetInnerHTML={"__html": chart_html}
        )

# 在数据详情中也显示设备信息和新增字段
if st.session_state.show_data_modal and chart_data:
    with st.expander("📊 数据详情", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**数据点总数:** {len(chart_data)}")
            st.markdown(f"**监控设备:** {device_sn}")
            if st.session_state.monitor_start_datetime:
                st.markdown(f"**监控开始:** {st.session_state.monitor_start_datetime}")
            if st.session_state.monitor_end_datetime:
                st.markdown(f"**监控结束:** {st.session_state.monitor_end_datetime}")
        
        # 创建数据表格
        df_display = pd.DataFrame(chart_data[-15:])  # 显示最近15条，更紧凑
        df_display['生理状态'] = df_display.apply(
            lambda row: '正常' if (60 <= row['heart_rate'] <= 100 and 12 <= row['breathing_rate'] <= 20) else '异常',
            axis=1
        )
        
        # 包含所有字段，显示8种状态
        if 'device_id' in df_display.columns:
            df_display = df_display[['time', 'datetime', 'heart_rate', 'heart_line', 'breathing_rate', 'breath_line', 'reconstruction_loss', 'status', 'device_id', '生理状态']]
            df_display.columns = ['时间', '完整日期时间', '心率(BPM)', '心线原始', '呼吸率(/min)', '呼吸线原始', '重构损失', '监测状态', '设备ID', '生理状态评估']
        else:
            df_display = df_display[['time', 'datetime', 'heart_rate', 'heart_line', 'breathing_rate', 'breath_line', 'reconstruction_loss', 'status', '生理状态']]
            df_display.columns = ['时间', '完整日期时间', '心率(BPM)', '心线原始', '呼吸率(/min)', '呼吸线原始', '重构损失', '监测状态', '生理状态评估']
        
        st.dataframe(df_display, use_container_width=True, height=300)

# 高级图表展开区域 - 修改为支持8种状态
if st.session_state.show_chart_modal and len(chart_data) > 1:
    with st.expander("📈 高级趋势图表", expanded=True):
        # 创建子图 - 5个图表
        fig = make_subplots(
            rows=5, cols=1,
            subplot_titles=('心率趋势', '心线原始信号', '呼吸率趋势', '呼吸线原始信号', '重构损失值趋势（8种状态）'),
            vertical_spacing=0.08  # 调整间距以适应5个图表
        )
        
        # 心率数据
        fig.add_trace(
            go.Scatter(
                x=[point['time'] for point in chart_data],
                y=[point['heart_rate'] for point in chart_data],
                mode='lines+markers',
                name='心率',
                line=dict(color='#FF4B4B', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # 心线原始数据
        fig.add_trace(
            go.Scatter(
                x=[point['time'] for point in chart_data],
                y=[point['heart_line'] for point in chart_data],
                mode='lines+markers',
                name='心线原始',
                line=dict(color='#FF8080', width=1.5),
                marker=dict(size=3)
            ),
            row=2, col=1
        )
        
        # 呼吸率数据
        fig.add_trace(
            go.Scatter(
                x=[point['time'] for point in chart_data],
                y=[point['breathing_rate'] for point in chart_data],
                mode='lines+markers',
                name='呼吸率',
                line=dict(color='#00CC88', width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )
        
        # 呼吸线原始数据
        fig.add_trace(
            go.Scatter(
                x=[point['time'] for point in chart_data],
                y=[point['breath_line'] for point in chart_data],
                mode='lines+markers',
                name='呼吸线原始',
                line=dict(color='#80CCAA', width=1.5),
                marker=dict(size=3)
            ),
            row=4, col=1
        )
        
        # 重构损失值数据 - 按8种状态着色
        status_groups = {}
        for point in chart_data:
            status = point['status']
            if status not in status_groups:
                status_groups[status] = {'times': [], 'values': []}
            status_groups[status]['times'].append(point['time'])
            status_groups[status]['values'].append(point['reconstruction_loss'])
        
        # 为每种状态创建散点图
        for status, data in status_groups.items():
            fig.add_trace(
                go.Scatter(
                    x=data['times'],
                    y=data['values'],
                    mode='markers',
                    name=status,
                    marker=dict(
                        color=STATUS_COLORS.get(status, '#4CAF50'), 
                        size=6 if status != '在床正常' else 4
                    )
                ),
                row=5, col=1
            )
        
        fig.update_layout(
            height=800,  # 适应5个图表的高度
            showlegend=True,
            title_text=f"设备 {device_sn} 生理参数监控趋势（含8种状态检测）",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        fig.update_xaxes(title_text="时间", row=5, col=1)
        fig.update_yaxes(title_text="心率 (BPM)", row=1, col=1)
        fig.update_yaxes(title_text="心线原始", row=2, col=1)
        fig.update_yaxes(title_text="呼吸率 (/min)", row=3, col=1)
        fig.update_yaxes(title_text="呼吸线原始", row=4, col=1)
        fig.update_yaxes(title_text="重构损失", row=5, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

# 自动刷新
if st.session_state.monitoring:
    time.sleep(0.5)  # 界面刷新频率，数据仍然1秒更新
    st.rerun()

# 底部说明 - 紧凑样式
st.markdown(f"""
<div style="border-top: 1px solid #e6e6e6; margin-top: 1rem; padding-top: 0.5rem;">
<details>
<summary style="cursor: pointer; font-weight: bold; color: #666;">✨ 功能特点</summary>
<div style="font-size: 0.85rem; margin-top: 0.5rem; color: #555;">
• <strong>设备监控:</strong> 设备编号 {device_sn}<br>
• <strong>访问方式:</strong> http://ip:port/real_time_vital/device_sn={device_sn}<br>
• <strong>完整监控:</strong> 心率 + 心线原始 + 呼吸率 + 呼吸线原始 + 8种状态检测<br>
• <strong>状态分类:</strong> 在床正常🟢 / 离床🟠 / 呼吸急促🔴 / 体动🔵 / 呼吸暂停🟣 / 清醒😊 / 浅睡眠😴 / 深睡眠😴<br>
• <strong>数据累加:</strong> 持续累积，支持50-1000个数据点<br>
• <strong>实时更新:</strong> 固定1秒更新频率<br>
• <strong>无闪动界面:</strong> Material-UI + 智能图表显示<br>
• <strong>并排显示:</strong> 心率与心线、呼吸率与呼吸线对比分析<br>
• <strong>8状态检测:</strong> 重构损失值分析 + 实时8种状态监控 + 彩色分类显示<br>
• <strong>展开式详情:</strong> 数据表格 + Plotly交互图表（含8种状态分类）<br>
• <strong>紧凑界面:</strong> 优化空间利用，专注监控体验<br>
• <strong>NEW 时间记录:</strong> 自动记录监控开始/结束时间（基于实时数据）<br>
• <strong>NEW 报告生成:</strong> 停止监控后自动生成健康报告，显示生成状态<br>
• <strong>NEW 报告查看:</strong> 点击"查看报告"按钮跳转到专门的报告页面<br>
• <strong>报告链接:</strong> http://localhost:{get_current_port()}/report?device_sn={device_sn}
</div>
</details>
</div>
""", unsafe_allow_html=True)

if not st.session_state.monitoring and len(st.session_state.data) == 0:
    st.markdown('<div style="background-color: #e3f2fd; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.9rem;">👆 点击"开始监控"开始实时数据累加展示（自动记录时间范围）</div>', unsafe_allow_html=True)
elif len(st.session_state.data) > 0:
    st.markdown(f'<div style="background-color: #e8f5e8; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.9rem;">💾 已累积 {len(st.session_state.data)} 个数据点，监控状态: {"🟢 运行中" if st.session_state.monitoring else "⏸️ 已暂停"}</div>', unsafe_allow_html=True)
    st.markdown('<div style="background-color: #fff3e0; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.85rem;">💡 点击 "📊 数据详情" 展开详细数据表格，点击 "📈 高级图表" 展开交互式分析图表，停止监控后可查看生成的健康报告</div>', unsafe_allow_html=True)