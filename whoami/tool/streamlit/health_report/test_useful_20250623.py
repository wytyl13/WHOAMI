#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/21 11:46
@Author  : weiyutao
@File    : test_.py
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

# 直接指定项目根目录
project_root = "/work/ai/WHOAMI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)



# 页面配置
st.set_page_config(
    page_title="Elements无闪动监控",
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
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💓 Elements 无闪动实时监控</h1>', unsafe_allow_html=True)

# 初始化状态
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
    st.session_state.data = []
    st.session_state.last_update = time.time()
    st.session_state.max_points = 200  # 可配置的最大数据点数
    st.session_state.show_data_modal = False  # 数据展开状态
    st.session_state.show_chart_modal = False  # 图表展开状态

if 'socket_manager' not in st.session_state:
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
    st.session_state.socket_manager = SocketServerManager(
        max_producers=10,      # 最大生产者数量
        max_consumers=15,      # 最大消费者数量
        production_queue_size=500,  # 生产队列大小
        consumer_tool_pool=consumer_tool_pool,
        sliding_window_size=60
    )
    st.session_state.socket_manager.start_socket_server(port=8000, backlog=5)


def get_available_devices():
    """获取所有可用的设备ID"""
    socket_manager = st.session_state.socket_manager
    if hasattr(socket_manager, 'latest_real_time_data'):
        return list(socket_manager.latest_real_time_data.keys())
    return []


def generate_data():
    """生成新数据点"""
    current_time = time.time()
    
    # 固定1秒更新一次
    if current_time - st.session_state.last_update < 1.0:
        return
    
    socket_manager = st.session_state.socket_manager
        
    selected_device_id = st.session_state.get('selected_device_id', None)
    if not selected_device_id or not hasattr(socket_manager, 'latest_real_time_data') or selected_device_id not in socket_manager.latest_real_time_data:
        # 伪造数据
        hr = 75 + np.sin(current_time * 0.1) * 8 + random.uniform(-3, 3)
        br = 16 + np.sin(current_time * 0.08) * 2 + random.uniform(-1, 1)
        
        data_point = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'timestamp': current_time,
            'heart_rate': max(60, min(100, round(hr, 1))),
            'breathing_rate': max(12, min(20, round(br, 1))),
            # 添加重构损失值相关字段
            'anomaly_status': random.choice([True, False]),  # 伪造异常状态
            'reconstruction_loss': random.uniform(0.1, 2.0)  # 伪造重构损失值
        }
        
        st.session_state.data.append(data_point)
    else:
        try:
            item = socket_manager.latest_real_time_data[selected_device_id]
            
            # 获取label数据
            label_data = None
            if hasattr(socket_manager, 'latest_real_time_label') and selected_device_id in socket_manager.latest_real_time_label:
                label_data = socket_manager.latest_real_time_label[selected_device_id]
            
            # 解析item数据（根据你的实际数据结构调整）
            data_point = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'timestamp': current_time,
                'heart_rate': item[3],  # 根据你的item结构调整
                'breathing_rate': item[1],  # 根据你的item结构调整
                'device_id': selected_device_id,
                # 添加label相关字段
                'anomaly_status': label_data[0] if label_data else False,  # bool值
                'reconstruction_loss': label_data[1] if label_data else 0.0  # float值
            }
            
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


# 设备选择器 - 放在右上角
col_title, col_device = st.columns([3, 1])

with col_title:
    st.markdown("")  # 占位

with col_device:
    # 获取可用设备
    available_devices = get_available_devices()
    
    if available_devices:
        # 添加"所有设备"选项
        device_options = ['模拟数据'] + available_devices
        
        # 设备选择下拉菜单
        selected_device = st.selectbox(
            "选择设备:",
            device_options,
            index=0,  # 默认选择模拟数据
            key="device_selector",
            help="选择要监控的设备"
        )
        
        # 保存选中的设备到session state
        if selected_device == '模拟数据':
            st.session_state.selected_device_id = None
        else:
            st.session_state.selected_device_id = selected_device
            
        # 如果切换了设备，清除旧数据
        if 'previous_device' not in st.session_state:
            st.session_state.previous_device = selected_device
        elif st.session_state.previous_device != selected_device:
            st.session_state.data = []  # 清除数据
            st.session_state.previous_device = selected_device
            st.info(f"已切换到设备: {selected_device}")
    else:
        st.info("暂无设备连接，使用模拟数据")
        st.session_state.selected_device_id = None

# 显示当前监控信息
current_device = st.session_state.get('selected_device_id', '模拟数据') or '模拟数据'
device_count = len(available_devices)


def create_reconstruction_loss_chart(title, rl_values, anomaly_statuses, times):
    """创建重构损失值和异常状态的复合图表"""
    svg_width = 700
    svg_height = 220
    padding_left = 50
    padding_right = 30
    padding_top = 25
    padding_bottom = 40
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
        points.append((x, y, anomaly_statuses[i]))
    
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
            
            <!-- 异常区域背景 -->"""
    
    # 添加异常区域的背景高亮
    for i in range(len(points)):
        if anomaly_statuses[i]:
            x_start = points[i][0] - (chart_width / max(1, len(points) - 1)) / 2 if len(points) > 1 else padding_left
            x_end = points[i][0] + (chart_width / max(1, len(points) - 1)) / 2 if len(points) > 1 else padding_left + chart_width
            svg_content += f"""
            <rect x="{max(padding_left, x_start)}" y="{padding_top}" width="{min(chart_width, x_end - x_start)}" height="{chart_height}" fill="#FFE6E6" opacity="0.3"/>"""
    
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
    
    # 数据折线（根据异常状态分段着色）
    if len(points) > 1:
        for i in range(len(points) - 1):
            x1, y1, status1 = points[i]
            x2, y2, status2 = points[i + 1]
            # 如果任一点异常，线段为红色；否则为蓝色
            line_color = "#FF4444" if status1 or status2 else "#4444FF"
            svg_content += f"""
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{line_color}" stroke-width="2" stroke-linecap="round"/>"""
    
    # 数据点（根据异常状态着色）
    svg_content += f"""
            <!-- 数据点 -->"""
    
    for x, y, is_anomaly in points:
        point_color = "#FF4444" if is_anomaly else "#4444FF"
        point_size = "4" if is_anomaly else "3"  # 异常点稍大
        svg_content += f"""
            <circle cx="{x}" cy="{y}" r="{point_size}" fill="{point_color}" stroke="white" stroke-width="1"/>"""
    
    # 图例和标签
    svg_content += f"""
            <!-- 图例 -->
            <circle cx="{padding_left + 20}" cy="{padding_top + 15}" r="3" fill="#4444FF" stroke="white" stroke-width="1"/>
            <text x="{padding_left + 28}" y="{padding_top + 18}" fill="#4444FF" font-size="9">正常</text>
            <circle cx="{padding_left + 70}" cy="{padding_top + 15}" r="3" fill="#FF4444" stroke="white" stroke-width="1"/>
            <text x="{padding_left + 78}" y="{padding_top + 18}" fill="#FF4444" font-size="9">异常</text>
            
            <!-- Y轴标签 -->
            <text x="15" y="{padding_top + chart_height/2}" fill="#333" font-size="10" text-anchor="middle" transform="rotate(-90, 15, {padding_top + chart_height/2})">Loss</text>
            
            <!-- X轴标签 -->
            <text x="{padding_left + chart_width/2}" y="{svg_height - 8}" fill="#333" font-size="10" text-anchor="middle">时间</text>
            
            <!-- 当前值显示 -->
            <text x="{svg_width - 15}" y="20" fill="#9C27B0" font-size="12" font-weight="bold" text-anchor="end">当前: {rl_values[-1]:.3f}</text>
            
            <!-- 当前状态显示 -->
            <text x="{svg_width - 15}" y="35" fill="{'#FF4444' if anomaly_statuses[-1] else '#4444FF'}" font-size="10" font-weight="bold" text-anchor="end">{'异常' if anomaly_statuses[-1] else '正常'}</text>
            
            <!-- 范围显示 -->
            <text x="{svg_width - 15}" y="50" fill="#999" font-size="9" text-anchor="end">范围: {y_min:.2f} - {y_max:.2f}</text>
            
        </svg>
    </div>"""
    
    return svg_content


def create_chart_html(chart_data):
    """创建带坐标轴的紧凑SVG图表"""
    times = [point['time'] for point in chart_data]
    hr_values = [point['heart_rate'] for point in chart_data]
    br_values = [point['breathing_rate'] for point in chart_data]
    # 添加重构损失值数据
    rl_values = [point['reconstruction_loss'] for point in chart_data]
    anomaly_statuses = [point['anomaly_status'] for point in chart_data]
    
    # 紧凑的图表尺寸
    svg_width = 700
    svg_height = 220
    padding_left = 50
    padding_right = 30
    padding_top = 25
    padding_bottom = 40
    
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
        y_min = max(0, min_val - margin)  # 心率和呼吸率不应该小于0
        y_max = max_val + margin
        
        # 确保范围不会太小
        if y_max - y_min < (default_max - default_min) * 0.3:
            center = (y_min + y_max) / 2
            half_range = (default_max - default_min) * 0.15
            y_min = max(0, center - half_range)
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
            <h4 style="color: {color}; margin-bottom: 3px; text-align: center; font-size: 1rem;">{title}</h4>
            <svg width="{svg_width}" height="{svg_height}" style="border: 1px solid #ddd; background: white;">
                <!-- 网格线 -->
                <defs>
                    <pattern id="grid" width="8" height="8" patternUnits="userSpaceOnUse">
                        <path d="M 8 0 L 0 0 0 8" fill="none" stroke="#f5f5f5" stroke-width="1"/>
                    </pattern>
                </defs>
                <rect x="{padding_left}" y="{padding_top}" width="{chart_width}" height="{chart_height}" fill="url(#grid)"/>
                
                <!-- Y轴 -->
                <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_height}" stroke="#333" stroke-width="1.5"/>
                
                <!-- X轴 -->
                <line x1="{padding_left}" y1="{padding_top + chart_height}" x2="{padding_left + chart_width}" y2="{padding_top + chart_height}" stroke="#333" stroke-width="1.5"/>
                
                <!-- Y轴刻度和标签 -->"""
        
        for tick_y, tick_value in y_ticks:
            svg_content += f"""
                <line x1="{padding_left - 3}" y1="{tick_y}" x2="{padding_left}" y2="{tick_y}" stroke="#333" stroke-width="1"/>
                <text x="{padding_left - 8}" y="{tick_y + 3}" fill="#666" font-size="9" text-anchor="end">{tick_value:.1f}</text>"""
        
        # X轴刻度和标签
        for tick_x, tick_time in x_ticks:
            svg_content += f"""
                <line x1="{tick_x}" y1="{padding_top + chart_height}" x2="{tick_x}" y2="{padding_top + chart_height + 3}" stroke="#333" stroke-width="1"/>
                <text x="{tick_x}" y="{padding_top + chart_height + 15}" fill="#666" font-size="9" text-anchor="middle">{tick_time}</text>"""
        
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
                <circle cx="{x}" cy="{y}" r="3" fill="{color}" stroke="white" stroke-width="1"/>"""
        
        # 轴标签
        svg_content += f"""
                <!-- Y轴标签 -->
                <text x="15" y="{padding_top + chart_height/2}" fill="#333" font-size="10" text-anchor="middle" transform="rotate(-90, 15, {padding_top + chart_height/2})">{unit}</text>
                
                <!-- X轴标签 -->
                <text x="{padding_left + chart_width/2}" y="{svg_height - 8}" fill="#333" font-size="10" text-anchor="middle">时间</text>
                
                <!-- 当前值显示 -->
                <text x="{svg_width - 15}" y="20" fill="{color}" font-size="12" font-weight="bold" text-anchor="end">当前: {values[-1]:.1f} {unit}</text>
                
                <!-- 范围显示 -->
                <text x="{svg_width - 15}" y="35" fill="#999" font-size="9" text-anchor="end">范围: {y_min:.1f} - {y_max:.1f}</text>
                
            </svg>
        </div>"""
        
        return svg_content
    
    # 生成三个图表
    hr_svg = create_chart_svg("心率趋势", hr_values, "#FF4B4B", 50, 110, "BPM")
    br_svg = create_chart_svg("呼吸率趋势", br_values, "#00CC88", 10, 25, "/min")
    rl_svg = create_reconstruction_loss_chart("重构损失值 & 异常检测", rl_values, anomaly_statuses, times)
    
    return hr_svg + br_svg + rl_svg


# 控制按钮和设置
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if st.button("开始监控", type="primary"):
        st.session_state.monitoring = True

with col2:
    if st.button("停止监控"):
        st.session_state.monitoring = False

with col3:
    if st.button("清除数据"):
        st.session_state.data = []

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


# 显示当前状态 - 紧凑样式
st.markdown(f"""
<div style="background-color: #f0f2f6; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.85rem;">
📊 <strong>当前数据点:</strong> {len(st.session_state.data)}/{st.session_state.max_points} | 
⏱️ <strong>更新频率:</strong> 1秒/次
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
        current_device_display = latest.get('device_id', '未知设备')
        
        # 顶部指标卡片 - 紧凑布局
        with mui.Grid(container=True, spacing=1.5):
            # 心率卡片
            with mui.Grid(item=True, xs=3):
                with mui.Card(elevation=2, sx={"height": "120px"}):
                    with mui.CardContent(sx={"padding": "12px", "&:last-child": {"paddingBottom": "12px"}}):
                        mui.Typography("心率", variant="subtitle2", color="textSecondary", sx={"fontSize": "0.8rem"})
                        mui.Typography(
                            f"{latest['heart_rate']} BPM", 
                            variant="h5", 
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
            
            # 呼吸率卡片
            with mui.Grid(item=True, xs=3):
                with mui.Card(elevation=2, sx={"height": "120px"}):
                    with mui.CardContent(sx={"padding": "12px", "&:last-child": {"paddingBottom": "12px"}}):
                        mui.Typography("呼吸率", variant="subtitle2", color="textSecondary", sx={"fontSize": "0.8rem"})
                        mui.Typography(
                            f"{latest['breathing_rate']} /min", 
                            variant="h5", 
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
            
            # 状态卡片
            with mui.Grid(item=True, xs=3):
                with mui.Card(elevation=2, sx={"height": "120px"}):
                    with mui.CardContent(sx={"padding": "12px", "&:last-child": {"paddingBottom": "12px"}}):
                        mui.Typography("状态", variant="subtitle2", color="textSecondary", sx={"fontSize": "0.8rem"})
                        status_text = "🟢 监控中" if st.session_state.monitoring else "⏸️ 已停止"
                        mui.Typography(status_text, variant="h6", sx={"margin": "4px 0"})
                        mui.Typography(
                            f"更新: {latest['time']}", 
                            variant="caption", 
                            color="textSecondary",
                            sx={"fontSize": "0.75rem"}
                        )

            # 重构损失卡片
            with mui.Grid(item=True, xs=3):
                with mui.Card(elevation=2, sx={"height": "120px"}):
                    with mui.CardContent(sx={"padding": "12px", "&:last-child": {"paddingBottom": "12px"}}):
                        mui.Typography("重构损失", variant="subtitle2", color="textSecondary", sx={"fontSize": "0.8rem"})
                        mui.Typography(
                            f"{latest['reconstruction_loss']:.3f}", 
                            variant="h5", 
                            color="secondary",
                            sx={"fontWeight": "bold", "margin": "4px 0"}
                        )
                        # 异常状态显示
                        status_text = "🔴 异常" if latest['anomaly_status'] else "🟢 正常"
                        status_color = "error" if latest['anomaly_status'] else "success"
                        mui.Typography(
                            status_text,
                            variant="caption",
                            color=status_color,
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


# 在数据详情中也显示设备信息
if st.session_state.show_data_modal and chart_data:
    with st.expander("📊 数据详情", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**数据点总数:** {len(chart_data)}")
            if chart_data:
                current_device_display = chart_data[-1].get('device_id', '未知设备')
                st.markdown(f"**当前设备:** {current_device_display}")
        
        # 创建数据表格
        df_display = pd.DataFrame(chart_data[-15:])  # 显示最近15条，更紧凑
        df_display['状态'] = df_display.apply(
            lambda row: '正常' if (60 <= row['heart_rate'] <= 100 and 12 <= row['breathing_rate'] <= 20) else '异常',
            axis=1
        )
        
        # 包含设备ID和重构损失信息
        if 'device_id' in df_display.columns:
            df_display = df_display[['time', 'heart_rate', 'breathing_rate', 'reconstruction_loss', 'anomaly_status', 'device_id', '状态']]
            df_display.columns = ['时间', '心率 (BPM)', '呼吸率 (/min)', '重构损失', '异常状态', '设备ID', '生理状态']
        else:
            df_display = df_display[['time', 'heart_rate', 'breathing_rate', 'reconstruction_loss', 'anomaly_status', '状态']]
            df_display.columns = ['时间', '心率 (BPM)', '呼吸率 (/min)', '重构损失', '异常状态', '生理状态']
        
        st.dataframe(df_display, use_container_width=True, height=300)


# 高级图表展开区域
if st.session_state.show_chart_modal and len(chart_data) > 1:
    with st.expander("📈 高级趋势图表", expanded=True):
        # 创建子图 - 增加更多间距
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('心率趋势', '呼吸率趋势', '重构损失值趋势'),
            vertical_spacing=0.15  # 增加垂直间距
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
            row=2, col=1
        )
        
        # 重构损失值数据 - 根据异常状态着色
        normal_points = [(point['time'], point['reconstruction_loss']) for point in chart_data if not point['anomaly_status']]
        anomaly_points = [(point['time'], point['reconstruction_loss']) for point in chart_data if point['anomaly_status']]
        
        if normal_points:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in normal_points],
                    y=[p[1] for p in normal_points],
                    mode='lines+markers',
                    name='正常',
                    line=dict(color='#4444FF', width=2),
                    marker=dict(size=4)
                ),
                row=3, col=1
            )
        
        if anomaly_points:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in anomaly_points],
                    y=[p[1] for p in anomaly_points],
                    mode='markers',
                    name='异常',
                    marker=dict(color='#FF4444', size=6)
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=600,  # 增加高度适应3个图表
            showlegend=True,
            title_text="生理参数监控趋势",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        fig.update_xaxes(title_text="时间", row=3, col=1)
        fig.update_yaxes(title_text="心率 (BPM)", row=1, col=1)
        fig.update_yaxes(title_text="呼吸率 (/min)", row=2, col=1)
        fig.update_yaxes(title_text="重构损失", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

# 自动刷新
if st.session_state.monitoring:
    time.sleep(0.5)  # 界面刷新频率，数据仍然1秒更新
    st.rerun()

# 底部说明 - 紧凑样式
st.markdown("""
<div style="border-top: 1px solid #e6e6e6; margin-top: 1rem; padding-top: 0.5rem;">
<details>
<summary style="cursor: pointer; font-weight: bold; color: #666;">✨ 功能特点</summary>
<div style="font-size: 0.85rem; margin-top: 0.5rem; color: #555;">
• <strong>数据累加:</strong> 持续累积，支持50-1000个数据点<br>
• <strong>实时更新:</strong> 固定1秒更新频率<br>
• <strong>无闪动界面:</strong> Material-UI + 智能图表显示<br>
• <strong>专业坐标轴:</strong> 完整刻度和标签<br>
• <strong>异常检测:</strong> 重构损失值分析 + 实时异常状态监控<br>
• <strong>展开式详情:</strong> 数据表格 + Plotly交互图表<br>
• <strong>紧凑界面:</strong> 优化空间利用，专注监控体验
</div>
</details>
</div>
""", unsafe_allow_html=True)

if not st.session_state.monitoring and len(st.session_state.data) == 0:
    st.markdown('<div style="background-color: #e3f2fd; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.9rem;">👆 点击"开始监控"开始实时数据累加展示</div>', unsafe_allow_html=True)
elif len(st.session_state.data) > 0:
    st.markdown(f'<div style="background-color: #e8f5e8; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.9rem;">💾 已累积 {len(st.session_state.data)} 个数据点，监控状态: {"🟢 运行中" if st.session_state.monitoring else "⏸️ 已暂停"}</div>', unsafe_allow_html=True)
    st.markdown('<div style="background-color: #fff3e0; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 0.85rem;">💡 点击 "📊 数据详情" 展开详细数据表格，点击 "📈 高级图表" 展开交互式分析图表</div>', unsafe_allow_html=True)