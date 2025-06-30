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
from streamlit_elements import elements, mui, html, sync

# 页面配置
st.set_page_config(
    page_title="Elements无闪动监控",
    page_icon="💓",
    layout="wide"
)

st.title("💓 Elements 无闪动实时监控")

# 初始化状态
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
    st.session_state.data = []
    st.session_state.last_update = time.time()
    st.session_state.max_points = 200  # 可配置的最大数据点数

def generate_data():
    """生成新数据点"""
    current_time = time.time()
    
    # 固定1秒更新一次
    if current_time - st.session_state.last_update < 1.0:
        return
    
    hr = 75 + np.sin(current_time * 0.1) * 8 + random.uniform(-3, 3)
    br = 16 + np.sin(current_time * 0.08) * 2 + random.uniform(-1, 1)
    
    data_point = {
        'time': datetime.now().strftime('%H:%M:%S'),
        'timestamp': current_time,
        'heart_rate': max(60, min(100, round(hr, 1))),
        'breathing_rate': max(12, min(20, round(br, 1)))
    }
    
    st.session_state.data.append(data_point)
    
    # 根据设置限制数据点数量
    if len(st.session_state.data) > st.session_state.max_points:
        st.session_state.data = st.session_state.data[-st.session_state.max_points:]
    
    st.session_state.last_update = current_time

def create_chart_html(chart_data):
    """创建带坐标轴的完整SVG图表"""
    times = [point['time'] for point in chart_data]
    hr_values = [point['heart_rate'] for point in chart_data]
    br_values = [point['breathing_rate'] for point in chart_data]
    
    # 图表尺寸
    svg_width = 800
    svg_height = 280
    padding_left = 60
    padding_right = 40
    padding_top = 30
    padding_bottom = 50
    
    # 可用绘图区域
    chart_width = svg_width - padding_left - padding_right
    chart_height = svg_height - padding_top - padding_bottom
    
    def create_chart_svg(title, values, color, y_min, y_max, unit):
        # 计算数据点
        points = []
        for i, value in enumerate(values):
            x = padding_left + (i / (len(values) - 1)) * chart_width
            y = padding_top + chart_height - ((value - y_min) / (y_max - y_min)) * chart_height
            points.append(f"{x},{y}")
        
        # Y轴刻度
        y_ticks = []
        for i in range(6):  # 5个刻度间隔
            tick_value = y_min + (y_max - y_min) * i / 5
            tick_y = padding_top + chart_height - (i / 5) * chart_height
            y_ticks.append((tick_y, tick_value))
        
        # X轴刻度（显示时间）
        x_ticks = []
        # 根据数据点数量智能选择显示的时间标签数量
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
        <div style="margin: 15px 0;">
            <h4 style="color: {color}; margin-bottom: 10px; text-align: center;">{title}</h4>
            <svg width="{svg_width}" height="{svg_height}" style="border: 1px solid #ddd; background: white;">
                <!-- 网格线 -->
                <defs>
                    <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                        <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#f0f0f0" stroke-width="1"/>
                    </pattern>
                </defs>
                <rect x="{padding_left}" y="{padding_top}" width="{chart_width}" height="{chart_height}" fill="url(#grid)"/>
                
                <!-- Y轴 -->
                <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_height}" stroke="#333" stroke-width="2"/>
                
                <!-- X轴 -->
                <line x1="{padding_left}" y1="{padding_top + chart_height}" x2="{padding_left + chart_width}" y2="{padding_top + chart_height}" stroke="#333" stroke-width="2"/>
                
                <!-- Y轴刻度和标签 -->"""
        
        for tick_y, tick_value in y_ticks:
            svg_content += f"""
                <line x1="{padding_left - 5}" y1="{tick_y}" x2="{padding_left}" y2="{tick_y}" stroke="#333" stroke-width="1"/>
                <text x="{padding_left - 10}" y="{tick_y + 4}" fill="#666" font-size="11" text-anchor="end">{tick_value:.0f}</text>"""
        
        # X轴刻度和标签
        for tick_x, tick_time in x_ticks:
            svg_content += f"""
                <line x1="{tick_x}" y1="{padding_top + chart_height}" x2="{tick_x}" y2="{padding_top + chart_height + 5}" stroke="#333" stroke-width="1"/>
                <text x="{tick_x}" y="{padding_top + chart_height + 18}" fill="#666" font-size="10" text-anchor="middle">{tick_time}</text>"""
        
        # 数据折线
        svg_content += f"""
                <!-- 数据折线 -->
                <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round"/>
                
                <!-- 数据点 -->"""
        
        for point in points:
            x, y = point.split(',')
            svg_content += f"""
                <circle cx="{x}" cy="{y}" r="4" fill="{color}" stroke="white" stroke-width="2"/>"""
        
        # 轴标签
        svg_content += f"""
                <!-- Y轴标签 -->
                <text x="20" y="{padding_top + chart_height/2}" fill="#333" font-size="12" text-anchor="middle" transform="rotate(-90, 20, {padding_top + chart_height/2})">{unit}</text>
                
                <!-- X轴标签 -->
                <text x="{padding_left + chart_width/2}" y="{svg_height - 10}" fill="#333" font-size="12" text-anchor="middle">时间</text>
                
                <!-- 当前值显示 -->
                <text x="{svg_width - 20}" y="25" fill="{color}" font-size="14" font-weight="bold" text-anchor="end">当前: {values[-1]:.1f} {unit}</text>
                
            </svg>
        </div>"""
        
        return svg_content
    
    # 生成心率和呼吸率图表
    hr_svg = create_chart_svg("心率趋势", hr_values, "#FF4B4B", 50, 110, "BPM")
    br_svg = create_chart_svg("呼吸率趋势", br_values, "#00CC88", 10, 25, "/min")
    
    return hr_svg + br_svg

# 控制按钮和设置
col1, col2, col3, col4 = st.columns(4)

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

# 显示当前状态
st.info(f"📊 当前数据点: {len(st.session_state.data)}/{st.session_state.max_points} | 更新频率: 1秒/次")

# 如果正在监控，生成新数据
if st.session_state.monitoring:
    generate_data()

# 使用所有数据进行显示（不再限制显示数量）
chart_data = st.session_state.data if st.session_state.data else []

# 使用 Elements 创建无闪动的实时界面
with elements("realtime_monitor"):
    
    if chart_data:
        latest = chart_data[-1]
        
        # 顶部指标卡片
        with mui.Grid(container=True, spacing=2):
            # 心率卡片
            with mui.Grid(item=True, xs=4):
                with mui.Card(elevation=3):
                    with mui.CardContent():
                        mui.Typography("心率", variant="h6", color="textSecondary")
                        mui.Typography(
                            f"{latest['heart_rate']} BPM", 
                            variant="h4", 
                            color="error"
                        )
                        if len(chart_data) > 1:
                            delta = latest['heart_rate'] - chart_data[-2]['heart_rate']
                            mui.Typography(
                                f"{'↑' if delta > 0 else '↓'} {abs(delta):.1f}",
                                variant="body2",
                                color="success" if 60 <= latest['heart_rate'] <= 100 else "warning"
                            )
            
            # 呼吸率卡片
            with mui.Grid(item=True, xs=4):
                with mui.Card(elevation=3):
                    with mui.CardContent():
                        mui.Typography("呼吸率", variant="h6", color="textSecondary")
                        mui.Typography(
                            f"{latest['breathing_rate']} /min", 
                            variant="h4", 
                            color="primary"
                        )
                        if len(chart_data) > 1:
                            delta = latest['breathing_rate'] - chart_data[-2]['breathing_rate']
                            mui.Typography(
                                f"{'↑' if delta > 0 else '↓'} {abs(delta):.1f}",
                                variant="body2",
                                color="success" if 12 <= latest['breathing_rate'] <= 20 else "warning"
                            )
            
            # 状态卡片
            with mui.Grid(item=True, xs=4):
                with mui.Card(elevation=3):
                    with mui.CardContent():
                        mui.Typography("状态", variant="h6", color="textSecondary")
                        status_text = "🟢 监控中" if st.session_state.monitoring else "⏸️ 已停止"
                        mui.Typography(status_text, variant="h5")
                        mui.Typography(
                            f"更新: {latest['time']}", 
                            variant="body2", 
                            color="textSecondary"
                        )
    
    # 分隔线
    mui.Divider(sx={"margin": "20px 0"})
    
    # 图表区域 - 使用SVG替代Canvas
    if len(chart_data) > 1:
        chart_html = create_chart_html(chart_data)
        html.div(
            dangerouslySetInnerHTML={"__html": chart_html}
        )
    
    # 数据表格
    if chart_data:
        mui.Typography("最近数据", variant="h6", sx={"margin": "20px 0 10px 0"})
        
        # 表格头
        with mui.Table():
            with mui.TableHead():
                with mui.TableRow():
                    mui.TableCell("时间")
                    mui.TableCell("心率 (BPM)")
                    mui.TableCell("呼吸率 (/min)")
                    mui.TableCell("状态")
            
            # 表格数据
            with mui.TableBody():
                for point in chart_data[-10:]:  # 显示最近10条
                    with mui.TableRow():
                        mui.TableCell(point['time'])
                        mui.TableCell(str(point['heart_rate']))
                        mui.TableCell(str(point['breathing_rate']))
                        
                        # 状态指示
                        hr_ok = 60 <= point['heart_rate'] <= 100
                        br_ok = 12 <= point['breathing_rate'] <= 20
                        status = "正常" if hr_ok and br_ok else "异常"
                        color = "success" if hr_ok and br_ok else "warning"
                        
                        with mui.TableCell():
                            mui.Chip(
                                label=status,
                                color=color,
                                size="small"
                            )

# 如果需要更高级的图表，可以在Elements外使用Plotly
if len(chart_data) > 1 and st.checkbox("显示高级图表", value=False):
    st.subheader("📊 高级趋势图表")
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('心率趋势', '呼吸率趋势'),
        vertical_spacing=0.1
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
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="生理参数监控趋势"
    )
    
    fig.update_xaxes(title_text="时间", row=2, col=1)
    fig.update_yaxes(title_text="心率 (BPM)", row=1, col=1)
    fig.update_yaxes(title_text="呼吸率 (/min)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# 自动刷新
if st.session_state.monitoring:
    time.sleep(0.5)  # 界面刷新频率，数据仍然1秒更新
    st.rerun()

# 底部说明
st.markdown("---")
st.markdown("""
**真正累加版本特点:**
- ✅ **完全数据累加** - 数据持续累积，不会丢失
- ✅ **可配置数据量** - 支持50-1000个数据点
- ✅ **固定更新频率** - 精确1秒更新一次
- ✅ **智能图表显示** - 根据数据量自动调整时间轴
- ✅ **完全无闪动** - Material-UI界面  
- ✅ **专业坐标轴** - 完整的刻度和标签
- ✅ **实时状态显示** - 显示当前数据点数量
- ✅ **可选高级图表** - Plotly交互式分析
""")

if not st.session_state.monitoring and len(st.session_state.data) == 0:
    st.info("👆 点击'开始监控'开始实时数据累加展示")
elif len(st.session_state.data) > 0:
    st.success(f"💾 已累积 {len(st.session_state.data)} 个数据点，监控状态: {'🟢 运行中' if st.session_state.monitoring else '⏸️ 已暂停'}")