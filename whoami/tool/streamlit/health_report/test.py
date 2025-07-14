# 首先安装依赖：pip install streamlit-autorefresh

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import random
from streamlit_autorefresh import st_autorefresh

# 页面配置
st.set_page_config(
    page_title="生理数据实时监控 - AutoRefresh",
    page_icon="💓",
    layout="wide"
)

st.title("💓 呼吸率与心率实时监控 (AutoRefresh)")

# 侧边栏设置
st.sidebar.header("监控设置")
auto_refresh = st.sidebar.checkbox("启用自动刷新", value=True)
refresh_interval = st.sidebar.slider("刷新间隔(毫秒)", 500, 5000, 1000)
max_points = st.sidebar.slider("最大数据点", 20, 200, 100)

# 自动刷新
if auto_refresh:
    count = st_autorefresh(interval=refresh_interval, key="datarefresh")

# 数据存储文件路径（实际应用中可以用数据库）
DATA_FILE = "vital_signs_data.csv"

def load_or_create_data():
    """加载或创建数据"""
    try:
        data = pd.read_csv(DATA_FILE)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    except FileNotFoundError:
        return pd.DataFrame({
            'timestamp': [],
            'heart_rate': [],
            'breathing_rate': []
        })

def save_data(data):
    """保存数据"""
    data.to_csv(DATA_FILE, index=False)

def generate_new_data_point():
    """生成新的数据点"""
    current_time = datetime.now()
    
    # 模拟心率数据（60-100 BPM）
    base_hr = 75
    hr_noise = np.sin(time.time() * 0.1) * 8 + random.uniform(-3, 3)
    heart_rate = max(60, min(100, base_hr + hr_noise))
    
    # 模拟呼吸率数据（12-20次/分钟）
    base_br = 16
    br_noise = np.sin(time.time() * 0.08) * 2.5 + random.uniform(-1.5, 1.5)
    breathing_rate = max(12, min(20, base_br + br_noise))
    
    return {
        'timestamp': current_time,
        'heart_rate': round(heart_rate, 1),
        'breathing_rate': round(breathing_rate, 1)
    }

# 加载现有数据
data = load_or_create_data()

# 如果启用自动刷新，添加新数据点
if auto_refresh:
    new_point = generate_new_data_point()
    new_row = pd.DataFrame([new_point])
    data = pd.concat([data, new_row], ignore_index=True)
    
    # 限制数据点数量
    if len(data) > max_points:
        data = data.tail(max_points)
    
    # 保存数据
    save_data(data)

# 清除数据按钮
if st.sidebar.button("清除所有数据"):
    data = pd.DataFrame({
        'timestamp': [],
        'heart_rate': [],
        'breathing_rate': []
    })
    save_data(data)
    st.rerun()

# 显示当前状态
col1, col2, col3 = st.columns(3)

if len(data) > 0:
    latest = data.iloc[-1]
    
    with col1:
        st.metric(
            "心率",
            f"{latest['heart_rate']:.1f} BPM",
            delta=f"{latest['heart_rate'] - data.iloc[-2]['heart_rate']:.1f}" if len(data) > 1 else None
        )
    
    with col2:
        st.metric(
            "呼吸率", 
            f"{latest['breathing_rate']:.1f} /min",
            delta=f"{latest['breathing_rate'] - data.iloc[-2]['breathing_rate']:.1f}" if len(data) > 1 else None
        )
    
    with col3:
        st.metric(
            "数据点数量",
            len(data),
            delta=1 if auto_refresh else 0
        )

# 状态显示
status_col1, status_col2 = st.columns(2)
with status_col1:
    if auto_refresh:
        st.success("🟢 实时监控中")
    else:
        st.info("⏸️ 监控已暂停")

with status_col2:
    if len(data) > 0:
        st.write(f"最后更新: {data.iloc[-1]['timestamp'].strftime('%H:%M:%S')}")

# 绘制图表
if len(data) > 1:
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['心率趋势 (BPM)', '呼吸率趋势 (次/分钟)'],
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    # 心率图
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['heart_rate'],
            mode='lines+markers',
            name='心率',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=6, color='#FF6B6B'),
            hovertemplate='<b>心率</b><br>时间: %{x}<br>数值: %{y:.1f} BPM<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 呼吸率图
    fig.add_trace(
        go.Scatter(
            x=data['timestamp'],
            y=data['breathing_rate'],
            mode='lines+markers',
            name='呼吸率',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=6, color='#4ECDC4'),
            hovertemplate='<b>呼吸率</b><br>时间: %{x}<br>数值: %{y:.1f} /min<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 添加正常范围线
    fig.add_hline(y=60, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=12, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # 更新布局
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="生理参数实时监控图表",
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # 更新坐标轴
    fig.update_xaxes(
        title_text="时间",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="心率 (BPM)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        range=[50, 110],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="呼吸率 (次/分钟)",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        range=[10, 25],
        row=2, col=1
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 数据统计
if len(data) > 0:
    st.subheader("📊 数据统计")
    
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.write("**心率统计:**")
        hr_stats = data['heart_rate'].describe()
        st.write(f"- 平均值: {hr_stats['mean']:.1f} BPM")
        st.write(f"- 最小值: {hr_stats['min']:.1f} BPM")
        st.write(f"- 最大值: {hr_stats['max']:.1f} BPM")
        st.write(f"- 标准差: {hr_stats['std']:.1f}")
    
    with stats_col2:
        st.write("**呼吸率统计:**")
        br_stats = data['breathing_rate'].describe()
        st.write(f"- 平均值: {br_stats['mean']:.1f} /min")
        st.write(f"- 最小值: {br_stats['min']:.1f} /min")
        st.write(f"- 最大值: {br_stats['max']:.1f} /min")
        st.write(f"- 标准差: {br_stats['std']:.1f}")

# 最近数据表格
if len(data) > 0:
    with st.expander("查看最近数据", expanded=False):
        recent_data = data.tail(20).copy()
        recent_data['时间'] = recent_data['timestamp'].dt.strftime('%H:%M:%S')
        recent_data['心率(BPM)'] = recent_data['heart_rate']
        recent_data['呼吸率(/min)'] = recent_data['breathing_rate']
        
        display_data = recent_data[['时间', '心率(BPM)', '呼吸率(/min)']]
        st.dataframe(display_data, use_container_width=True)

# 底部信息
st.markdown("---")
st.markdown("💡 **使用说明:** 勾选侧边栏的'启用自动刷新'开始实时监控，调整刷新间隔控制更新频率。")