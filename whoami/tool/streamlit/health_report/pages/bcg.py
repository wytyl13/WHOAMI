import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# 页面配置
st.set_page_config(
    page_title="医疗健康监测仪表盘",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    
    .stApp {
        background-color: #34455d;
    }
    
    .metric-container {
        background-color: #213046;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-top: 3px solid;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .heart-rate {
        border-top-color: #ef4444;
    }
    
    .resp-rate {
        border-top-color: #3b82f6;
    }
    
    .comprehensive {
        border-top-color: #22c55e;
    }
    
    .header-container {
        background-color: #213046;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    
    .status-normal {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 14px;
        display: inline-block;
    }
    
    .status-warning {
        background-color: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 14px;
        display: inline-block;
    }
    
    .big-number {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
    
    .heart-rate-color {
        color: #ef4444;
    }
    
    .resp-rate-color {
        color: #3b82f6;
    }
    
    h1, h2, h3 {
        color: white;
    }
    
    .stMetric {
        color: white;
    }
    
    div[data-testid="metric-container"] {
        background-color: #213046;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
    st.session_state.ecg_data = []
    st.session_state.resp_data = []
    st.session_state.hr_history = []
    st.session_state.rr_history = []
    st.session_state.time_history = []

# 生成模拟数据的函数
def generate_ecg_data(length=100):
    """生成模拟心电图数据"""
    data = []
    for i in range(length):
        t = i / 10
        ecg_pattern = (0.7 * np.sin(t * 5) + 
                      0.3 * np.sin(t * 20) + 
                      0.1 * np.sin(t * 60) +
                      random.uniform(-0.1, 0.1))
        data.append(ecg_pattern)
    return data

def generate_resp_data(length=100):
    """生成模拟呼吸波形数据"""
    data = []
    for i in range(length):
        t = i / 30
        resp_pattern = np.sin(t) + random.uniform(-0.1, 0.1)
        data.append(resp_pattern)
    return data

def get_current_vitals():
    """获取当前生命体征数据"""
    return {
        'heart_rate': random.randint(70, 85),
        'resp_rate': random.randint(14, 18),
        'status': '正常范围',
        'comprehensive_status': '生命体征稳定',
        'warning': '需关注血氧饱和度'
    }

# 页面标题和患者信息
st.markdown("""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2>患者: 张伟 (ID: P202309001)</h2>
            <p>ICU-3床 | 主治医师: 李华</p>
        </div>
        <div style="text-align: right;">
            <h3>{}</h3>
            <div style="color: #4ade80;">● 系统状态: 在线</div>
        </div>
    </div>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# 创建列布局
col1, col2, col3 = st.columns([1, 2, 1])

# 更新数据
current_vitals = get_current_vitals()

# 左侧：生命体征数据
with col1:
    st.markdown("### 生命体征监测")
    
    # 心率卡片
    st.markdown(f"""
    <div class="metric-container heart-rate">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 24px; margin-right: 10px;">❤️</span>
            <h3>心率</h3>
        </div>
        <div class="big-number heart-rate-color">{current_vitals['heart_rate']}</div>
        <div class="status-normal">正常范围</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 呼吸率卡片
    st.markdown(f"""
    <div class="metric-container resp-rate">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 24px; margin-right: 10px;">🫁</span>
            <h3>呼吸率</h3>
        </div>
        <div class="big-number resp-rate-color">{current_vitals['resp_rate']}</div>
        <div class="status-normal">正常范围</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 综合指标卡片
    st.markdown(f"""
    <div class="metric-container comprehensive">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 24px; margin-right: 10px;">📊</span>
            <h3>综合指标</h3>
        </div>
        <div style="text-align: center; padding: 20px 0;">
            <div class="status-normal" style="margin-bottom: 10px;">{current_vitals['comprehensive_status']}</div>
            <div class="status-warning">{current_vitals['warning']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 中间：实时波形图
with col2:
    st.markdown("### 实时波形监测")
    
    # 生成新的波形数据
    if len(st.session_state.ecg_data) >= 200:
        st.session_state.ecg_data = st.session_state.ecg_data[-100:]
    if len(st.session_state.resp_data) >= 200:
        st.session_state.resp_data = st.session_state.resp_data[-100:]
    
    # 添加新数据点
    st.session_state.ecg_data.extend(generate_ecg_data(5))
    st.session_state.resp_data.extend(generate_resp_data(5))
    
    # 创建心电图
    fig_waves = make_subplots(
        rows=2, cols=1,
        subplot_titles=('实时心电图波形', '实时呼吸波形'),
        vertical_spacing=0.1
    )
    
    # ECG波形
    fig_waves.add_trace(
        go.Scatter(
            y=st.session_state.ecg_data[-100:],
            mode='lines',
            name='ECG',
            line=dict(color='#4ade80', width=2),
            fill='tonexty',
            fillcolor='rgba(74, 222, 128, 0.3)'
        ),
        row=1, col=1
    )
    
    # 呼吸波形
    fig_waves.add_trace(
        go.Scatter(
            y=st.session_state.resp_data[-100:],
            mode='lines',
            name='呼吸',
            line=dict(color='#60a5fa', width=2),
            fill='tonexty',
            fillcolor='rgba(96, 165, 250, 0.3)'
        ),
        row=2, col=1
    )
    
    fig_waves.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(33,48,70,1)',
        font=dict(color='white'),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    fig_waves.update_xaxes(showgrid=False, showticklabels=False)
    fig_waves.update_yaxes(showgrid=True, gridcolor='#394c66')
    
    st.plotly_chart(fig_waves, use_container_width=True)

# 右侧：趋势图
with col3:
    st.markdown("### 生命体征趋势")
    
    # 更新趋势数据
    current_time = datetime.now()
    if len(st.session_state.time_history) == 0 or (current_time - st.session_state.last_update).seconds >= 5:
        st.session_state.time_history.append(current_time)
        st.session_state.hr_history.append(current_vitals['heart_rate'])
        st.session_state.rr_history.append(current_vitals['resp_rate'])
        st.session_state.last_update = current_time
        
        # 保持最近20个数据点
        if len(st.session_state.time_history) > 20:
            st.session_state.time_history = st.session_state.time_history[-20:]
            st.session_state.hr_history = st.session_state.hr_history[-20:]
            st.session_state.rr_history = st.session_state.rr_history[-20:]
    
    # 心率趋势图
    if len(st.session_state.hr_history) > 0:
        fig_hr = go.Figure()
        fig_hr.add_trace(
            go.Scatter(
                x=st.session_state.time_history,
                y=st.session_state.hr_history,
                mode='lines+markers',
                name='心率',
                line=dict(color='#ef4444', width=2),
                marker=dict(size=6),
                fill='tonexty',
                fillcolor='rgba(239, 68, 68, 0.3)'
            )
        )
        
        fig_hr.update_layout(
            title="心率趋势 (bpm)",
            height=250,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(33,48,70,1)',
            font=dict(color='white', size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            title_font_color='#ef4444'
        )
        
        fig_hr.update_xaxes(showgrid=False)
        fig_hr.update_yaxes(showgrid=True, gridcolor='#394c66', range=[60, 100])
        
        st.plotly_chart(fig_hr, use_container_width=True)
    
    # 呼吸率趋势图
    if len(st.session_state.rr_history) > 0:
        fig_rr = go.Figure()
        fig_rr.add_trace(
            go.Scatter(
                x=st.session_state.time_history,
                y=st.session_state.rr_history,
                mode='lines+markers',
                name='呼吸率',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6),
                fill='tonexty',
                fillcolor='rgba(59, 130, 246, 0.3)'
            )
        )
        
        fig_rr.update_layout(
            title="呼吸率趋势 (次/分)",
            height=250,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(33,48,70,1)',
            font=dict(color='white', size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            title_font_color='#3b82f6'
        )
        
        fig_rr.update_xaxes(showgrid=False)
        fig_rr.update_yaxes(showgrid=True, gridcolor='#394c66', range=[12, 20])
        
        st.plotly_chart(fig_rr, use_container_width=True)

# 底部控制栏
st.markdown("---")
col_footer1, col_footer2 = st.columns([1, 1])

with col_footer1:
    st.markdown(f"**最后更新:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col_footer2:
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("暂停更新", key="pause"):
            st.success("更新已暂停")
    
    with col_btn2:
        if st.button("打印报告", key="print"):
            st.success("报告打印中...")
    
    with col_btn3:
        if st.button("导出数据", key="export"):
            # 创建数据导出
            data_dict = {
                'timestamp': st.session_state.time_history,
                'heart_rate': st.session_state.hr_history,
                'resp_rate': st.session_state.rr_history
            }
            df = pd.DataFrame(data_dict)
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载CSV",
                data=csv,
                file_name=f"vital_signs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col_btn4:
        if st.button("系统设置", key="settings"):
            st.info("系统设置功能开发中...")

# 自动刷新
time.sleep(1)
st.rerun()