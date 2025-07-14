import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import re

# 添加项目路径
project_root = "/work/ai/WHOAMI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据库模型和提供者
from whoami.tool.streamlit.health_report.table.device_data import DeviceData
from whoami.provider.sql_provider import SqlProvider

# 页面配置
st.set_page_config(
    page_title="设备数据 - 睡眠健康管理系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 用户数据文件路径
USER_DATA_FILE = os.path.join(project_root, "data", "users.json")
SQL_CONFIG_PATH = '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml'

def load_users():
    """加载用户数据"""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def get_sleep_data_provider():
    """获取睡眠数据提供者"""
    try:
        from whoami.tool.real_time_vital_analyze.sleep_statistics_model import SleepStatistics
        
        sleep_data_provider = SqlProvider(
            model=SleepStatistics,
            sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
        )
        return sleep_data_provider
    except Exception as e:
        st.error(f"睡眠详细数据连接失败: {str(e)}")
        return None

def parse_duration_to_hours(duration_str):
    """将时长字符串转换为小时数"""
    if not duration_str or duration_str == 'NULL':
        return 0.0
    
    try:
        # 假设格式为 "HH:MM:SS" 或 "MM:SS" 或直接是小时数
        if ':' in duration_str:
            parts = duration_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                return hours + minutes/60 + seconds/3600
            elif len(parts) == 2:  # MM:SS
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes/60 + seconds/3600
        else:
            # 直接是数字，假设为小时
            return float(duration_str)
    except:
        return 0.0

def get_sleep_statistics_data(device_sn):
    """获取最近30天的睡眠统计数据"""
    try:
        # 获取数据提供者
        sql_provider = get_sleep_data_provider()
        if not sql_provider:
            return pd.DataFrame()
        
        # 计算30天前的日期
        # thirty_days_ago = datetime.now() - timedelta(days=30)
        
        # 查询数据
        records = sql_provider.get_record_by_condition(
            condition={
                "device_sn": device_sn,
                # "sleep_start_time": {"min": thirty_days_ago.strftime('%Y-%m-%d %H:%M:%S')}
            },
            fields=None
        )
        # st.write(f"records: ------------- {records}")
        if not records:
            st.warning("未找到该设备的睡眠数据")
            return pd.DataFrame()
        
        # 转换为DataFrame并处理数据
        sleep_data = []
        for record in records:
            # 处理时长数据
            total_duration = parse_duration_to_hours(record.get('total_duration', '0'))
            deep_sleep_duration = parse_duration_to_hours(record.get('deep_sleep_duration', '0'))
            light_sleep_duration = parse_duration_to_hours(record.get('light_sleep_duration', '0'))
            
            sleep_data.append({
                'date': record.get('sleep_start_time', datetime.now()).strftime('%Y-%m-%d') if record.get('sleep_start_time') else datetime.now().strftime('%Y-%m-%d'),
                'sleep_duration': total_duration,
                'deep_sleep_duration': deep_sleep_duration,
                'light_sleep_duration': light_sleep_duration,
                'avg_breath_rate': record.get('avg_breath_rate', 0) or 0,
                'avg_heart_rate': record.get('avg_heart_rate', 0) or 0,
                'body_movement_count': record.get('body_movement_count', 0) or 0,
                'apnea_count': record.get('apnea_count', 0) or 0,
                'sleep_start_time': record.get('sleep_start_time'),
                'sleep_end_time': record.get('sleep_end_time'),
                'bed_time': record.get('bed_time'),
                'wake_time': record.get('wake_time')
            })
        
        return pd.DataFrame(sleep_data)
        
    except Exception as e:
        st.error(f"获取睡眠数据失败: {str(e)}")
        return pd.DataFrame()

def check_login():
    """检查登录状态"""
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.error("请先登录！")
        st.switch_page("pages/login.py")
        return False
    return True

# CSS样式
st.markdown("""
<style>
/* 全局样式 */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* 侧边栏样式 */
.css-1d391kg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* 主要内容区域 */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* 卡片样式 */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border-left: 4px solid #667eea;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 10px 0;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

/* 用户头部样式 */
.user-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px 30px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    position: relative;
}

.user-avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px;
    font-size: 1.5rem;
}

/* 设备信息样式 */
.device-info {
    background: rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    text-align: left;
}

.device-info-item {
    margin: 8px 0;
    font-size: 0.9rem;
    display: flex;
    justify-content: space-between;
}

.device-info-label {
    font-weight: bold;
    min-width: 80px;
}

.device-info-value {
    color: rgba(255, 255, 255, 0.9);
}

/* 状态指示器 */
.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-good { background-color: #28a745; }
.status-warning { background-color: #ffc107; }
.status-danger { background-color: #dc3545; }

/* 图表容器 */
.chart-container {
    background: white;
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)



def parse_duration_to_minutes(duration_str):
    """
    将 'X小时Y分Z秒' 格式的字符串转换为总分钟数
    """
    # 检查空值
    if pd.isna(duration_str) or duration_str == '' or duration_str is None:
        return 0
    
    # 确保是字符串类型
    if not isinstance(duration_str, str):
        # 如果是数字，直接返回0或转换为字符串
        if isinstance(duration_str, (int, float)):
            return int(duration_str) if not pd.isna(duration_str) else 0
        else:
            # 尝试转换为字符串
            try:
                duration_str = str(duration_str)
            except:
                return 0
    
    # 使用正则表达式提取小时、分钟、秒
    try:
        hours = re.search(r'(\d+)小时', duration_str)
        minutes = re.search(r'(\d+)分', duration_str)
        seconds = re.search(r'(\d+)秒', duration_str)
        
        total_minutes = 0
        if hours:
            total_minutes += int(hours.group(1)) * 60
        if minutes:
            total_minutes += int(minutes.group(1))
        if seconds:
            # 秒数四舍五入到分钟，但用户说不需要秒，所以可以忽略
            # total_minutes += round(int(seconds.group(1)) / 60)
            pass
        
        return total_minutes
    except Exception as e:
        print(f"解析时间格式出错: {duration_str}, 错误: {e}")
        return 0

    
    
def minutes_to_hour_minute_format(total_minutes):
    """
    将总分钟数转换为 'X小时Y分' 格式
    """
    if total_minutes == 0:
        return '0小时0分'
    
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    
    return f'{hours}小时{minutes}分'


def calculate_sleep_duration_averages(sleep_df):
    """
    计算睡眠时长的平均值
    """
    # 需要计算平均值的时长字段
    duration_columns = [
        'total_duration',
        'in_bed_duration', 
        'out_bed_duration',
        'deep_sleep_duration',
        'light_sleep_duration',
        'awake_duration'
    ]
    
    results = {}
    
    # 计算现有字段的平均值
    for col in duration_columns:
        if col in sleep_df.columns:
            # 转换为分钟数
            minutes_list = sleep_df[col].apply(parse_duration_to_minutes)
            
            # 计算平均值
            avg_minutes = minutes_list.mean()
            
            # 转换回时间格式
            avg_formatted = minutes_to_hour_minute_format(avg_minutes)
            
            results[f'avg_{col}'] = avg_formatted
            
            print(f"{col} 平均值: {avg_formatted}")
    
    # 特别计算睡眠时长（深睡+浅睡）
    if 'deep_sleep_duration' in sleep_df.columns and 'light_sleep_duration' in sleep_df.columns:
        # 将深睡和浅睡时长转换为分钟数
        deep_minutes = sleep_df['deep_sleep_duration'].apply(parse_duration_to_minutes)
        light_minutes = sleep_df['light_sleep_duration'].apply(parse_duration_to_minutes)
        
        # 计算总睡眠时长（深睡+浅睡）
        total_sleep_minutes = deep_minutes + light_minutes
        
        # 计算平均睡眠时长
        avg_sleep_minutes = total_sleep_minutes.mean()
        avg_sleep_formatted = minutes_to_hour_minute_format(avg_sleep_minutes)
        
        results['avg_sleep_duration'] = avg_sleep_formatted
        print(f"sleep_duration (深睡+浅睡) 平均值: {avg_sleep_formatted}")
    
    return results


def main():
    """主函数"""
    
    # 检查登录状态
    if not check_login():
        return
    
    # 获取设备信息（从session_state获取）
    device_info = st.session_state.get('selected_device')
    if not device_info:
        st.error("未找到设备信息，请重新选择设备")
        if st.button("返回设备列表"):
            st.switch_page("pages/user_dashboard.py")
        return
    
    device_sn = device_info.get('device_code')
    if not device_sn:
        st.error("设备编号不存在")
        return
    
    # 获取真实睡眠数据
    sleep_df = get_sleep_statistics_data(device_sn)
    sleep_df_average = calculate_sleep_duration_averages(sleep_df)
    # 获取用户数据
    username = st.session_state.username
    users = load_users()
    user_data = users.get(username, {})
    
    # 显示名称：优先显示姓名，否则显示用户名
    display_name = user_data.get('name', username) if user_data.get('name') else username
    
    # 侧边栏
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; color: white;">
            <div style="font-size: 3rem; margin-bottom: 10px;">📊</div>
            <h3 style="margin: 0; color: white;">设备数据分析</h3>
            <p style="margin: 5px 0; color: rgba(255,255,255,0.8);">{display_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 导航菜单
        menu_options = {
            "📊 数据概览": "overview",
            "😴 睡眠监控": "sleep_monitor",
            "📈 健康报告": "health_report",
            "🔧 设备管理": "device_manage"
        }
        
        selected_menu = st.selectbox("功能导航", list(menu_options.keys()))
        
        st.markdown("---")
        
        # 快速统计
        st.markdown("### 📈 快速统计")
        if not sleep_df.empty:
            avg_sleep = sleep_df_average['avg_sleep_duration']
            avg_heart_rate = sleep_df['avg_heart_rate'].mean()
            avg_breath_rate = sleep_df['avg_breath_rate'].mean()
            
            st.metric("平均睡眠时长", f"{avg_sleep}")
            st.metric("平均心率", f"{avg_heart_rate:.0f}次/分")
            st.metric("平均呼吸率", f"{avg_breath_rate:.0f}次/分")
        else:
            st.info("暂无数据")
        
        st.markdown("---")
        
        # 返回按钮
        if st.button("🔙 返回设备列表", use_container_width=True):
            st.switch_page("pages/user_dashboard.py")
        
        # 退出登录
        if st.button("🚪 退出登录", use_container_width=True):
            # 清除登录状态
            for key in ['logged_in', 'username', 'user_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.switch_page("pages/login.py")
    
    
    # 主内容区域 - 用户信息和设备信息
    st.markdown(f"""
    <div class="user-header">
        <div class="user-avatar">📊</div>
        <h2 style="margin: 0;">睡眠健康数据分析</h2>
        <p style="margin: 5px 0;">用户：{display_name} | 今日：{datetime.now().strftime('%Y年%m月%d日')}</p>
    </div>
    """, unsafe_allow_html=True)

    # 单独添加跳转按钮
    # st.markdown(f"""
    # <div style="text-align: right; margin-top: -60px; margin-bottom: 20px; margin-right: 20px; position: relative; z-index: 10;">
    #     <a href="http://1.71.15.121:8000/real_time_vital?device_sn={device_sn}" 
    #     target="_blank" 
    #     style="display: inline-block; padding: 8px 16px; background-color: rgba(255, 255, 255, 0.9); color: #667eea; text-decoration: none; border-radius: 8px; font-size: 14px; font-weight: 500; border: 1px solid rgba(102, 126, 234, 0.3); box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    #         📊 实时监控
    #     </a>
    # </div>
    # """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📊 实时监控", use_container_width=True):
        # 保存当前设备信息到 session_state
            st.session_state.from_device_data = True
            st.session_state.return_device_info = device_info
            st.session_state.current_device_sn = device_sn  # 保存设备编号
            # 跳转到实时监控页面（不带URL参数）
            st.switch_page("pages/real_time_vital.py")


    # 设备信息卡片 - 保持原样，不做任何修改
    st.markdown(f"""
    <div class="device-info" style="background: white; padding: 20px; border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="margin: 0 0 15px 0; color: #333;">📱 设备信息</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div><strong>设备编号:</strong> {device_info.get('device_code', 'N/A')}</div>
            <div><strong>使用场景:</strong> {device_info.get('scene', 'N/A')}</div>
            <div><strong>WiFi名称:</strong> {device_info.get('wifi_name', 'N/A')}</div>
            <div><strong>设备状态:</strong> {'🟢 在线' if device_info.get('is_online', False) else '🔴 离线'}</div>
            <div><strong>数据记录:</strong> {len(sleep_df)} 条睡眠记录</div>
            <div><strong>创建时间:</strong> {str(device_info.get('create_time', 'N/A'))[:16] if device_info.get('create_time') else 'N/A'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 根据选择显示不同内容
    if selected_menu == "📊 数据概览":
        show_overview(sleep_df, sleep_df_average)
    elif selected_menu == "😴 睡眠监控":
        show_sleep_monitor(sleep_df)
    elif selected_menu == "📈 健康报告":
        show_health_report(sleep_df, sleep_df_average)
    elif selected_menu == "🔧 设备管理":
        show_device_manage(device_info)

def show_overview(sleep_df, sleep_df_average):
    """显示数据概览"""
    st.header("📊 数据概览")
    
    if sleep_df.empty:
        st.warning("暂无睡眠数据，请确保设备正常工作并同步数据")
        return
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_breath_rate = sleep_df['avg_breath_rate'].mean()
        status = "good" if 12 <= avg_breath_rate <= 20 else "warning" if 10 <= avg_breath_rate <= 25 else "danger"
        st.markdown(f"""
        <div class="metric-card">
            <div><span class="status-indicator status-{status}"></span>平均呼吸率</div>
            <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{avg_breath_rate:.1f}</div>
            <div style="font-size: 0.9rem; color: #666;">次/分钟</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_heart_rate = sleep_df['avg_heart_rate'].mean()
        status = "good" if 60 <= avg_heart_rate <= 100 else "warning" if 50 <= avg_heart_rate <= 120 else "danger"
        st.markdown(f"""
        <div class="metric-card">
            <div><span class="status-indicator status-{status}"></span>平均心率</div>
            <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{avg_heart_rate:.0f}</div>
            <div style="font-size: 0.9rem; color: #666;">次/分钟</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_sleep = sleep_df_average['avg_sleep_duration']
        # status = "good" if avg_sleep >= 7 else "warning" if avg_sleep >= 6 else "danger"
        status = "good"

        st.markdown(f"""
        <div class="metric-card">
            <div><span class="status-indicator status-{status}"></span>平均睡眠时长</div>
            <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{avg_sleep}</div>
            <div style="font-size: 0.9rem; color: #666;">小时</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_deep = sleep_df_average['avg_deep_sleep_duration']
        # status = "good" if avg_deep >= 1.2 else "warning" if avg_deep >= 1.0 else "danger"
        status = "good"
        st.markdown(f"""
        <div class="metric-card">
            <div><span class="status-indicator status-{status}"></span>平均深度睡眠</div>
            <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{avg_deep}</div>
            <div style="font-size: 0.9rem; color: #666;">小时</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 30天睡眠趋势
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📈 30天睡眠趋势分析")
    
    # 按睡眠开始时间排序
    sleep_df_sorted = sleep_df.sort_values('sleep_start_time')
    
    fig = go.Figure()
    
    # 添加理想睡眠时间参考线
    fig.add_hline(
        y=8, 
        line_dash="dot", 
        line_color="rgba(102, 126, 234, 0.3)",
        annotation_text="理想睡眠时长 (8小时)",
        annotation_position="top right",
        annotation=dict(
            font=dict(size=12, color="rgba(102, 126, 234, 0.6)"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(102, 126, 234, 0.3)",
            borderwidth=1
        )
    )
    
    # 添加轻度睡眠时长（填充区域）
    fig.add_trace(go.Scatter(
        x=sleep_df_sorted['sleep_start_time'],
        y=sleep_df_sorted['light_sleep_duration'],
        mode='lines',
        name='轻度睡眠',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 193, 7, 0.2)',
        hovertemplate='<b>轻度睡眠</b><br>' +
                      '日期: %{x|%Y-%m-%d}<br>' +
                      '时长: %{y:.1f}小时<extra></extra>',
        showlegend=True
    ))
    
    # 添加深度睡眠时长（填充区域）
    fig.add_trace(go.Scatter(
        x=sleep_df_sorted['sleep_start_time'],
        y=sleep_df_sorted['deep_sleep_duration'],
        mode='lines',
        name='深度睡眠',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor='rgba(40, 167, 69, 0.3)',
        hovertemplate='<b>深度睡眠</b><br>' +
                      '日期: %{x|%Y-%m-%d}<br>' +
                      '时长: %{y:.1f}小时<extra></extra>',
        showlegend=True
    ))
    
    # 添加总睡眠时长趋势线（主要线条）
    fig.add_trace(go.Scatter(
        x=sleep_df_sorted['sleep_start_time'],
        y=sleep_df_sorted['sleep_duration'],
        mode='lines+markers',
        name='总睡眠时长',
        line=dict(
            color='#667eea', 
            width=4,
            shape='spline',
            smoothing=0.3
        ),
        marker=dict(
            size=10,
            color='#667eea',
            symbol='circle',
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>总睡眠时长</b><br>' +
                      '日期: %{x|%Y-%m-%d}<br>' +
                      '时长: %{y:.1f}小时<br>' +
                      '<extra></extra>',
        showlegend=True
    ))
    
    # 添加深度睡眠趋势线
    fig.add_trace(go.Scatter(
        x=sleep_df_sorted['sleep_start_time'],
        y=sleep_df_sorted['deep_sleep_duration'],
        mode='lines+markers',
        name='深度睡眠时长',
        line=dict(
            color='#28a745', 
            width=3,
            shape='spline',
            smoothing=0.3,
            dash='dot'
        ),
        marker=dict(
            size=8,
            color='#28a745',
            symbol='diamond',
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>深度睡眠时长</b><br>' +
                      '日期: %{x|%Y-%m-%d}<br>' +
                      '时长: %{y:.1f}小时<br>' +
                      '<extra></extra>',
        showlegend=True
    ))
    
    # 更新布局样式
    fig.update_layout(
        title=dict(
            text='<b>睡眠时长与深度睡眠趋势</b>',
            x=0.5,
            font=dict(size=20, color='#2c3e50', family="Arial Black")
        ),
        xaxis=dict(
            title=dict(
                text='<b>日期</b>',
                font=dict(size=14, color='#34495e')
            ),
            tickformat='%m月%d日',
            tickangle=45,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showline=True,
            linewidth=2,
            linecolor='rgba(128, 128, 128, 0.3)',
            mirror=True
        ),
        yaxis=dict(
            title=dict(
                text='<b>睡眠时长 (小时)</b>',
                font=dict(size=14, color='#34495e')
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showline=True,
            linewidth=2,
            linecolor='rgba(128, 128, 128, 0.3)',
            mirror=True,
            range=[0, max(sleep_df_sorted['sleep_duration'].max() * 1.1, 10)]
        ),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        height=450,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            font_size=12,
            font_family="Arial"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=60, t=80, b=100),
        font=dict(family="Arial, sans-serif")
    )
    
    # 添加注释
    if len(sleep_df_sorted) > 0:
        latest_sleep = sleep_df_sorted.iloc[-1]['sleep_duration']
        avg_sleep = sleep_df_average['avg_sleep_duration']
        # trend = "上升" if latest_sleep > avg_sleep else "下降"
        # trend_color = "#28a745" if latest_sleep > avg_sleep else "#dc3545"
        trend = "上升"
        trend_color = "#28a745"
        
        fig.add_annotation(
            x=sleep_df_sorted.iloc[-1]['sleep_start_time'],
            y=latest_sleep,
            text=f"最新: {latest_sleep:.1f}h<br>趋势: {trend}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=trend_color,
            ax=50,
            ay=-50,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=trend_color,
            borderwidth=2,
            font=dict(size=11, color=trend_color)
        )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_sleep_monitor(sleep_df):
    """显示睡眠监控"""
    st.header("😴 睡眠监控")
    
    if sleep_df.empty:
        st.warning("暂无睡眠数据")
        return
    
    # 睡眠质量分析
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("💓 生理指标趋势")
        
        # 按睡眠开始时间排序
        sleep_df_sorted = sleep_df.sort_values('sleep_start_time')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sleep_df_sorted['sleep_start_time'],
            y=sleep_df_sorted['avg_heart_rate'],
            mode='lines+markers',
            name='平均心率',
            line=dict(color='#e74c3c', width=2),
            hovertemplate='<b>平均心率</b><br>' +
                          '时间: %{x}<br>' +
                          '心率: %{y:.0f}次/分<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=sleep_df_sorted['sleep_start_time'],
            y=sleep_df_sorted['avg_breath_rate'],
            mode='lines+markers',
            name='平均呼吸率',
            line=dict(color='#3498db', width=2),
            yaxis='y2',
            hovertemplate='<b>平均呼吸率</b><br>' +
                          '时间: %{x}<br>' +
                          '呼吸率: %{y:.0f}次/分<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title='睡眠开始时间',
            yaxis_title='心率 (次/分)',
            yaxis2=dict(
                title='呼吸率 (次/分)',
                overlaying='y',
                side='right'
            ),
            template='plotly_white',
            height=400,
            xaxis=dict(
                tickformat='%Y-%m-%d %H:%M',
                tickangle=45
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🛏️ 睡眠行为分析")
        
        # 计算平均值
        avg_movement = sleep_df['body_movement_count'].mean()
        avg_apnea = sleep_df['apnea_count'].mean()
        
        # 显示指标
        st.metric("平均翻身次数", f"{avg_movement:.1f}次")
        st.metric("平均呼吸暂停", f"{avg_apnea:.1f}次")
        
        # 绘制柱状图
        sleep_df_sorted = sleep_df.sort_values('sleep_start_time')
        
        fig = go.Figure(data=[
            go.Bar(name='翻身次数', 
                   x=sleep_df_sorted['sleep_start_time'], 
                   y=sleep_df_sorted['body_movement_count'], 
                   marker_color='#f39c12',
                   hovertemplate='<b>翻身次数</b><br>' +
                                 '时间: %{x}<br>' +
                                 '次数: %{y}<extra></extra>'),
            go.Bar(name='呼吸暂停', 
                   x=sleep_df_sorted['sleep_start_time'], 
                   y=sleep_df_sorted['apnea_count'], 
                   marker_color='#e74c3c',
                   hovertemplate='<b>呼吸暂停</b><br>' +
                                 '时间: %{x}<br>' +
                                 '次数: %{y}<extra></extra>')
        ])
        
        fig.update_layout(
            title='睡眠行为统计',
            xaxis_title='睡眠开始时间',
            yaxis_title='次数',
            template='plotly_white',
            height=300,
            xaxis=dict(
                tickformat='%Y-%m-%d %H:%M',
                tickangle=45
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_health_report(sleep_df, sleep_df_average):
    """显示健康报告"""
    st.header("📈 健康报告")
    
    if sleep_df.empty:
        st.warning("暂无数据生成健康报告")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("💊 健康建议")
        
        avg_sleep = sleep_df_average['avg_sleep_duration']
        avg_heart_rate = sleep_df['avg_heart_rate'].mean()
        avg_breath_rate = sleep_df['avg_breath_rate'].mean()
        avg_deep_sleep = sleep_df_average['avg_deep_sleep_duration']
        
        # 生成健康建议
        suggestions = []
        
        if avg_sleep < 7:
            suggestions.append("😴 建议增加睡眠时间，每晚保证7-9小时睡眠")
        
        if avg_deep_sleep < 1.2:
            suggestions.append("🛏️ 深度睡眠时间不足，建议睡前避免剧烈运动")
        
        if avg_heart_rate > 100:
            suggestions.append("💓 心率偏高，建议咨询医生并适当运动")
        
        if avg_breath_rate < 12 or avg_breath_rate > 20:
            suggestions.append("🫁 呼吸频率异常，建议关注呼吸健康")
        
        if not suggestions:
            st.success("🎉 您的睡眠状况很好！继续保持健康的睡眠习惯。")
        else:
            st.warning("⚠️ 发现以下需要注意的问题：")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 睡眠质量评分")
        
        # 计算综合得分
        sleep_score = min(100, (
            (avg_sleep / 8 * 30) +  # 睡眠时长权重30%
            (avg_deep_sleep / 2 * 25) +  # 深度睡眠权重25%
            (1 - min(sleep_df['apnea_count'].mean() / 10, 1)) * 25 +  # 呼吸暂停权重25%
            (1 - min(sleep_df['body_movement_count'].mean() / 20, 1)) * 20  # 翻身次数权重20%
        ))
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sleep_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "睡眠健康得分"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "#ff6b6b"},
                    {'range': [50, 80], 'color': "#ffc107"},
                    {'range': [80, 100], 'color': "#28a745"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_device_manage(device_info):
    """显示设备管理"""
    st.header("🔧 设备管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📱 设备信息")
        
        st.info(f"**设备编号:** {device_info.get('device_code', 'N/A')}")
        st.info(f"**使用场景:** {device_info.get('scene', 'N/A')}")
        st.info(f"**WiFi名称:** {device_info.get('wifi_name', 'N/A')}")
        st.info(f"**设备状态:** {'🟢 在线' if device_info.get('is_online', False) else '🔴 离线'}")
        st.info(f"**创建时间:** {device_info.get('create_time', 'N/A')}")
        st.info(f"**更新时间:** {device_info.get('update_time', 'N/A')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("⚙️ 设备操作")
        
        if st.button("🔄 同步数据", use_container_width=True):
            st.info("数据同步功能开发中...")
        
        if st.button("📊 导出数据", use_container_width=True):
            st.info("数据导出功能开发中...")
        
        if st.button("🔧 设备配置", use_container_width=True):
            st.info("设备配置功能开发中...")
        
        if st.button("🆘 设备重启", use_container_width=True):
            st.warning("设备重启功能开发中...")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()