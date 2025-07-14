import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# 添加项目路径
project_root = "/work/ai/WHOAMI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据库模型和提供者
from whoami.tool.streamlit.health_report.table.device_data import DeviceData
from whoami.provider.sql_provider import SqlProvider

# 页面配置
st.set_page_config(
    page_title="用户主页 - 睡眠健康管理系统",
    page_icon="🏠",
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

def get_device_info(device_sn):
    """根据设备编号获取设备信息"""
    try:
        sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        result = sql_provider.get_record_by_condition(condition={"device_code": device_sn})
        return result
    except Exception as e:
        st.error(f"获取设备信息失败: {str(e)}")
        return None

def update_device_info(device_sn, scene, wifi_name, wifi_password):
    """更新设备信息"""
    try:
        sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        update_data = {
            "scene": scene,
            "wifi_name": wifi_name, 
            "wifi_password": wifi_password,
            "update_time": datetime.now()
        }
        result = sql_provider.update_record_by_condition(
            condition={"device_code": device_sn},
            update_data=update_data
        )
        return result
    except Exception as e:
        st.error(f"更新设备信息失败: {str(e)}")
        return False

def generate_sleep_data():
    """生成模拟睡眠数据"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    sleep_data = []
    
    for date in dates:
        # 模拟睡眠数据
        sleep_duration = np.random.normal(7.5, 1.2)  # 平均7.5小时，标准差1.2
        sleep_quality = np.random.randint(60, 100)  # 睡眠质量60-100
        deep_sleep = np.random.normal(1.5, 0.3)  # 深度睡眠时间
        light_sleep = sleep_duration - deep_sleep - np.random.normal(1.0, 0.2)  # 浅度睡眠
        rem_sleep = np.random.normal(1.2, 0.2)  # REM睡眠
        
        sleep_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'sleep_duration': max(4, min(12, sleep_duration)),
            'sleep_quality': sleep_quality,
            'deep_sleep': max(0.5, deep_sleep),
            'light_sleep': max(2, light_sleep),
            'rem_sleep': max(0.5, rem_sleep),
            'bedtime': f"{np.random.randint(21, 24)}:{np.random.randint(0, 59):02d}",
            'wake_time': f"{np.random.randint(6, 9)}:{np.random.randint(0, 59):02d}"
        })
    
    return pd.DataFrame(sleep_data)

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

/* 用户头部样式 - 缩小尺寸 */
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
    padding: 10px;
    border-radius: 10px;
    margin-top: 10px;
    text-align: left;
}

.device-info-item {
    margin: 5px 0;
    font-size: 0.9rem;
}

/* 设备信息修改按钮 */
.edit-device-btn {
    position: absolute;
    top: 15px;
    right: 15px;
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: white;
    border-radius: 8px;
    padding: 5px 10px;
    cursor: pointer;
    font-size: 0.8rem;
}

.edit-device-btn:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* 功能按钮样式 */
.feature-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 15px 25px;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
    margin: 5px 0;
    font-weight: 600;
}

.feature-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
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

/* 数据统计卡片 */
.stat-card {
    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin: 10px 0;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

.stat-value {
    font-size: 2.5rem;
    font-weight: bold;
    margin: 10px 0;
}

.stat-label {
    font-size: 1rem;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

def show_device_edit_modal(device_info):
    """显示设备信息编辑弹窗"""
    if st.button("⚙️ 设备设置", key="edit_device_btn"):
        st.session_state.show_device_modal = True
    
    if getattr(st.session_state, 'show_device_modal', False):
        with st.container():
            st.markdown("---")
            st.subheader("🔧 编辑设备信息")
            
            col1, col2 = st.columns(2)
            with col1:
                new_scene = st.text_input("场景", value=device_info.get('scene', '') if device_info else '')
                new_wifi_name = st.text_input("WiFi名称", value=device_info.get('wifi_name', '') if device_info else '')
            
            with col2:
                new_wifi_password = st.text_input("WiFi密码", 
                                                value=device_info.get('wifi_password', '') if device_info else '', 
                                                type="password")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💾 保存", type="primary", use_container_width=True):
                    device_sn = st.query_params.get("device_sn", "")
                    if device_sn and update_device_info(device_sn, new_scene, new_wifi_name, new_wifi_password):
                        st.success("设备信息更新成功！")
                        st.session_state.show_device_modal = False
                        st.rerun()
                    else:
                        st.error("更新失败，请重试！")
            
            with col2:
                if st.button("❌ 取消", use_container_width=True):
                    st.session_state.show_device_modal = False
                    st.rerun()
            
            st.markdown("---")

def main():
    """主函数"""
    
    # 检查登录状态
    if not check_login():
        return
    
    # 获取URL参数中的设备编号
    device_sn = st.query_params.get("device_sn", "")
    
    # 获取设备信息
    device_info = None
    if device_sn:
        device_info = get_device_info(device_sn)
    
    # 获取用户数据
    username = st.session_state.username
    users = load_users()
    user_data = users.get(username, {})
    
    # 显示名称：优先显示姓名，否则显示用户名
    display_name = user_data.get('name', username) if user_data.get('name') else username
    
    # 生成睡眠数据
    sleep_df = generate_sleep_data()
    
    # 侧边栏
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; color: white;">
            <div style="font-size: 3rem; margin-bottom: 10px;">👤</div>
            <h3 style="margin: 0; color: white;">欢迎回来</h3>
            <p style="margin: 5px 0; color: rgba(255,255,255,0.8);">{display_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 导航菜单
        menu_options = {
            "📊 数据概览": "overview",
            "😴 睡眠监控": "sleep_monitor",
            "📈 健康报告": "health_report",
            "⚙️ 系统设置": "settings",
            "👤 个人资料": "profile"
        }
        
        selected_menu = st.selectbox("功能导航", list(menu_options.keys()))
        
        st.markdown("---")
        
        # 快速统计
        st.markdown("### 📈 快速统计")
        avg_sleep = sleep_df['sleep_duration'].mean()
        avg_quality = sleep_df['sleep_quality'].mean()
        
        st.metric("平均睡眠时长", f"{avg_sleep:.1f}小时")
        st.metric("平均睡眠质量", f"{avg_quality:.0f}分")
        
        st.markdown("---")
        
        # 退出登录
        if st.button("🚪 退出登录", use_container_width=True):
            # 清除登录状态
            for key in ['logged_in', 'username', 'user_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.switch_page("pages/login.py")
    
    # 主内容区域 - 用户信息和设备信息
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # 设备信息HTML
        device_info_html = ""
        if device_info:
            device_info_html = f"""
            <div class="device-info">
                <div class="device-info-item"><strong>设备编号:</strong> {device_info.get('device_code', 'N/A')}</div>
                <div class="device-info-item"><strong>场景:</strong> {device_info.get('scene', 'N/A')}</div>
                <div class="device-info-item"><strong>WiFi名称:</strong> {device_info.get('wifi_name', 'N/A')}</div>
                <div class="device-info-item"><strong>WiFi密码:</strong> {'●' * len(device_info.get('wifi_password', '')) if device_info.get('wifi_password') else 'N/A'}</div>
            </div>
            """
        elif device_sn:
            device_info_html = f"""
            <div class="device-info">
                <div class="device-info-item"><strong>设备编号:</strong> {device_sn}</div>
                <div class="device-info-item" style="color: #ffcccb;"><strong>状态:</strong> 设备信息未找到</div>
            </div>
            """
        
        st.markdown(f"""
        <div class="user-header">
            <div class="user-avatar">🌙</div>
            <h2 style="margin: 0;">欢迎使用睡眠健康管理系统</h2>
            <p style="margin: 5px 0;">用户：{display_name} | 今日：{datetime.now().strftime('%Y年%m月%d日')}</p>
            {device_info_html}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 设备编辑按钮
        if device_sn:
            show_device_edit_modal(device_info)
    
    # 根据选择显示不同内容
    if selected_menu == "📊 数据概览":
        show_overview(sleep_df)
    elif selected_menu == "😴 睡眠监控":
        show_sleep_monitor(sleep_df)
    elif selected_menu == "📈 健康报告":
        show_health_report(sleep_df)
    elif selected_menu == "⚙️ 系统设置":
        show_settings()
    elif selected_menu == "👤 个人资料":
        show_profile(user_data)

def show_overview(sleep_df):
    """显示数据概览"""
    st.header("📊 数据概览")
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_sleep = sleep_df['sleep_duration'].mean()
        status = "good" if avg_sleep >= 7 else "warning" if avg_sleep >= 6 else "danger"
        st.markdown(f"""
        <div class="metric-card">
            <div><span class="status-indicator status-{status}"></span>平均睡眠时长</div>
            <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{avg_sleep:.1f}小时</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_quality = sleep_df['sleep_quality'].mean()
        status = "good" if avg_quality >= 80 else "warning" if avg_quality >= 70 else "danger"
        st.markdown(f"""
        <div class="metric-card">
            <div><span class="status-indicator status-{status}"></span>平均睡眠质量</div>
            <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{avg_quality:.0f}分</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_deep = sleep_df['deep_sleep'].mean()
        status = "good" if avg_deep >= 1.2 else "warning" if avg_deep >= 1.0 else "danger"
        st.markdown(f"""
        <div class="metric-card">
            <div><span class="status-indicator status-{status}"></span>平均深度睡眠</div>
            <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{avg_deep:.1f}小时</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        consistency = len(sleep_df[sleep_df['sleep_duration'].between(7, 9)]) / len(sleep_df) * 100
        status = "good" if consistency >= 70 else "warning" if consistency >= 50 else "danger"
        st.markdown(f"""
        <div class="metric-card">
            <div><span class="status-indicator status-{status}"></span>睡眠规律性</div>
            <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{consistency:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 睡眠趋势图
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📈 30天睡眠趋势")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sleep_df['date'],
        y=sleep_df['sleep_duration'],
        mode='lines+markers',
        name='睡眠时长',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=sleep_df['date'],
        y=sleep_df['sleep_quality'] / 10,  # 缩放到与睡眠时长相近的范围
        mode='lines+markers',
        name='睡眠质量 (×10)',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='睡眠时长与质量趋势',
        xaxis_title='日期',
        yaxis_title='睡眠时长 (小时)',
        yaxis2=dict(
            title='睡眠质量 (分)',
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_sleep_monitor(sleep_df):
    """显示睡眠监控"""
    st.header("😴 睡眠监控")
    
    # 睡眠阶段分析
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🔄 睡眠阶段分析")
    
    # 饼图显示睡眠阶段分布
    latest_data = sleep_df.iloc[-1]
    
    fig = go.Figure(data=[go.Pie(
        labels=['深度睡眠', '浅度睡眠', 'REM睡眠'],
        values=[latest_data['deep_sleep'], latest_data['light_sleep'], latest_data['rem_sleep']],
        hole=0.3,
        marker_colors=['#667eea', '#ff9a9e', '#fecfef']
    )])
    
    fig.update_layout(
        title='最近一晚睡眠阶段分布',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 睡眠质量热力图
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🌡️ 睡眠质量热力图")
    
    # 创建热力图数据
    sleep_df['weekday'] = pd.to_datetime(sleep_df['date']).dt.day_name()
    sleep_df['week'] = pd.to_datetime(sleep_df['date']).dt.isocalendar().week
    
    heatmap_data = sleep_df.pivot_table(
        values='sleep_quality',
        index='weekday',
        columns='week',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale='RdYlBu_r',
        title='每周睡眠质量分布',
        labels={'color': '睡眠质量分数'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_health_report(sleep_df):
    """显示健康报告"""
    st.header("📈 健康报告")
    
    # 健康评估
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("💊 健康建议")
        
        avg_sleep = sleep_df['sleep_duration'].mean()
        avg_quality = sleep_df['sleep_quality'].mean()
        
        if avg_sleep >= 7 and avg_quality >= 80:
            st.success("🎉 您的睡眠状况很好！继续保持健康的睡眠习惯。")
        elif avg_sleep >= 6 and avg_quality >= 70:
            st.warning("⚠️ 您的睡眠状况一般，建议调整作息时间。")
        else:
            st.error("🚨 您的睡眠质量需要改善，建议咨询医生。")
        
        st.markdown("### 改善建议：")
        st.markdown("""
        - 🕘 保持规律的睡眠时间
        - 🚫 睡前避免使用电子设备
        - 🌡️ 保持卧室温度适宜
        - 🥛 睡前避免咖啡因
        - 🏃 适度运动，但不要在睡前进行
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📊 睡眠得分")
        
        # 计算综合得分
        sleep_score = (avg_sleep / 8 * 40 + avg_quality / 100 * 60)
        
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

def show_settings():
    """显示系统设置"""
    st.header("⚙️ 系统设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔔 通知设置")
        
        enable_notifications = st.checkbox("启用通知", value=True)
        if enable_notifications:
            st.time_input("就寝提醒时间", value=datetime.strptime("22:00", "%H:%M").time())
            st.time_input("起床提醒时间", value=datetime.strptime("07:00", "%H:%M").time())
        
        st.subheader("🎯 睡眠目标")
        target_sleep = st.slider("目标睡眠时长 (小时)", 6.0, 10.0, 8.0, 0.5)
        target_quality = st.slider("目标睡眠质量 (分)", 70, 100, 80, 5)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📱 数据同步")
        
        sync_options = st.multiselect(
            "选择同步的设备",
            ["Apple Watch", "Fitbit", "小米手环", "华为手环"],
            default=["Apple Watch"]
        )
        
        st.subheader("📊 数据导出")
        export_format = st.selectbox("导出格式", ["CSV", "Excel", "PDF"])
        
        if st.button("导出数据", use_container_width=True):
            st.success("数据导出成功！")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_profile(user_data):
    """显示个人资料"""
    st.header("👤 个人资料")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📋 基本信息")
        
        username = st.text_input("用户名", value=st.session_state.username, disabled=True)
        email = st.text_input("邮箱", value=user_data.get('email', ''))
        phone = st.text_input("手机号", value=user_data.get('phone', ''))
        
        st.subheader("🎂 个人信息")
        age = st.number_input("年龄", min_value=1, max_value=120, value=30)
        gender = st.selectbox("性别", ["男", "女", "其他"])
        height = st.number_input("身高 (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("体重 (kg)", min_value=30, max_value=200, value=70)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📅 账户信息")
        
        st.info(f"注册时间: {user_data.get('create_time', '未知')}")
        st.info(f"最后更新: {user_data.get('update_time', '未知')}")
        st.info(f"账户状态: {user_data.get('status', '未知')}")
        
        st.subheader("🔐 安全设置")
        if st.button("修改密码", use_container_width=True):
            st.info("密码修改功能开发中...")
        
        if st.button("注销账户", use_container_width=True):
            st.error("账户注销功能开发中...")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("保存修改", use_container_width=True, type="primary"):
            st.success("个人信息更新成功！")
    with col2:
        if st.button("重置", use_container_width=True):
            st.rerun()

if __name__ == "__main__":
    main()