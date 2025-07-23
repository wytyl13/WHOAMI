import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta
import time

# 添加项目路径
project_root = "/work/ai/WHOAMI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据库模型和提供者
from whoami.tool.streamlit.health_report.table.device_data import DeviceData
from whoami.tool.real_time_vital_analyze.sleep_data_state import SleepDataState
from whoami.provider.sql_provider import SqlProvider

# 页面配置
st.set_page_config(
    page_title="社区智能体 - 用户首页",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 配置文件路径
USER_DATA_FILE = os.path.join(project_root, "data", "users.json")
SQL_CONFIG_PATH = '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml'

# CSS样式
st.markdown("""
<style>

/* 隐藏侧边栏 */
section[data-testid="stSidebar"] {
    display: none !important;
}

.css-1d391kg, .css-17lntkn, .css-1rs6os, .css-10trblm,
.css-12oz5g7, .css-1outpf7, .css-1y4p8pa, .css-1lcbmhc,
.css-1v0mbdj, .css-1cypcdb, .css-17eq0hr, .css-zt5igj {
    display: none !important;
}

button[kind="header"] {
    display: none !important;
}
/* 隐藏侧边栏 */


/* 全局样式 */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

/* 隐藏默认边距 */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* 用户欢迎头部样式 */
.welcome-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px 40px;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.welcome-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    z-index: 1;
}

.welcome-content {
    position: relative;
    z-index: 2;
}

.welcome-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0 0 10px 0;
    text-align: center;
}

.welcome-subtitle {
    font-size: 1.2rem;
    margin: 0;
    text-align: center;
    opacity: 0.9;
}

/* 设备管理标题 */
.section-title {
    background: white;
    padding: 20px 30px;
    border-radius: 15px;
    margin: 20px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 5px solid #667eea;
}

.section-title h2 {
    margin: 0;
    color: #333;
    font-size: 1.8rem;
    font-weight: 600;
}

.section-title .icon {
    color: #667eea;
    margin-right: 10px;
}

/* 设备列表容器 */
.device-list-container {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    margin: 20px 0;
}

.device-list-title {
    font-size: 1.4rem;
    color: #333;
    margin-bottom: 25px;
    font-weight: 600;
}

/* 设备卡片网格 */
.device-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

/* 设备卡片样式 */
.device-card {
    background: white;
    border: 2px solid #e74c3c;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    box-shadow: 0 4px 12px rgba(231, 76, 60, 0.15);
}

.device-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 35px rgba(231, 76, 60, 0.25);
    border-color: #c0392b;
}

.device-card.offline {
    border-color: #dc3545;
    background: rgba(220, 53, 69, 0.05);
}

.device-card.offline:hover {
    border-color: #c82333;
    box-shadow: 0 12px 35px rgba(220, 53, 69, 0.25);
}

.device-name {
    font-weight: 700;
    color: #e74c3c;
    font-size: 1.3rem;
    margin-bottom: 12px;
}

.device-card.offline .device-name {
    color: #dc3545;
}

.device-code {
    font-size: 0.9rem;
    color: #666;
    font-family: 'Monaco', 'Menlo', monospace;
    margin-bottom: 8px;
}

.device-wifi {
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 8px;
}

.device-status {
    font-size: 0.9rem;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 20px;
    display: inline-block;
}

/* 设备状态指示器动画 */
.status-online {
    background: rgba(40, 167, 69, 0.1);
    color: #28a745;
    animation: pulse-green 2s infinite;
}

.status-offline {
    background: rgba(220, 53, 69, 0.1);
    color: #dc3545;
    animation: pulse-red 2s infinite;
}

@keyframes pulse-green {
    0% {
        background: rgba(40, 167, 69, 0.1);
    }
    50% {
        background: rgba(40, 167, 69, 0.2);
    }
    100% {
        background: rgba(40, 167, 69, 0.1);
    }
}

@keyframes pulse-red {
    0% {
        background: rgba(220, 53, 69, 0.1);
    }
    50% {
        background: rgba(220, 53, 69, 0.2);
    }
    100% {
        background: rgba(220, 53, 69, 0.1);
    }
}

/* 状态更新提示 */
.status-update-info {
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
    padding: 10px 15px;
    border-radius: 8px;
    font-size: 0.9rem;
    margin-bottom: 15px;
    border-left: 3px solid #667eea;
}

/* 添加设备卡片 */
.add-device-card {
    border: 2px dashed #28a745;
    background: rgba(40, 167, 69, 0.05);
    border-radius: 15px;
    height: 160px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
}

.add-device-card:hover {
    background: rgba(40, 167, 69, 0.1);
    transform: translateY(-8px);
    box-shadow: 0 12px 35px rgba(40, 167, 69, 0.15);
}

.add-icon {
    font-size: 3rem;
    color: #28a745;
    margin-bottom: 10px;
}

.add-text {
    color: #28a745;
    font-weight: 600;
    font-size: 1rem;
}

/* 设备操作按钮 */
.device-actions {
    margin-top: 15px;
    display: flex;
    gap: 8px;
    justify-content: center;
}

.action-btn {
    padding: 6px 12px;
    border: none;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s ease;
    font-weight: 500;
}

.btn-view {
    background: #17a2b8;
    color: white;
}

.btn-edit {
    background: #ffc107;
    color: #212529;
}

.btn-delete {
    background: #dc3545;
    color: white;
}

.btn-reconnect {
    background: #fd7e14;
    color: white;
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* 离线设备提示 */
.offline-banner {
    background: linear-gradient(135deg, #fd7e14 0%, #f85032 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    margin: 10px 0;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.offline-banner-icon {
    font-size: 1.2rem;
    margin-right: 10px;
}

.offline-banner-text {
    flex: 1;
}

.offline-banner-action {
    margin-left: 15px;
}

/* 按钮样式 */
.btn {
    padding: 12px 24px;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 5px;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.btn-secondary {
    background: #6c757d;
    color: white;
}

.btn-success {
    background: #28a745;
    color: white;
}

.btn-danger {
    background: #dc3545;
    color: white;
}

.btn-warning {
    background: #fd7e14;
    color: white;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

/* 消息样式 */
.message {
    padding: 15px 20px;
    border-radius: 10px;
    margin: 15px 0;
    font-weight: 500;
}

.message-success {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.message-error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.message-warning {
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffeaa7;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .device-grid {
        grid-template-columns: 1fr;
    }
    
    .welcome-title {
        font-size: 2rem;
    }
    
    .welcome-subtitle {
        font-size: 1rem;
    }
}



/* 用户欢迎头部样式 - 修改为三列布局 */
.welcome-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px 40px;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.welcome-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    z-index: 1;
}

/* 左侧按钮区域 */
.header-left {
    flex: 0 0 auto;
    z-index: 3;
}

/* 中间欢迎内容区域 */
.welcome-content {
    flex: 1;
    position: relative;
    z-index: 2;
    text-align: center;
}

/* 右侧空白区域，保持平衡 */
.header-right {
    flex: 0 0 auto;
    width: 120px; /* 与左侧按钮宽度相等，保持平衡 */
}

.welcome-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0 0 10px 0;
}

.welcome-subtitle {
    font-size: 1.2rem;
    margin: 0;
    opacity: 0.9;
}

/* 返回按钮样式 */
.return-btn {
    background: rgba(255, 255, 255, 0.15);
    color: white;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 12px;
    padding: 12px 20px;
    font-size: 1rem;
    font-weight: 600;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    white-space: nowrap;
}

.return-btn:hover {
    background: rgba(255, 255, 255, 0.25);
    border-color: rgba(255, 255, 255, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    color: white;
    text-decoration: none;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .welcome-header {
        flex-direction: column;
        gap: 20px;
        padding: 25px 20px;
    }
    
    .header-left {
        order: -1;
        align-self: flex-start;
    }
    
    .welcome-content {
        order: 0;
    }
    
    .header-right {
        display: none;
    }
    
    .welcome-title {
        font-size: 2rem;
    }
    
    .welcome-subtitle {
        font-size: 1rem;
    }
    
    .return-btn {
        padding: 10px 16px;
        font-size: 0.9rem;
    }
}

/* 小屏幕进一步优化 */
@media (max-width: 480px) {
    .welcome-header {
        padding: 20px 15px;
    }
    
    .welcome-title {
        font-size: 1.8rem;
    }
    
    .return-btn {
        padding: 8px 12px;
        font-size: 0.85rem;
    }
}



</style>
""", unsafe_allow_html=True)

# 数据库操作函数
@st.cache_data(ttl=300)  # 缓存5分钟
def load_users():
    """加载用户数据"""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def check_device_online_status(device_code):
    """检查设备在线状态"""
    try:
        # 获取当前时间戳
        current_timestamp = datetime.now().timestamp()
        # 检查最近30分钟内的数据（30分钟 = 30 * 60 = 1800秒）
        start_timestamp = current_timestamp - 1800
        
        sql_provider = SqlProvider(
            model=SleepDataState, 
            sql_config_path=SQL_CONFIG_PATH
        )
        
        # 查询最近30分钟内的数据
        recent_data = sql_provider.get_record_by_condition(
            condition={
                "device_id": device_code,
                "timestamp": {"min": start_timestamp, "max": current_timestamp}
            },
            fields=["timestamp", "state"]
        )
        
        # 如果有数据，说明设备在线
        return len(recent_data) > 0
        
    except Exception as e:
        st.warning(f"检查设备 {device_code} 在线状态失败: {str(e)}")
        return False

def update_device_online_status(device_code, is_online):
    """更新设备在线状态到数据库"""
    try:
        sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        
        # 首先根据device_code查找设备记录，确认设备存在
        device_records = sql_provider.get_record_by_condition(
            condition={"device_code": device_code}
        )
        
        if not device_records:
            st.warning(f"未找到设备 {device_code}")
            return False
            
        # 准备更新数据，包含device_code作为唯一字段
        update_data = {
            "status": 'active' if is_online else 'inactive',
            "update_time": datetime.now()
        }
        
        # 使用upsert_record_by_unique_field更新记录
        result = sql_provider.update_record_enhanced(
            record_id=device_records[0]["id"],
            data=update_data,
        )
        return result is not None
        
    except Exception as e:
        st.warning(f"更新设备 {device_code} 状态失败: {str(e)}")
        return False

def get_user_devices(username):
    """获取用户的所有设备并更新在线状态"""
    try:
        sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        results = sql_provider.get_record_by_condition(
            condition={"username": username},
            fields=["id", "device_code", "scene", "wifi_name", "wifi_password", "status", "create_time", "update_time"]
        )
        print(results)
        devices = []
        for result in results:
            # 检查设备在线状态
            is_online = check_device_online_status(result["device_code"])
            
            # 如果当前数据库中的状态与实际在线状态不一致，则更新
            current_status = 'active' if is_online else 'inactive'
            if result["status"] != current_status:
                update_device_online_status(result["device_code"], is_online)
                # 更新本地状态
                result["status"] = current_status
            
            # 添加实时在线状态标识
            result["is_online"] = is_online
            devices.append(result)
        return devices
    except Exception as e:
        st.error(f"获取设备列表失败: {str(e)}")
        return []

def update_device_info(device_sn, update_data):
    """更新设备信息"""
    try:
        sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        
        # 首先根据device_code查找设备记录，确认设备存在
        device_records = sql_provider.get_record_by_condition(
            condition={"device_code": device_sn}
        )
        
        if not device_records:
            st.error(f"未找到设备 {device_sn}")
            return False
            
        update_data["update_time"] = datetime.now()
        # 使用upsert_record_by_unique_field更新记录
        
        result = sql_provider.update_record_enhanced(
            record_id=device_records[0]["id"],
            data=update_data
        )
        return result is not None
        
    except Exception as e:
        st.error(f"更新设备信息失败: {str(e)}")
        return False

def delete_device(device_sn):
    """删除设备（硬删除）"""
    try:
        sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        
        # 使用条件删除
        deleted_count = sql_provider.delete_records_by_condition(
            condition={"device_code": device_sn}
        )
        
        return deleted_count > 0
        
    except Exception as e:
        st.error(f"删除设备失败: {str(e)}")
        return False

def check_login():
    """检查登录状态"""
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.error("请先登录！")
        st.switch_page("pages/login.py")
        return False
    return True


def show_welcome_header(username):
    """显示欢迎头部"""
    users = load_users()
    user_data = users.get(username, {})
    display_name = user_data.get('name', username) if user_data.get('name') else username
    
    current_date = datetime.now().strftime('%Y年%m月%d日')
    
    st.markdown(f"""
    <div class="welcome-header">
        <div class="header-left">
            <a href="https://1.71.15.121:8000/login" class="return-btn" target="_self">
                🔙 返回登录
            </a>
        </div>
        <div class="welcome-content">
            <h1 class="welcome-title">欢迎使用社区智能体</h1>
            <p class="welcome-subtitle">用户：{display_name} | 今日：{current_date}</p>
        </div>
        <div class="header-right">
            <!-- 右侧空白区域，保持平衡 -->
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 处理登录状态清除的JavaScript
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const returnBtn = document.querySelector('.return-btn');
        if (returnBtn) {
            returnBtn.addEventListener('click', function(e) {
                // 清除本地存储
                localStorage.clear();
                sessionStorage.clear();
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)




def show_device_card(device):
    """显示设备卡片"""
    # 使用实时在线状态
    is_online = device.get('is_online', device['status'] == 'active')
    status_class = 'status-online' if is_online else 'status-offline'
    status_text = '在线' if is_online else '离线'
    card_class = 'device-card' if is_online else 'device-card offline'
    
    # 截断设备编号显示
    device_code_display = device['device_code'][:12] + '...' if len(device['device_code']) > 12 else device['device_code']
    
    # 添加实时状态指示器
    status_indicator = '🟢' if is_online else '🔴'
    
    card_html = f"""
    <div class="{card_class}">
        <div class="device-name">{device['scene'] or '设备'}</div>
        <div class="device-code">{device_code_display}</div>
        <div class="device-wifi">{device['wifi_name'] or '未配置WiFi'}</div>
        <div class="device-status {status_class}">{status_indicator} {status_text}</div>
    </div>
    """
    
    return card_html

def show_add_device_card():
    """显示添加设备卡片"""
    return """
    <div class="add-device-card">
        <div class="add-icon">+</div>
        <div class="add-text">添加设备</div>
    </div>
    """

def show_offline_devices_banner(offline_devices):
    """显示离线设备横幅提示"""
    if offline_devices:
        offline_count = len(offline_devices)
        st.markdown(f"""
        <div class="offline-banner">
            <div class="offline-banner-icon">⚠️</div>
            <div class="offline-banner-text">
                检测到 {offline_count} 台设备离线，建议进行重新连接
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_device_management(username):
    """显示设备管理"""
    # 获取用户设备列表
    devices = get_user_devices(username)
    # 统计离线设备
    offline_devices = [device for device in devices if not device.get('is_online', False)]
    
    # 设备管理标题
    st.markdown("""
    <div class="section-title">
        <h2><span class="icon">📱</span>设备管理</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示离线设备提示
    show_offline_devices_banner(offline_devices)
    
    # 设备列表容器
    st.markdown('<div class="device-list-container">', unsafe_allow_html=True)
    st.markdown('<div class="device-list-title">我的设备列表</div>', unsafe_allow_html=True)
    
    # 添加状态更新提示和刷新按钮
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="status-update-info">
            💡 设备状态已实时更新（基于最近30分钟内的数据活动）
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🔄 刷新状态", key="refresh_status", use_container_width=True):
            # 清除缓存，重新获取设备状态
            st.cache_data.clear()
            st.rerun()
    
    # 使用columns布局显示设备
    if devices:
        # 计算列数，每行最多3个设备
        cols_per_row = 3
        rows = (len(devices) + 1 + cols_per_row - 1) // cols_per_row  # +1 for add button
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                device_idx = row * cols_per_row + col_idx
                
                with cols[col_idx]:
                    if device_idx < len(devices):
                        device = devices[device_idx]
                        st.markdown(show_device_card(device), unsafe_allow_html=True)
                        
                        # 添加不可见的点击区域
                        if st.button("查看详情", key=f"view_detail_{device['device_code']}", 
                                    use_container_width=True):
                            # 将设备信息存储到session_state中传递给下一个页面
                            st.session_state.selected_device = device
                            st.session_state.device_code = device['device_code']
                            st.switch_page("pages/device_data.py")
                        
                        # 设备操作按钮
                        is_online = device.get('is_online', False)
                        
                        if is_online:
                            # 在线设备：查看、编辑、删除
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("查看", key=f"view_{device['device_code']}", use_container_width=True):
                                    st.session_state.view_device = device
                            with col2:
                                if st.button("编辑", key=f"edit_{device['device_code']}", use_container_width=True):
                                    st.session_state.edit_device = device
                            with col3:
                                if st.button("删除", key=f"delete_{device['device_code']}", use_container_width=True):
                                    if st.session_state.get('confirm_delete') != device['device_code']:
                                        st.session_state.confirm_delete = device['device_code']
                                        st.rerun()
                        else:
                            # 离线设备：重新连接、编辑、删除
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("🔗 重连", key=f"reconnect_{device['device_code']}", use_container_width=True):
                                    # 设置重连参数到session state
                                    st.session_state.connection_mode = "reconnect"
                                    st.session_state.connection_device_code = device['device_code']
                                    # 跳转到设备连接页面
                                    st.switch_page("pages/device_connection.py")
                            with col2:
                                if st.button("编辑", key=f"edit_{device['device_code']}", use_container_width=True):
                                    st.session_state.edit_device = device
                            with col3:
                                if st.button("删除", key=f"delete_{device['device_code']}", use_container_width=True):
                                    if st.session_state.get('confirm_delete') != device['device_code']:
                                        st.session_state.confirm_delete = device['device_code']
                                        st.rerun()
                        
                    elif device_idx == len(devices):
                        st.markdown(show_add_device_card(), unsafe_allow_html=True)
                        if st.button("🔗 连接新设备", key="add_device_btn", use_container_width=True):
                            # 设置添加设备参数到session state
                            st.session_state.connection_mode = "add"
                            st.session_state.connection_device_code = None
                            # 跳转到设备连接页面
                            st.switch_page("pages/device_connection.py")
                    else:
                        st.empty()
    else:
        # 如果没有设备，只显示添加按钮
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(show_add_device_card(), unsafe_allow_html=True)
            if st.button("🔗 连接新设备", key="add_device_btn", use_container_width=True):
                # 设置添加设备参数到session state
                st.session_state.connection_mode = "add"
                st.session_state.connection_device_code = None
                # 跳转到设备连接页面
                st.switch_page("pages/device_connection.py")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_device_modals(username):
    """显示设备相关的模态框"""
    # 查看设备详情
    if 'view_device' in st.session_state:
        device = st.session_state.view_device
        with st.expander(f"📱 设备详情 - {device['scene'] or '未命名设备'}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**设备编号:** {device['device_code']}")
                st.write(f"**使用场景:** {device['scene'] or 'N/A'}")
                st.write(f"**WiFi名称:** {device['wifi_name'] or 'N/A'}")
            with col2:
                # 显示实时在线状态
                is_online = device.get('is_online', device['status'] == 'active')
                status_emoji = '🟢' if is_online else '🔴'
                status_text = '在线' if is_online else '离线'
                st.write(f"**实时状态:** {status_emoji} {status_text}")
                st.write(f"**创建时间:** {device['create_time']}")
                st.write(f"**更新时间:** {device['update_time']}")
            
            # 添加状态说明和重连选项
            if is_online:
                st.success("✅ 设备在最近30分钟内有数据活动，状态正常")
            else:
                st.warning("⚠️ 设备在最近30分钟内无数据活动，可能已离线")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔗 重新连接设备", key="reconnect_from_detail"):
                        # 设置重连参数到session state
                        st.session_state.connection_mode = "reconnect"
                        st.session_state.connection_device_code = device['device_code']
                        # 跳转到设备连接页面
                        st.switch_page("pages/device_connection.py")
            
            if st.button("关闭", key="close_view_device"):
                del st.session_state.view_device
                st.rerun()
    
    # 编辑设备
    if 'edit_device' in st.session_state:
        device = st.session_state.edit_device
        with st.expander(f"✏️ 编辑设备 - {device['scene'] or '未命名设备'}", expanded=True):
            with st.form(f"edit_device_form_{device['device_code']}"):
                col1, col2 = st.columns(2)
                with col1:
                    new_scene = st.text_input("使用场景", value=device['scene'] or '')
                    new_wifi_name = st.text_input("WiFi名称", value=device['wifi_name'] or '')
                with col2:
                    new_wifi_password = st.text_input("WiFi密码", value=device['wifi_password'] or '', type="password")
                    new_status = st.selectbox("设备状态", ['active', 'inactive'], 
                                           index=0 if device['status'] == 'active' else 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("💾 保存", use_container_width=True)
                with col2:
                    if st.form_submit_button("❌ 取消", use_container_width=True):
                        del st.session_state.edit_device
                        st.rerun()
                
                if submitted:
                    update_data = {
                        'scene': new_scene,
                        'wifi_name': new_wifi_name,
                        'wifi_password': new_wifi_password,
                        'status': new_status
                    }
                    
                    if update_device_info(device['device_code'], update_data):
                        st.success("设备信息更新成功！")
                        del st.session_state.edit_device
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("更新失败，请重试！")
    
    # 确认删除
    if 'confirm_delete' in st.session_state:
        device_code = st.session_state.confirm_delete
        st.warning(f"⚠️ 确定要删除设备 {device_code} 吗？此操作不可恢复！")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("确认删除", key="confirm_delete_btn", use_container_width=True):
                if delete_device(device_code):
                    st.success("设备删除成功！")
                    del st.session_state.confirm_delete
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("删除失败，请重试！")
        with col2:
            if st.button("取消", key="cancel_delete_btn", use_container_width=True):
                del st.session_state.confirm_delete
                st.rerun()







def show_ai_chat_button_bake():
    """显示AI聊天按钮 - 使用Streamlit原生组件"""
    import streamlit.components.v1 as components
    
    # 获取当前用户名
    username = st.session_state.get('username', 'testuser')
    
    # 方案1：使用完整的HTML文档结构
    ai_button_html = f"""
    <!DOCTYPE html>
    <html>
    
    <head>
        <meta charset="utf-8">
        <style>
            .ai-chat-toggle {
                position: fixed !important;
                top: 20px !important;
                right: 20px !important;
                width: 70px !important;
                height: 70px !important;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                border-radius: 50% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
                cursor: pointer !important;
                z-index: 2147483647 !important;
                transition: all 0.3s ease !important;
                color: white !important;
                font-size: 32px !important;
                border: 4px solid rgba(255, 255, 255, 0.2) !important;
                user-select: none !important;
                font-family: Arial, sans-serif !important;
            }
            
            .ai-chat-toggle:hover {
                transform: translateY(-8px) scale(1.15) !important;
                box-shadow: 0 20px 40px rgba(102, 126, 234, 0.6) !important;
            }
            
            .ai-chat-toggle.active {
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
            }
        </style>
    </head>
    
    <body>
        <!-- AI聊天按钮 -->
        <div class="ai-chat-toggle" id="aiChatToggle" onclick="toggleAIChat()">
            🤖
        </div>
        
        <!-- 聊天窗口 -->
        <div class="chat-container" id="chatContainer">
            <div class="chat-header">
                <span>🤖 社区智能体助手</span>
                <button class="chat-close-btn" onclick="closeAIChat()">×</button>
            </div>
            <iframe 
                class="chat-iframe" 
                src="https://1.71.15.121:5001/login"
                frameborder="0"
                allow="microphone">
            </iframe>
        </div>
        
        <script>
            let chatWindow = null;
            function toggleAIChat() {{
                const toggle = document.getElementById('aiChatToggle');
                const chatContainer = document.getElementById('chatContainer');
                const toggle = document.getElementById('aiChatToggle');
                
                if (chatWindow && !chatWindow.closed) {{
                    // 如果窗口已存在且未关闭，则关闭它
                    chatWindow.close();
                    toggle.classList.remove('active');
                    toggle.innerHTML = '🤖';
                }} else {{
                    // 打开新的聊天窗口
                    openAIChat();
                }}
            }}
            
            function openAIChat() {{
                const toggle = document.getElementById('aiChatToggle');
    
                // 窗口配置
                const windowFeatures = [
                    'width=800',
                    'height=600',
                    'left=' + (screen.width - 850),  // 靠右显示
                    'top=50',                        // 靠上显示
                    'resizable=yes',
                    'scrollbars=yes',
                    'status=no',
                    'menubar=no',
                    'toolbar=no',
                    'location=no',
                    'directories=no'
                ].join(',');
                
                // 打开新窗口
                chatWindow = window.open(
                    'https://1.71.15.121:5001/login',  // 你的聊天页面URL
                    'aiChatWindow',
                    windowFeatures
                );
                
                // 监听窗口关闭事件
                const checkClosed = setInterval(() => {{
                    if (chatWindow.closed) {{
                        toggle.classList.remove('active');
                        toggle.innerHTML = '🤖';
                        chatWindow = null;
                        clearInterval(checkClosed);
                    }}
                }}, 1000);
                
                // 更新按钮状态
                toggle.classList.add('active');
                toggle.innerHTML = '✕';
                
                // 聚焦到新窗口
                if (chatWindow) {{
                    chatWindow.focus();
                }}
            }}
            
            function closeAIChat() {{
                if (chatWindow && !chatWindow.closed) {{
                    chatWindow.close();
                }}
            }}
            
            // ESC键关闭聊天
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'Escape') {{
                    closeAIChat();
                }}
            }});
            
            // 确保元素在页面加载后可见
            window.addEventListener('load', function() {{
                const toggle = document.getElementById('aiChatToggle');
                if (toggle) {{
                    toggle.style.display = 'flex';
                }}
            }});
        </script>
    </body>
    </html>
    """

    # 使用更大的高度值确保按钮可见
    components.html(ai_button_html, height=600, scrolling=False)




def show_ai_chat_button():
    """显示AI聊天按钮 - 使用Streamlit原生组件，支持可调整大小的对话框"""
    import streamlit.components.v1 as components
    
    # 获取当前用户名
    username = st.session_state.get('username', 'testuser')
    
    # 增强版AI按钮HTML
    ai_button_html = f"""
    <!DOCTYPE html>
    <html>
    
    
    <head>
        <meta charset="utf-8">
        <style>
            .ai-chat-toggle {{
                position: fixed !important;
                top: 10px !important;
                right: 10px !important;
                width: 50px !important;
                height: 50px !important;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                border-radius: 50% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
                cursor: pointer !important;
                z-index: 2147483647 !important;
                transition: all 0.3s ease !important;
                color: white !important;
                font-size: 16px !important;
                border: 4px solid rgba(255, 255, 255, 0.2) !important;
                user-select: none !important;
                font-family: Arial, sans-serif !important;
            }}
            
            .ai-chat-toggle:hover {{
                transform: translateY(-8px) scale(1.15) !important;
                box-shadow: 0 20px 40px rgba(102, 126, 234, 0.6) !important;
            }}
            
            .ai-chat-toggle.active {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
            }}
        </style>
    </head>
    
    
    <body>
        <!-- AI聊天按钮 -->
        <div class="ai-chat-toggle" id="aiChatToggle" onclick="toggleAIChat()">
            🤖
        </div>
        
        <script>
            let isFullscreen = false;
            let originalSize = {{ width: 800, height: 800 }};
            let isDragging = false;
            let isResizing = false;
            let dragOffset = {{ x: 0, y: 0 }};
            let chatWindow = null;
            function toggleAIChat() {{
                const toggle = document.getElementById('aiChatToggle');
    
                if (chatWindow && !chatWindow.closed) {{
                    // 如果窗口已存在且未关闭，则关闭它
                    chatWindow.close();
                    toggle.classList.remove('active');
                    toggle.innerHTML = '🤖';
                }} else {{
                    // 打开新的聊天窗口
                    openAIChat();
                }}
            }}
            
            function openAIChat() {{
                const toggle = document.getElementById('aiChatToggle');
    
                // 窗口配置
                const windowWidth = 1000;
                const windowHeight = 800;
                const windowFeatures = [
                    'width=' + windowWidth,
                    'height=' + windowHeight,
                    'left=' + ((screen.width - windowWidth) / 2),   // 水平居中
                    'top=' + ((screen.height - windowHeight) / 2),  // 垂直居中
                    'resizable=yes',
                    'scrollbars=yes',
                    'status=no',
                    'menubar=no',
                    'toolbar=no',
                    'location=no',
                    'directories=no'
                ].join(',');
                
                // 打开新窗口
                chatWindow = window.open(
                    'https://1.71.15.121:5001/login',  // 你的聊天页面URL
                    'aiChatWindow',
                    windowFeatures
                );
                
                // 监听窗口关闭事件
                const checkClosed = setInterval(() => {{
                    if (chatWindow.closed) {{
                        toggle.classList.remove('active');
                        toggle.innerHTML = '🤖';
                        chatWindow = null;
                        clearInterval(checkClosed);
                    }}
                }}, 1000);
                
                // 更新按钮状态
                toggle.classList.add('active');
                toggle.innerHTML = '✕';
                
                // 聚焦到新窗口
                if (chatWindow) {{
                    chatWindow.focus();
                }}
            }}
            
            function closeAIChat() {{
                if (chatWindow && !chatWindow.closed) {{
                    chatWindow.close();
                }}
            }}
            
            function toggleFullscreen() {{
                const chatContainer = document.getElementById('chatContainer');
                const fullscreenIcon = document.getElementById('fullscreenIcon');
                
                if (!isFullscreen) {{
                    // 进入全屏模式
                    originalSize.width = chatContainer.offsetWidth;
                    originalSize.height = chatContainer.offsetHeight;
                    
                    chatContainer.style.width = '95vw';
                    chatContainer.style.height = '90vh';
                    chatContainer.style.top = '5vh';
                    chatContainer.style.left = '2.5vw';
                    chatContainer.style.right = 'auto';
                    chatContainer.style.bottom = 'auto';
                    
                    fullscreenIcon.innerHTML = '⛶';
                    isFullscreen = true;
                }} else {{
                    // 退出全屏模式
                    chatContainer.style.width = originalSize.width + 'px';
                    chatContainer.style.height = originalSize.height + 'px';
                    chatContainer.style.top = 'auto';
                    chatContainer.style.left = 'auto';
                    chatContainer.style.right = '20px';
                    chatContainer.style.bottom = '100px';
                    
                    fullscreenIcon.innerHTML = '⛶';
                    isFullscreen = false;
                }}
                
                updateSizeIndicator();
            }}
            
            function resetSize() {{
                const chatContainer = document.getElementById('chatContainer');
                
                if (isFullscreen) {{
                    toggleFullscreen();
                }} else {{
                    chatContainer.style.width = '800px';
                    chatContainer.style.height = '800px';
                    originalSize = {{ width: 800, height: 800 }};
                }}
                
                updateSizeIndicator();
            }}
            
            function updateSizeIndicator() {{
                const chatContainer = document.getElementById('chatContainer');
                const sizeIndicator = document.getElementById('sizeIndicator');
                
                if (chatContainer && sizeIndicator) {{
                    const width = chatContainer.offsetWidth;
                    const height = chatContainer.offsetHeight;
                    sizeIndicator.textContent = `${{width}}×${{height}}`;
                }}
            }}
            
            // 拖拽和调整大小功能
            function initDragAndResize() {{
                const chatContainer = document.getElementById('chatContainer');
                const chatHeader = document.getElementById('chatHeader');
                const resizeHandle = document.getElementById('resizeHandle');
                
                // 拖拽头部移动窗口
                chatHeader.addEventListener('mousedown', (e) => {{
                    if (isFullscreen || e.target.closest('.chat-control-btn')) return;
                    
                    isDragging = true;
                    const rect = chatContainer.getBoundingClientRect();
                    dragOffset.x = e.clientX - rect.left;
                    dragOffset.y = e.clientY - rect.top;
                    
                    chatContainer.style.transition = 'none';
                    document.addEventListener('mousemove', handleDrag);
                    document.addEventListener('mouseup', stopDrag);
                    e.preventDefault();
                }});
                
                // 拖拽右下角调整大小
                resizeHandle.addEventListener('mousedown', (e) => {{
                    if (isFullscreen) return;
                    
                    isResizing = true;
                    chatContainer.classList.add('resizing');
                    const rect = chatContainer.getBoundingClientRect();
                    const startX = e.clientX;
                    const startY = e.clientY;
                    const startWidth = rect.width;
                    const startHeight = rect.height;
                    
                    chatContainer.style.transition = 'none';
                    document.addEventListener('mousemove', handleResize);
                    document.addEventListener('mouseup', stopResize);
                    e.preventDefault();
                    e.stopPropagation();
                    
                    function handleResize(e) {{
                        if (!isResizing) return;
                        
                        const newWidth = startWidth + (e.clientX - startX);
                        const newHeight = startHeight + (e.clientY - startY);
                        
                        // 限制最小和最大尺寸
                        const minWidth = 400;
                        const minHeight = 300;
                        const maxWidth = window.innerWidth * 0.9;
                        const maxHeight = window.innerHeight * 0.8;
                        
                        const constrainedWidth = Math.max(minWidth, Math.min(newWidth, maxWidth));
                        const constrainedHeight = Math.max(minHeight, Math.min(newHeight, maxHeight));
                        
                        chatContainer.style.width = constrainedWidth + 'px';
                        chatContainer.style.height = constrainedHeight + 'px';
                        
                        // 实时更新尺寸指示器
                        updateSizeIndicator();
                        
                        // 确保窗口不超出视口
                        const containerRect = chatContainer.getBoundingClientRect();
                        if (containerRect.right > window.innerWidth) {{
                            const overflow = containerRect.right - window.innerWidth;
                            const currentLeft = parseInt(chatContainer.style.left) || (window.innerWidth - containerRect.width - 20);
                            chatContainer.style.left = Math.max(0, currentLeft - overflow) + 'px';
                            chatContainer.style.right = 'auto';
                        }}
                        
                        if (containerRect.bottom > window.innerHeight) {{
                            const overflow = containerRect.bottom - window.innerHeight;
                            const currentTop = parseInt(chatContainer.style.top) || (window.innerHeight - containerRect.height - 100);
                            chatContainer.style.top = Math.max(0, currentTop - overflow) + 'px';
                            chatContainer.style.bottom = 'auto';
                        }}
                    }}
                    
                    function stopResize() {{
                        isResizing = false;
                        chatContainer.classList.remove('resizing');
                        chatContainer.style.transition = '';
                        document.removeEventListener('mousemove', handleResize);
                        document.removeEventListener('mouseup', stopResize);
                        
                        // 更新原始尺寸记录
                        originalSize.width = chatContainer.offsetWidth;
                        originalSize.height = chatContainer.offsetHeight;
                    }}
                }});
                
                function handleDrag(e) {{
                    if (!isDragging || isResizing) return;
                    
                    const newX = e.clientX - dragOffset.x;
                    const newY = e.clientY - dragOffset.y;
                    
                    // 限制在视口内
                    const maxX = window.innerWidth - chatContainer.offsetWidth;
                    const maxY = window.innerHeight - chatContainer.offsetHeight;
                    
                    const constrainedX = Math.max(0, Math.min(newX, maxX));
                    const constrainedY = Math.max(0, Math.min(newY, maxY));
                    
                    chatContainer.style.left = constrainedX + 'px';
                    chatContainer.style.top = constrainedY + 'px';
                    chatContainer.style.right = 'auto';
                    chatContainer.style.bottom = 'auto';
                }}
                
                function stopDrag() {{
                    isDragging = false;
                    chatContainer.style.transition = '';
                    document.removeEventListener('mousemove', handleDrag);
                    document.removeEventListener('mouseup', stopDrag);
                }}
                
                // 添加鼠标样式提示
                resizeHandle.addEventListener('mouseenter', () => {{
                    resizeHandle.style.cursor = 'se-resize';
                }});
                
                // 监听窗口大小变化以更新尺寸指示器
                const resizeObserver = new ResizeObserver(() => {{
                    if (!isResizing) {{
                        updateSizeIndicator();
                    }}
                }});
                
                resizeObserver.observe(chatContainer);
            }}
            
            // ESC键关闭聊天
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'Escape') {{
                    closeAIChat();
                }}
            }});
            
            // 页面加载完成后初始化
            window.addEventListener('load', function() {{
                const toggle = document.getElementById('aiChatToggle');
                const resizeHandle = document.getElementById('resizeHandle');
                
                if (toggle) {{
                    toggle.style.display = 'flex';
                }}
                
                // 确保调整大小手柄可见
                if (resizeHandle) {{
                    resizeHandle.style.display = 'flex';
                    console.log('Resize handle initialized');
                }}
                
                initDragAndResize();
                
                // 添加一些调试信息
                console.log('AI Chat window initialized with resize functionality');
            }});
        </script>
    </body>
    </html>
    """

    # 使用全屏高度确保元素可以覆盖整个页面
    components.html(ai_button_html, height=70, scrolling=False)




def show_ai_chat_button_iframe():
    """显示AI聊天按钮 - 使用Streamlit原生组件，支持可调整大小的对话框"""
    import streamlit.components.v1 as components
    
    # 获取当前用户名
    username = st.session_state.get('username', 'testuser')
    
    # 增强版AI按钮HTML
    ai_button_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
            }}
            
            .ai-chat-toggle {{
                position: fixed !important;
                top: 20px !important;
                right: 20px !important;
                width: 70px !important;
                height: 70px !important;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                border-radius: 50% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
                cursor: pointer !important;
                z-index: 999999 !important;
                transition: all 0.3s ease !important;
                color: white !important;
                font-size: 32px !important;
                border: 4px solid rgba(255, 255, 255, 0.2) !important;
                user-select: none !important;
                font-family: Arial, sans-serif !important;
            }}

            .ai-chat-toggle:hover {{
                transform: translateY(-8px) scale(1.15) !important;
                box-shadow: 0 20px 40px rgba(102, 126, 234, 0.6) !important;
                background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            }}

            .ai-chat-toggle.active {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
                transform: scale(1.1) !important;
            }}

            .chat-container {{
                position: fixed !important;
                bottom: 100px !important;
                right: 20px !important;
                width: 1000px !important;
                height: 800px !important;
                min-width: 400px !important;
                min-height: 300px !important;
                max-width: 90vw !important;
                max-height: 80vh !important;
                background: white !important;
                border-radius: 20px !important;
                box-shadow: 0 15px 50px rgba(0,0,0,0.3) !important;
                z-index: 999998 !important;
                display: none !important;
                overflow: hidden !important;
                border: 3px solid #667eea !important;
                animation: slideUp 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
            }}

            .chat-container.show {{
                display: block !important;
            }}

            .chat-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                padding: 20px 25px !important;
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
                font-weight: 600 !important;
                font-size: 18px !important;
                font-family: Arial, sans-serif !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
                cursor: move !important;
                user-select: none !important;
            }}

            .chat-title {{
                display: flex !important;
                align-items: center !important;
                gap: 10px !important;
            }}

            .chat-controls {{
                display: flex !important;
                gap: 10px !important;
                align-items: center !important;
            }}

            .chat-control-btn {{
                background: rgba(255, 255, 255, 0.2) !important;
                border: none !important;
                color: white !important;
                font-size: 18px !important;
                cursor: pointer !important;
                padding: 8px !important;
                width: 35px !important;
                height: 35px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                border-radius: 8px !important;
                transition: all 0.2s ease !important;
                font-weight: bold !important;
            }}

            .chat-control-btn:hover {{
                background: rgba(255, 255, 255, 0.3) !important;
                transform: scale(1.05) !important;
            }}

            .resize-handle {{
                position: absolute !important;
                bottom: 0 !important;
                right: 0 !important;
                width: 25px !important;
                height: 25px !important;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                cursor: se-resize !important;
                border-radius: 25px 0 20px 0 !important;
                z-index: 1000 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                transition: all 0.2s ease !important;
            }}

            .resize-handle:hover {{
                width: 30px !important;
                height: 30px !important;
                background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            }}

            .resize-handle::before {{
                content: '' !important;
                position: absolute !important;
                bottom: 6px !important;
                right: 6px !important;
                width: 8px !important;
                height: 8px !important;
                background: repeating-linear-gradient(
                    -45deg,
                    rgba(255, 255, 255, 0.8) 0px,
                    rgba(255, 255, 255, 0.8) 1px,
                    transparent 1px,
                    transparent 3px
                ) !important;
                border-radius: 1px !important;
            }}

            .resize-handle::after {{
                content: '' !important;
                position: absolute !important;
                bottom: 3px !important;
                right: 3px !important;
                width: 12px !important;
                height: 12px !important;
                background: repeating-linear-gradient(
                    -45deg,
                    rgba(255, 255, 255, 0.6) 0px,
                    rgba(255, 255, 255, 0.6) 1px,
                    transparent 1px,
                    transparent 2px
                ) !important;
                border-radius: 2px !important;
            }}

            .chat-iframe {{
                width: 100% !important;
                height: calc(100% - 70px) !important;
                border: none !important;
                border-radius: 0 0 20px 20px !important;
            }}

            .size-indicator {{
                position: absolute !important;
                top: 10px !important;
                right: 160px !important;
                background: rgba(255, 255, 255, 0.2) !important;
                color: white !important;
                padding: 6px 12px !important;
                border-radius: 15px !important;
                font-size: 12px !important;
                font-family: monospace !important;
                opacity: 0 !important;
                transition: opacity 0.3s ease !important;
                backdrop-filter: blur(10px) !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
            }}

            .chat-container:hover .size-indicator,
            .chat-container.resizing .size-indicator {{
                opacity: 1 !important;
            }}

            .chat-container.resizing {{
                box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4) !important;
                border-color: #764ba2 !important;
            }}

            .chat-container.resizing .size-indicator {{
                background: rgba(102, 126, 234, 0.9) !important;
                transform: scale(1.05) !important;
            }}

            @keyframes slideUp {{
                from {{
                    transform: translateY(100%) scale(0.8);
                    opacity: 0;
                }}
                to {{
                    transform: translateY(0) scale(1);
                    opacity: 1;
                }}
            }}

            @keyframes slideDown {{
                from {{
                    transform: translateY(0) scale(1);
                    opacity: 1;
                }}
                to {{
                    transform: translateY(100%) scale(0.8);
                    opacity: 0;
                }}
            }}

            .chat-container.hiding {{
                animation: slideDown 0.3s ease-in forwards !important;
            }}

            /* 移动设备适配 */
            @media (max-width: 768px) {{
                .chat-container {{
                    width: calc(100vw - 20px) !important;
                    height: calc(100vh - 120px) !important;
                    bottom: 100px !important;
                    right: 10px !important;
                    left: 10px !important;
                    min-width: unset !important;
                    min-height: unset !important;
                }}
                
                .ai-chat-toggle {{
                    bottom: 20px !important;
                    right: 20px !important;
                    width: 60px !important;
                    height: 60px !important;
                    font-size: 28px !important;
                }}

                .chat-header {{
                    padding: 15px 20px !important;
                    font-size: 16px !important;
                }}

                .resize-handle {{
                    display: none !important;
                }}
                
                .chat-control-btn {{
                    font-size: 16px !important;
                    width: 32px !important;
                    height: 32px !important;
                }}
            }}

            /* 桌面端拖拽优化 */
            @media (min-width: 769px) {{
                .chat-container {{
                    transition: none !important;
                }}
            }}

            /* 自定义滚动条样式 */
            .chat-container ::-webkit-scrollbar {{
                width: 6px !important;
            }}

            .chat-container ::-webkit-scrollbar-track {{
                background: #f1f1f1 !important;
                border-radius: 3px !important;
            }}

            .chat-container ::-webkit-scrollbar-thumb {{
                background: #667eea !important;
                border-radius: 3px !important;
            }}

            .chat-container ::-webkit-scrollbar-thumb:hover {{
                background: #764ba2 !important;
            }}
        </style>
    </head>
    <body>
        <!-- AI聊天按钮 -->
        <div class="ai-chat-toggle" id="aiChatToggle" onclick="toggleAIChat()">
            🤖
        </div>
        
        <!-- 聊天窗口 -->
        <div class="chat-container" id="chatContainer">
            <div class="chat-header" id="chatHeader">
                <div class="chat-title">
                    <span>🤖 社区智能体助手</span>
                </div>
                <div class="size-indicator" id="sizeIndicator">
                    600×700
                </div>
                <div class="chat-controls">
                    <button class="chat-control-btn" onclick="toggleFullscreen()" title="全屏/窗口模式">
                        <span id="fullscreenIcon">⛶</span>
                    </button>
                    <button class="chat-control-btn" onclick="resetSize()" title="重置大小">
                        ↺
                    </button>
                    <button class="chat-control-btn" onclick="closeAIChat()" title="关闭">
                        ×
                    </button>
                </div>
            </div>
            <iframe 
                class="chat-iframe" 
                src="https://1.71.15.121:5001/login"
                frameborder="0"
                allow="microphone">
            </iframe>
            <div class="resize-handle" id="resizeHandle"></div>
        </div>
        
        <script>
            let isFullscreen = false;
            let originalSize = {{ width: 600, height: 700 }};
            let isDragging = false;
            let isResizing = false;
            let dragOffset = {{ x: 0, y: 0 }};
            
            function toggleAIChat() {{
                const chatContainer = document.getElementById('chatContainer');
                const toggle = document.getElementById('aiChatToggle');
                
                if (chatContainer.classList.contains('show')) {{
                    closeAIChat();
                }} else {{
                    openAIChat();
                }}
            }}
            
            function openAIChat() {{
                const chatContainer = document.getElementById('chatContainer');
                const toggle = document.getElementById('aiChatToggle');
                
                chatContainer.classList.remove('hiding');
                chatContainer.classList.add('show');
                toggle.classList.add('active');
                toggle.innerHTML = '✕';
                
                updateSizeIndicator();
            }}
            
            function closeAIChat() {{
                const chatContainer = document.getElementById('chatContainer');
                const toggle = document.getElementById('aiChatToggle');
                
                chatContainer.classList.add('hiding');
                toggle.classList.remove('active');
                toggle.innerHTML = '🤖';
                
                setTimeout(() => {{
                    chatContainer.classList.remove('show', 'hiding');
                }}, 300);
            }}
            
            function toggleFullscreen() {{
                const chatContainer = document.getElementById('chatContainer');
                const fullscreenIcon = document.getElementById('fullscreenIcon');
                
                if (!isFullscreen) {{
                    // 进入全屏模式
                    originalSize.width = chatContainer.offsetWidth;
                    originalSize.height = chatContainer.offsetHeight;
                    
                    chatContainer.style.width = '95vw';
                    chatContainer.style.height = '90vh';
                    chatContainer.style.top = '5vh';
                    chatContainer.style.left = '2.5vw';
                    chatContainer.style.right = 'auto';
                    chatContainer.style.bottom = 'auto';
                    
                    fullscreenIcon.innerHTML = '⛶';
                    isFullscreen = true;
                }} else {{
                    // 退出全屏模式
                    chatContainer.style.width = originalSize.width + 'px';
                    chatContainer.style.height = originalSize.height + 'px';
                    chatContainer.style.top = 'auto';
                    chatContainer.style.left = 'auto';
                    chatContainer.style.right = '20px';
                    chatContainer.style.bottom = '100px';
                    
                    fullscreenIcon.innerHTML = '⛶';
                    isFullscreen = false;
                }}
                
                updateSizeIndicator();
            }}
            
            function resetSize() {{
                const chatContainer = document.getElementById('chatContainer');
                
                if (isFullscreen) {{
                    toggleFullscreen();
                }} else {{
                    chatContainer.style.width = '600px';
                    chatContainer.style.height = '700px';
                    originalSize = {{ width: 600, height: 700 }};
                }}
                
                updateSizeIndicator();
            }}
            
            function updateSizeIndicator() {{
                const chatContainer = document.getElementById('chatContainer');
                const sizeIndicator = document.getElementById('sizeIndicator');
                
                if (chatContainer && sizeIndicator) {{
                    const width = chatContainer.offsetWidth;
                    const height = chatContainer.offsetHeight;
                    sizeIndicator.textContent = `${{width}}×${{height}}`;
                }}
            }}
            
            // 拖拽和调整大小功能
            function initDragAndResize() {{
                const chatContainer = document.getElementById('chatContainer');
                const chatHeader = document.getElementById('chatHeader');
                const resizeHandle = document.getElementById('resizeHandle');
                
                // 拖拽头部移动窗口
                chatHeader.addEventListener('mousedown', (e) => {{
                    if (isFullscreen || e.target.closest('.chat-control-btn')) return;
                    
                    isDragging = true;
                    const rect = chatContainer.getBoundingClientRect();
                    dragOffset.x = e.clientX - rect.left;
                    dragOffset.y = e.clientY - rect.top;
                    
                    chatContainer.style.transition = 'none';
                    document.addEventListener('mousemove', handleDrag);
                    document.addEventListener('mouseup', stopDrag);
                    e.preventDefault();
                }});
                
                // 拖拽右下角调整大小
                resizeHandle.addEventListener('mousedown', (e) => {{
                    if (isFullscreen) return;
                    
                    isResizing = true;
                    chatContainer.classList.add('resizing');
                    const rect = chatContainer.getBoundingClientRect();
                    const startX = e.clientX;
                    const startY = e.clientY;
                    const startWidth = rect.width;
                    const startHeight = rect.height;
                    
                    chatContainer.style.transition = 'none';
                    document.addEventListener('mousemove', handleResize);
                    document.addEventListener('mouseup', stopResize);
                    e.preventDefault();
                    e.stopPropagation();
                    
                    function handleResize(e) {{
                        if (!isResizing) return;
                        
                        const newWidth = startWidth + (e.clientX - startX);
                        const newHeight = startHeight + (e.clientY - startY);
                        
                        // 限制最小和最大尺寸
                        const minWidth = 400;
                        const minHeight = 300;
                        const maxWidth = window.innerWidth * 0.9;
                        const maxHeight = window.innerHeight * 0.8;
                        
                        const constrainedWidth = Math.max(minWidth, Math.min(newWidth, maxWidth));
                        const constrainedHeight = Math.max(minHeight, Math.min(newHeight, maxHeight));
                        
                        chatContainer.style.width = constrainedWidth + 'px';
                        chatContainer.style.height = constrainedHeight + 'px';
                        
                        // 实时更新尺寸指示器
                        updateSizeIndicator();
                        
                        // 确保窗口不超出视口
                        const containerRect = chatContainer.getBoundingClientRect();
                        if (containerRect.right > window.innerWidth) {{
                            const overflow = containerRect.right - window.innerWidth;
                            const currentLeft = parseInt(chatContainer.style.left) || (window.innerWidth - containerRect.width - 20);
                            chatContainer.style.left = Math.max(0, currentLeft - overflow) + 'px';
                            chatContainer.style.right = 'auto';
                        }}
                        
                        if (containerRect.bottom > window.innerHeight) {{
                            const overflow = containerRect.bottom - window.innerHeight;
                            const currentTop = parseInt(chatContainer.style.top) || (window.innerHeight - containerRect.height - 100);
                            chatContainer.style.top = Math.max(0, currentTop - overflow) + 'px';
                            chatContainer.style.bottom = 'auto';
                        }}
                    }}
                    
                    function stopResize() {{
                        isResizing = false;
                        chatContainer.classList.remove('resizing');
                        chatContainer.style.transition = '';
                        document.removeEventListener('mousemove', handleResize);
                        document.removeEventListener('mouseup', stopResize);
                        
                        // 更新原始尺寸记录
                        originalSize.width = chatContainer.offsetWidth;
                        originalSize.height = chatContainer.offsetHeight;
                    }}
                }});
                
                function handleDrag(e) {{
                    if (!isDragging || isResizing) return;
                    
                    const newX = e.clientX - dragOffset.x;
                    const newY = e.clientY - dragOffset.y;
                    
                    // 限制在视口内
                    const maxX = window.innerWidth - chatContainer.offsetWidth;
                    const maxY = window.innerHeight - chatContainer.offsetHeight;
                    
                    const constrainedX = Math.max(0, Math.min(newX, maxX));
                    const constrainedY = Math.max(0, Math.min(newY, maxY));
                    
                    chatContainer.style.left = constrainedX + 'px';
                    chatContainer.style.top = constrainedY + 'px';
                    chatContainer.style.right = 'auto';
                    chatContainer.style.bottom = 'auto';
                }}
                
                function stopDrag() {{
                    isDragging = false;
                    chatContainer.style.transition = '';
                    document.removeEventListener('mousemove', handleDrag);
                    document.removeEventListener('mouseup', stopDrag);
                }}
                
                // 添加鼠标样式提示
                resizeHandle.addEventListener('mouseenter', () => {{
                    resizeHandle.style.cursor = 'se-resize';
                }});
                
                // 监听窗口大小变化以更新尺寸指示器
                const resizeObserver = new ResizeObserver(() => {{
                    if (!isResizing) {{
                        updateSizeIndicator();
                    }}
                }});
                
                resizeObserver.observe(chatContainer);
            }}
            
            // ESC键关闭聊天
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'Escape') {{
                    closeAIChat();
                }}
            }});
            
            // 页面加载完成后初始化
            window.addEventListener('load', function() {{
                const toggle = document.getElementById('aiChatToggle');
                const resizeHandle = document.getElementById('resizeHandle');
                
                if (toggle) {{
                    toggle.style.display = 'flex';
                }}
                
                // 确保调整大小手柄可见
                if (resizeHandle) {{
                    resizeHandle.style.display = 'flex';
                    console.log('Resize handle initialized');
                }}
                
                initDragAndResize();
                
                // 添加一些调试信息
                console.log('AI Chat window initialized with resize functionality');
            }});
        </script>
    </body>
    </html>
    """

    # 使用更大的高度值确保所有元素可见
    components.html(ai_button_html, height=800, scrolling=False)






def main():
    """主函数"""
    
    # 检查登录状态
    if not check_login():
        return
    
    # 获取URL参数或session中的用户名
    query_params = st.query_params
    username = query_params.get("user_name", st.session_state.get('username', 'testuser'))
    
    # 更新session state
    st.session_state.username = username
    # 显示右下角AI聊天按钮
    show_ai_chat_button()
    
    # 显示欢迎头部
    show_welcome_header(username)
    
    # 显示设备管理
    show_device_management(username)
    
    # 显示设备相关的模态框
    show_device_modals(username)
    show_ai_chat_button_iframe()
    


if __name__ == "__main__":
    main()