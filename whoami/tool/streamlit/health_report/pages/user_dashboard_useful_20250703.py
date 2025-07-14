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
    page_title="睡眠健康管理系统 - 用户首页",
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

.device-name {
    font-weight: 700;
    color: #e74c3c;
    font-size: 1.3rem;
    margin-bottom: 12px;
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
    border: 2px dashed #e74c3c;
    background: rgba(231, 76, 60, 0.05);
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
    background: rgba(231, 76, 60, 0.1);
    transform: translateY(-8px);
    box-shadow: 0 12px 35px rgba(231, 76, 60, 0.15);
}

.add-icon {
    font-size: 3rem;
    color: #e74c3c;
    margin-bottom: 10px;
}

.add-text {
    color: #e74c3c;
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

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* 模态框样式 */
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
}

.modal-content {
    background: white;
    padding: 30px;
    border-radius: 20px;
    max-width: 500px;
    width: 90%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

/* 表单样式 */
.form-group {
    margin-bottom: 20px;
}

.form-label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: #333;
}

.form-input {
    width: 100%;
    padding: 12px;
    border: 2px solid #e1e5e9;
    border-radius: 10px;
    font-size: 1rem;
    transition: border-color 0.2s ease;
}

.form-input:focus {
    border-color: #667eea;
    outline: none;
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
            
            devices.append(result)
        return devices
    except Exception as e:
        st.error(f"获取设备列表失败: {str(e)}")
        return []

def add_device(device_data):
    """添加新设备"""
    try:
        sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
        
        # 准备设备数据，包含所有必要字段
        upsert_data = {
            'device_code': device_data['device_code'],  # 唯一字段
            'scene': device_data['scene'],
            'wifi_name': device_data['wifi_name'],
            'wifi_password': device_data['wifi_password'],
            'username': device_data['username'],
            'status': 'active',
            'creator': device_data['username'],
            'create_time': datetime.now(),
            'update_time': datetime.now()
        }
        
        # 使用upsert_record_by_unique_field，如果设备已存在会更新，不存在会插入
        result = sql_provider.add_record(
            data=upsert_data,
        )
        return result is not None
        
    except Exception as e:
        st.error(f"添加设备失败: {str(e)}")
        return False

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
        <div class="welcome-content">
            <h1 class="welcome-title">欢迎使用睡眠健康管理系统</h1>
            <p class="welcome-subtitle">用户：{display_name} | 今日：{current_date}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_device_card(device):
    """显示设备卡片"""
    # 使用实时在线状态
    is_online = device.get('is_online', device['status'] == 'active')
    status_class = 'status-online' if is_online else 'status-offline'
    status_text = '在线' if is_online else '离线'
    
    # 截断设备编号显示
    device_code_display = device['device_code'][:12] + '...' if len(device['device_code']) > 12 else device['device_code']
    
    # 添加实时状态指示器
    status_indicator = '🟢' if is_online else '🔴'
    
    card_html = f"""
    <div class="device-card">
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

def show_device_management(username):
    """显示设备管理"""
    # 获取用户设备列表
    devices = get_user_devices(username)
    
    # 设备管理标题
    st.markdown("""
    <div class="section-title">
        <h2><span class="icon">📱</span>设备管理</h2>
    </div>
    """, unsafe_allow_html=True)
    
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
                        
                        # 设备操作按钮
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
                        
                    elif device_idx == len(devices):
                        st.markdown(show_add_device_card(), unsafe_allow_html=True)
                        if st.button("添加设备", key="add_device_btn", use_container_width=True):
                            st.session_state.show_add_device = True
                    else:
                        st.empty()
    else:
        # 如果没有设备，只显示添加按钮
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(show_add_device_card(), unsafe_allow_html=True)
            if st.button("添加设备", key="add_device_btn", use_container_width=True):
                st.session_state.show_add_device = True
    
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
            
            # 添加状态说明
            if is_online:
                st.success("✅ 设备在最近30分钟内有数据活动，状态正常")
            else:
                st.warning("⚠️ 设备在最近30分钟内无数据活动，可能已离线")
            
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
    
    # 添加新设备
    if st.session_state.get('show_add_device', False):
        with st.expander("➕ 添加新设备", expanded=True):
            with st.form("add_device_form"):
                col1, col2 = st.columns(2)
                with col1:
                    device_code = st.text_input("设备编号*", placeholder="请输入设备编号")
                    scene = st.text_input("使用场景", placeholder="如：客厅、卧室等")
                with col2:
                    wifi_name = st.text_input("WiFi名称")
                    wifi_password = st.text_input("WiFi密码", type="password")
                
                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("💾 添加设备", use_container_width=True)
                with col2:
                    if st.form_submit_button("❌ 取消", use_container_width=True):
                        st.session_state.show_add_device = False
                        st.rerun()
                
                if submitted:
                    if device_code:
                        device_data = {
                            'device_code': device_code,
                            'scene': scene,
                            'wifi_name': wifi_name,
                            'wifi_password': wifi_password,
                            'username': username
                        }
                        
                        if add_device(device_data):
                            st.success("设备添加成功！")
                            st.session_state.show_add_device = False
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("添加失败，请检查设备编号是否重复！")
                    else:
                        st.error("请输入设备编号！")
    
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
    
    # 显示欢迎头部
    show_welcome_header(username)
    
    # 显示设备管理
    show_device_management(username)
    
    # 显示设备相关的模态框
    show_device_modals(username)

if __name__ == "__main__":
    main()