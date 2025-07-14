import streamlit as st
import sys
import os
import json
import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

# 添加项目路径
project_root = "/work/ai/WHOAMI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入相关模块
from whoami.tool.streamlit.health_report.table.device_data import DeviceData
from whoami.provider.sql_provider import SqlProvider

# 页面配置
st.set_page_config(
    page_title="设备连接 - 睡眠健康管理系统",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 配置文件路径
SQL_CONFIG_PATH = '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml'

# CSS样式
st.markdown("""
<style>
/* 全局样式 */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

/* 连接页面头部 */
.connection-header {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 30px 40px;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    text-align: center;
}

.connection-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 10px 0;
}

.connection-subtitle {
    font-size: 1.1rem;
    margin: 0;
    opacity: 0.9;
}

/* 步骤容器 */
.steps-container {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    margin: 20px 0;
}

/* 步骤指示器 */
.step-indicator {
    display: flex;
    justify-content: space-between;
    margin-bottom: 40px;
    position: relative;
}

.step-indicator::before {
    content: '';
    position: absolute;
    top: 25px;
    left: 50px;
    right: 50px;
    height: 3px;
    background: #e1e5e9;
    z-index: 1;
}

.step {
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
    z-index: 2;
    flex: 1;
}

.step-number {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 1.2rem;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}

.step-inactive {
    background: #e1e5e9;
    color: #6c757d;
}

.step-active {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    animation: pulse-blue 2s infinite;
}

.step-completed {
    background: #28a745;
    color: white;
}

.step-error {
    background: #dc3545;
    color: white;
}

.step-title {
    font-weight: 600;
    color: #333;
    text-align: center;
    font-size: 0.9rem;
}

@keyframes pulse-blue {
    0% { box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4); }
    50% { box-shadow: 0 4px 25px rgba(79, 172, 254, 0.6); }
    100% { box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4); }
}

/* 步骤内容区域 */
.step-content {
    background: #f8f9fa;
    padding: 25px;
    border-radius: 15px;
    margin: 20px 0;
    border-left: 4px solid #4facfe;
}

.step-content-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
}

.step-icon {
    margin-right: 10px;
    font-size: 1.5rem;
}

/* 蓝牙设备列表 */
.bluetooth-device {
    background: white;
    border: 2px solid #e1e5e9;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.bluetooth-device:hover {
    border-color: #4facfe;
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.2);
}

.bluetooth-device.selected {
    border-color: #4facfe;
    background: rgba(79, 172, 254, 0.05);
}

.device-info {
    flex: 1;
}

.device-name {
    font-weight: 600;
    color: #333;
    font-size: 1.1rem;
}

.device-id {
    color: #666;
    font-size: 0.9rem;
    font-family: 'Monaco', 'Menlo', monospace;
}

.device-signal {
    color: #28a745;
    font-size: 0.9rem;
}

/* 网络配置表单 */
.wifi-form {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #e1e5e9;
}

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
    padding: 12px 15px;
    border: 2px solid #e1e5e9;
    border-radius: 10px;
    font-size: 1rem;
    transition: border-color 0.2s ease;
}

.form-input:focus {
    border-color: #4facfe;
    outline: none;
    box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
}

/* 连接状态 */
.connection-status {
    text-align: center;
    padding: 30px;
}

.status-icon {
    font-size: 4rem;
    margin-bottom: 20px;
}

.status-text {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.status-detail {
    color: #666;
    font-size: 1rem;
}

/* 进度条 */
.progress-container {
    background: #e1e5e9;
    border-radius: 10px;
    height: 8px;
    margin: 20px 0;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    border-radius: 10px;
    transition: width 0.3s ease;
    animation: progress-pulse 2s infinite;
}

@keyframes progress-pulse {
    0% { opacity: 0.8; }
    50% { opacity: 1; }
    100% { opacity: 0.8; }
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
    text-decoration: none;
    display: inline-block;
    text-align: center;
}

.btn-primary {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
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

.btn-secondary {
    background: #6c757d;
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

.message-info {
    background: #d6edff;
    color: #0c5460;
    border: 1px solid #b8daff;
}

.message-warning {
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffeaa7;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .step-indicator {
        flex-direction: column;
        gap: 20px;
    }
    
    .step-indicator::before {
        display: none;
    }
    
    .bluetooth-device {
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
    }
}
</style>
""", unsafe_allow_html=True)

class DeviceConnectionManager:
    """设备连接管理器"""
    
    def __init__(self):
        self.sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
    
    def get_device_wifi_config(self, device_code: str) -> Optional[Dict[str, str]]:
        """获取设备的WiFi配置信息"""
        try:
            device_records = self.sql_provider.get_record_by_condition(
                condition={"device_code": device_code},
                fields=["wifi_name", "wifi_password"]
            )
            
            if device_records:
                device = device_records[0]
                return {
                    "wifi_name": device.get("wifi_name", ""),
                    "wifi_password": device.get("wifi_password", "")
                }
            return None
        except Exception as e:
            st.error(f"获取设备WiFi配置失败: {str(e)}")
            return None
    
    def save_device_wifi_config(self, device_code: str, wifi_name: str, wifi_password: str) -> bool:
        """保存设备的WiFi配置"""
        try:
            # 更新设备的WiFi配置
            device_records = self.sql_provider.get_record_by_condition(
                condition={"device_code": device_code}
            )
            
            if device_records:
                update_data = {
                    "wifi_name": wifi_name,
                    "wifi_password": wifi_password,
                    "update_time": datetime.now()
                }
                
                result = self.sql_provider.update_record_enhanced(
                    record_id=device_records[0]["id"],
                    data=update_data
                )
                return result is not None
            return False
        except Exception as e:
            st.error(f"保存WiFi配置失败: {str(e)}")
            return False

def simulate_bluetooth_scan():
    """模拟蓝牙扫描"""
    # 模拟的蓝牙设备列表
    mock_devices = [
        {
            "name": "SleepDevice-001",
            "id": "12:34:56:78:9A:BC",
            "signal": "-45 dBm",
            "device_type": "sleep_monitor"
        },
        {
            "name": "SleepDevice-002", 
            "id": "12:34:56:78:9A:BD",
            "signal": "-52 dBm",
            "device_type": "sleep_monitor"
        },
        {
            "name": "Unknown Device",
            "id": "12:34:56:78:9A:BE",
            "signal": "-68 dBm",
            "device_type": "unknown"
        }
    ]
    return mock_devices

def simulate_bluetooth_connect(device_id: str) -> bool:
    """模拟蓝牙连接"""
    # 模拟连接延迟
    time.sleep(2)
    # 模拟连接成功率（90%）
    import random
    return random.random() > 0.1

def simulate_wifi_configure(device_id: str, wifi_name: str, wifi_password: str) -> bool:
    """模拟WiFi配置"""
    # 模拟配置延迟
    time.sleep(3)
    # 模拟配置成功率（85%）
    import random
    return random.random() > 0.15

def simulate_data_sync(device_id: str) -> bool:
    """模拟数据同步"""
    # 模拟同步延迟
    time.sleep(2)
    # 模拟同步成功率（95%）
    import random
    return random.random() > 0.05

def show_step_indicator(current_step: int, step_states: Dict[int, str]):
    """显示步骤指示器"""
    steps = [
        {"number": 1, "title": "蓝牙连接"},
        {"number": 2, "title": "网络配置"},
        {"number": 3, "title": "数据同步"}
    ]
    
    step_html = '<div class="step-indicator">'
    
    for step in steps:
        step_num = step["number"]
        state = step_states.get(step_num, "inactive")
        
        if step_num == current_step and state == "active":
            css_class = "step-active"
        elif state == "completed":
            css_class = "step-completed"
        elif state == "error":
            css_class = "step-error"
        else:
            css_class = "step-inactive"
        
        step_html += f'''
        <div class="step">
            <div class="step-number {css_class}">
                {"✓" if state == "completed" else "✗" if state == "error" else step_num}
            </div>
            <div class="step-title">{step["title"]}</div>
        </div>
        '''
    
    step_html += '</div>'
    st.markdown(step_html, unsafe_allow_html=True)

def show_bluetooth_step():
    """显示蓝牙连接步骤"""
    st.markdown("""
    <div class="step-content">
        <div class="step-content-title">
            <span class="step-icon">📶</span>
            第一步：蓝牙设备连接
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("正在扫描附近的睡眠监测设备...")
    
    with col2:
        if st.button("🔄 重新扫描", key="rescan_bluetooth"):
            st.session_state.bluetooth_devices = simulate_bluetooth_scan()
            st.rerun()
    
    # 显示扫描进度
    if 'bluetooth_scanning' in st.session_state and st.session_state.bluetooth_scanning:
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        st.session_state.bluetooth_scanning = False
        st.session_state.bluetooth_devices = simulate_bluetooth_scan()
        st.rerun()
    
    # 获取蓝牙设备列表
    if 'bluetooth_devices' not in st.session_state:
        st.session_state.bluetooth_scanning = True
        st.session_state.bluetooth_devices = simulate_bluetooth_scan()
        st.rerun()
    
    devices = st.session_state.bluetooth_devices
    
    if devices:
        st.write("找到以下设备，请选择要连接的设备：")
        
        for device in devices:
            is_selected = st.session_state.get('selected_bluetooth_device') == device['id']
            
            device_html = f'''
            <div class="bluetooth-device {'selected' if is_selected else ''}" onclick="selectDevice('{device['id']}')">
                <div class="device-info">
                    <div class="device-name">{device['name']}</div>
                    <div class="device-id">设备ID: {device['id']}</div>
                    <div class="device-signal">信号强度: {device['signal']}</div>
                </div>
            </div>
            '''
            
            st.markdown(device_html, unsafe_allow_html=True)
            
            # 使用按钮来选择设备
            if st.button(f"选择 {device['name']}", key=f"select_{device['id']}"):
                st.session_state.selected_bluetooth_device = device['id']
                st.session_state.selected_device_name = device['name']
                st.rerun()
        
        # 连接按钮
        if 'selected_bluetooth_device' in st.session_state:
            st.success(f"已选择设备：{st.session_state.selected_device_name}")
            
            col1, col2, col3 = st.columns(3)
            with col2:
                if st.button("🔗 连接设备", key="connect_bluetooth", use_container_width=True):
                    with st.spinner("正在连接蓝牙设备..."):
                        success = simulate_bluetooth_connect(st.session_state.selected_bluetooth_device)
                        
                        if success:
                            st.session_state.connection_step = 2
                            st.session_state.step_states[1] = "completed"
                            st.session_state.step_states[2] = "active"
                            st.success("蓝牙连接成功！")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.session_state.step_states[1] = "error"
                            st.error("蓝牙连接失败，请重试！")
    else:
        st.warning("未找到可用的设备，请确保设备已开启并处于可连接状态。")

def show_wifi_step(connection_manager: DeviceConnectionManager, device_code: str = None, is_reconnect: bool = False):
    """显示WiFi配置步骤"""
    st.markdown("""
    <div class="step-content">
        <div class="step-content-title">
            <span class="step-icon">📶</span>
            第二步：网络配置
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 如果是重连模式且有设备代码，尝试获取已保存的WiFi配置
    saved_wifi_config = None
    if is_reconnect and device_code:
        saved_wifi_config = connection_manager.get_device_wifi_config(device_code)
    
    with st.form("wifi_config_form"):
        if saved_wifi_config:
            st.info("🔍 检测到已保存的WiFi配置，您可以直接使用或修改后使用")
            
        col1, col2 = st.columns(2)
        
        with col1:
            wifi_name = st.text_input(
                "WiFi名称*", 
                value=saved_wifi_config.get("wifi_name", "") if saved_wifi_config else "",
                placeholder="请输入WiFi网络名称"
            )
        
        with col2:
            wifi_password = st.text_input(
                "WiFi密码*", 
                value=saved_wifi_config.get("wifi_password", "") if saved_wifi_config else "",
                type="password",
                placeholder="请输入WiFi密码"
            )
        
        if saved_wifi_config:
            st.write("💡 **提示：** 如果WiFi配置无误，可以直接点击配置按钮")
        
        col1, col2, col3 = st.columns(3)
        with col2:
            submitted = st.form_submit_button("📡 配置网络", use_container_width=True)
        
        if submitted:
            if wifi_name and wifi_password:
                with st.spinner("正在配置设备网络..."):
                    # 模拟网络配置过程
                    success = simulate_wifi_configure(
                        st.session_state.selected_bluetooth_device, 
                        wifi_name, 
                        wifi_password
                    )
                    
                    if success:
                        st.session_state.connection_step = 3
                        st.session_state.step_states[2] = "completed"
                        st.session_state.step_states[3] = "active"
                        st.session_state.wifi_name = wifi_name
                        st.session_state.wifi_password = wifi_password
                        st.success("网络配置成功！")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.session_state.step_states[2] = "error"
                        st.error("网络配置失败，请检查WiFi信息是否正确！")
            else:
                st.error("请填写完整的WiFi信息！")

def show_sync_step(connection_manager: DeviceConnectionManager, device_code: str = None, is_reconnect: bool = False):
    """显示数据同步步骤"""
    st.markdown("""
    <div class="step-content">
        <div class="step-content-title">
            <span class="step-icon">🔄</span>
            第三步：数据同步
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("正在与设备建立数据连接并同步配置...")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("🚀 开始同步", key="start_sync", use_container_width=True):
            with st.spinner("正在同步数据..."):
                # 模拟数据同步过程
                progress_bar = st.progress(0)
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.03)
                
                success = simulate_data_sync(st.session_state.selected_bluetooth_device)
                
                if success:
                    st.session_state.step_states[3] = "completed"
                    st.session_state.connection_complete = True
                    
                    # 如果有WiFi配置，保存到数据库
                    if hasattr(st.session_state, 'wifi_name') and device_code:
                        connection_manager.save_device_wifi_config(
                            device_code, 
                            st.session_state.wifi_name, 
                            st.session_state.wifi_password
                        )
                    
                    st.success("设备连接完成！")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.step_states[3] = "error"
                    st.error("数据同步失败，请重试！")

def show_connection_complete():
    """显示连接完成"""
    st.markdown("""
    <div class="connection-status">
        <div class="status-icon">✅</div>
        <div class="status-text">设备连接成功！</div>
        <div class="status-detail">您的设备已成功连接并配置完成</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 重新连接", key="reconnect_device", use_container_width=True):
            # 重置连接状态
            reset_connection_state()
            st.rerun()
    
    with col2:
        if st.button("📊 查看设备", key="view_device_status", use_container_width=True):
            st.switch_page("pages/user_dashboard.py")
    
    with col3:
        if st.button("🏠 返回首页", key="back_to_home", use_container_width=True):
            st.switch_page("pages/user_dashboard.py")

def reset_connection_state():
    """重置连接状态"""
    # 清除所有连接相关的session state
    keys_to_remove = [
        'connection_step', 'step_states', 'bluetooth_devices', 
        'selected_bluetooth_device', 'selected_device_name',
        'wifi_name', 'wifi_password', 'connection_complete',
        'bluetooth_scanning'
    ]
    
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    # 初始化连接状态
    st.session_state.connection_step = 1
    st.session_state.step_states = {1: "active", 2: "inactive", 3: "inactive"}

def main():
    """主函数"""
    # 检查登录状态
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.error("请先登录！")
        st.switch_page("pages/login.py")
        return
    
    # 获取URL参数
    query_params = st.query_params
    device_code = query_params.get("device_code")
    mode = query_params.get("mode", "add")  # add 或 reconnect
    
    # 初始化连接管理器
    connection_manager = DeviceConnectionManager()
    
    # 显示页面头部
    if mode == "reconnect":
        title = "设备重新连接"
        subtitle = f"正在重新连接设备: {device_code}"
    else:
        title = "添加新设备"
        subtitle = "请按照以下步骤连接您的睡眠监测设备"
    
    st.markdown(f"""
    <div class="connection-header">
        <h1 class="connection-title">{title}</h1>
        <p class="connection-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 初始化连接状态
    if 'connection_step' not in st.session_state:
        reset_connection_state()
    
    # 显示步骤容器
    st.markdown('<div class="steps-container">', unsafe_allow_html=True)
    
    # 显示步骤指示器
    show_step_indicator(st.session_state.connection_step, st.session_state.step_states)
    
    # 根据当前步骤显示对应内容
    if st.session_state.get('connection_complete', False):
        show_connection_complete()
    elif st.session_state.connection_step == 1:
        show_bluetooth_step()
    elif st.session_state.connection_step == 2:
        show_wifi_step(connection_manager, device_code, mode == "reconnect")
    elif st.session_state.connection_step == 3:
        show_sync_step(connection_manager, device_code, mode == "reconnect")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 返回按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ 返回设备管理", key="back_to_devices"):
            st.switch_page("pages/user_dashboard.py")

if __name__ == "__main__":
    main()