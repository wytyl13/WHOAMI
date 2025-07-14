import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import json
import time
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

    def save_device_connection(self, device_info: Dict[str, Any]) -> bool:
        """保存设备连接信息"""
        try:
            connection_data = {
                "device_code": device_info.get("id"),
                "device_name": device_info.get("name", "未知设备"),
                "connection_type": "bluetooth" if not device_info.get("manual") else "manual",
                "connection_status": "connected",
                "is_manual": device_info.get("manual", False),
                "create_time": datetime.now(),
                "update_time": datetime.now()
            }
            
            result = self.sql_provider.insert_record_enhanced(data=connection_data)
            return result is not None
        except Exception as e:
            st.error(f"保存设备连接信息失败: {str(e)}")
            return False
    
    def get_connected_devices(self) -> list:
        """获取已连接的设备列表"""
        try:
            devices = self.sql_provider.get_record_by_condition(
                condition={"connection_status": "connected"},
                fields=["device_code", "device_name", "connection_type", "is_manual", "create_time"]
            )
            return devices or []
        except Exception as e:
            st.error(f"获取设备列表失败: {str(e)}")
            return []

    def save_complete_device_info(self, device_info: Dict[str, Any], debug_mode: bool = False) -> bool:
        """保存完整的设备信息到数据库"""
        try:
            if debug_mode:
                st.write("### 🔍 调试日志信息")
                st.write("**步骤1：** 接收到的设备信息")
                st.json(device_info)
            
            device_code = device_info.get("device_code")
            if debug_mode:
                st.write(f"**步骤2：** 提取设备编号 = `{device_code}`")
            
            if not device_code:
                st.error("❌ 设备编号不能为空")
                return False
            
            # 获取用户信息
            user_data = st.session_state.get("user_data", {})
            user_id = user_data.get("id") if user_data else None
            username = st.session_state.get("username", "")
            
            if debug_mode:
                st.write("**步骤3：** 当前用户信息")
                st.write(f"- 用户名: `{username}`")
                st.write(f"- 用户ID: `{user_id}`")
            
            # 检查现有设备
            try:
                existing_devices = self.sql_provider.get_record_by_condition(
                    condition={"device_code": device_code}
                )
                if debug_mode:
                    st.write(f"**步骤4：** 查询现有设备 - 找到 {len(existing_devices) if existing_devices else 0} 条记录")
                
                # 软删除现有设备
                if existing_devices:
                    for i, device in enumerate(existing_devices):
                        result = self.sql_provider.update_record_enhanced(
                            record_id=device["id"],
                            data={"deleted": True, "update_time": datetime.now()}
                        )
                        if debug_mode:
                            st.write(f"- 设备{i+1} (ID={device['id']}) 软删除结果: {result}")
                            
            except Exception as e:
                st.error(f"❌ 查询/删除现有设备失败: {str(e)}")
                if debug_mode:
                    st.write(f"**错误详情:** {e}")
            
            # 准备新设备数据
            device_data = {
                "device_code": device_code,
                "scene": device_info.get("scene", "睡眠监测"),
                "wifi_name": device_info.get("wifi_name", ""),
                "wifi_password": device_info.get("wifi_password", ""),
                "username": username,
                "user_id": user_id,
                "status": "active",
            }
            
            if debug_mode:
                st.write("**步骤5：** 准备插入的数据")
                st.json(device_data)
            
            # 执行数据库插入
            try:
                result = self.sql_provider.add_record(device_data)
                if debug_mode:
                    st.write(f"**步骤6：** 插入操作返回结果: {result}")
                
                if result:
                    st.success(f"✅ 设备信息已成功保存到数据库: {device_code}")
                    
                    # 验证插入结果
                    if debug_mode:
                        verify_devices = self.sql_provider.get_record_by_condition(
                            condition={"device_code": device_code, "deleted": False}
                        )
                        st.write(f"**步骤7：** 验证查询结果 - 找到 {len(verify_devices) if verify_devices else 0} 条记录")
                    
                    return True
                else:
                    st.error("❌ 数据库添加记录失败")
                    return False
                    
            except Exception as e:
                st.error(f"❌ 数据库插入操作异常: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())
                return False
                
        except Exception as e:
            st.error(f"❌ 保存设备信息失败: {str(e)}")
            if debug_mode:
                import traceback
                st.code(traceback.format_exc())
            return False


def get_complete_bluetooth_html():
    """生成完整的蓝牙连接和配网HTML组件"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>蓝牙设备连接和配网</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                background: #f5f7fa;
                margin: 0;
                padding: 15px;
                min-height: 100vh;
                line-height: 1.6;
            }

            .container {
                max-width: 100%;
                margin: 0 auto;
            }

            .step-indicator {
                display: flex;
                justify-content: center;
                margin-bottom: 30px;
                gap: 10px;
            }

            .step {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 20px;
                border-radius: 25px;
                background: #e1e5e9;
                color: #6c757d;
                font-weight: 500;
                transition: all 0.3s ease;
            }

            .step.active {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                box-shadow: 0 5px 15px rgba(79, 172, 254, 0.3);
            }

            .step.completed {
                background: #28a745;
                color: white;
            }

            .card {
                background: white;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                border: 1px solid #e1e5e9;
            }

            .card h3 {
                margin: 0 0 20px 0;
                color: #333;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .scan-modes {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }

            .scan-mode {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 12px;
                padding: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                border: 2px solid transparent;
            }

            .scan-mode:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }

            .scan-mode.selected {
                border-color: #ffd700;
                box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
            }

            .scan-mode-title {
                font-weight: bold;
                font-size: 1.1rem;
                margin-bottom: 8px;
            }

            .scan-mode-desc {
                font-size: 0.9rem;
                opacity: 0.9;
            }

            .btn {
                padding: 12px 24px;
                border: none;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 5px;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                text-decoration: none;
            }

            .btn-primary {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
            }

            .btn-success {
                background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
                color: white;
            }

            .btn-warning {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }

            .btn-secondary {
                background: #6c757d;
                color: white;
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            }

            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .device-list {
                display: grid;
                gap: 15px;
            }

            .device-item {
                background: #f8f9fa;
                border: 2px solid #e1e5e9;
                border-radius: 12px;
                padding: 20px;
                transition: all 0.3s ease;
                cursor: pointer;
            }

            .device-item:hover {
                border-color: #4facfe;
                box-shadow: 0 5px 15px rgba(79, 172, 254, 0.2);
                transform: translateY(-2px);
            }

            .device-item.selected {
                border-color: #28a745;
                background: rgba(40, 167, 69, 0.05);
                box-shadow: 0 5px 20px rgba(40, 167, 69, 0.3);
            }

            .device-info {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .device-details h4 {
                margin: 0 0 5px 0;
                color: #333;
                font-size: 1.1rem;
            }

            .device-details p {
                margin: 2px 0;
                color: #666;
                font-size: 0.9rem;
            }

            .config-form {
                display: grid;
                gap: 20px;
                margin: 20px 0;
            }

            .form-group {
                display: flex;
                flex-direction: column;
            }

            .form-label {
                font-weight: 600;
                margin-bottom: 8px;
                color: #333;
                display: flex;
                align-items: center;
                gap: 5px;
            }

            .form-input {
                padding: 12px 15px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                font-size: 1rem;
                transition: border-color 0.2s ease;
            }

            .form-input:focus {
                border-color: #4facfe;
                outline: none;
                box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
            }

            .form-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }

            .progress-section {
                text-align: center;
                padding: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 15px;
                margin: 20px 0;
            }

            .progress-bar {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 25px;
                height: 8px;
                margin: 20px 0;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #ffd700 0%, #ffed4e 100%);
                border-radius: 25px;
                transition: width 0.3s ease;
                width: 0%;
            }

            .config-result {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                border-left: 4px solid #28a745;
            }

            .config-result h4 {
                margin: 0 0 10px 0;
                color: #28a745;
            }

            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }

            .info-item {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                text-align: center;
            }

            .info-item strong {
                display: block;
                color: #333;
                margin-bottom: 5px;
            }

            .status-box {
                padding: 15px 20px;
                border-radius: 10px;
                margin: 15px 0;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .status-success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }

            .status-error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }

            .status-info {
                background: #d6edff;
                color: #0c5460;
                border: 1px solid #b8daff;
            }

            .status-warning {
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }

            .protocol-info {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }

            .protocol-info h4 {
                margin: 0 0 15px 0;
            }

            .protocol-commands {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 8px;
                font-family: monospace;
                font-size: 0.9rem;
            }

            .hidden {
                display: none !important;
            }

            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #4facfe;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .celebration {
                text-align: center;
                padding: 40px;
                background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
                color: white;
                border-radius: 15px;
                margin: 20px 0;
            }

            .celebration-icon {
                font-size: 4rem;
                margin-bottom: 20px;
                animation: bounce 2s infinite;
            }

            @keyframes bounce {
                0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
                40% { transform: translateY(-20px); }
                60% { transform: translateY(-10px); }
            }

            @media (max-width: 768px) {
                .form-row {
                    grid-template-columns: 1fr;
                }
                
                .step-indicator {
                    flex-direction: column;
                    align-items: center;
                }
                
                .device-info {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 15px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- 步骤指示器 -->
            <div class="step-indicator">
                <div id="step-1" class="step active">
                    <span>1️⃣</span>
                    <span>扫描设备</span>
                </div>
                <div id="step-2" class="step">
                    <span>2️⃣</span>
                    <span>连接设备</span>
                </div>
                <div id="step-3" class="step">
                    <span>3️⃣</span>
                    <span>设备配网</span>
                </div>
                <div id="step-4" class="step">
                    <span>4️⃣</span>
                    <span>配置完成</span>
                </div>
            </div>

            <!-- 第一步：设备扫描 -->
            <div id="scan-section" class="card">
                <h3>🔍 第一步：扫描蓝牙设备</h3>
                
                <div class="scan-modes">
                    <div class="scan-mode selected" data-mode="aerosense">
                        <div class="scan-mode-title">🎯 AeroSense专用</div>
                        <div class="scan-mode-desc">精确扫描AeroSense设备</div>
                    </div>
                    <div class="scan-mode" data-mode="health">
                        <div class="scan-mode-title">❤️ 健康设备</div>
                        <div class="scan-mode-desc">扫描健康监测设备</div>
                    </div>
                    <div class="scan-mode" data-mode="all">
                        <div class="scan-mode-title">📡 通用扫描</div>
                        <div class="scan-mode-desc">扫描所有蓝牙设备</div>
                    </div>
                    <div class="scan-mode" data-mode="manual">
                        <div class="scan-mode-title">✏️ 手动输入</div>
                        <div class="scan-mode-desc">手动添加设备信息</div>
                    </div>
                </div>

                <div style="text-align: center; margin: 20px 0;">
                    <button id="scan-btn" class="btn btn-primary">
                        🔍 开始扫描设备
                    </button>
                    <button id="manual-btn" class="btn btn-secondary hidden">
                        ✏️ 手动添加设备
                    </button>
                </div>

                <!-- 扫描状态 -->
                <div id="scan-status" class="hidden"></div>
            </div>

            <!-- 手动输入设备 -->
            <div id="manual-section" class="card hidden">
                <h3>✏️ 手动添加设备</h3>
                <div class="config-form">
                    <div class="form-group">
                        <label class="form-label">📱 设备名称 *</label>
                        <input type="text" id="manual-name" class="form-input" placeholder="例：AeroSense睡眠监测仪">
                    </div>
                    <div class="form-group">
                        <label class="form-label">🔗 设备MAC地址 (可选)</label>
                        <input type="text" id="manual-mac" class="form-input" placeholder="例：A0:76:4E:47:6A:4C">
                    </div>
                    <div style="text-align: center;">
                        <button id="add-manual-btn" class="btn btn-success">
                            ➕ 添加设备
                        </button>
                    </div>
                </div>
            </div>

            <!-- 第二步：设备列表和连接 -->
            <div id="device-section" class="card hidden">
                <h3>📱 第二步：选择并连接设备</h3>
                <div id="device-list" class="device-list"></div>
            </div>

            <!-- 第三步：设备配网 -->
            <div id="config-section" class="card hidden">
                <h3>⚙️ 第三步：AeroSense设备配网</h3>
                
                <!-- AeroSense协议说明 -->
                <div class="protocol-info">
                    <h4>📋 AeroSense BLE配网协议</h4>
                    <div class="protocol-commands">
• WiFi配置: (网络名, 密码) → 返回 0x01 成功<br>
• 服务器配置: [IP, 端口] → 返回 0x01 成功<br>
• 查询IP: i, → 返回设备IP地址<br>
• 查询版本: v, → 返回版本号<br>
• 查询MAC: m, → 返回MAC地址<br>
• 静态IP配置: s,IP;网关;子网;DNS1;DNS2
                    </div>
                </div>

                <div class="config-form">
                    <div class="form-group">
                        <label class="form-label">📶 WiFi网络名称 (SSID) *</label>
                        <input type="text" id="wifi-ssid" class="form-input" placeholder="sxkj" value="sxkj">
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">🔐 WiFi密码 *</label>
                        <input type="password" id="wifi-password" class="form-input" placeholder="88888888" value="88888888">
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">🌐 服务器IP地址</label>
                            <input type="text" id="server-ip" class="form-input" placeholder="1.71.15.121" value="1.71.15.121">
                        </div>
                        <div class="form-group">
                            <label class="form-label">🚪 服务器端口</label>
                            <input type="text" id="server-port" class="form-input" placeholder="8888" value="8888">
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">📏 工作距离 (米)</label>
                            <input type="number" id="working-distance" class="form-input" 
                                placeholder="1.5" value="1.5" min="0.6" max="3.0" step="0.1">
                            <small style="color: #666; font-size: 0.9rem;">检测范围：0.6-3.0米，推荐1.5米</small>
                        </div>
                        <div class="form-group">
                            <label class="form-label">📊 采样间隔 (秒)</label>
                            <input type="number" id="report-interval" class="form-input" 
                                placeholder="1" value="1" min="1" max="60" readonly>
                            <small style="color: #666; font-size: 0.9rem;">固定1秒一次数据上报</small>
                        </div>
                    </div>
                    
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">🏠 应用场景</label>
                            <input type="text" id="scene" class="form-input" 
                                placeholder="睡眠监测" value="睡眠监测">
                            <small style="color: #666; font-size: 0.9rem;">输入设备应用场景，如：睡眠监测、健康监测、老人看护等</small>
                        </div>
                    </div>

                    <div style="text-align: center; margin: 20px 0;">
                        <button id="query-device-btn" class="btn btn-warning">
                            🔍 查询设备信息
                        </button>
                        <button id="get-device-id-btn" class="btn btn-info">
                            🆔 获取设备ID
                        </button>
                        <button id="get-device-sn-btn" class="btn btn-secondary">
                            📋 获取设备序列号
                        </button>
                        <button id="config-device-btn" class="btn btn-success">
                            📡 开始配网
                        </button>
                        <button id="skip-config-btn" class="btn btn-secondary">
                            ⏭️ 跳过配网
                        </button>
                    </div>
                </div>

                <!-- 配网进度 -->
                <div id="config-progress" class="progress-section hidden">
                    <h4 id="progress-title">📡 正在配置设备...</h4>
                    <div class="progress-bar">
                        <div id="progress-fill" class="progress-fill"></div>
                    </div>
                    <p id="progress-text">准备开始配网...</p>
                </div>

                <!-- 设备信息显示 -->
                <div id="device-info-section" class="config-result hidden">
                    <h4>📊 设备信息</h4>
                    <div id="device-info-grid" class="info-grid"></div>
                </div>
                
                <!-- 服务器连接状态 -->
                <div id="server-status-section" class="config-result hidden">
                    <h4>🌐 服务器连接状态</h4>
                    <div class="info-grid">
                        <div class="info-item">
                            <strong>服务器地址</strong>
                            <span id="server-address">未配置</span>
                        </div>
                        <div class="info-item">
                            <strong>配网状态</strong>
                            <span id="device-status">未开始</span>
                        </div>
                    </div>
                    <p id="server-status-text">配网完成后，设备将自动连接到您指定的服务器</p>
                </div>
            </div>

            <!-- 第四步：配置完成 -->
            <div id="completion-section" class="card hidden">
                <div class="celebration">
                    <div class="celebration-icon">🎉</div>
                    <h2>AeroSense设备配网成功！</h2>
                    <p>您的睡眠监测设备已成功连接WiFi网络并配置完成</p>
                    
                    <div style="margin-top: 30px;">
                        <button id="finish-btn" class="btn btn-success" style="font-size: 1.2rem; padding: 15px 30px;">
                            ✅ 完成配置
                        </button>
                    </div>
                </div>
            </div>

            <!-- 消息显示区域 -->
            <div id="messages"></div>
        </div>

        <script>
            // 检查真正的Streamlit对象是否存在
            if (typeof Streamlit === 'undefined') {
                console.log('⚠️ 真正的Streamlit对象不存在，创建模拟版本');
                window.Streamlit = {
                    setComponentValue: function(value) {
                        console.log('🎭 模拟Streamlit.setComponentValue:', value);
                        // 标记这是模拟版本
                        this._isSimulated = true;
                    },
                    _isSimulated: true
                };
            } else {
                console.log('✅ 真正的Streamlit对象已加载');
                window.Streamlit._isSimulated = false;
            }
        
            class AeroSenseConfigManager {
                constructor() {
                    this.currentStep = 1;
                    this.selectedMode = 'aerosense';
                    this.device = null;
                    this.server = null;
                    this.characteristic = null;
                    this.notifyCharacteristic = null;
                    this.foundDevices = new Map();
                    this.deviceInfo = {};
                    this.responseHandlers = new Map();
                    
                    this.init();
                }

                init() {
                    this.setupEventListeners();
                    this.checkBrowserSupport();
                    console.log('🚀 AeroSense配网管理器初始化完成');
                }

                setupEventListeners() {
                    // 扫描模式选择
                    document.querySelectorAll('.scan-mode').forEach(mode => {
                        mode.onclick = () => {
                            document.querySelectorAll('.scan-mode').forEach(m => m.classList.remove('selected'));
                            mode.classList.add('selected');
                            this.selectedMode = mode.dataset.mode;
                            
                            const manualBtn = document.getElementById('manual-btn');
                            if (this.selectedMode === 'manual') {
                                manualBtn.classList.remove('hidden');
                                this.showManualSection();
                            } else {
                                manualBtn.classList.add('hidden');
                                this.hideManualSection();
                            }
                        };
                    });

                    // 按钮事件
                    document.getElementById('scan-btn').onclick = () => this.startScan();
                    document.getElementById('manual-btn').onclick = () => this.showManualSection();
                    document.getElementById('add-manual-btn').onclick = () => this.addManualDevice();
                    document.getElementById('query-device-btn').onclick = () => this.queryDeviceInfo();
                    document.getElementById('get-device-id-btn').onclick = () => this.getDeviceId();
                    document.getElementById('get-device-sn-btn').onclick = () => this.getDeviceSerialNumber();
                    document.getElementById('config-device-btn').onclick = () => this.startConfiguration();
                    document.getElementById('skip-config-btn').onclick = () => this.skipConfiguration();
                    
                    const finishBtn = document.getElementById('finish-btn');
                    if (finishBtn) {
                        finishBtn.onclick = () => this.finishConfiguration();
                    }
                }

                checkBrowserSupport() {
                    if (!navigator.bluetooth) {
                        this.showMessage('❌ 您的浏览器不支持Web Bluetooth API。请使用Chrome、Edge或Opera浏览器。', 'error');
                        document.getElementById('scan-btn').disabled = true;
                        return false;
                    }
                    return true;
                }

                async startScan() {
                    if (!this.checkBrowserSupport()) return;
                    
                    console.log(`🔍 开始扫描 - 模式: ${this.selectedMode}`);
                    this.showScanStatus();
                    
                    try {
                        const options = this.getScanOptions();
                        const device = await navigator.bluetooth.requestDevice(options);
                        
                        this.foundDevices.set(device.id, device);
                        if (device.name && device.name.includes('AeroSense')) {
                            this.showMessage('✅ 找到AeroSense设备，可以进行配网', 'success');
                        } else {
                            this.showMessage('⚠️ 请确认这是AeroSense设备', 'warning');
                        }
                        
                        
                        this.displayDevice(device);
                        this.showMessage(`✅ 找到设备: ${device.name || '未知设备'}`, 'success');
                        
                    } catch (error) {
                        console.error('扫描失败:', error);
                        this.handleScanError(error);
                    } finally {
                        this.hideScanStatus();
                    }
                }

                getScanOptions() {
                    switch(this.selectedMode) {
                        case 'aerosense':
                            return {
                                filters: [
                                    { namePrefix: 'AeroSense' },
                                    { namePrefix: 'Aero' },
                                    { services: ['6e400001-b5a3-f393-e0a9-e50e24dcca9e'] }
                                ],
                                optionalServices: ['6e400001-b5a3-f393-e0a9-e50e24dcca9e', 'heart_rate', 'battery_service', 'device_information']
                            };
                        case 'health':
                            return {
                                filters: [
                                    { namePrefix: 'Sleep' },
                                    { namePrefix: 'Health' },
                                    { services: ['heart_rate', 'battery_service'] }
                                ],
                                optionalServices: ['heart_rate', 'battery_service', 'device_information']
                            };
                        default:
                            return {
                                acceptAllDevices: true,
                                optionalServices: ['heart_rate', 'battery_service', 'device_information']
                            };
                    }
                }

                handleScanError(error) {
                    let message = '';
                    let suggestion = '';
                    
                    if (error.name === 'NotFoundError') {
                        message = '未找到符合条件的设备';
                        suggestion = '💡 建议：1) 确保AeroSense设备处于配对模式 2) 尝试通用扫描 3) 使用手动输入';
                    } else if (error.name === 'NotAllowedError') {
                        message = '用户拒绝了蓝牙访问权限';
                        suggestion = '💡 建议：在浏览器地址栏点击蓝牙图标，允许访问权限';
                    } else {
                        message = error.message;
                        suggestion = '💡 建议：检查设备和浏览器设置';
                    }
                    
                    this.showMessage(`❌ 扫描失败: ${message}`, 'error');
                    if (suggestion) {
                        this.showMessage(suggestion, 'warning');
                    }
                }

                showManualSection() {
                    document.getElementById('manual-section').classList.remove('hidden');
                }

                hideManualSection() {
                    document.getElementById('manual-section').classList.add('hidden');
                }

                addManualDevice() {
                    const name = document.getElementById('manual-name').value.trim();
                    const mac = document.getElementById('manual-mac').value.trim();
                    
                    if (!name) {
                        this.showMessage('请输入设备名称', 'error');
                        return;
                    }
                    
                    const mockDevice = {
                        id: mac || `manual-${Date.now()}`,
                        name: name,
                        manual: true,
                        gatt: { connected: false }
                    };
                    
                    this.foundDevices.set(mockDevice.id, mockDevice);
                    this.displayDevice(mockDevice);
                    this.showMessage('✅ 手动设备已添加！', 'success');
                    
                    document.getElementById('manual-name').value = '';
                    document.getElementById('manual-mac').value = '';
                    this.hideManualSection();
                }

                displayDevice(device) {
                    const deviceSection = document.getElementById('device-section');
                    const deviceList = document.getElementById('device-list');
                    
                    deviceSection.classList.remove('hidden');
                    
                    const deviceDiv = document.createElement('div');
                    deviceDiv.className = 'device-item';
                    deviceDiv.innerHTML = `
                        <div class="device-info">
                            <div class="device-details">
                                <h4>${device.name || '未知设备'} ${device.manual ? '(手动添加)' : ''}</h4>
                                <p><strong>设备ID:</strong> ${device.id}</p>
                                <p><strong>连接状态:</strong> ${device.gatt?.connected ? '已连接' : '未连接'}</p>
                                <p><strong>设备类型:</strong> ${device.manual ? '手动设备' : 'BLE设备'}</p>
                            </div>
                            <div>
                                <button class="btn btn-success" onclick="configManager.selectDevice('${device.id}')">
                                    🔗 选择此设备
                                </button>
                            </div>
                        </div>
                    `;
                    
                    deviceList.appendChild(deviceDiv);
                    this.updateStep(2);
                }

                async selectDevice(deviceId) {
                    this.device = this.foundDevices.get(deviceId);
                    
                    document.querySelectorAll('.device-item').forEach(item => {
                        item.classList.remove('selected');
                    });
                    event.target.closest('.device-item').classList.add('selected');
                    
                    if (!this.device.manual) {
                        await this.connectDevice();
                    } else {
                        this.showConfigSection();
                    }
                }

                async connectDevice() {
                    try {
                        console.log(`连接设备: ${this.device.name}`);
                        
                        if (!this.device.gatt) {
                            throw new Error('设备不支持GATT连接');
                        }

                        this.server = await this.device.gatt.connect();
                        console.log('GATT连接成功');
                        
                        await this.findCommunicationCharacteristics();
                        
                        this.showMessage('✅ 设备连接成功！', 'success');
                        if (this.characteristic && this.notifyCharacteristic) {
                            this.showMessage('✅ 蓝牙通信通道建立成功，可以配网', 'success');
                        } else {
                            this.showMessage('⚠️ 蓝牙连接不完整，配网可能失败', 'warning');
                        }
                        this.showConfigSection();
                        
                    } catch (error) {
                        console.error('连接失败:', error);
                        this.showMessage(`❌ 连接失败: ${error.message}`, 'error');
                        this.showConfigSection();
                    }
                }

                async findCommunicationCharacteristics() {
                    try {
                        const services = await this.server.getPrimaryServices();
                        console.log(`找到 ${services.length} 个服务`);
                        
                        for (const service of services) {
                            const characteristics = await service.getCharacteristics();
                            for (const char of characteristics) {
                                const props = char.properties;
                                
                                if ((props.write || props.writeWithoutResponse) && !this.characteristic) {
                                    this.characteristic = char;
                                }
                                
                                if ((props.notify || props.indicate) && !this.notifyCharacteristic) {
                                    this.notifyCharacteristic = char;
                                    await this.setupNotifications();
                                }
                            }
                        }
                        
                    } catch (error) {
                        console.error('特征值查找失败:', error);
                    }
                }

                async setupNotifications() {
                    try {
                        this.notifyCharacteristic.addEventListener('characteristicvaluechanged', (event) => {
                            const dataView = event.target.value;
                            const bytes = new Uint8Array(dataView.buffer);
                            const value = Array.from(bytes)
                                .map(byte => byte.toString(16).padStart(2, '0').toUpperCase())
                                .join(' ');
                            
                            if (bytes.length > 10 && bytes[0] === 0x13 && bytes[1] === 0x01) {
                                this.handleBinaryResponse(bytes);
                            } else {
                                this.handleDeviceResponse(value);
                            }
                        });
                        
                        await this.notifyCharacteristic.startNotifications();
                        console.log('✅ 通知监听已启动');
                        
                    } catch (error) {
                        console.error('通知设置失败:', error);
                    }
                }

                showConfigSection() {
                    document.getElementById('config-section').classList.remove('hidden');
                    this.updateStep(3);
                }

                async queryDeviceInfo() {
                    if (!this.device) {
                        this.showMessage('请先选择设备', 'error');
                        return;
                    }

                    if (this.device.manual) {
                        this.showMessage('手动设备无法查询信息', 'warning');
                        return;
                    }

                    try {
                        this.showMessage('🔍 正在查询设备信息...', 'info');
                        
                        await this.sendCommand('i,');
                        await this.delay(1000);
                        await this.sendCommand('v,');
                        await this.delay(1000);
                        await this.sendCommand('m,');
                        await this.delay(1000);
                        
                        this.displayDeviceInfo();
                        this.showMessage('✅ 设备信息查询完成', 'success');
                        
                    } catch (error) {
                        console.error('查询失败:', error);
                        this.showMessage(`❌ 查询失败: ${error.message}`, 'error');
                    }
                }

                async getDeviceId() {
                    if (!this.device) {
                        this.showMessage('请先选择设备', 'error');
                        return;
                    }

                    if (this.device.manual) {
                        const deviceId = this.device.id;
                        this.showMessage(`📱 手动设备ID: ${deviceId}`, 'info');
                        this.deviceInfo.mac = this.deviceInfo.mac || deviceId;
                        this.deviceInfo.radar_id = this.deviceInfo.radar_id || deviceId;
                        return;
                    }

                    try {
                        this.showMessage('🔍 正在获取设备ID...', 'info');
                        
                        await this.sendCommand('m,');
                        await this.delay(2000);
                        
                        if (!this.deviceInfo.mac) {
                            await this.sendCommand('v,');
                            await this.delay(1000);
                        }
                        
                        const deviceId = this.deviceInfo.mac || this.deviceInfo.radar_id || this.device.id;
                        
                        if (deviceId) {
                            this.showMessage(`✅ 设备ID: ${deviceId}`, 'success');
                        } else {
                            const fallbackId = `device_${Date.now()}`;
                            this.deviceInfo.mac = fallbackId;
                            this.showMessage(`📱 使用备用ID: ${fallbackId}`, 'info');
                        }
                        
                    } catch (error) {
                        console.error('获取设备ID失败:', error);
                        this.showMessage(`❌ 获取设备ID失败: ${error.message}`, 'error');
                        
                        const fallbackId = `device_${Date.now()}`;
                        this.deviceInfo.mac = fallbackId;
                        this.showMessage(`📱 使用备用ID: ${fallbackId}`, 'warning');
                    }
                }

                async getDeviceSerialNumber() {
                    if (!this.device) {
                        this.showMessage('请先选择设备', 'error');
                        return;
                    }

                    if (this.device.manual) {
                        const deviceSN = this.device.id;
                        this.showMessage(`📋 手动设备序列号: ${deviceSN}`, 'info');
                        return;
                    }

                    try {
                        this.showMessage('🔍 正在获取设备序列号...', 'info');
                        
                        await this.sendRadarIdCommand();
                        await this.delay(3000);
                        
                        if (this.deviceInfo.radar_id) {
                            this.showMessage(`✅ 设备序列号: ${this.deviceInfo.radar_id}`, 'success');
                        } else {
                            this.showMessage('⚠️ 未能获取到设备序列号，请重试', 'warning');
                        }
                        
                    } catch (error) {
                        console.error('获取设备序列号失败:', error);
                        this.showMessage(`❌ 获取设备序列号失败: ${error.message}`, 'error');
                    }
                }

                async sendRadarIdCommand() {
                    if (!this.characteristic) {
                        console.log(`模拟发送Radar ID命令`);
                        setTimeout(() => {
                            this.deviceInfo.radar_id = "13F61B9D100040711117956107";
                            this.displayDeviceInfo();
                        }, 1000);
                        return;
                    }

                    try {
                        const command = this.createRadarIdCommand();
                        
                        if (this.characteristic.properties.writeWithoutResponse) {
                            await this.characteristic.writeValueWithoutResponse(command);
                        } else {
                            await this.characteristic.writeValue(command);
                        }
                        
                        console.log(`✅ Radar ID命令发送成功`);
                        
                    } catch (error) {
                        console.error(`Radar ID命令发送失败: ${error.message}`);
                        throw error;
                    }
                }

                createRadarIdCommand() {
                    const command = new Uint8Array([
                        0x13, 0x01,             // magic, ver
                        0x01, 0x01,             // type, cmd  
                        0x00, 0x00, 0x00, 0x01, // req_id
                        0x00, 0x0A,             // timeout
                        0x00, 0x00, 0x00, 0x06, // content_len = 6
                        0x04, 0x10,             // func_tag = 0x0410 (Get Radar ID)
                        0x00, 0x00, 0x00, 0x00  // data (空数据)
                    ]);
                    
                    return command;
                }

                async startConfiguration() {
                    const wifiSSID = document.getElementById('wifi-ssid').value.trim();
                    const wifiPassword = document.getElementById('wifi-password').value.trim();
                    const serverIP = document.getElementById('server-ip').value.trim() || '1.71.15.121';
                    const serverPort = document.getElementById('server-port').value.trim() || '8888';
                    
                    let workingDistance = 1.5;
                    const workingDistanceElement = document.getElementById('working-distance');
                    if (workingDistanceElement) {
                        workingDistance = parseFloat(workingDistanceElement.value) || 1.5;
                    }
                    
                    if (!wifiSSID || !wifiPassword) {
                        this.showMessage('请填写完整的WiFi信息！', 'error');
                        return;
                    }

                    
                    // 检查是否为真实蓝牙设备
                    if (this.device.manual || !this.characteristic) {
                        this.showMessage('❌ 请使用真实蓝牙设备进行配网，手动模式无法配网到真实设备', 'error');
                        return;
                    }

                    // 检查WiFi名称格式
                    if (wifiSSID.includes(',') || wifiSSID.includes('(') || wifiSSID.includes(')')) {
                        this.showMessage('❌ WiFi名称不能包含逗号或括号等特殊字符', 'error');
                        return;
                    }

                    // 检查密码长度
                    if (wifiPassword.length < 8 && wifiPassword.length > 0) {
                        this.showMessage('❌ WiFi密码长度不能少于8位', 'error');
                        return;
                    }

                    // 检查服务器IP格式
                    const ipRegex = /^(\d{1,3}\.){3}\d{1,3}$/;
                    if (!ipRegex.test(serverIP)) {
                        this.showMessage('❌ 服务器IP格式错误', 'error');
                        return;
                    }

                    // 检查端口号
                    const port = parseInt(serverPort);
                    if (port < 1 || port > 65535) {
                        this.showMessage('❌ 端口号必须在1-65535之间', 'error');
                        return;
                    }

                    this.showMessage('⚠️ 请确认WiFi是2.4GHz网络，设备不支持5GHz', 'warning');
                    
                    
                    
                    try {
                        this.showConfigProgress();
                        
                        this.updateProgress(25, '📶 配置WiFi网络...');
                        const wifiCommand = `(${wifiSSID.trim()},${wifiPassword.trim()})`;
                        
                        if (!this.device.manual) {
                            await this.sendCommand(wifiCommand);
                            await this.delay(3000);
                        }
                        
                        this.updateProgress(50, '🌐 配置服务器连接...');
                        const serverCommand = `[${serverIP.trim()},${serverPort.trim()}]`;
                        
                        
                        if (!this.device.manual) {
                            await this.sendCommand(serverCommand);
                            await this.delay(2000);  // 保留原有的延迟
                            
                            // 【在这里添加新的验证代码】
                            this.updateProgress(65, '📋 验证配网设置...');
                            
                            // 查询设备IP确认WiFi连接
                            await this.sendCommand('i,');
                            await this.delay(3000);
                            
                            // 查询设备版本确认通信正常
                            await this.sendCommand('v,');
                            await this.delay(2000);
                            
                            // 查询设备MAC地址
                            await this.sendCommand('m,');
                            await this.delay(2000);
                            
                            this.showMessage('✅ 配网命令已发送，正在验证...', 'info');
                        }

                        
                        
                        if (workingDistanceElement && !this.device.manual) {
                            this.updateProgress(60, '📏 设置工作距离...');
                            await this.setWorkingDistance(workingDistance);
                            await this.delay(2000);
                        }
                        // 添加采样间隔设置
                        const reportIntervalElement = document.getElementById('report-interval');
                        if (reportIntervalElement && !this.device.manual) {
                            this.updateProgress(65, '📊 设置采样间隔...');
                            const reportInterval = parseInt(reportIntervalElement.value) || 1;
                            await this.setReportInterval(reportInterval);
                            await this.delay(2000);
                        }
                        this.updateProgress(70, '📡 断开蓝牙，设备将连接WiFi...');
                        
                        
                        if (!this.device.manual) {
                            try {
                                if (this.server && this.server.connected) {
                                    await this.server.disconnect();
                                }
                                if (this.device && this.device.gatt && this.device.gatt.connected) {
                                    this.device.gatt.disconnect();
                                }
                                this.showMessage('✅ 蓝牙已断开，设备将连接WiFi并向服务器发送数据', 'success');
                            } catch (error) {
                                console.log('蓝牙断开:', error);
                            }
                        }

                        await this.delay(15000);
                        this.updateProgress(85, '🔍 等待设备连接服务器...');
                        this.showMessage('🔍 请检查设备指示灯状态：', 'info');
                        this.showMessage('💡 蓝灯常亮 = 配网成功', 'success');
                        this.showMessage('💡 红灯常亮 = WiFi连接失败', 'error');
                        this.showMessage('💡 蓝灯闪烁 = 仍在配网中', 'warning');

                        this.updateProgress(85, '🔍 等待设备连接服务器...');
                        this.showMessage(`📡 设备将连接到服务器 ${serverIP}:${serverPort}`, 'info');
                        await this.delay(3000);

                        this.updateProgress(100, '✅ 配网完成！');
                        
                        await this.delay(1000);
                        this.hideConfigProgress();
                        this.showCompletionSection();
                        this.showMessage('🎉 AeroSense设备配网成功！', 'success');
                        
                        this.updateServerStatus(serverIP, serverPort, '配网完成，等待连接');
                        
                        this.saveConfigData({
                            wifi_ssid: wifiSSID,
                            wifi_password: wifiPassword,
                            server_ip: serverIP,
                            server_port: serverPort,
                            working_distance: workingDistance
                        });
                        
                    } catch (error) {
                        console.error('配网失败:', error);
                        this.showMessage(`❌ 配网失败: ${error.message}`, 'error');
                        this.hideConfigProgress();
                    }
                }

                async setWorkingDistance(distance) {
                    if (!this.characteristic) {
                        console.log(`模拟设置工作距离: ${distance}米`);
                        return;
                    }

                    try {
                        const command = this.createWorkingDistanceCommand(distance);
                        
                        if (this.characteristic.properties.writeWithoutResponse) {
                            await this.characteristic.writeValueWithoutResponse(command);
                        } else {
                            await this.characteristic.writeValue(command);
                        }
                        
                        console.log(`✅ 工作距离设置成功: ${distance}米`);
                        
                    } catch (error) {
                        console.error(`工作距离设置失败: ${error.message}`);
                        throw error;
                    }
                }



                async setReportInterval(intervalSeconds) {
                    if (!this.characteristic) {
                        console.log(`模拟设置采样间隔: 每${intervalSeconds}秒一次`);
                        return;
                    }

                    try {
                        // 根据协议文档，1秒对应值20，所以公式是: intervalSeconds * 20
                        const intervalValue = intervalSeconds * 20;
                        const command = this.createReportIntervalCommand(intervalValue);
                        
                        if (this.characteristic.properties.writeWithoutResponse) {
                            await this.characteristic.writeValueWithoutResponse(command);
                        } else {
                            await this.characteristic.writeValue(command);
                        }
                        
                        console.log(`✅ 采样间隔设置成功: 每${intervalSeconds}秒一次 (协议值: ${intervalValue})`);
                        
                    } catch (error) {
                        console.error(`采样间隔设置失败: ${error.message}`);
                        throw error;
                    }
                }


                createReportIntervalCommand(intervalValue) {
                    // 将intervalValue转换为big endian格式的4字节整数
                    const dataBytes = new ArrayBuffer(4);
                    const dataView = new DataView(dataBytes);
                    dataView.setUint32(0, intervalValue, false); // false表示big endian
                    
                    const command = new Uint8Array([
                        0x13, 0x01,             // magic, ver
                        0x01, 0x01,             // type, cmd  
                        0x00, 0x00, 0x00, 0x01, // req_id
                        0x00, 0x0A,             // timeout
                        0x00, 0x00, 0x00, 0x06, // content_len = 6
                        0x03, 0xE9,             // func_tag = 0x03E9 (Report Interval)
                        ...new Uint8Array(dataBytes) // interval value
                    ]);
                    
                    return command;
                }
                
                
                


                createWorkingDistanceCommand(distance) {
                    const distanceBytes = new ArrayBuffer(4);
                    const distanceView = new DataView(distanceBytes);
                    distanceView.setFloat32(0, distance, false);
                    
                    const command = new Uint8Array([
                        0x13, 0x01,             // magic, ver
                        0x01, 0x01,             // type, cmd  
                        0x00, 0x00, 0x00, 0x01, // req_id
                        0x00, 0x0A,             // timeout
                        0x00, 0x00, 0x00, 0x06, // content_len = 6
                        0x03, 0xEB,             // func_tag = 0x03eb (Set Working Distance)
                        ...new Uint8Array(distanceBytes) // distance float
                    ]);
                    
                    return command;
                }

                async sendCommand(command) {
                    if (!this.characteristic) {
                        console.log(`模拟发送命令: ${command}`);
                        return;
                    }

                    try {
                        const encoder = new TextEncoder();
                        const data = encoder.encode(command);
                        
                        if (this.characteristic.properties.writeWithoutResponse) {
                            await this.characteristic.writeValueWithoutResponse(data);
                        } else {
                            await this.characteristic.writeValue(data);
                        }
                        
                        console.log(`✅ 命令发送成功: ${command}`);
                        
                    } catch (error) {
                        console.error(`命令发送失败: ${error.message}`);
                        throw error;
                    }
                }

                handleDeviceResponse(response) {
                    const trimmed = response.trim();
                    console.log(`处理设备回复: "${trimmed}"`);
                    
                    if (trimmed === '0x01' || trimmed === '1') {
                        this.showMessage('✅ 设备操作成功！', 'success');
                    } else if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(trimmed)) {
                        this.deviceInfo.ip = trimmed;
                        this.showMessage(`📍 设备IP: ${trimmed}`, 'info');
                    } else if (/^[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}$/.test(trimmed)) {
                        this.deviceInfo.mac = trimmed;
                        this.showMessage(`🔗 设备MAC: ${trimmed}`, 'info');
                    } else if (/^[A-Fa-f0-9]{26}$/.test(trimmed)) {
                        this.deviceInfo.radar_id = trimmed;
                        this.showMessage(`📋 设备序列号: ${trimmed}`, 'info');
                    } else {
                        this.showMessage(`📨 设备回复: ${trimmed}`, 'info');
                    }
                    
                    this.displayDeviceInfo();
                }

                handleBinaryResponse(bytes) {
                    try {
                        if (bytes.length < 16) return;
                        
                        const contentLen = (bytes[10] << 24) | (bytes[11] << 16) | (bytes[12] << 8) | bytes[13];
                        const funcTag = (bytes[14] << 8) | bytes[15];
                        
                        console.log(`📦 二进制响应解析: funcTag=0x${funcTag.toString(16).padStart(4, '0')}`);
                        
                        if (funcTag === 0x0410) {
                            const radarIdBytes = bytes.slice(16, 16 + 13);
                            const radarId = Array.from(radarIdBytes)
                                .map(byte => byte.toString(16).padStart(2, '0').toUpperCase())
                                .join('');
                            
                            this.deviceInfo.radar_id = radarId;
                            this.showMessage(`📋 设备序列号: ${radarId}`, 'success');
                            this.displayDeviceInfo();
                        }
                        
                    } catch (error) {
                        console.error('二进制响应解析失败:', error);
                    }
                }

                displayDeviceInfo() {
                    const infoSection = document.getElementById('device-info-section');
                    const infoGrid = document.getElementById('device-info-grid');
                    
                    if (Object.keys(this.deviceInfo).length === 0) return;
                    
                    infoSection.classList.remove('hidden');
                    infoGrid.innerHTML = '';
                    
                    Object.entries(this.deviceInfo).forEach(([key, value]) => {
                        const itemDiv = document.createElement('div');
                        itemDiv.className = 'info-item';
                        itemDiv.innerHTML = `
                            <strong>${this.getInfoLabel(key)}</strong>
                            <span>${value}</span>
                        `;
                        infoGrid.appendChild(itemDiv);
                    });
                }

                getInfoLabel(key) {
                    const labels = {
                        ip: '📍 IP地址',
                        version: '🔢 版本号',
                        mac: '🔗 MAC地址',
                        radar_id: '📋 设备序列号'
                    };
                    return labels[key] || key;
                }

                skipConfiguration() {
                    this.showCompletionSection();
                    this.showMessage('⏭️ 已跳过设备配网', 'info');
                }

                showConfigProgress() {
                    document.getElementById('config-progress').classList.remove('hidden');
                }

                hideConfigProgress() {
                    document.getElementById('config-progress').classList.add('hidden');
                }

                updateProgress(percentage, text) {
                    document.getElementById('progress-fill').style.width = percentage + '%';
                    document.getElementById('progress-text').textContent = text;
                }

                showCompletionSection() {
                    document.getElementById('completion-section').classList.remove('hidden');
                    this.updateStep(4);
                }

                updateServerStatus(serverIP, serverPort, status = '配网完成') {
                    const serverStatusSection = document.getElementById('server-status-section');
                    const serverAddress = document.getElementById('server-address');
                    const deviceStatus = document.getElementById('device-status');
                    
                    if (serverStatusSection) {
                        serverStatusSection.classList.remove('hidden');
                    }
                    if (serverAddress) {
                        serverAddress.textContent = `${serverIP}:${serverPort}`;
                    }
                    if (deviceStatus) {
                        deviceStatus.textContent = status;
                    }
                }

                saveConfigData(config) {
                    this.configData = config;
                }

                
                
                finishConfiguration() {
                    console.log('🚀 finishConfiguration 开始执行');
                    
                    const deviceInfo = {
                        device_code: this.deviceInfo.radar_id,
                        wifi_name: document.getElementById('wifi-ssid').value.trim(),
                        wifi_password: document.getElementById('wifi-password').value.trim(),
                        working_distance: parseFloat(document.getElementById('working-distance')?.value || '1.5'),
                        scene: document.getElementById('scene').value.trim() || '睡眠监测'
                    };

                    // 从window.currentUser获取用户信息
                    if (window.currentUser) {
                        deviceInfo.username = window.currentUser.username;
                        deviceInfo.user_id = window.currentUser.user_id;
                    } else {
                        // 备用方案：从URL参数获取
                        const urlParams = new URLSearchParams(window.location.search);
                        deviceInfo.username = urlParams.get('username') || '';
                        deviceInfo.user_id = parseInt(urlParams.get('user_id') || '0');
                    }

                    if (!deviceInfo.device_code) {
                        this.showMessage('❌ 无法获取设备编号，请重新连接设备', 'error');
                        return;
                    }

                    if (!deviceInfo.wifi_name || !deviceInfo.wifi_password) {
                        this.showMessage('❌ 请填写完整的WiFi信息', 'error');
                        return;
                    }

                    this.showMessage('💾 正在保存设备信息...', 'info');
                    console.log('📋 准备传递的设备信息:', deviceInfo);

                    // 修改这部分 - 添加重试机制和更好的错误处理
                    const saveDeviceInfo = async (retryCount = 0) => {
                        const maxRetries = 3;
                        
                        try {
                            const response = await fetch('https://1.71.15.121:8889/api/save_device', {  // 改为相对路径
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                    'Accept': 'application/json',
                                },
                                body: JSON.stringify(deviceInfo),
                            });

                            if (!response.ok) {
                                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                            }

                            const data = await response.json();
                            // ... 成功处理的代码保持不变 ...
                            
                        } catch (error) {
                            console.error(`❌ 保存设备信息失败 (尝试${retryCount + 1}/${maxRetries + 1}):`, error);
                            
                            if (retryCount < maxRetries) {
                                this.showMessage(`🔄 保存失败，正在重试... (${retryCount + 1}/${maxRetries})`, 'warning');
                                setTimeout(() => saveDeviceInfo(retryCount + 1), 2000);
                                return;
                            }
                            
                            // 最终失败，使用备用方案
                            this.showMessage(`❌ 保存失败: ${error.message}`, 'error');
                            this.showMessage('💡 正在使用备用保存方案...', 'info');
                            this.fallbackSave(deviceInfo);
                        }
                    };
                    saveDeviceInfo();
                }
                
                
                
                // 添加备用保存方案
                fallbackSave(deviceInfo) {
                    try {
                        // 通过Streamlit组件值传递
                        if (window.Streamlit && !window.Streamlit._isSimulated) {
                            window.Streamlit.setComponentValue({
                                action: 'save_device',
                                data: deviceInfo,
                                timestamp: new Date().toISOString()
                            });
                            this.showMessage('✅ 设备信息已通过备用方案保存', 'success');
                        } else {
                            this.showDeviceInfoForManualSave(deviceInfo);
                        }
                    } catch (error) {
                        console.error('备用保存方案失败:', error);
                        this.showDeviceInfoForManualSave(deviceInfo);
                    }
                }
                
                

                showDeviceInfoForManualSave(deviceInfo) {
                    // 在页面上显示设备信息，供用户手动保存
                    const messagesDiv = document.getElementById('messages');
                    const infoDiv = document.createElement('div');
                    infoDiv.className = 'config-result';
                    infoDiv.innerHTML = `
                        <h4>📋 设备配网信息</h4>
                        <p><strong>请手动复制以下信息到下方表单中保存：</strong></p>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
                            <p><strong>设备编号:</strong> ${deviceInfo.device_code}</p>
                            <p><strong>WiFi名称:</strong> ${deviceInfo.wifi_name}</p>
                            <p><strong>WiFi密码:</strong> ${deviceInfo.wifi_password}</p>
                            <p><strong>服务器IP:</strong> ${deviceInfo.server_ip}</p>
                            <p><strong>服务器端口:</strong> ${deviceInfo.server_port}</p>
                            <p><strong>工作距离:</strong> ${deviceInfo.working_distance}米</p>
                        </div>
                        <button class="btn btn-primary" onclick="this.parentElement.remove()">✅ 已复制信息</button>
                    `;
                    messagesDiv.appendChild(infoDiv);
                    
                    this.showMessage('💡 请手动复制上方信息到下方的手动配网表单中', 'warning');
                }

                attemptUrlRedirect(deviceInfo) {
                    try {
                        console.log('🔄 尝试URL参数重定向');
                        const urlParams = new URLSearchParams(window.location.search);
                        urlParams.set('action', 'save_device_info');
                        urlParams.set('device_code', deviceInfo.device_code);
                        urlParams.set('wifi_name', deviceInfo.wifi_name);
                        urlParams.set('wifi_password', deviceInfo.wifi_password);
                        urlParams.set('server_ip', deviceInfo.server_ip);
                        urlParams.set('server_port', deviceInfo.server_port);
                        urlParams.set('working_distance', deviceInfo.working_distance);
                        urlParams.set('scene', deviceInfo.scene);
                        
                        // 构造完整的URL，保持当前页面路径
                        const baseUrl = window.location.protocol + '//' + window.location.host + window.location.pathname;
                        const newUrl = baseUrl + '?' + urlParams.toString();
                        console.log('🔗 准备重定向到:', newUrl);
                        
                        this.showMessage('🔄 正在保存设备信息... 如3秒后未跳转，请使用上方显示的手动信息', 'info');
                        
                        // 尝试重定向，但有超时保护
                        const redirectTimer = setTimeout(() => {
                            console.log('🔄 执行重定向');
                            window.location.replace(newUrl);
                        }, 2000);
                        
                        // 5秒后如果还在当前页面，取消重定向
                        setTimeout(() => {
                            clearTimeout(redirectTimer);
                            console.log('⚠️ 重定向超时，请使用手动方式');
                            this.showMessage('⚠️ 自动保存失败，请使用上方显示的设备信息手动保存', 'warning');
                        }, 5000);
                        
                    } catch (error) {
                        console.error('❌ URL重定向失败:', error);
                        this.showMessage('❌ 自动保存失败，请使用手动方式', 'error');
                    }
                }

                updateStep(step) {
                    for (let i = 1; i <= 4; i++) {
                        const stepElement = document.getElementById(`step-${i}`);
                        stepElement.classList.remove('active', 'completed');
                        
                        if (i < step) {
                            stepElement.classList.add('completed');
                        } else if (i === step) {
                            stepElement.classList.add('active');
                        }
                    }
                    this.currentStep = step;
                }

                showScanStatus() {
                    const statusDiv = document.getElementById('scan-status');
                    statusDiv.classList.remove('hidden');
                    statusDiv.innerHTML = `
                        <div class="status-box status-info">
                            <span class="loading"></span>
                            正在扫描设备... (${this.getScanModeText()})
                        </div>
                    `;
                }

                hideScanStatus() {
                    setTimeout(() => {
                        document.getElementById('scan-status').classList.add('hidden');
                    }, 2000);
                }

                getScanModeText() {
                    const texts = {
                        aerosense: 'AeroSense专用扫描',
                        health: '健康设备扫描',
                        all: '通用设备扫描',
                        manual: '手动输入模式'
                    };
                    return texts[this.selectedMode] || '未知模式';
                }

                showMessage(message, type = 'info') {
                    const messagesDiv = document.getElementById('messages');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `status-box status-${type}`;
                    
                    const icon = {
                        'success': '✅',
                        'error': '❌',
                        'warning': '⚠️',
                        'info': 'ℹ️'
                    }[type] || 'ℹ️';
                    
                    messageDiv.innerHTML = `${icon} ${message}`;
                    messagesDiv.appendChild(messageDiv);
                    
                    setTimeout(() => {
                        if (messageDiv.parentNode) {
                            messageDiv.parentNode.removeChild(messageDiv);
                        }
                    }, 6000);
                }

                delay(ms) {
                    return new Promise(resolve => setTimeout(resolve, ms));
                }
            }

            // 初始化
            let configManager;
            document.addEventListener('DOMContentLoaded', () => {
                configManager = new AeroSenseConfigManager();
                console.log('✅ AeroSense配网系统已启动');
            });
        </script>
    </body>
    </html>
    """
    
    return html_content


def handle_device_save():
    """处理设备保存逻辑"""
    query_params = st.query_params
    
    if query_params.get("action") == "save_device_info":
        st.title("💾 设备信息保存")
        
        # 从URL参数获取设备信息
        device_info = {
            "device_code": query_params.get("device_code", ""),
            "wifi_name": query_params.get("wifi_name", ""),
            "wifi_password": query_params.get("wifi_password", ""),
            "server_ip": query_params.get("server_ip", "1.71.15.121"),
            "server_port": query_params.get("server_port", "8888"),
            "working_distance": query_params.get("working_distance", "1.5"),
            "scene": query_params.get("scene", "睡眠监测")
        }
        
        # 显示设备信息
        with st.expander("📋 接收到的设备信息", expanded=True):
            st.json(device_info)
        
        # 验证必要字段
        if not device_info["device_code"]:
            st.error("❌ 设备编号为空，无法保存")
            if st.button("🔄 返回重新配网"):
                st.query_params.clear()
                st.rerun()
            return True
        
        if not device_info["wifi_name"]:
            st.error("❌ WiFi名称为空，无法保存")
            if st.button("🔄 返回重新配网"):
                st.query_params.clear()
                st.rerun()
            return True
        
        # 保存设备信息
        try:
            connection_manager = DeviceConnectionManager()
            with st.spinner("正在保存设备信息到数据库..."):
                success = connection_manager.save_complete_device_info(device_info, debug_mode=True)
            
            if success:
                st.success("🎉 设备信息保存成功！")
                st.balloons()
                
                # 显示保存成功的信息摘要
                with st.expander("✅ 保存成功的设备信息", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**设备编号:** {device_info['device_code']}")
                        st.write(f"**WiFi名称:** {device_info['wifi_name']}")
                        st.write(f"**服务器地址:** {device_info['server_ip']}:{device_info['server_port']}")
                    with col2:
                        st.write(f"**工作距离:** {device_info['working_distance']}米")
                        st.write(f"**应用场景:** {device_info['scene']}")
                        st.write(f"**保存时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 提供操作选项
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🏠 返回设备管理", use_container_width=True, type="primary"):
                        st.switch_page("pages/user_dashboard.py")
                with col2:
                    if st.button("🔄 继续配网其他设备", use_container_width=True):
                        st.query_params.clear()
                        st.rerun()
                        
            else:
                st.error("❌ 设备信息保存失败，请查看上方的详细日志")
                
                # 提供操作选项
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 重新尝试保存", use_container_width=True):
                        st.rerun()
                with col2:
                    if st.button("🔄 返回重新配网", use_container_width=True):
                        st.query_params.clear()
                        st.rerun()
                        
        except Exception as e:
            st.error(f"❌ 保存过程出错: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        return True
    return False



def main():
    """主函数"""
    # 检查登录状态
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.error("请先登录！")
        st.switch_page("pages/login.py")
        return
    
    
    
    st.title("🔗 AeroSense设备连接与配网")
    
    # 重要提示
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="margin: 0 0 10px 0;">🚀 完整配网流程</h4>
        <p style="margin: 5px 0;">1️⃣ <strong>扫描设备</strong> - 支持AeroSense专用扫描</p>
        <p style="margin: 5px 0;">2️⃣ <strong>连接设备</strong> - BLE通信建立</p>
        <p style="margin: 5px 0;">3️⃣ <strong>设备配网</strong> - WiFi和服务器配置</p>
        <p style="margin: 5px 0;">4️⃣ <strong>配置完成</strong> - 自动保存到数据库</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 获取用户信息
    user_data = st.session_state.get("user_data", {})
    username = st.session_state.get("username", "")
    user_id = user_data.get("id", 0) if user_data else 0
    
    # 嵌入完整的配网组件
    try:
        # 修改HTML内容，注入用户信息
        html_content = get_complete_bluetooth_html()
        user_script = f"""
        <script>
            window.currentUser = {{
                username: '{username}',
                user_id: {user_id}
            }};
            console.log('👤 用户信息已注入:', window.currentUser);
        </script>
        """
        html_content = html_content.replace('</head>', user_script + '</head>')
        
        components.html(
            html_content,
            height=1000,
            scrolling=True
        )
            
    except Exception as e:
        st.error(f"❌ 加载配网组件失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # 底部操作按钮
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ 返回设备管理", use_container_width=True):
            st.switch_page("pages/user_dashboard.py")
    
    with col2:
        if st.button("🔄 刷新页面", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("🧪 测试API连接", use_container_width=True):
            try:
                import requests
                response = requests.get('https://1.71.15.121:8889/api/health', timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ {data.get('message', 'API服务正常')}")
                else:
                    st.error(f"❌ API服务异常: {response.status_code}")
            except Exception as e:
                st.error(f"❌ API连接失败，请确保8889端口已启动: {str(e)}")




if __name__ == "__main__":
    main()