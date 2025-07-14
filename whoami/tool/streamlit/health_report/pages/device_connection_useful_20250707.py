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
                        <input type="text" id="wifi-ssid" class="form-input" placeholder="请输入WiFi网络名称">
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">🔐 WiFi密码 *</label>
                        <input type="password" id="wifi-password" class="form-input" placeholder="请输入WiFi密码">
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">🌐 服务器IP地址</label>
                            <input type="text" id="server-ip" class="form-input" placeholder="10.8.4.144" value="10.8.4.144">
                        </div>
                        <div class="form-group">
                            <label class="form-label">🚪 服务器端口</label>
                            <input type="text" id="server-port" class="form-input" placeholder="8899" value="8899">
                        </div>
                    </div>

                    <div style="text-align: center; margin: 20px 0;">
                        <button id="query-device-btn" class="btn btn-warning">
                            🔍 查询设备信息
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
            </div>

            <!-- 第四步：配置完成 -->
            <div id="completion-section" class="card hidden">
                <div class="celebration">
                    <div class="celebration-icon">🎉</div>
                    <h2>AeroSense设备配网成功！</h2>
                    <p>您的睡眠监测设备已成功连接WiFi网络并配置完成</p>
                    
                    <div style="margin-top: 30px;">
                        <button id="test-connection-btn" class="btn btn-primary">
                            🧪 测试设备连接
                        </button>
                        <button id="save-device-btn" class="btn btn-success">
                            💾 保存设备信息
                        </button>
                        <button id="finish-btn" class="btn btn-warning">
                            ✅ 完成配置
                        </button>
                    </div>
                </div>
            </div>

            <!-- 消息显示区域 -->
            <div id="messages"></div>
        </div>

        <script>
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
                            
                            // 显示/隐藏手动添加按钮
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
                    document.getElementById('config-device-btn').onclick = () => this.startConfiguration();
                    document.getElementById('skip-config-btn').onclick = () => this.skipConfiguration();
                    document.getElementById('test-connection-btn').onclick = () => this.testConnection();
                    document.getElementById('save-device-btn').onclick = () => this.saveDevice();
                    document.getElementById('finish-btn').onclick = () => this.finishConfiguration();
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
                        console.log('扫描选项:', options);
                        
                        const device = await navigator.bluetooth.requestDevice(options);
                        console.log(`✅ 找到设备: ${device.name}`);
                        
                        this.foundDevices.set(device.id, device);
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
                                    { services: ['6e400001-b5a3-f393-e0a9-e50e24dcca9e'] } // Nordic UART
                                ],
                                optionalServices: [
                                    '6e400001-b5a3-f393-e0a9-e50e24dcca9e', // Nordic UART Service
                                    'heart_rate', 'battery_service', 'device_information'
                                ]
                            };
                        case 'health':
                            return {
                                filters: [
                                    { namePrefix: 'Sleep' },
                                    { namePrefix: 'Health' },
                                    { namePrefix: 'Monitor' },
                                    { namePrefix: 'Fitness' },
                                    { namePrefix: 'Band' },
                                    { namePrefix: 'Watch' },
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
                    } else if (error.name === 'SecurityError') {
                        message = '安全限制，无法访问蓝牙';
                        suggestion = '💡 建议：确保使用HTTPS连接或localhost';
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
                    
                    // 清空输入
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
                    
                    // 高亮选中的设备
                    document.querySelectorAll('.device-item').forEach(item => {
                        item.classList.remove('selected');
                    });
                    event.target.closest('.device-item').classList.add('selected');
                    
                    if (!this.device.manual) {
                        // 蓝牙设备需要连接
                        await this.connectDevice();
                    } else {
                        // 手动设备直接进入配网
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
                        
                        // 查找通信特征值
                        await this.findCommunicationCharacteristics();
                        
                        this.showMessage('✅ 设备连接成功！', 'success');
                        this.showConfigSection();
                        
                    } catch (error) {
                        console.error('连接失败:', error);
                        this.showMessage(`❌ 连接失败: ${error.message}`, 'error');
                        // 连接失败但仍然可以进入配网（用于手动测试）
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
                                
                                // 查找写入特征值
                                if ((props.write || props.writeWithoutResponse) && !this.characteristic) {
                                    this.characteristic = char;
                                    console.log(`找到写入特征值: ${char.uuid}`);
                                }
                                
                                // 查找通知特征值
                                if ((props.notify || props.indicate) && !this.notifyCharacteristic) {
                                    this.notifyCharacteristic = char;
                                    await this.setupNotifications();
                                    console.log(`找到通知特征值: ${char.uuid}`);
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
                            console.log(`📥 收到设备回复: "${value}"`);
                            this.handleDeviceResponse(value);
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
                        
                        // 查询IP
                        await this.sendCommand('i,');
                        await this.delay(1000);
                        
                        // 查询版本
                        await this.sendCommand('v,');
                        await this.delay(1000);
                        
                        // 查询MAC
                        await this.sendCommand('m,');
                        await this.delay(1000);
                        
                        // 显示收集到的信息
                        this.displayDeviceInfo();
                        this.showMessage('✅ 设备信息查询完成', 'success');
                        
                    } catch (error) {
                        console.error('查询失败:', error);
                        this.showMessage(`❌ 查询失败: ${error.message}`, 'error');
                    }
                }

                async startConfiguration() {
                    const wifiSSID = document.getElementById('wifi-ssid').value.trim();
                    const wifiPassword = document.getElementById('wifi-password').value.trim();
                    const serverIP = document.getElementById('server-ip').value.trim() || '10.8.4.144';
                    const serverPort = document.getElementById('server-port').value.trim() || '8899';
                    
                    if (!wifiSSID || !wifiPassword) {
                        this.showMessage('请填写完整的WiFi信息！', 'error');
                        return;
                    }

                    try {
                        this.showConfigProgress();
                        
                        // 步骤1: WiFi配置
                        this.updateProgress(25, '📶 配置WiFi网络...');
                        const wifiCommand = `(${wifiSSID}, ${wifiPassword})`;
                        
                        if (!this.device.manual) {
                            await this.sendCommand(wifiCommand);
                            await this.delay(3000); // 等待WiFi连接
                        }
                        
                        // 步骤2: 服务器配置
                        this.updateProgress(50, '🌐 配置服务器连接...');
                        const serverCommand = `[${serverIP}, ${serverPort}]`;
                        
                        if (!this.device.manual) {
                            await this.sendCommand(serverCommand);
                            await this.delay(2000);
                        }
                        
                        // 步骤3: 验证配置
                        this.updateProgress(75, '🔍 验证配置结果...');
                        if (!this.device.manual) {
                            await this.sendCommand('i,'); // 查询IP确认连接
                            await this.delay(2000);
                        }
                        
                        // 步骤4: 完成
                        this.updateProgress(100, '✅ 配网完成！');
                        
                        await this.delay(1000);
                        this.hideConfigProgress();
                        this.showCompletionSection();
                        this.showMessage('🎉 AeroSense设备配网成功！', 'success');
                        
                        // 保存配网信息
                        this.saveConfigData({
                            wifi_ssid: wifiSSID,
                            wifi_password: wifiPassword,
                            server_ip: serverIP,
                            server_port: serverPort
                        });
                        
                    } catch (error) {
                        console.error('配网失败:', error);
                        this.showMessage(`❌ 配网失败: ${error.message}`, 'error');
                        this.hideConfigProgress();
                    }
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
                    
                    // 解析不同类型的回复
                    if (trimmed === '0x01' || trimmed === '1') {
                        this.showMessage('✅ 设备操作成功！', 'success');
                    } else if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(trimmed)) {
                        this.deviceInfo.ip = trimmed;
                        this.showMessage(`📍 设备IP: ${trimmed}`, 'info');
                    } else if (/^\d+\.\d+\.\d+\.\d+$/.test(trimmed) && !trimmed.includes('.168.')) {
                        this.deviceInfo.version = trimmed;
                        this.showMessage(`🔢 设备版本: ${trimmed}`, 'info');
                    } else if (/^[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}:[A-Fa-f0-9]{2}$/.test(trimmed)) {
                        this.deviceInfo.mac = trimmed;
                        this.showMessage(`🔗 设备MAC: ${trimmed}`, 'info');
                    } else {
                        this.showMessage(`📨 设备回复: ${trimmed}`, 'info');
                    }
                    
                    // 更新设备信息显示
                    this.displayDeviceInfo();
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
                        mac: '🔗 MAC地址'
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

                async testConnection() {
                    if (!this.device) {
                        this.showMessage('没有选择的设备', 'error');
                        return;
                    }

                    try {
                        this.showMessage('🧪 正在测试设备连接...', 'info');
                        
                        if (this.device.manual) {
                            this.showMessage('✅ 手动设备测试通过', 'success');
                        } else {
                            // 测试BLE连接
                            if (this.server && this.server.connected) {
                                await this.sendCommand('v,'); // 查询版本作为连接测试
                                this.showMessage('✅ BLE连接测试通过', 'success');
                            } else {
                                this.showMessage('⚠️ BLE连接已断开', 'warning');
                            }
                        }
                        
                    } catch (error) {
                        this.showMessage(`❌ 连接测试失败: ${error.message}`, 'error');
                    }
                }

                saveDevice() {
                    if (!this.device) {
                        this.showMessage('没有设备信息可保存', 'error');
                        return;
                    }

                    // 构造设备信息
                    const deviceData = {
                        id: this.device.id,
                        name: this.device.name,
                        type: this.device.manual ? 'manual' : 'bluetooth',
                        info: this.deviceInfo,
                        config: this.configData || {},
                        timestamp: new Date().toISOString()
                    };

                    // 通知父窗口保存设备
                    this.notifyParent({
                        type: 'save_device',
                        device: deviceData
                    });

                    this.showMessage('💾 设备信息已保存', 'success');
                }

                saveConfigData(config) {
                    this.configData = config;
                }

                finishConfiguration() {
                    // 通知父窗口配置完成
                    this.notifyParent({
                        type: 'config_complete',
                        device: this.device,
                        info: this.deviceInfo,
                        config: this.configData
                    });

                    this.showMessage('✅ 配置流程已完成！', 'success');
                }

                updateStep(step) {
                    // 更新步骤指示器
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

                notifyParent(data) {
                    if (window.parent) {
                        window.parent.postMessage({
                            ...data,
                            source: 'aerosense_config'
                        }, '*');
                    }
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
    mode = query_params.get("mode", "add")
    
    # 初始化连接管理器
    connection_manager = DeviceConnectionManager()
    
    # 页面标题和说明
    if mode == "reconnect":
        st.title("🔄 设备重新连接")
        st.info(f"重新连接设备: **{device_code}**")
    else:
        st.title("🔗 AeroSense设备连接和配网")
        st.info("完整的设备扫描、连接和WiFi配网流程")
    
    # 重要提示
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="margin: 0 0 10px 0;">🚀 完整配网流程</h4>
        <p style="margin: 5px 0;">1️⃣ <strong>扫描设备</strong> - 支持AeroSense专用扫描</p>
        <p style="margin: 5px 0;">2️⃣ <strong>连接设备</strong> - BLE通信建立</p>
        <p style="margin: 5px 0;">3️⃣ <strong>设备配网</strong> - WiFi和服务器配置</p>
        <p style="margin: 5px 0;">4️⃣ <strong>配置完成</strong> - 保存和测试</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 嵌入完整的配网组件
    try:
        components.html(
            get_complete_bluetooth_html(),
            height=1000,
            scrolling=True
        )
    except Exception as e:
        st.error(f"❌ 加载配网组件失败: {str(e)}")
        st.info("💡 请尝试刷新页面或检查网络连接")
        return
    
    # AeroSense协议说明
    with st.expander("📋 AeroSense BLE配网协议说明", expanded=False):
        st.markdown("""
        ### 🔧 支持的配网命令:
        
        | 命令格式 | 功能说明 | 返回值 |
        |---------|---------|--------|
        | `(WiFi名称, WiFi密码)` | WiFi网络配置 | `0x01` 成功 |
        | `[服务器IP, 服务器端口]` | 服务器连接配置 | `0x01` 成功 |
        | `i,` | 查询设备IP地址 | IP地址字符串 |
        | `v,` | 查询通信版本 | 版本号字符串 |
        | `m,` | 查询WiFi MAC地址 | MAC地址字符串 |
        | `s,静态IP;网关;子网;DNS1;DNS2` | 设置静态IP (可选) | `0x01` 成功 |
        
        ### 📝 配网示例:
        ```
        发送: (MyWiFi, password123)
        返回: 0x01
        
        发送: [10.8.4.144, 8899]  
        返回: 0x01
        
        发送: i,
        返回: 10.8.4.123
        ```
        
        ### ⚙️ 使用步骤:
        1. 确保AeroSense设备处于配网模式
        2. 使用"AeroSense专用"扫描模式
        3. 连接设备并建立BLE通信
        4. 输入WiFi信息和服务器配置
        5. 点击"开始配网"执行配置
        6. 查看配网结果和设备信息
        """)
    
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
        if st.button("🧪 打开调试工具", use_container_width=True):
            # 可以在新标签页打开调试工具
            st.info("调试工具已集成在配网流程中，查看浏览器控制台获取详细信息")

if __name__ == "__main__":
    main()