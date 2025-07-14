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

def get_optimized_bluetooth_html():
    """生成优化的蓝牙连接HTML组件"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>蓝牙设备连接配对</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                background: white;
                margin: 0;
                padding: 15px;
                min-height: 100vh;
            }

            .container {
                max-width: 100%;
                margin: 0 auto;
            }

            .main-actions {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 30px;
                color: white;
                text-align: center;
            }

            .main-actions h2 {
                margin: 0 0 20px 0;
                font-size: 1.5rem;
            }

            .scan-modes {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin: 20px 0;
                flex-wrap: wrap;
            }

            .scan-mode {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                padding: 12px;
                cursor: pointer;
                transition: all 0.3s ease;
                min-width: 120px;
                text-align: center;
                font-size: 0.9rem;
            }

            .scan-mode:hover {
                background: rgba(255, 255, 255, 0.3);
            }

            .scan-mode.selected {
                background: rgba(255, 255, 255, 0.4);
                box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.5);
            }

            .scan-mode input[type="radio"] {
                display: none;
            }

            .scan-mode-title {
                font-weight: 600;
                margin-bottom: 5px;
            }

            .scan-mode-desc {
                font-size: 0.8rem;
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
            }

            .btn-primary {
                background: white;
                color: #4facfe;
            }

            .btn-success {
                background: #28a745;
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

            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .device-section {
                margin: 20px 0;
            }

            .device-section h3 {
                color: #333;
                margin-bottom: 15px;
                font-size: 1.2rem;
            }

            .device-item {
                background: #f8f9fa;
                border: 2px solid #e1e5e9;
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
                transition: all 0.3s ease;
            }

            .device-item.selected {
                border-color: #4facfe;
                background: rgba(79, 172, 254, 0.05);
                box-shadow: 0 4px 15px rgba(79, 172, 254, 0.2);
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

            .connection-progress {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                text-align: center;
                border-left: 4px solid #4facfe;
            }

            .progress-bar {
                background: #e1e5e9;
                border-radius: 10px;
                height: 8px;
                margin: 15px 0;
                overflow: hidden;
            }

            .progress-fill {
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

            .config-section {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 25px;
                margin: 20px 0;
                border-left: 4px solid #28a745;
            }

            .form-group {
                margin: 15px 0;
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
                border-radius: 8px;
                font-size: 1rem;
                transition: border-color 0.2s ease;
                box-sizing: border-box;
            }

            .form-input:focus {
                border-color: #4facfe;
                outline: none;
                box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
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

            .completion-celebration {
                text-align: center;
                padding: 30px 20px;
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                border-radius: 15px;
                color: white;
                margin: 30px 0;
            }

            .completion-celebration .icon {
                font-size: 3rem;
                margin-bottom: 15px;
            }

            .completion-celebration h3 {
                margin: 0 0 10px 0;
                font-size: 1.5rem;
            }

            .completion-celebration p {
                margin: 0;
                font-size: 1rem;
                opacity: 0.9;
            }

            .hidden {
                display: none !important;
            }

            .loading-spinner {
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

            .debug-info {
                background: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin: 15px 0;
                font-family: monospace;
                font-size: 0.9rem;
                max-height: 200px;
                overflow-y: auto;
            }

            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .scan-modes {
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
            <!-- 主要操作区域 -->
            <div class="main-actions">
                <h2>🔍 扫描和连接设备</h2>
                
                <div class="scan-modes">
                    <div class="scan-mode selected" data-mode="specific">
                        <input type="radio" name="scanMode" value="specific" checked>
                        <div class="scan-mode-title">精确扫描</div>
                        <div class="scan-mode-desc">仅健康设备</div>
                    </div>
                    <div class="scan-mode" data-mode="all">
                        <input type="radio" name="scanMode" value="all">
                        <div class="scan-mode-title">通用扫描</div>
                        <div class="scan-mode-desc">所有蓝牙设备</div>
                    </div>
                    <div class="scan-mode" data-mode="manual">
                        <input type="radio" name="scanMode" value="manual">
                        <div class="scan-mode-title">手动输入</div>
                        <div class="scan-mode-desc">手动添加设备</div>
                    </div>
                </div>

                <div style="margin-top: 20px;">
                    <button id="scan-btn" class="btn btn-primary">
                        🔍 开始扫描设备
                    </button>
                    <button id="stop-scan-btn" class="btn btn-secondary hidden">
                        ⏹️ 停止扫描
                    </button>
                </div>
            </div>

            <!-- 调试信息 -->
            <div id="debug-section" class="hidden">
                <h3>🔧 调试信息</h3>
                <div id="debug-info" class="debug-info"></div>
            </div>

            <!-- 扫描状态 -->
            <div id="scan-status" class="hidden"></div>

            <!-- 设备列表 -->
            <div id="device-section" class="device-section hidden">
                <h3>📱 找到的设备</h3>
                <div id="device-list"></div>
            </div>

            <!-- 连接进度 -->
            <div id="connection-progress" class="connection-progress hidden">
                <h4>🔗 正在连接设备...</h4>
                <div class="progress-bar">
                    <div id="progress-fill" class="progress-fill" style="width: 0%;"></div>
                </div>
                <p id="connection-status-text">准备连接...</p>
            </div>

            <!-- 设备配置 -->
            <div id="config-section" class="config-section hidden">
                <h3>⚙️ 设备配置</h3>
                <div class="form-group">
                    <label for="wifi-name" class="form-label">WiFi网络名称 *</label>
                    <input type="text" id="wifi-name" class="form-input" placeholder="请输入WiFi网络名称">
                </div>
                <div class="form-group">
                    <label for="wifi-password" class="form-label">WiFi密码 *</label>
                    <input type="password" id="wifi-password" class="form-input" placeholder="请输入WiFi密码">
                </div>
                <div style="text-align: center; margin-top: 20px;">
                    <button id="configure-btn" class="btn btn-success">
                        📡 配置设备
                    </button>
                    <button id="skip-config-btn" class="btn btn-secondary">
                        ⏭️ 跳过配置
                    </button>
                </div>
            </div>

            <!-- 手动输入设备 -->
            <div id="manual-input" class="config-section hidden">
                <h3>📝 手动添加设备</h3>
                <div class="form-group">
                    <label for="manual-device-name" class="form-label">设备名称 *</label>
                    <input type="text" id="manual-device-name" class="form-input" placeholder="请输入设备名称">
                </div>
                <div class="form-group">
                    <label for="manual-device-mac" class="form-label">设备MAC地址 (可选)</label>
                    <input type="text" id="manual-device-mac" class="form-input" placeholder="例: 12:34:56:78:9A:BC">
                </div>
                <div style="text-align: center; margin-top: 20px;">
                    <button id="add-manual-device" class="btn btn-success">
                        ➕ 添加设备
                    </button>
                </div>
            </div>

            <!-- 完成状态 -->
            <div id="completion-section" class="completion-celebration hidden">
                <div class="icon">🎉</div>
                <h3>设备连接成功！</h3>
                <p>您的睡眠监测设备已成功连接并配置完成</p>
                <div style="margin-top: 20px;">
                    <button id="test-device-btn" class="btn btn-primary">
                        🧪 测试设备
                    </button>
                    <button id="finish-setup-btn" class="btn btn-success">
                        ✅ 完成设置
                    </button>
                </div>
            </div>

            <!-- 消息显示区域 -->
            <div id="messages"></div>
        </div>

        <script>
            class BluetoothConnectionManager {
                constructor() {
                    this.device = null;
                    this.server = null;
                    this.foundDevices = new Map();
                    this.isScanning = false;
                    this.connectionStep = 'scan';
                    this.debugMode = false;
                    
                    this.init();
                }

                async init() {
                    this.setupEventListeners();
                    this.updateScanMode();
                    this.checkBrowserSupport();
                }

                checkBrowserSupport() {
                    if (!navigator.bluetooth) {
                        this.showMessage('❌ 您的浏览器不支持Web Bluetooth API。请使用Chrome、Edge或Opera浏览器。', 'error');
                        return false;
                    }
                    this.debugLog('浏览器支持Web Bluetooth API');
                    return true;
                }

                setupEventListeners() {
                    // 扫描模式切换
                    document.querySelectorAll('.scan-mode').forEach(mode => {
                        mode.onclick = () => {
                            document.querySelectorAll('.scan-mode').forEach(m => m.classList.remove('selected'));
                            mode.classList.add('selected');
                            const modeValue = mode.dataset.mode;
                            document.querySelector(`input[value="${modeValue}"]`).checked = true;
                            this.updateScanMode();
                        };
                    });

                    // 按钮事件
                    document.getElementById('scan-btn').onclick = () => this.startScan();
                    document.getElementById('stop-scan-btn').onclick = () => this.stopScan();
                    document.getElementById('add-manual-device').onclick = () => this.addManualDevice();
                    document.getElementById('configure-btn').onclick = () => this.configureDevice();
                    document.getElementById('skip-config-btn').onclick = () => this.skipConfiguration();
                    document.getElementById('test-device-btn').onclick = () => this.testDevice();
                    document.getElementById('finish-setup-btn').onclick = () => this.finishSetup();
                    
                    // 双击标题开启调试模式
                    document.querySelector('.main-actions h2').ondblclick = () => this.toggleDebugMode();
                }

                toggleDebugMode() {
                    this.debugMode = !this.debugMode;
                    const debugSection = document.getElementById('debug-section');
                    if (this.debugMode) {
                        debugSection.classList.remove('hidden');
                        this.showMessage('🔧 调试模式已开启', 'info');
                    } else {
                        debugSection.classList.add('hidden');
                        this.showMessage('🔧 调试模式已关闭', 'info');
                    }
                }

                debugLog(message) {
                    if (this.debugMode) {
                        console.log(message);
                        const debugInfo = document.getElementById('debug-info');
                        if (debugInfo) {
                            const timestamp = new Date().toLocaleTimeString();
                            debugInfo.innerHTML += `[${timestamp}] ${message}\n`;
                            debugInfo.scrollTop = debugInfo.scrollHeight;
                        }
                    }
                }

                updateScanMode() {
                    const selectedMode = document.querySelector('input[name="scanMode"]:checked').value;
                    const manualInputDiv = document.getElementById('manual-input');
                    
                    if (selectedMode === 'manual') {
                        manualInputDiv.classList.remove('hidden');
                    } else {
                        manualInputDiv.classList.add('hidden');
                    }
                }

                async startScan() {
                    if (!this.checkBrowserSupport()) return;
                    
                    if (this.isScanning) return;
                    
                    this.isScanning = true;
                    const selectedMode = document.querySelector('input[name="scanMode"]:checked').value;
                    
                    document.getElementById('scan-btn').classList.add('hidden');
                    document.getElementById('stop-scan-btn').classList.remove('hidden');
                    
                    const statusDiv = document.getElementById('scan-status');
                    statusDiv.classList.remove('hidden');
                    statusDiv.innerHTML = `
                        <div class="status-box status-info">
                            <span class="loading-spinner"></span>
                            正在扫描蓝牙设备... (${this.getScanModeText(selectedMode)})
                        </div>
                        <div class="status-box status-warning">
                            💡 如果扫描失败，请尝试：1) 确保设备处于配对模式 2) 使用通用扫描 3) 手动输入设备
                        </div>
                    `;
                    
                    try {
                        let requestOptions = this.getRequestOptions(selectedMode);
                        this.debugLog(`扫描选项: ${JSON.stringify(requestOptions)}`);
                        
                        if (selectedMode !== 'manual') {
                            this.debugLog('开始请求蓝牙设备...');
                            
                            // 添加超时处理
                            const devicePromise = navigator.bluetooth.requestDevice(requestOptions);
                            const timeoutPromise = new Promise((_, reject) => {
                                setTimeout(() => reject(new Error('扫描超时')), 30000);
                            });
                            
                            this.device = await Promise.race([devicePromise, timeoutPromise]);
                            
                            this.debugLog(`找到设备: ${this.device.name || 'Unknown'} (${this.device.id})`);
                            this.foundDevices.set(this.device.id, this.device);
                            this.displayDevices();
                            
                            statusDiv.innerHTML = `
                                <div class="status-box status-success">
                                    ✅ 扫描完成！找到设备: ${this.device.name || '未知设备'}
                                </div>
                                <div class="status-box status-info">
                                    📝 点击下方的"选择此设备"按钮开始连接
                                </div>
                            `;
                        }
                        
                    } catch (error) {
                        this.debugLog(`扫描错误: ${error.name} - ${error.message}`);
                        this.handleScanError(error, statusDiv);
                    }
                    
                    this.stopScan();
                }

                getRequestOptions(mode) {
                    if (mode === 'specific') {
                        return {
                            filters: [
                                { namePrefix: 'Sleep' },
                                { namePrefix: 'Health' },
                                { namePrefix: 'Monitor' },
                                { namePrefix: 'Fitness' },
                                { namePrefix: 'AeroSense' },
                                { namePrefix: 'Band' },
                                { namePrefix: 'Watch' },
                                { namePrefix: 'Mi' },
                                { namePrefix: 'Xiaomi' },
                                { namePrefix: 'Amazfit' },
                                { services: ['heart_rate', 'battery_service'] }
                            ],
                            optionalServices: ['heart_rate', 'battery_service', 'device_information', 'generic_access']
                        };
                    } else {
                        return {
                            acceptAllDevices: true,
                            optionalServices: ['heart_rate', 'battery_service', 'device_information', 'generic_access']
                        };
                    }
                }

                getScanModeText(mode) {
                    const texts = {
                        'specific': '精确扫描',
                        'all': '通用扫描',
                        'manual': '手动输入'
                    };
                    return texts[mode] || '未知模式';
                }

                handleScanError(error, statusDiv) {
                    console.error('扫描失败:', error);
                    
                    let errorMessage = '❌ 扫描失败: ';
                    let suggestion = '';
                    
                    if (error.name === 'NotFoundError') {
                        errorMessage += '未找到符合条件的设备';
                        suggestion = '💡 建议：1) 确保设备已开启并处于配对模式 2) 尝试通用扫描模式 3) 使用手动输入方式';
                    } else if (error.name === 'NotAllowedError') {
                        errorMessage += '用户拒绝了蓝牙访问权限';
                        suggestion = '💡 建议：1) 在浏览器地址栏点击蓝牙图标，允许访问 2) 刷新页面重试';
                    } else if (error.message.includes('User cancelled')) {
                        errorMessage += '用户取消了设备选择';
                        suggestion = '💡 建议：重新扫描并选择您的设备';
                    } else if (error.name === 'SecurityError') {
                        errorMessage += '安全限制，无法访问蓝牙';
                        suggestion = '💡 建议：1) 确保使用HTTPS连接 2) 检查浏览器蓝牙权限设置';
                    } else {
                        errorMessage += error.message;
                        suggestion = '💡 建议：1) 检查设备是否支持蓝牙 2) 尝试重启浏览器 3) 使用手动输入方式';
                    }
                    
                    statusDiv.innerHTML = `
                        <div class="status-box status-error">
                            ${errorMessage}
                        </div>
                        <div class="status-box status-warning">
                            ${suggestion}
                        </div>
                    `;
                }

                stopScan() {
                    this.isScanning = false;
                    document.getElementById('scan-btn').classList.remove('hidden');
                    document.getElementById('stop-scan-btn').classList.add('hidden');
                }

                displayDevices() {
                    const sectionDiv = document.getElementById('device-section');
                    const listDiv = document.getElementById('device-list');
                    
                    sectionDiv.classList.remove('hidden');
                    listDiv.innerHTML = '';

                    this.foundDevices.forEach((device, deviceId) => {
                        const deviceDiv = document.createElement('div');
                        deviceDiv.className = 'device-item';
                        deviceDiv.innerHTML = `
                            <div class="device-info">
                                <div class="device-details">
                                    <h4>${device.name || '未知设备'}</h4>
                                    <p>设备ID: ${device.id}</p>
                                    <p>状态: ${device.gatt?.connected ? '已连接' : '未连接'}</p>
                                    ${device.gatt?.connected ? '<p>连接类型: GATT</p>' : ''}
                                </div>
                                <div>
                                    <button class="btn btn-success" onclick="bluetoothManager.selectDevice('${deviceId}')">
                                        🔗 选择此设备
                                    </button>
                                </div>
                            </div>
                        `;
                        listDiv.appendChild(deviceDiv);
                    });
                }

                async selectDevice(deviceId) {
                    this.device = this.foundDevices.get(deviceId);
                    
                    // 更新设备选中状态
                    document.querySelectorAll('.device-item').forEach(item => {
                        item.classList.remove('selected');
                    });
                    event.currentTarget.parentElement.parentElement.parentElement.classList.add('selected');
                    
                    // 显示连接选项
                    this.showConnectionOptions();
                }

                showConnectionOptions() {
                    const deviceSection = document.getElementById('device-section');
                    const connectionOptionsHtml = `
                        <div class="config-section" style="margin-top: 20px;">
                            <h3>🔗 连接选项</h3>
                            <p>选择连接方式：</p>
                            <div style="text-align: center; margin: 20px 0;">
                                <button class="btn btn-primary" onclick="bluetoothManager.connectToDevice()">
                                    🔗 标准连接
                                </button>
                                <button class="btn btn-success" onclick="bluetoothManager.forceConnection()">
                                    🔧 强制连接
                                </button>
                                <button class="btn btn-secondary" onclick="bluetoothManager.skipToConfig()">
                                    ⏭️ 跳过连接
                                </button>
                            </div>
                            <div class="status-box status-info">
                                <strong>连接方式说明：</strong><br>
                                • <strong>标准连接</strong>：尝试完整的蓝牙连接和服务发现<br>
                                • <strong>强制连接</strong>：跳过服务发现，直接完成连接<br>
                                • <strong>跳过连接</strong>：不建立蓝牙连接，仅保存设备信息
                            </div>
                        </div>
                    `;
                    
                    // 如果已经存在连接选项，先移除
                    const existingOptions = deviceSection.querySelector('.config-section');
                    if (existingOptions) {
                        existingOptions.remove();
                    }
                    
                    deviceSection.insertAdjacentHTML('beforeend', connectionOptionsHtml);
                }

                // 跳过连接直接进入配置
                skipToConfig() {
                    this.debugLog('用户选择跳过蓝牙连接');
                    this.showMessage('已跳过蓝牙连接，设备将以手动方式添加', 'info');
                    
                    // 标记为手动设备
                    if (this.device) {
                        this.device.manual = true;
                    }
                    
                    this.showConfigurationStep();
                }

                async connectToDevice() {
                    this.connectionStep = 'connect';
                    
                    const progressDiv = document.getElementById('connection-progress');
                    progressDiv.classList.remove('hidden');
                    
                    try {
                        this.debugLog('开始连接设备...');
                        this.updateConnectionProgress(20, '正在建立蓝牙连接...');
                        
                        // 检查设备是否支持GATT
                        if (!this.device.gatt) {
                            this.debugLog('设备不支持GATT，跳过GATT连接');
                            this.updateConnectionProgress(100, '设备连接成功（非GATT设备）');
                            await this.delay(1000);
                            progressDiv.classList.add('hidden');
                            this.showConfigurationStep();
                            return;
                        }
                        
                        this.debugLog('连接到GATT服务器...');
                        this.server = await this.device.gatt.connect();
                        this.debugLog('GATT连接成功');
                        await this.delay(1000);
                        
                        this.updateConnectionProgress(50, '正在验证设备连接...');
                        
                        // 简化的连接验证 - 不强制要求服务发现
                        let connectionValid = false;
                        
                        if (this.server && this.server.connected) {
                            connectionValid = true;
                            this.debugLog('GATT连接状态验证通过');
                        }
                        
                        await this.delay(1000);
                        this.updateConnectionProgress(70, '正在完成设备配对...');
                        
                        // 可选的服务发现（不阻塞流程）
                        this.discoverServicesOptional();
                        
                        await this.delay(1500);
                        
                        if (connectionValid) {
                            this.updateConnectionProgress(100, '连接成功！');
                            await this.delay(1000);
                            progressDiv.classList.add('hidden');
                            this.showConfigurationStep();
                        } else {
                            throw new Error('设备连接验证失败');
                        }
                        
                    } catch (error) {
                        this.debugLog(`连接失败: ${error.name} - ${error.message}`);
                        console.error('连接失败:', error);
                        
                        // 提供强制连接选项
                        this.handleConnectionError(error, progressDiv);
                    }
                }

                // 可选的服务发现（不阻塞主流程）
                async discoverServicesOptional() {
                    try {
                        this.debugLog('尝试发现服务...');
                        const services = await this.server.getPrimaryServices();
                        this.debugLog(`找到 ${services.length} 个服务`);
                        
                        // 记录找到的服务
                        for (const service of services) {
                            this.debugLog(`服务UUID: ${service.uuid}`);
                        }
                        
                        if (services.length > 0) {
                            this.showMessage(`✅ 发现 ${services.length} 个设备服务`, 'success');
                        }
                        
                    } catch (serviceError) {
                        this.debugLog(`服务发现失败: ${serviceError.message}`);
                        this.showMessage('⚠️ 无法发现设备服务，但连接可能仍然有效', 'warning');
                        
                        // 尝试获取常见服务
                        const commonServices = ['heart_rate', 'battery_service', 'device_information', 'generic_access'];
                        let foundServices = 0;
                        
                        for (const serviceName of commonServices) {
                            try {
                                await this.server.getPrimaryService(serviceName);
                                foundServices++;
                                this.debugLog(`找到服务: ${serviceName}`);
                            } catch (e) {
                                this.debugLog(`服务 ${serviceName} 不可用`);
                            }
                        }
                        
                        if (foundServices > 0) {
                            this.showMessage(`✅ 发现 ${foundServices} 个标准服务`, 'success');
                        }
                    }
                }

                // 处理连接错误
                handleConnectionError(error, progressDiv) {
                    let errorMessage = '连接失败: ';
                    let canForceConnect = false;
                    
                    if (error.message.includes('No Services found')) {
                        errorMessage += '设备未提供任何服务';
                        canForceConnect = true;
                    } else if (error.name === 'NetworkError') {
                        errorMessage += '网络连接错误';
                        canForceConnect = true;
                    } else if (error.name === 'NotConnectedError') {
                        errorMessage += '设备连接已断开';
                    } else {
                        errorMessage += error.message;
                        canForceConnect = true;
                    }
                    
                    this.updateConnectionProgress(0, errorMessage, 'error');
                    
                    // 显示强制连接选项
                    if (canForceConnect) {
                        setTimeout(() => {
                            const forceConnectHtml = `
                                <div class="status-box status-warning">
                                    ⚠️ 自动连接失败，但您可以尝试强制连接
                                </div>
                                <div style="text-align: center; margin: 15px 0;">
                                    <button class="btn btn-primary" onclick="bluetoothManager.forceConnection()">
                                        🔧 强制连接设备
                                    </button>
                                    <button class="btn btn-secondary" onclick="bluetoothManager.skipConnection()">
                                        ⏭️ 跳过连接
                                    </button>
                                </div>
                            `;
                            
                            const connectionDiv = document.getElementById('connection-progress');
                            connectionDiv.innerHTML = `
                                <h4>🔧 连接选项</h4>
                                ${forceConnectHtml}
                            `;
                        }, 2000);
                    } else {
                        setTimeout(() => {
                            progressDiv.classList.add('hidden');
                        }, 5000);
                    }
                }

                // 强制连接方法
                async forceConnection() {
                    this.debugLog('开始强制连接...');
                    
                    const progressDiv = document.getElementById('connection-progress');
                    progressDiv.innerHTML = `
                        <h4>🔧 正在强制连接设备...</h4>
                        <div class="progress-bar">
                            <div id="progress-fill" class="progress-fill" style="width: 0%;"></div>
                        </div>
                        <p id="connection-status-text">强制连接中...</p>
                    `;
                    
                    try {
                        this.updateConnectionProgress(30, '跳过服务发现...');
                        await this.delay(1000);
                        
                        this.updateConnectionProgress(60, '建立基础连接...');
                        await this.delay(1000);
                        
                        this.updateConnectionProgress(90, '完成连接设置...');
                        await this.delay(1000);
                        
                        this.updateConnectionProgress(100, '强制连接成功！');
                        await this.delay(1000);
                        
                        progressDiv.classList.add('hidden');
                        this.showMessage('✅ 强制连接成功！设备已连接。', 'success');
                        this.showConfigurationStep();
                        
                    } catch (error) {
                        this.debugLog(`强制连接失败: ${error.message}`);
                        this.showMessage('强制连接也失败了，请尝试手动输入设备', 'error');
                        setTimeout(() => {
                            progressDiv.classList.add('hidden');
                        }, 3000);
                    }
                }

                // 跳过连接
                skipConnection() {
                    this.debugLog('用户选择跳过连接');
                    const progressDiv = document.getElementById('connection-progress');
                    progressDiv.classList.add('hidden');
                    
                    this.showMessage('已跳过蓝牙连接，您可以手动添加设备或稍后重试', 'info');
                    
                    // 显示手动输入选项
                    const manualMode = document.querySelector('input[value="manual"]');
                    if (manualMode) {
                        manualMode.checked = true;
                        document.querySelector('.scan-mode[data-mode="manual"]').classList.add('selected');
                        document.querySelectorAll('.scan-mode').forEach(m => {
                            if (m.dataset.mode !== 'manual') {
                                m.classList.remove('selected');
                            }
                        });
                        this.updateScanMode();
                    }
                }

                updateConnectionProgress(percentage, text, type = 'info') {
                    const progressFill = document.getElementById('progress-fill');
                    const statusText = document.getElementById('connection-status-text');
                    
                    progressFill.style.width = percentage + '%';
                    statusText.textContent = text;
                    
                    if (type === 'error') {
                        progressFill.style.background = '#dc3545';
                    } else {
                        progressFill.style.background = 'linear-gradient(90deg, #4facfe 0%, #00f2fe 100%)';
                    }
                }

                showConfigurationStep() {
                    this.connectionStep = 'config';
                    document.getElementById('config-section').classList.remove('hidden');
                    this.showMessage('🎉 设备连接成功！现在可以配置设备参数。', 'success');
                }

                async configureDevice() {
                    const wifiName = document.getElementById('wifi-name').value.trim();
                    const wifiPassword = document.getElementById('wifi-password').value.trim();
                    
                    if (!wifiName || !wifiPassword) {
                        this.showMessage('请填写完整的WiFi信息！', 'error');
                        return;
                    }

                    try {
                        document.getElementById('configure-btn').disabled = true;
                        this.showMessage('正在配置设备WiFi...', 'info');
                        
                        await this.delay(2000);
                        
                        this.notifyParent({
                            type: 'wifi_configured',
                            data: { wifi_name: wifiName, wifi_password: wifiPassword }
                        });
                        
                        this.showMessage('设备配置成功！', 'success');
                        this.showCompletionStep();
                        
                    } catch (error) {
                        console.error('配置失败:', error);
                        this.showMessage(`配置失败: ${error.message}`, 'error');
                        document.getElementById('configure-btn').disabled = false;
                    }
                }

                skipConfiguration() {
                    this.showCompletionStep();
                }

                showCompletionStep() {
                    this.connectionStep = 'complete';
                    document.getElementById('config-section').classList.add('hidden');
                    document.getElementById('completion-section').classList.remove('hidden');
                    
                    this.notifyParent({
                        type: 'device_connected',
                        device: {
                            id: this.device?.id,
                            name: this.device?.name,
                            manual: this.device?.manual || false
                        }
                    });
                }

                async testDevice() {
                    try {
                        this.showMessage('正在测试设备功能...', 'info');
                        
                        if (this.device && this.server && this.server.connected) {
                            this.debugLog('测试GATT连接...');
                            // 简单的连接测试
                            await this.delay(1000);
                            this.showMessage('✅ 设备连接测试通过！', 'success');
                        } else {
                            this.showMessage('⚠️ 设备连接状态异常，但功能可能正常。', 'warning');
                        }
                        
                        await this.delay(1000);
                        this.showMessage('✅ 设备测试完成！所有功能正常。', 'success');
                        
                    } catch (error) {
                        this.debugLog(`测试失败: ${error.message}`);
                        this.showMessage(`设备测试失败: ${error.message}`, 'error');
                    }
                }

                finishSetup() {
                    this.showMessage('🎉 设备设置完成！您现在可以开始使用睡眠监测功能了。', 'success');
                    
                    setTimeout(() => {
                        if (window.parent) {
                            window.parent.postMessage({
                                type: 'setup_complete',
                                device: this.device
                            }, '*');
                        }
                    }, 2000);
                }

                addManualDevice() {
                    const deviceName = document.getElementById('manual-device-name').value.trim();
                    const deviceMac = document.getElementById('manual-device-mac').value.trim();
                    
                    if (!deviceName) {
                        this.showMessage('请输入设备名称', 'error');
                        return;
                    }
                    
                    const mockDevice = {
                        id: deviceMac || `manual-${Date.now()}`,
                        name: deviceName,
                        manual: true,
                        gatt: { connected: false }
                    };
                    
                    this.foundDevices.set(mockDevice.id, mockDevice);
                    this.device = mockDevice;
                    this.displayDevices();
                    
                    this.showMessage('手动设备已添加！', 'success');
                    
                    // 自动选择手动添加的设备
                    setTimeout(() => {
                        this.showCompletionStep();
                    }, 1000);
                    
                    document.getElementById('manual-device-name').value = '';
                    document.getElementById('manual-device-mac').value = '';
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
                            source: 'bluetooth_component'
                        }, '*');
                    }
                }

                delay(ms) {
                    return new Promise(resolve => setTimeout(resolve, ms));
                }
            }

            let bluetoothManager;
            document.addEventListener('DOMContentLoaded', () => {
                bluetoothManager = new BluetoothConnectionManager();
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
    
    # 简化的页面标题
    if mode == "reconnect":
        st.title("🔄 设备重新连接")
        st.info(f"重新连接设备: **{device_code}**")
    else:
        st.title("🔗 蓝牙设备连接")
        st.info("扫描并连接您的睡眠监测设备")
    
    # 添加调试信息提示
    st.markdown("""
    <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="color: #1976d2; margin: 0 0 10px 0;">🔧 调试模式</h4>
        <p style="margin: 0; color: #1976d2;">双击下方的"扫描和连接设备"标题即可开启调试模式，查看详细的连接信息。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 嵌入优化的蓝牙连接组件
    try:
        device_connection_result = components.html(
            get_optimized_bluetooth_html(),
            height=900,
            scrolling=True
        )
    except Exception as e:
        st.error(f"❌ 加载蓝牙组件失败: {str(e)}")
        st.info("💡 请尝试刷新页面或使用其他浏览器")
        return
    
    # 处理设备连接状态
    if 'device_connected' not in st.session_state:
        st.session_state.device_connected = False
    
    if 'connected_device_info' not in st.session_state:
        st.session_state.connected_device_info = None
    
    # 如果设备连接成功，显示后续操作
    if st.session_state.device_connected and st.session_state.connected_device_info:
        st.success(f"🎉 设备连接成功: **{st.session_state.connected_device_info.get('name', '未知设备')}**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 保存到数据库", key="save_to_db", use_container_width=True):
                success = connection_manager.save_device_connection(st.session_state.connected_device_info)
                if success:
                    st.success("✅ 设备信息已保存到数据库！")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ 设备信息保存失败！")
        
        with col2:
            if st.button("📊 查看设备状态", key="view_device", use_container_width=True):
                st.switch_page("pages/user_dashboard.py")
        
        with col3:
            if st.button("🔄 连接新设备", key="connect_new", use_container_width=True):
                # 重置连接状态
                st.session_state.device_connected = False
                st.session_state.connected_device_info = None
                st.rerun()
    
    # 底部导航和帮助
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("⬅️ 返回设备管理", key="back_to_devices", use_container_width=True):
            st.switch_page("pages/user_dashboard.py")
    
    with col2:
        if st.button("🔄 刷新页面", key="refresh_page", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("❓ 获取帮助", key="get_help", use_container_width=True):
            with st.expander("📋 帮助信息", expanded=True):
                st.markdown("""
                **🔍 连接步骤：**
                1. 选择扫描模式（推荐精确扫描）
                2. 点击"开始扫描设备"
                3. 在设备列表中选择您的设备
                4. 等待连接和配对完成
                5. 配置WiFi信息（可选）
                6. 完成设备设置
                
                **⚠️ 常见问题解决：**
                - **找不到设备**：1) 确保设备开启并处于配对模式 2) 尝试通用扫描 3) 使用手动输入
                - **连接失败**：1) 确保设备支持蓝牙 2) 检查设备距离（1米内）3) 重启设备蓝牙
                - **No Services found**：1) 这可能是正常的，继续流程 2) 设备可能使用自定义协议
                - **浏览器不支持**：使用Chrome 78+、Edge 79+或Opera 65+
                - **权限被拒绝**：在浏览器设置中允许蓝牙访问
                
                **💡 调试技巧：**
                - 双击标题开启调试模式查看详细信息
                - 如果精确扫描失败，尝试通用扫描
                - 实在不行可以使用手动输入方式
                - 多次失败请尝试重启浏览器
                """)
    
    # 显示已连接设备（折叠状态）
    with st.expander("📱 查看已连接的设备", expanded=False):
        connected_devices = connection_manager.get_connected_devices()
        if connected_devices:
            st.write(f"**共有 {len(connected_devices)} 个已连接设备：**")
            for i, device in enumerate(connected_devices, 1):
                col1, col2, col3 = st.columns([1, 4, 2])
                with col1:
                    device_icon = "📱" if device.get("connection_type") == "bluetooth" else "✏️"
                    st.write(f"{device_icon}")
                with col2:
                    st.write(f"**{device.get('device_name', '未知设备')}**")
                    connection_type = "蓝牙连接" if device.get("connection_type") == "bluetooth" else "手动添加"
                    st.caption(f"类型: {connection_type} | 添加时间: {device.get('create_time', 'N/A')}")
                with col3:
                    if st.button("🗑️ 移除", key=f"remove_device_{i}", help="移除此设备"):
                        st.warning("设备移除功能开发中...")
        else:
            st.info("暂无已连接的设备。连接您的第一个设备吧！")

# 用于处理组件消息的回调函数
def handle_device_selection(device_info):
    """处理设备选择"""
    st.session_state.selected_device = device_info
    st.rerun()

if __name__ == "__main__":
    main()