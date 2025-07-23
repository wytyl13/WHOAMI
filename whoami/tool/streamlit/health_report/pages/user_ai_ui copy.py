"""
优化的社区管理助手聊天界面
- 全屏布局，无滚轮
- 历史消息区域可滚动
- 输入框固定在底部
- 文件上传和发送功能
"""

import streamlit as st
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO

# 页面配置
st.set_page_config(
    page_title="社区管理助手",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def apply_fullscreen_styles():
    """应用全屏样式，消除滚轮"""
    st.markdown("""
    <style>
        /* 隐藏Streamlit默认元素 */
        .stDeployButton {display: none !important;}
        footer {display: none !important;}
        .stApp > header {display: none !important;}
        .stMainBlockContainer {padding: 0 !important;}
        
        /* 全屏布局 */
        .main .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
            height: 100vh !important;
            overflow: hidden !important;
        }
        
        /* 主容器 */
        .main-chat-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            background-color: #ffffff;
        }
        
        /* 历史消息区域 */
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #e1e5e9;
            display: flex;
            flex-direction: column;
        }
        
        /* 输入区域 - 固定在底部 */
        .input-section {
            flex-shrink: 0;
            background-color: #ffffff;
            border-top: 1px solid #e1e5e9;
            padding: 20px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            z-index: 100;
            position: relative;
        }
        
        /* 消息气泡样式 */
        .message-bubble {
            margin: 10px 0;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
            animation: fadeIn 0.3s ease-in;
        }
        
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            margin-right: 0;
        }
        
        .assistant-message {
            background-color: #f1f3f4;
            color: #333;
            margin-left: 0;
            margin-right: auto;
        }
        
        .system-message {
            background-color: #e8f5e8;
            color: #2d5a2d;
            margin-left: auto;
            margin-right: auto;
            text-align: center;
            max-width: 90%;
        }
        
        /* 文件附件样式 */
        .file-attachment {
            background-color: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 8px;
            padding: 8px 12px;
            margin: 5px 0;
            display: inline-block;
            font-size: 14px;
        }
        
        /* 输入区域布局 */
        .input-container {
            display: flex;
            align-items: center;
            gap: 12px;
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 8px;
            border: 1px solid #e1e5e9;
        }
        
        /* 上传按钮样式 */
        .upload-button {
            background-color: #f1f3f4;
            border: none;
            border-radius: 8px;
            padding: 10px;
            cursor: pointer;
            font-size: 18px;
            color: #5f6368;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s;
        }
        
        .upload-button:hover {
            background-color: #e8eaed;
        }
        
        /* 发送按钮样式 */
        .send-button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 16px;
            cursor: pointer;
            font-size: 16px;
            height: 44px;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: background-color 0.2s;
            white-space: nowrap;
        }
        
        .send-button:hover {
            background-color: #0056b3;
        }
        
        .send-button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        
        /* 文本区域样式 */
        .stTextArea {
            flex: 1;
        }
        
        .stTextArea > div > div > textarea {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            resize: none !important;
            background-color: transparent !important;
            font-size: 16px !important;
            padding: 8px 12px !important;
            min-height: 44px !important;
            max-height: 120px !important;
        }
        
        /* 文件预览区域 */
        .file-preview {
            margin-bottom: 12px;
            padding: 12px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e1e5e9;
        }
        
        .file-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background-color: #e3f2fd;
            border-radius: 6px;
            margin: 4px 0;
            border: 1px solid #2196f3;
        }
        
        .file-info {
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
        }
        
        .remove-file {
            color: #f44336;
            cursor: pointer;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        
        .remove-file:hover {
            background-color: #ffebee;
        }
        
        /* 空状态样式 */
        .empty-state {
            text-align: center;
            color: #666;
            padding: 60px 20px;
            font-size: 16px;
        }
        
        /* 用户信息栏 */
        .user-info {
            background-color: #e3f2fd;
            padding: 15px 20px;
            border-bottom: 1px solid #2196f3;
            color: #1976d2;
            flex-shrink: 0;
        }
        
        /* 动画效果 */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* 滚动条样式 */
        .messages-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .messages-container::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        .messages-container::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        .messages-container::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        /* 隐藏文件上传器 */
        .stFileUploader {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_files" not in st.session_state:
        st.session_state.current_files = []
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user_001"

def get_file_icon(file_type: str) -> str:
    """根据文件类型返回对应图标"""
    if file_type.startswith('image/'):
        return "🖼️"
    elif file_type == 'application/pdf':
        return "📄"
    elif file_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
        return "📊"
    elif file_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        return "📝"
    elif file_type.startswith('text/'):
        return "📋"
    else:
        return "📎"

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def add_message(role: str, content: str, files: List[Dict] = None):
    """添加消息到历史记录"""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "files": files or []
    }
    st.session_state.messages.append(message)

def display_messages():
    """显示历史消息"""
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size: 48px; margin-bottom: 20px;">💬</div>
            <div>欢迎使用社区管理助手</div>
            <div style="margin-top: 8px; font-size: 14px; color: #888;">
                您可以发送消息或上传文件开始对话
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    for message in st.session_state.messages:
        # 确定消息样式类
        if message["role"] == "user":
            bubble_class = "user-message"
        elif message["role"] == "assistant":
            bubble_class = "assistant-message"
        else:
            bubble_class = "system-message"
        
        # 显示消息内容
        message_html = f"""
        <div class="message-bubble {bubble_class}">
            <div>{message["content"]}</div>
        """
        
        # 显示附件
        if message.get("files"):
            message_html += '<div style="margin-top: 8px;">'
            for file_info in message["files"]:
                icon = get_file_icon(file_info.get("type", ""))
                name = file_info.get("name", "unknown")
                size = format_file_size(file_info.get("size", 0))
                message_html += f"""
                <div class="file-attachment">
                    {icon} {name} ({size})
                </div>
                """
            message_html += '</div>'
        
        # 显示时间戳
        message_html += f"""
            <div style="font-size: 12px; color: rgba(255,255,255,0.7) if '{bubble_class}' == 'user-message' else rgba(0,0,0,0.5); margin-top: 4px;">
                {message["timestamp"]}
            </div>
        </div>
        """
        
        st.markdown(message_html, unsafe_allow_html=True)

def display_file_preview():
    """显示文件预览"""
    if not st.session_state.current_files:
        return
    
    st.markdown("""
    <div class="file-preview">
        <div style="font-weight: bold; margin-bottom: 8px; color: #333;">
            📎 待发送文件：
        </div>
    """, unsafe_allow_html=True)
    
    for i, file_info in enumerate(st.session_state.current_files):
        icon = get_file_icon(file_info.get("type", ""))
        name = file_info.get("name", "unknown")
        size = format_file_size(file_info.get("size", 0))
        
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.markdown(f"""
            <div class="file-item">
                <div class="file-info">
                    <span>{icon}</span>
                    <span><strong>{name}</strong></span>
                    <span style="color: #666;">({size})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("✕", key=f"remove_file_{i}", help="删除文件"):
                st.session_state.current_files.pop(i)
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def handle_file_upload(uploaded_files) -> List[Dict]:
    """
    处理文件上传的函数 - 供您修改
    
    Args:
        uploaded_files: Streamlit上传的文件对象列表
    
    Returns:
        List[Dict]: 处理后的文件信息列表
    
    TODO: 在这里添加您的文件处理逻辑
    - 保存文件到指定目录
    - 文件类型验证
    - 病毒扫描
    - 文件大小限制
    - 生成文件ID/路径
    """
    processed_files = []
    
    for uploaded_file in uploaded_files:
        # 这里是您可以修改的文件处理逻辑
        file_info = {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "type": uploaded_file.type,
            "content": uploaded_file.getvalue(),  # 文件内容
            "file_id": f"file_{int(time.time())}_{uploaded_file.name}",  # 生成文件ID
            # 在这里添加更多您需要的字段
        }
        processed_files.append(file_info)
        
        # 示例：保存文件到本地（您可以修改为其他逻辑）
        # save_path = f"uploads/{file_info['file_id']}"
        # with open(save_path, "wb") as f:
        #     f.write(uploaded_file.getvalue())
    
    return processed_files

def handle_send_message(user_input: str, files: List[Dict]) -> str:
    """
    处理发送消息的函数 - 供您修改
    
    Args:
        user_input: 用户输入的文本
        files: 上传的文件信息列表
    
    Returns:
        str: AI助手的回复内容
    
    TODO: 在这里添加您的消息处理逻辑
    - 调用AI模型API
    - 处理文件内容
    - 生成回复
    - 保存到数据库
    """
    # 这里是您可以修改的消息处理逻辑
    
    # 示例回复逻辑（请根据需要修改）
    if files:
        file_names = [f["name"] for f in files]
        response = f"我已收到您的消息和以下文件：{', '.join(file_names)}。\n\n"
        response += f"消息内容：{user_input}\n\n"
        response += "正在处理您的文件，请稍等..."
        
        # 在这里添加文件处理逻辑
        # process_files(files)
        
    else:
        response = f"收到您的消息：{user_input}\n\n这是一个固定的回复，请在 handle_send_message 函数中修改为您的实际处理逻辑。"
    
    # 在这里添加更多处理逻辑
    # - 调用AI模型
    # - 保存到数据库
    # - 发送通知等
    
    return response

def render_chat_interface():
    """渲染聊天界面"""
    # 应用样式
    apply_fullscreen_styles()
    
    # 主容器
    st.markdown('<div class="main-chat-container">', unsafe_allow_html=True)
    
    # 用户信息栏
    username = st.query_params.get("username", "访客用户")
    st.markdown(f"""
    <div class="user-info">
        <strong>👋 欢迎，{username}！</strong>
        <span style="margin-left: 20px;">社区管理助手已就绪</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 历史消息区域
    st.markdown('<div class="messages-container" id="messages-container">', unsafe_allow_html=True)
    display_messages()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 输入区域
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # 文件预览
    display_file_preview()
    
    # 隐藏的文件上传器
    uploaded_files = st.file_uploader(
        "上传文件",
        type=['txt', 'doc', 'docx', 'pdf', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'],
        accept_multiple_files=True,
        key=f"hidden_uploader_{st.session_state.file_uploader_key}"
    )
    
    # 处理新上传的文件
    if uploaded_files:
        new_files = handle_file_upload(uploaded_files)
        for file_info in new_files:
            # 检查是否已存在
            if not any(f["name"] == file_info["name"] for f in st.session_state.current_files):
                st.session_state.current_files.append(file_info)
        
        # 清空上传器
        st.session_state.file_uploader_key += 1
        st.rerun()
    
    # 输入框区域
    col1, col2, col3 = st.columns([1, 8, 1.5])
    
    with col1:
        # 上传按钮
        if st.button("📎", key="upload_btn", help="上传文件"):
            st.session_state.file_uploader_key += 1
            st.rerun()
    
    with col2:
        # 文本输入 - 修复高度为68px（Streamlit最小要求）
        user_input = st.text_area(
            "输入消息",
            placeholder="请输入您的消息...",
            height=68,  # 修改：从60改为68，满足Streamlit最小高度要求
            label_visibility="collapsed",
            key="message_input"
        )
    
    with col3:
        # 发送按钮
        can_send = bool(user_input.strip() or st.session_state.current_files)
        
        if st.button("🚀 发送", disabled=not can_send, key="send_btn", type="primary"):
            # 准备消息内容
            message_content = user_input.strip() or "发送了文件"
            current_files = st.session_state.current_files.copy()
            
            # 添加用户消息
            add_message("user", message_content, current_files)
            
            # 处理消息并获取回复
            assistant_response = handle_send_message(user_input, current_files)
            
            # 添加助手回复
            add_message("assistant", assistant_response)
            
            # 清空输入
            st.session_state.current_files = []
            
            # 重新运行以更新界面
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 自动滚动到底部的JavaScript
    st.markdown("""
    <script>
        setTimeout(function() {
            const messagesContainer = document.getElementById('messages-container');
            if (messagesContainer) {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }, 100);
    </script>
    """, unsafe_allow_html=True)

def main():
    """主函数"""
    # 初始化会话状态
    initialize_session_state()
    
    # 渲染聊天界面
    render_chat_interface()

if __name__ == "__main__":
    main()