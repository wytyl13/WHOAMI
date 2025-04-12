"""
CSS styles for the SX AI application.
"""

import streamlit as st

def apply_base_styles():
    """Apply base styles to the application."""
    st.markdown("""
    <style>
        /* 侧边栏样式 */
        .css-1d391kg, .css-1lcbmhc {
            background-color: #f5f5f5;
        }
        
        /* 隐藏原始标题 */
        .css-zt5igj {
            display: none;
        }
        
        /* 自定义标题栏 */
        .header-container {
            display: flex;
            align-items: center;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* 新会话按钮样式 */
        .new-chat-container {
            display: flex;
            align-items: center;
            padding: 0.5rem 1rem;
            margin: 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            background-color: #f0f0f0;
        }
        
        .new-chat-container:hover {
            background-color: #e0e0e0;
        }
        
        .new-chat-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: #ff5f5f;
            color: white;
            font-size: 16px;
            margin-right: 10px;
        }
        
        .new-chat-text {
            font-size: 14px;
        }
        
        /* 历史会话项样式 */
        .chat-item {
            padding: 0.5rem 1rem;
            margin: 0.2rem 0;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .chat-item:hover {
            background-color: #e6e6e6;
        }
        
        /* 删除按钮样式 */
        .delete-btn {
            color: #888;
            background: none;
            border: none;
            float: right;
            cursor: pointer;
        }
        
        /* 会话内容容器 */
        .chat-container {
            margin-left: 20px;
        }
        
        /* 隐藏向左箭头按钮 */
        button[aria-label="Collapse sidebar"] {
            display: none;
        }
        
        /* 添加自定义展开/折叠侧边栏按钮 */
        .sidebar-toggle {
            position: fixed;
            left: 0;
            top: 40px;
            width: 30px;
            height: 30px;
            background-color: #f5f5f5;
            border: none;
            border-radius: 0 4px 4px 0;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            cursor: pointer;
        }
        
        /* 输入框样式 */
        .stTextInput input, .stTextArea textarea {
            border-radius: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_sidebar_styles():
    """Apply styles specific to the sidebar."""
    st.markdown("""
    <style>
        /* 隐藏默认的标题区域 */
        header[data-testid="stHeader"] {
            display: none;
        }
        
        /* "+" 按钮样式 - 棕色背景 */
        div.stButton > button:first-child:has(div:contains("➕")) {
            background-color: #8B4513 !important; /* 棕色 */
            color: white !important;
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            min-width: 32px !important;
            margin-right: 10px !important;
            border: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* 确保按钮内的文字居中显示 */
        div.stButton > button:first-child:has(div:contains("➕")) div {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-weight: bold !important;
            font-size: 18px !important;
        }
        
        /* 确保"+"按钮完全可见 */
        button[data-testid="baseButton-secondary"] {
            line-height: 1 !important;
            font-size: 22px !important;
            font-weight: bold !important;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_toggle_sidebar_script():
    """Apply the script for toggling the sidebar."""
    st.markdown("""
    <style>
        /* 仅针对特定的箭头进行隐藏 */
        [data-testid="stHeader"] button svg {
            display: none !important;
        }
        
        /* 隐藏默认的侧边栏折叠按钮 */
        [data-testid="collapsedControl"], [data-testid="expandedControl"] {
            display: none !important;
        }
        
        /* 专门隐藏左上角的箭头 */
        header > div > button > svg {
            display: none !important;
        }
        
        /* 自定义侧边栏切换按钮 */
        #custom-sidebar-toggle {
            position: fixed;
            top: 0.5rem;
            left: 0;
            z-index: 999;
            background: white;
            border: none;
            padding: 0.5rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            border-radius: 0 4px 4px 0;
            transition: all 0.3s ease;
        }
        
        #custom-sidebar-toggle.expanded {
            left: 21rem; /* 侧边栏展开时移动按钮 */
        }
        
        #sidebar-title {
            margin-left: 0.5rem;
            font-weight: bold;
            display: none;
        }
        
        #custom-sidebar-toggle.expanded #sidebar-title {
            display: inline;
        }
        
        /* 保证自定义按钮可见 */
        #toggle-icon {
            display: inline !important;
        }
    </style>

    <!-- 自定义的侧边栏切换按钮 -->
    <div id="custom-sidebar-toggle" style="display: flex;">
        <span id="toggle-icon">→</span>
        <span id="sidebar-title">SX AI</span>
    </div>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // 获取侧边栏元素
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        const toggleBtn = document.getElementById('custom-sidebar-toggle');
        const toggleIcon = document.getElementById('toggle-icon');
        let isExpanded = true; // 默认为展开状态
        
        // 隐藏左上角的箭头
        function hideHeaderArrow() {
            const headerArrows = document.querySelectorAll('header > div > button > svg');
            headerArrows.forEach(arrow => {
                arrow.style.display = 'none';
            });
        }
        
        // 初始化函数
        function init() {
            // 隐藏左上角箭头
            hideHeaderArrow();
            
            // 检查侧边栏状态
            if (sidebar) {
                const sidebarWidth = sidebar.offsetWidth;
                isExpanded = sidebarWidth > 100;
                
                if (isExpanded) {
                    toggleBtn.classList.add('expanded');
                    toggleIcon.textContent = '←';
                } else {
                    toggleBtn.classList.remove('expanded');
                    toggleIcon.textContent = '→';
                }
            }
        }
        
        // 切换侧边栏状态
        function toggleSidebar() {
            if (sidebar) {
                if (isExpanded) {
                    // 收起侧边栏
                    sidebar.style.transform = 'translateX(-100%)';
                    sidebar.style.marginLeft = '-100%';
                    toggleBtn.classList.remove('expanded');
                    toggleIcon.textContent = '→';
                } else {
                    // 展开侧边栏
                    sidebar.style.transform = 'translateX(0)';
                    sidebar.style.marginLeft = '0';
                    toggleBtn.classList.add('expanded');
                    toggleIcon.textContent = '←';
                }
                
                isExpanded = !isExpanded;
            }
        }
        
        // 监听自定义按钮点击事件
        toggleBtn.addEventListener('click', toggleSidebar);
        
        // 定期检查并隐藏可能动态加载的箭头
        setInterval(hideHeaderArrow, 500);
        
        // 初始化执行
        setTimeout(init, 500);
    });
    </script>
    """, unsafe_allow_html=True)

def apply_chat_styles():
    """Apply styles for the chat interface to match the screenshot."""
    st.markdown("""
    <style>
        /* Chat header - blue design */
        .chat-header {
            background-color: #00A3E0;
            color: white;
            padding: 15px;
            text-align: center;
            border-radius: 10px 10px 0 0;
            margin-bottom: 20px;
        }
        
        /* Avatar container */
        .avatar-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 10px;
        }
        
        /* Greeting text */
        .greeting-text {
            color: white;
            margin-top: 10px;
            font-size: 16px;
        }
        
        /* Chat content area */
        .chat-content-area {
            height: calc(100vh - 210px);
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 80px; /* Space for input box */
            background-color: #f5f5f5;
            border-radius: 0 0 10px 10px;
        }
        
        /* User message bubble */
        .stChatMessage[data-testid="stChatMessage"] .user {
            background-color: #0097E6 !important;
            color: white !important;
            border-radius: 18px !important;
            padding: 10px 15px !important;
            max-width: 80% !important;
            margin-left: auto !important;
            margin-right: 10px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        }
        
        /* Assistant message bubble */
        .stChatMessage[data-testid="stChatMessage"] .assistant {
            background-color: white !important;
            color: #333 !important;
            border-radius: 18px !important;
            padding: 10px 15px !important;
            max-width: 80% !important;
            margin-right: auto !important;
            margin-left: 10px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        }
        
        /* Remove avatar from chat bubbles */
        .stChatMessage [data-testid="stChatMessageAvatar"] {
            display: none !important;
        }
        
        /* Chat input container - fixed at bottom */
        .chat-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 15px;
            background: white;
            z-index: 1000; /* Increased z-index */
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            margin-left: 250px; /* Adjusted for sidebar width */
        }
        
        /* When sidebar is collapsed, adjust margin */
        @media screen and (max-width: 992px) {
            .chat-input-container {
                margin-left: 0;
            }
        }
        
        /* Chat input style */
        .stChatInput > div {
            border-radius: 20px !important;
            border: 1px solid #e0e0e0 !important;
            background-color: #f8f8f8 !important;
            width: 100% !important;
        }
        
        /* Chat input text */
        .stChatInput input {
            font-size: 14px !important;
            padding: 10px 15px !important;
        }
        
        /* Send button */
        .stChatInput button {
            background-color: #00A3E0 !important;
            border-radius: 50% !important;
        }
        
        /* Disabled send button */
        .stChatInput button:disabled {
            background-color: #cccccc !important;
            opacity: 0.7 !important;
        }
        
        /* Make sure the chat content area doesn't get covered by input box */
        .chat-content-area {
            padding-bottom: 100px !important; /* Increased padding to avoid content being hidden behind input */
        }
        
        /* Audio button styling */
        div.stButton > button:first-child:has(div:contains("🎤")) {
            background-color: #00A3E0 !important;
            color: white !important;
            border-radius: 50% !important;
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            border: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        }
        
        /* Hide the audio recorder component but keep functionality */
        div[data-testid="stAudioRecorder"] {
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            pointer-events: auto !important;
        }
        
        /* File uploader styling */
        .stFileUploader > div {
            padding: 0 !important;
        }
        
        .stFileUploader > div > div {
            display: none !important;
        }
        
        .stFileUploader > div > button {
            background-color: transparent !important;
            border: none !important;
            color: #00A3E0 !important;
            font-size: 18px !important;
            padding: 5px !important;
            border-radius: 50% !important;
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Emoji button */
        .emoji-button {
            background: none;
            border: none;
            font-size: 18px;
            cursor: pointer;
            color: #999;
            padding: 5px;
        }
        
        /* Online status text style */
        .online-status {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 5px;
            font-size: 12px;
            color: white;
        }

        /* Add specific selector for the main content area to ensure proper spacing */
        .main .block-container {
            padding-bottom: 100px !important;
        }
    </style>

    <script>
    // Scroll to bottom of chat when loaded
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const chatContent = document.querySelector('.chat-content-area');
            if (chatContent) {
                chatContent.scrollTop = chatContent.scrollHeight;
            }
        }, 500);

        // Also add an event listener for the sidebar toggle to adjust input position
        const sidebarToggle = document.getElementById('custom-sidebar-toggle');
        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', function() {
                const inputContainer = document.querySelector('.chat-input-container');
                if (inputContainer) {
                    // Toggle margin based on sidebar state
                    if (toggleBtn.classList.contains('expanded')) {
                        inputContainer.style.marginLeft = '250px';
                    } else {
                        inputContainer.style.marginLeft = '0';
                    }
                }
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)
