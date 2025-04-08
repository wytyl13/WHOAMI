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
    """Apply styles for the chat interface."""
    st.markdown("""
    <style>
        /* 聊天内容区域样式 */
        .chat-content-area {
            height: calc(100vh - 180px);
            overflow-y: auto;
            padding-bottom: 100px;
            margin-bottom: 20px;
        }
        
        /* 固定底部聊天输入框 */
        .chat-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 15px;
            background: white;
            z-index: 99;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
            border-top: 1px solid #f0f0f0;
        }
        
        /* 确保内容不被底部固定输入框遮挡 */
        .main-content {
            margin-bottom: 80px;
        }
        
        /* 聊天输入框样式 */
        .stChatInput {
            border-radius: 30px !important;
            padding: 10px 20px !important;
        }
        
        .stChatInput > div {
            background-color: #f5f5f5 !important;
            border-radius: 30px !important;
            border: 1px solid #e0e0e0 !important;
            box-shadow: none !important;
        }
        
        .stChatInput input {
            font-size: 16px !important;
        }
        
        /* 发送按钮样式 */
        .stChatInput button {
            border-radius: 50% !important;
            background-color: #f5f5f5 !important;
        }
        
        /* 录音相关样式 */
        /* 录音按钮样式 - 红色背景 */
        div.stButton > button:first-child:has(div:contains("🎤")) {
            background-color: #FF5F5F !important;
            color: white !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            padding: 0 !important;
            min-width: 40px !important;
            border: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* 停止录音按钮样式 */
        div.stButton > button:first-child:has(div:contains("⏹️")) {
            background-color: #FF0000 !important;
            color: white !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            padding: 0 !important;
            min-width: 40px !important;
            border: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* 录音状态指示器样式 */
        .recording-indicator {
            position: fixed;
            bottom: 75px;
            left: 50%;
            transform: translateX(-50%);
            background-color: #FF5F5F;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            z-index: 100;
            display: flex;
            align-items: center;
            animation: pulse 1.5s infinite;
            box-shadow: 0 2px 10px rgba(255, 0, 0, 0.3);
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .recording-indicator:before {
            content: "";
            display: inline-block;
            width: 10px;
            height: 10px;
            background-color: red;
            border-radius: 50%;
            margin-right: 10px;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        
        /* 音频录制组件样式 */
        .stAudioRecorder {
            margin-top: 10px !important;
        }
        
        .stAudioRecorder > div {
            background-color: transparent !important;
            border: none !important;
        }
        
        /* 隐藏多余的元素 */
        .stButton button svg {
            display: none !important;
        }
        
        /* 上传按钮样式 */
        .stFileUploader {
            margin-top: 5px !important;
        }
        
        .stFileUploader > div {
            background-color: transparent !important;
            border: none !important;
        }
        
        /* 去除滚动条 */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        /* 隐藏音频录制组件的用户界面，但保留功能 */
        div[data-testid="stAudioRecorder"] {
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            pointer-events: auto !important; /* 仍然允许互动 */
        }

        /* 录音/停止按钮样式统一 */
        div.stButton > button:first-child:has(div:contains("🎤")),
        div.stButton > button:first-child:has(div:contains("⏹️")) {
            background-color: #FF5F5F !important;
            color: white !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            padding: 0 !important;
            min-width: 40px !important;
            border: none !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        /* 完全隐藏的容器 */
        .completely-hidden {
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            position: absolute !important;
            left: -9999px !important;
            top: -9999px !important;
            opacity: 0 !important;
            pointer-events: none !important;
            display: none !important;
            overflow: hidden !important;
        }

        /* 隐藏audio-recorder组件的所有可见部分 */
        .audio-recorder {
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            position: absolute !important;
            left: -9999px !important;
            top: -9999px !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            pointer-events: auto !important;  /* 保留交互性 */
        }

        /* 确保录音按钮可以被点击，即使它是隐藏的 */
        .audio-recorder button {
            pointer-events: auto !important;
            position: fixed !important;
            left: -9999px !important;
            top: -9999px !important;
            z-index: -1 !important;
        }

        /* 覆盖Streamlit的默认样式，确保隐藏 */
        [data-testid="stAudioRecorder"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            overflow: hidden !important;
            max-height: 0 !important;
            max-width: 0 !important;
            opacity: 0 !important;
            pointer-events: auto !important;  /* 保留交互性 */
        }

        /* 确保音频录制组件的容器也被隐藏 */
        [data-testid="stAudioRecorder"] > div {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
        }

        /* 确保任何可能出现的弹出窗口也被隐藏 */
        .stAudioRecorderPlayback {
            display: none !important;
            visibility: hidden !important;
        }

        /* 只保留record按钮的交互功能，但隐藏视觉上的所有内容 */
        .audio-recorder .record-button {
            opacity: 0 !important;
            position: fixed !important;
            left: -9999px !important;
            top: -9999px !important;
        }
        
    </style>

    <script>
    // 页面加载完成后滚动到底部
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(function() {
            const chatContent = document.querySelector('.chat-content-area');
            if (chatContent) {
                chatContent.scrollTop = chatContent.scrollHeight;
            }
        }, 500);
    });
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <script>
        // 函数：查找并点击录音按钮
        function findAndClickRecordButton() {
            // 查找audio-recorder组件内的所有按钮
            const recorderButtons = document.querySelectorAll('.audio-recorder button');
            if (recorderButtons.length > 0) {
                console.log('找到录音按钮，进行点击');
                // 触发点击事件
                recorderButtons[0].click();
                return true;
            }
            return false;
        }
        
        // 创建一个MutationObserver来监视DOM变化
        const observer = new MutationObserver(function(mutations) {
            // 当DOM变化时，尝试找到并点击录音按钮
            if (document.querySelector('.recording-indicator')) {
                if (findAndClickRecordButton()) {
                    console.log('成功点击录音按钮');
                } else {
                    console.log('未找到录音按钮，将继续监视');
                }
            }
        });
        
        // 开始监视整个文档的变化
        document.addEventListener('DOMContentLoaded', function() {
            observer.observe(document.body, { 
                childList: true, 
                subtree: true 
            });
            
            // 页面加载后，检查是否有recording-indicator，如果有则尝试点击录音按钮
            setTimeout(function() {
                if (document.querySelector('.recording-indicator')) {
                    findAndClickRecordButton();
                }
            }, 500);
        });
    </script>
    """, unsafe_allow_html=True)
