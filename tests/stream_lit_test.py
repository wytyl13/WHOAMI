import streamlit as st
import requests
from typing import Optional, List, Dict
import uuid

# 页面配置
st.set_page_config(page_title="SX AI", layout="wide")

# 自定义CSS样式，模仿Claude界面
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

# API 配置
API_BASE_URL = "http://1.71.15.121:8888"
DEFAULT_USER_ID = "13D6F349200080712111957107"

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = ""  # 空字符串表示新会话
if "user_id" not in st.session_state:
    st.session_state.user_id = DEFAULT_USER_ID
    
# 初次加载时获取历史会话
if not st.session_state.conversations:
    try:
        response = requests.post(
            f"{API_BASE_URL}/get_conversation_history",
            json={"user_id": st.session_state.user_id},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            print(f"初始化获取历史会话响应: {data}")  # 调试信息
            
            # 根据返回的数据结构创建对话列表
            conversations = []
            for conv in data.get("data", []):
                conv_id = conv.get("conversation_id", "")
                if conv_id and conv.get("messages"):
                    # 使用第一条消息作为标题
                    messages = conv.get("messages", [])
                    if messages:
                        first_msg = messages[0].get("content", "New chat") if isinstance(messages[0], dict) else "New chat"
                        title = first_msg[:20] + "..." if len(first_msg) > 20 else first_msg
                        conversations.append({
                            "id": conv_id,
                            "title": title
                        })
                        print(f"添加会话: {conv_id}, 标题: {title}")  # 调试信息
            st.session_state.conversations = conversations
    except Exception as e:
        st.error(f"初始化历史会话出错: {str(e)}")
        import traceback
        print(traceback.format_exc())  # 详细错误信息

# 侧边栏 - Claude风格
with st.sidebar:
    # 隐藏默认的标题
    st.markdown("""
    <style>
        /* 隐藏默认标题区域 */
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
    </style>
    """, unsafe_allow_html=True)
    
    # 标题和新建会话按钮在同一行
    st.markdown("<h2 style='margin-bottom: 20px;'>SX AI</h2>", unsafe_allow_html=True)
    
    # 新建会话按钮样式
    st.markdown("""
    <style>
        /* 确保"+"按钮完全可见 */
        button[data-testid="baseButton-secondary"] {
            line-height: 1 !important;
            font-size: 22px !important;
            font-weight: bold !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        # 特殊处理加号按钮，确保内容可见
        new_chat_clicked = st.button("➕", key="new_chat_button", use_container_width=True)
    with col2:
        st.markdown("<div style='padding-top: 8px;'>New chat</div>", unsafe_allow_html=True)
    
    if new_chat_clicked:
        # 保存当前会话到历史
        if st.session_state.messages:
            # 确保当前会话已加入会话列表
            if not any(conv["id"] == st.session_state.current_conversation_id for conv in st.session_state.conversations):
                first_msg = st.session_state.messages[0]["content"] if st.session_state.messages else "New chat"
                # 使用第一条消息的前20个字符作为会话标题
                title = first_msg[:20] + "..." if len(first_msg) > 20 else first_msg
                st.session_state.conversations.append({
                    "id": st.session_state.current_conversation_id,
                    "title": title
                })
        
        # 创建新会话
        st.session_state.current_conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    # 历史会话标题
    st.markdown("<h3 style='margin-top:20px; font-size:16px;'>Recents</h3>", unsafe_allow_html=True)
    
    # 显示会话列表
    for idx, conversation in enumerate(st.session_state.conversations):
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(f"{conversation['title']}", key=f"conv_{conversation['id']}", 
                        use_container_width=True):
                # 加载选中的会话
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/get_conversation_history",
                        json={
                            "user_id": st.session_state.user_id,
                            "conversation_id": conversation["id"]
                        },
                        headers={"Content-Type": "application/json"}
                    )
                    if response.status_code == 200:
                        data = response.json().get("data", [])
                        st.session_state.messages = data[0].get("messages", [])
                        st.session_state.current_conversation_id = conversation["id"]
                        st.rerun()
                    else:
                        st.error(f"加载会话失败: {response.text}")
                except Exception as e:
                    st.error(f"加载会话出错: {str(e)}")
        
        with col2:
            if st.button("🗑️", key=f"del_{conversation['id']}"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/truncate_conversation_history",
                        json={
                            "user_id": st.session_state.user_id,
                            "conversation_id": conversation["id"]
                        },
                        headers={"Content-Type": "application/json"}
                    )
                    if response.status_code == 200:
                        # 从本地列表中移除
                        st.session_state.conversations.pop(idx)
                        st.success("会话已删除!")
                        # 如果删除的是当前会话，则清空当前会话
                        if st.session_state.current_conversation_id == conversation["id"]:
                            st.session_state.messages = []
                            st.session_state.current_conversation_id = str(uuid.uuid4())
                        st.rerun()
                    else:
                        st.error(f"删除会话失败: {response.text}")
                except Exception as e:
                    st.error(f"删除会话出错: {str(e)}")

# 添加隐藏箭头的样式和自定义侧边栏按钮
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

# 创建聊天内容区域和固定底部输入框
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
    
    /* 隐藏多余的元素 */
    .stButton button svg {
        display: none !important;
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
# 主内容区域
def main():
    # 创建一个容器用于滚动内容区域
    chat_content = st.container()
    
    # 在容器内显示聊天历史
    with chat_content:
        st.markdown('<div class="chat-content-area">', unsafe_allow_html=True)
        
        # 显示当前会话标题
        current_title = "New chat"
        for conv in st.session_state.conversations:
            if conv["id"] == st.session_state.current_conversation_id:
                current_title = conv["title"]
                break
        
        # 显示聊天历史
        if st.session_state.messages:
            for message in st.session_state.messages:
                # 确保message是字典并包含role和content键
                if isinstance(message, dict) and "role" in message and "content" in message:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                else:
                    print(f"跳过格式不正确的消息: {message}")
        else:
            st.info("没有聊天记录，开始新对话吧！")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 创建固定在底部的输入框容器
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 显示助手思考中状态
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("思考中...")
            
            # 调用您的API
            try:
                response = requests.post(
                    f"{API_BASE_URL}/chat_health_report",
                    json={
                        "question": prompt,
                        "user_id": st.session_state.user_id,
                        "conversation_id": st.session_state.current_conversation_id  # 添加会话ID
                    },
                    headers={"Content-Type": "application/json"}
                )
                result = response.json()
                print(result)
                # 从响应中提取助手回复
                assistant_response = result.get("data", "抱歉，我无法连接到服务。")
                if isinstance(assistant_response, dict) and "data" in assistant_response:
                    assistant_response = assistant_response["data"]
            except Exception as e:
                assistant_response = f"发生错误: {str(e)}"
            
            # 更新助手消息
            message_placeholder.markdown(assistant_response)
        
        # 添加助手消息到历史
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # 如果是新会话的第一条消息，添加到会话列表并立即刷新历史
        if not any(conv["id"] == st.session_state.current_conversation_id for conv in st.session_state.conversations):
            # 使用用户的第一条消息作为会话标题
            title = prompt[:20] + "..." if len(prompt) > 20 else prompt
            
            # 添加到本地会话列表
            st.session_state.conversations.append({
                "id": st.session_state.current_conversation_id,
                "title": title
            })
            
            # 立即从服务器刷新历史会话列表
            try:
                response = requests.post(
                    f"{API_BASE_URL}/get_conversation_history",
                    json={"user_id": st.session_state.user_id},
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    data = response.json()
                    print(f"刷新历史会话响应: {data}")
                    
                    # 根据返回的数据结构创建对话列表
                    conversations = []
                    for conv in data.get("data", []):
                        conv_id = conv.get("conversation_id", "")
                        if conv_id and conv.get("messages"):
                            messages = conv.get("messages", [])
                            if messages:
                                first_msg = messages[0].get("content", "New chat") if isinstance(messages[0], dict) else "New chat"
                                title = first_msg[:20] + "..." if len(first_msg) > 20 else first_msg
                                conversations.append({
                                    "id": conv_id,
                                    "title": title
                                })
                    st.session_state.conversations = conversations
            except Exception as e:
                print(f"刷新历史会话出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
            
            # 强制重新渲染页面以刷新历史会话列表
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 调用主函数
main()