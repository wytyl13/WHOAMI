"""
Chat interface components and functionality for the SX AI application.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from audio_recorder_streamlit import audio_recorder
import api
import utils
import audio
import io

def initialize_chat_state():
    """Initialize session state variables for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audio_text" not in st.session_state:
        st.session_state.audio_text = ""
    # 初始化音频状态
    audio.init_audio_state()

def render_chat_interface():
    """Render the chat interface."""
    # Create a container for the scrollable chat content
    chat_content = st.container()
    
    # Display chat history in the container
    with chat_content:
        st.markdown('<div class="chat-content-area">', unsafe_allow_html=True)
        
        # Display the current conversation title
        current_title = get_current_conversation_title()
        
        # Display chat messages
        if st.session_state.messages:
            for message in st.session_state.messages:
                if utils.is_valid_message(message):
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                else:
                    print(f"跳过格式不正确的消息: {message}")
        else:
            st.info("没有聊天记录，开始新对话吧！")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create fixed input area at the bottom
    render_chat_input()


# 在chat.py的render_chat_input函数中
def render_chat_input():
    """Render the chat input area at the bottom of the screen."""
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # 使用三列布局
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col1:
        # 最简单的方式：直接使用Streamlit的音频录制组件
        audio_bytes = audio_recorder(key="simple_audio_recorder", 
                                      pause_threshold=2.0,
                                      sample_rate=16000)
        
        if audio_bytes is not None:
            # 处理录音数据
            text = audio.handle_audio_upload(io.BytesIO(audio_bytes))
            if text:
                st.session_state.audio_text = text
                st.rerun()
    
    with col2:
        # 文件上传组件
        audio_file = st.file_uploader("上传音频", type=["wav", "mp3"], key="audio_uploader", 
                                     label_visibility="collapsed", accept_multiple_files=False)
        
        if audio_file is not None:
            text = audio.handle_audio_upload(audio_file)
            if text:
                st.session_state.audio_text = text
                st.session_state.audio_uploader = None
                st.rerun()
    
    with col3:
        # 聊天输入框
        if st.session_state.audio_text:
            prompt = st.session_state.audio_text
            st.info(f"语音识别结果: {prompt}")
            handle_new_message(prompt)
            st.session_state.audio_text = ""
        else:
            prompt = st.chat_input("请输入您的问题...", key="chat_input")
            if prompt:
                handle_new_message(prompt)
    
    st.markdown('</div>', unsafe_allow_html=True)
    

def handle_new_message(prompt: str):
    """
    Handle a new user message.
    
    Args:
        prompt: The user's message text.
    """
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Show thinking state and get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中...")
        
        # Get response from API
        assistant_response = api.send_chat_message(
            prompt, 
            st.session_state.user_id, 
            st.session_state.current_conversation_id
        )
        
        # Update with actual response
        message_placeholder.markdown(assistant_response)
    
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    # Handle new conversation case
    handle_first_message_in_new_conversation(prompt)

def handle_first_message_in_new_conversation(prompt: str):
    """
    Handle special case when this is the first message in a new conversation.
    
    Args:
        prompt: The user's message that started this conversation.
    """
    if not any(conv["id"] == st.session_state.current_conversation_id for conv in st.session_state.conversations):
        # Use the first message as the conversation title
        title = utils.format_chat_title(prompt)
        
        # Add to local conversation list
        st.session_state.conversations.append({
            "id": st.session_state.current_conversation_id,
            "title": title
        })
        
        # Refresh conversation list from server
        refresh_conversation_list()
        
        # Force re-render
        st.rerun()

def get_current_conversation_title() -> str:
    """
    Get the title of the current conversation.
    
    Returns:
        The conversation title or "New chat" if not found.
    """
    current_title = "New chat"
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_conversation_id:
            current_title = conv["title"]
            break
    return current_title

def refresh_conversation_list():
    """Refresh the conversation list from the server."""
    try:
        conversations_data = api.get_conversation_history(st.session_state.user_id)
        
        # Format the conversation list
        conversations = []
        for conv in conversations_data:
            conv_id = conv.get("conversation_id", "")
            if conv_id and conv.get("messages"):
                messages = conv.get("messages", [])
                if messages:
                    first_msg = messages[0].get("content", "New chat") if isinstance(messages[0], dict) else "New chat"
                    title = utils.format_chat_title(first_msg)
                    conversations.append({
                        "id": conv_id,
                        "title": title
                    })
        
        # Update session state
        st.session_state.conversations = conversations
    except Exception as e:
        print(f"刷新历史会话出错: {str(e)}")
        import traceback
        print(traceback.format_exc())