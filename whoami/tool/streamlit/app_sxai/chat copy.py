"""
Chat interface components and functionality for the SX AI application.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from audio_recorder_streamlit import audio_recorder
import requests

import api
import utils

def initialize_chat_state():
    """Initialize session state variables for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Audio recorder state
    if "recorder_audio_data" not in st.session_state:
        st.session_state.recorder_audio_data = None
    if "recorder_processed_text" not in st.session_state:
        st.session_state.recorder_processed_text = None
    if "last_sent_text" not in st.session_state:
        st.session_state.last_sent_text = None
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "show_transcription" not in st.session_state:
        st.session_state.show_transcription = False
    if "transcription_status" not in st.session_state:
        st.session_state.transcription_status = ""
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = ""

def apply_chat_styles():
    """Apply styles specific to the chat interface."""
    st.markdown("""
    <style>
        /* Chat message styling */
        .stChatMessage {
            padding: 10px 15px !important;
            border-radius: 10px !important;
            margin-bottom: 10px !important;
        }
        
        /* User message styling - Right alignment */
        [data-testid="stChatMessageContent"][data-test="chatAvatarIconUser"] > div,
        [data-testid="stChatMessageContent"] div[data-testid="chatAvatarIconUser"] + div {
            display: flex !important;
            justify-content: flex-end !important;
        }
        
        /* Force user message container to right */
        .stChatMessage[data-testid="chat-message-user"] {
            background-color: #e1f5fe !important;
            float: right !important;
            clear: both !important;
            max-width: 80% !important;
        }
        
        /* Assistant message styling - Left alignment */
        .stChatMessage[data-testid="chat-message-assistant"] {
            background-color: #f5f5f5 !important;
            float: left !important;
            clear: both !important;
            max-width: 80% !important;
        }
        
        /* Fix for chat container to handle floats */
        .chat-content-area::after {
            content: "";
            display: table;
            clear: both;
        }
        
        /* Chat input styling */
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
        
        /* Send button styling */
        .stChatInput button {
            border-radius: 50% !important;
            background-color: #f5f5f5 !important;
        }
        
        /* Hide audio recorder container */
        .stAudio {
            display: none !important;
        }
        
        /* Hide audio player element */
        audio {
            display: none !important;
        }
        
        /* Voice recording button styling */
        .voice-record-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: #f1f1f1;
            border: 1px solid #e0e0e0;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            margin-right: 10px;
            transition: background-color 0.3s;
        }
        
        .voice-record-button:hover {
            background-color: #e0e0e0;
        }
        
        .voice-record-button.recording {
            background-color: #ffcdd2;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.7);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(255, 82, 82, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(255, 82, 82, 0);
            }
        }
        
        /* Styling for input area with voice record button */
        .input-with-voice {
            display: flex;
            align-items: center;
            background-color: #f5f5f5;
            border-radius: 30px;
            padding: 5px 10px;
            border: 1px solid #e0e0e0;
        }
        
        /* Transcription result styling */
        .transcription-result {
            margin-top: 5px;
            font-size: 12px;
            color: #666;
            font-style: italic;
        }
        
        /* 状态指示器样式 */
        .status-indicator {
            text-align: center;
            font-size: 12px;
            color: #555;
            margin-bottom: 5px;
            padding: 3px 0;
        }
        
        /* 转录状态样式 */
        .transcription-status {
            color: #2196f3;
        }
        
        /* 处理状态样式 */
        .processing-status {
            color: #ff9800;
        }
        
        /* 确保只有一个状态显示 */
        .stChatMessage .stMarkdown p {
            margin-bottom: 0 !important;
        }
        
        /* 隐藏spinner */
        .stSpinner {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_chat_interface():
    """Render the chat interface."""
    # Apply chat specific styles
    apply_chat_styles()
    
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
    render_chat_input_with_voice()

def transcribe_with_sensevoice(audio_bytes):
    """使用SenseVoice转录音频数据"""
    try:
        # 构建API请求 - 直接传输二进制数据
        files = {
            "audio_file": ("recording.wav", audio_bytes, "audio/wav")
        }
        
        # 设置转录状态
        st.session_state.transcription_status = "转录中..."
        
        # 发送请求到SenseVoice服务
        response = requests.post(
            "http://localhost:8818/transcribe/",
            files=files
        )
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            # 清除转录状态
            st.session_state.transcription_status = ""
            return result["text"]
        else:
            print(f"API请求失败: {response.status_code} - {response.text}")
            # 清除转录状态并设置错误
            st.session_state.transcription_status = "转录失败"
            return None
    except Exception as e:
        print(f"转录出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # 清除转录状态并设置错误
        st.session_state.transcription_status = "转录失败"
        return None

def render_chat_input_with_voice():
    """Render the chat input area with voice recording at the bottom of the screen."""
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # 显示转录状态（如果有）
    if st.session_state.transcription_status:
        st.markdown(f'<div class="status-indicator transcription-status">{st.session_state.transcription_status}</div>', unsafe_allow_html=True)
    
    # 自定义布局，包括语音录制按钮和文本输入
    col1, col2 = st.columns([10, 1])
    
    with col2:
        # 音频录制组件 - 隐藏默认UI
        audio_bytes = audio_recorder(
            key="chat_audio_recorder", 
            pause_threshold=2.0,
            sample_rate=16000,
            text="",  # 隐藏默认录制文本
            recording_color="#e65100",
            neutral_color="#2196f3"
        )
        
        # 处理录音结果
        if audio_bytes is not None and audio_bytes != st.session_state.recorder_audio_data:
            # 更新录音数据
            st.session_state.recorder_audio_data = audio_bytes
            st.session_state.is_recording = False
            
            # 转录处理 - 不使用st.spinner，使用自定义状态显示
            transcribed_text = transcribe_with_sensevoice(audio_bytes)
            
            if transcribed_text:
                st.session_state.recorder_processed_text = transcribed_text
                
                # 如果转录文本与上次发送的不同，发送到聊天
                if transcribed_text != st.session_state.last_sent_text:
                    # 清除转录状态，避免状态显示冲突
                    st.session_state.transcription_status = ""
                    handle_new_message(transcribed_text)
                    st.session_state.last_sent_text = transcribed_text
                    st.rerun()
    
    with col1:
        # 文本输入
        if prompt := st.chat_input("请输入您的问题..."):
            # 确保清除任何可能的转录状态
            st.session_state.transcription_status = ""
            handle_new_message(prompt)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_chat_input():
    """Render the original chat input area at the bottom of the screen."""
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("请输入您的问题..."):
        handle_new_message(prompt)
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_new_message(prompt: str):
    """
    Handle a new user message.
    
    Args:
        prompt: The user's message text.
    """
    # 清除所有状态
    st.session_state.transcription_status = ""
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 设置处理状态
    st.session_state.processing_status = "思考中..."
    
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
        
        # 清除处理状态
        st.session_state.processing_status = ""
        
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