"""
音频处理功能模块 - 使用Streamlit的内置组件。
"""

import time
import streamlit as st
import io

def init_audio_state():
    """初始化音频相关的会话状态"""
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "recorded_audio" not in st.session_state:
        st.session_state.recorded_audio = None
    if "audio_processed" not in st.session_state:
        st.session_state.audio_processed = False

def toggle_recording():
    """切换录音状态"""
    if not st.session_state.is_recording:
        # 开始录音
        st.session_state.is_recording = True
        st.session_state.audio_data = None
        st.session_state.recorded_audio = None
        st.session_state.audio_processed = False
    else:
        # 停止录音
        st.session_state.is_recording = False
        # 在这里不处理录音，等待录音数据接收后再处理

def process_recording(audio_bytes):
    """处理录音数据（仅在收到录音数据时调用）"""
    if audio_bytes and not st.session_state.audio_processed:
        # 显示处理状态
        with st.spinner("正在转换录音为文本..."):
            # 这里应该调用实际的语音识别API
            # 目前我们只是模拟处理
            time.sleep(1)
            
            # 将录音数据保存到session state
            st.session_state.recorded_audio = audio_bytes
            st.session_state.audio_data = audio_bytes
            st.session_state.audio_processed = True  # 标记为已处理，防止重复处理
            
            # 实际应用中，此处应调用语音识别API
            return "你是谁"
    
    return None

def handle_audio_upload(audio_file):
    """
    处理上传的音频文件。
    
    Args:
        audio_file: 上传的音频文件对象
        
    Returns:
        固定的文本字符串"你是谁"
    """
    if audio_file is not None:
        # 显示处理状态
        with st.spinner("正在转换音频为文本..."):
            # 这里应该调用实际的语音识别API
            # 目前我们只是模拟处理
            time.sleep(1)
            
            # 可以记录音频数据以备将来使用
            st.session_state.audio_data = audio_file.getvalue()
            
            # 实际应用中，此处应调用语音识别API
            return "你是谁"
    
    return None