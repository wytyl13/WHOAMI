"""
Text-to-Speech functionality for the SX AI application.
"""

import requests
import json
import time
import streamlit as st
import traceback
from typing import Optional, Dict, Any, Tuple

# TTS API配置
TTS_API_URL = "http://1.71.15.121:3000/tts"

def generate_tts(text: str, 
                style: str = "希望你以后能够做的比我还好呦。", 
                instruct: str = "用四川话说这句话", 
                speed: float = 1.3) -> Optional[str]:
    """
    调用TTS API生成语音。
    
    Args:
        text: 要转换的文本
        style: 语音风格
        instruct: 语音指令（如方言）
        speed: 语速
        
    Returns:
        task_id 用于后续查询语音生成状态，如果失败则返回None
    """
    try:
        payload = {
            "text": text,
            "style": style,
            "instruct": instruct,
            "speed": speed,
            "use_batch": True
        }
        
        response = requests.post(TTS_API_URL, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            if response_data.get("success"):
                return response_data.get("task_id")
        
        print(f"TTS API调用失败: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        print(f"TTS请求出错: {str(e)}")
        print(traceback.format_exc())
        return None

def check_tts_status(task_id: str) -> Tuple[str, Optional[str]]:
    """
    检查TTS任务的状态。
    
    Args:
        task_id: TTS任务ID
    
    Returns:
        Tuple[status, audio_url]
        status: 状态 ('processing', 'completed', 'failed')
        audio_url: 完成后的音频URL，未完成时为None
    """
    try:
        check_url = f"http://1.71.15.121:3000/status/{task_id}"
        response = requests.get(check_url)
        
        if response.status_code == 200:
            status_data = response.json()
            
            # 根据API返回的数据解析状态
            if status_data.get("status") == "completed" and status_data.get("completed"):
                return "completed", status_data.get("audio_url")
            else:
                return "processing", None
        
        print(f"检查TTS状态失败: {response.status_code} - {response.text}")
        return "failed", None
    except Exception as e:
        print(f"检查TTS状态出错: {str(e)}")
        print(traceback.format_exc())
        return "failed", None

def initialize_tts_state():
    """初始化TTS相关的会话状态变量。"""
    if "tts_task_id" not in st.session_state:
        st.session_state.tts_task_id = None
    if "tts_status" not in st.session_state:
        st.session_state.tts_status = None
    if "tts_audio_url" not in st.session_state:
        st.session_state.tts_audio_url = None
    if "tts_checking" not in st.session_state:
        st.session_state.tts_checking = False

def update_tts_status():
    """
    更新TTS任务状态，如果有正在处理的任务。
    
    当任务完成时，更新会话状态中的音频URL。
    """
    if st.session_state.tts_task_id and st.session_state.tts_status == "processing":
        status, audio_url = check_tts_status(st.session_state.tts_task_id)
        
        st.session_state.tts_status = status
        
        if status == "completed" and audio_url:
            st.session_state.tts_audio_url = audio_url
            st.session_state.tts_task_id = None
            # 强制重新渲染页面以显示音频
            st.rerun()
        elif status == "failed":
            st.session_state.tts_task_id = None