"""
API-related functions for the SX AI application.
"""

import requests
from typing import Dict, List, Any, Optional
import traceback
import streamlit as st
from config import API_BASE_URL
import time
import re


def get_conversation_history(user_id: str, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get conversation history from the API.
    
    Args:
        user_id: The user ID.
        conversation_id: Optional conversation ID. If provided, only returns that conversation.
        
    Returns:
        List of conversation data.
    """
    try:
        payload = {"user_id": user_id}
        if conversation_id:
            payload["conversation_id"] = conversation_id
            
        response = requests.post(
            f"{API_BASE_URL}/get_conversation_history",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        else:
            st.error(f"获取会话历史失败: {response.text}")
            return []
    except Exception as e:
        st.error(f"获取会话历史出错: {str(e)}")
        print(traceback.format_exc())
        return []


def delete_conversation(user_id: str, conversation_id: str) -> bool:
    """
    Delete a conversation.
    
    Args:
        user_id: The user ID.
        conversation_id: The conversation ID to delete.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/truncate_conversation_history",
            json={
                "user_id": user_id,
                "conversation_id": conversation_id
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"删除会话失败: {response.text}")
            return False
    except Exception as e:
        st.error(f"删除会话出错: {str(e)}")
        print(traceback.format_exc())
        return False


def send_chat_message(prompt: str, user_id: str, conversation_id: str) -> str:
    """
    Send a chat message to the API and get a response.
    
    Args:
        prompt: The user's message.
        user_id: The user ID.
        conversation_id: The conversation ID.
        
    Returns:
        The assistant's response text.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat_health_report",
            json={
                "question": prompt,
                "user_id": user_id,
                "conversation_id": conversation_id
            },
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        
        # Extract the assistant response from the result
        assistant_response = result.get("data", "抱歉，我无法连接到服务。")
        if isinstance(assistant_response, dict) and "data" in assistant_response:
            assistant_response = assistant_response["data"]
            
        return assistant_response
    except Exception as e:
        error_message = f"发生错误: {str(e)}"
        print(traceback.format_exc())
        return error_message
    

def text_to_speech(text: str) -> bytes:
    """
    将文本转换为语音。
    
    Args:
        text: 要转换的文本
        
    Returns:
        音频数据的字节流
    """
    try:
        # 这里使用百度文本转语音API作为示例
        # 你需要替换为你选择的API服务
        import requests
        import json
        import base64
        
        # 百度API参数 - 需要替换为你自己的API密钥
        API_KEY = "YOUR_BAIDU_API_KEY"
        SECRET_KEY = "YOUR_BAIDU_SECRET_KEY"
        
        # 获取百度API访问令牌
        token_url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET_KEY}"
        token_response = requests.get(token_url)
        token = token_response.json().get("access_token")
        
        # 调用文本转语音API
        tts_url = f"https://tsn.baidu.com/text2audio"
        params = {
            "tex": text,
            "tok": token,
            "cuid": "streamlit_app",
            "ctp": 1,
            "lan": "zh",  # 中文
            "spd": 5,     # 语速，范围0-15
            "pit": 5,     # 音调，范围0-15
            "vol": 15,    # 音量，范围0-15
            "per": 0,     # 发音人，0为女声，1为男声，3为情感男声
            "aue": 3      # 音频格式，3为mp3格式
        }
        
        response = requests.post(tts_url, params=params)
        
        # 检查是否返回音频数据
        if response.headers.get("Content-Type") == "audio/mp3":
            return response.content
        else:
            st.error(f"语音合成失败: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"语音合成出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None
    
    
    
    
def generate_tts(text: str) -> Optional[str]:
    """
    生成文本的语音合成，并返回音频URL。
    直接调用TTS API并等待结果返回，简化流程。
    
    Args:
        text: 要转换为语音的文本
        
    Returns:
        音频URL，如果失败则返回None
    """
    try:
        # 调试信息
        print(f"开始为文本生成语音: {text[:50]}...")
        
        # 第一步：调用TTS API开始任务
        tts_url = "http://1.71.15.121:3000/tts"
        payload = {
            "text": text,
            "style": "希望你以后能够做的比我还好呦。",
            # "instruct": "用陕西话回答",
            "speed": 1.3,
            "use_batch": True
        }
        
        response = requests.post(tts_url, json=payload)
        if response.status_code != 200:
            print(f"TTS API调用失败: {response.status_code} - {response.text}")
            return None
            
        result = response.json()
        print(f"TTS API返回: {result}")
        
        if not result.get("success"):
            print("TTS任务创建失败")
            return None
            
        task_id = result.get("task_id")
        if not task_id:
            print("未获取到任务ID")
            return None
            
        # 第二步：轮询检查任务状态直到完成
        check_url = f"http://1.71.15.121:3000/status/{task_id}"
        max_attempts = 30  # 最多等待30次
        
        for attempt in range(max_attempts):
            print(f"检查TTS任务状态，尝试 {attempt+1}/{max_attempts}")
            
            try:
                status_response = requests.get(check_url)
                if status_response.status_code != 200:
                    print(f"检查状态失败: {status_response.status_code} - {status_response.text}")
                    time.sleep(1)
                    continue
                    
                status_data = status_response.json()
                print(f"状态检查返回: {status_data}")
                
                if status_data.get("status") == "completed" and status_data.get("completed"):
                    audio_url = status_data.get("audio_url")
                    old_prefix = r"http://1\.71\.15\.121:3000"
                    new_prefix = r"https://1.71.15.121:5001/ai/chat_sys/chat_health_report"
                    pattern = f"{old_prefix}(/[^'\"\s]*)"
                    audio_url = re.sub(pattern, f"{new_prefix}\\1", audio_url)
                    print(f"成功获取音频URL: {audio_url}")
                    return audio_url
                    
                # 如果任务仍在处理中，等待后再次检查
                if status_data.get("status") == "processing":
                    print("任务处理中，等待1秒后重试...")
                    time.sleep(1)
                    continue
                    
                # 其他情况，可能是任务失败
                print(f"任务可能失败: {status_data}")
                return None
                
            except Exception as e:
                print(f"检查状态时出错: {str(e)}")
                time.sleep(1)
                continue
                
        print(f"达到最大尝试次数，放弃等待")
        return None
        
    except Exception as e:
        print(f"TTS处理过程中出错: {str(e)}")
        print(traceback.format_exc())
        return None