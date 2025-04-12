"""
API-related functions for the SX AI application using NiceGUI.
"""

import requests
from typing import Dict, List, Any, Optional
import traceback
from nicegui import ui

from config import API_BASE_URL

async def get_conversation_history(user_id: str, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
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
            ui.notify(f"获取会话历史失败: {response.text}", color="negative")
            return []
    except Exception as e:
        ui.notify(f"获取会话历史出错: {str(e)}", color="negative")
        print(traceback.format_exc())
        return []

async def delete_conversation(user_id: str, conversation_id: str) -> bool:
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
            ui.notify(f"删除会话失败: {response.text}", color="negative")
            return False
    except Exception as e:
        ui.notify(f"删除会话出错: {str(e)}", color="negative")
        print(traceback.format_exc())
        return False

async def send_chat_message(prompt: str, user_id: str, conversation_id: str) -> str:
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