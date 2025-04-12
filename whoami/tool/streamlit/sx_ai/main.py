"""
Main entry point for the SX AI application using NiceGUI.
"""

from nicegui import ui, app
import os
import uuid
import requests
import datetime
import asyncio
import base64
import json
from typing import Dict, List, Any, Optional

# ===== Configuration =====
API_BASE_URL = "http://1.71.15.121:8888"
DEFAULT_USER_ID = "13D6F349200080712111957107"

# ===== Utility Functions =====
def create_conversation_id() -> str:
    """Generate a new unique conversation ID."""
    return str(uuid.uuid4())

def format_chat_title(message: str, max_length: int = 20) -> str:
    """Format a chat title from the first message."""
    if len(message) > max_length:
        return message[:max_length] + "..."
    return message

# ===== API Functions =====
async def get_conversation_history(user_id: str, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get conversation history from the API."""
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
        return []

async def delete_conversation(user_id: str, conversation_id: str) -> bool:
    """Delete a conversation."""
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
        return False

async def send_chat_message(prompt: str, user_id: str, conversation_id: str) -> str:
    """Send a chat message to the API and get a response."""
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
        return error_message

# ===== State Management =====
class State:
    def __init__(self):
        self.user_id = DEFAULT_USER_ID
        self.current_conversation_id = create_conversation_id()
        self.messages = []
        self.conversations = []
        self.audio_text = ""
        self.waiting_for_response = False
        self.is_recording = False
        self.audio_data = None
        self.recorded_audio = None

state = State()

# ===== Audio Handling =====
async def process_audio_data(audio_data: bytes) -> str:
    """
    Process audio data to extract text.
    In a real implementation, this would call a speech recognition API.
    
    For now, we'll return a placeholder text.
    """
    # Simulate processing delay
    await asyncio.sleep(1)
    return "你是谁"

# ===== Chat Functions =====
async def handle_new_message(prompt: str):
    """Handle a new user message."""
    if not prompt or state.waiting_for_response:
        return
    
    # Set waiting state
    state.waiting_for_response = True
    
    # Add user message to history
    state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.datetime.now().isoformat()})
    
    # Add to UI
    with chat_messages:
        ui.chat_message(prompt, name="你", send=True)
    
    # Show thinking state
    thinking_msg = ui.chat_message("思考中...", name="SX AI")
    
    # Get response from API
    assistant_response = await send_chat_message(prompt, state.user_id, state.current_conversation_id)
    
    # Remove thinking message and add real response
    thinking_msg.delete()
    with chat_messages:
        ui.chat_message(assistant_response, name="SX AI")
    
    # Add assistant message to history
    state.messages.append({"role": "assistant", "content": assistant_response, "timestamp": datetime.datetime.now().isoformat()})
    
    # Handle first message in new conversation
    await handle_first_message_in_new_conversation(prompt)
    
    # Reset waiting state
    state.waiting_for_response = False
    
    # Clear input field
    input_field.value = ""

async def handle_first_message_in_new_conversation(prompt: str):
    """Handle special case when this is the first message in a new conversation."""
    if not any(conv["id"] == state.current_conversation_id for conv in state.conversations):
        # Use the first message as the conversation title
        title = format_chat_title(prompt)
        
        # Add to local conversation list
        state.conversations.append({
            "id": state.current_conversation_id,
            "title": title
        })
        
        # Refresh conversation list
        await refresh_conversation_list()
        
        # Update the UI
        await rebuild_sidebar()

async def refresh_conversation_list():
    """Refresh the conversation list from the server."""
    try:
        conversations_data = await get_conversation_history(state.user_id)
        
        # Format the conversation list
        conversations = []
        for conv in conversations_data:
            conv_id = conv.get("conversation_id", "")
            if conv_id and conv.get("messages"):
                messages = conv.get("messages", [])
                if messages:
                    first_msg = messages[0].get("content", "New chat") if isinstance(messages[0], dict) else "New chat"
                    title = format_chat_title(first_msg)
                    conversations.append({
                        "id": conv_id,
                        "title": title
                    })
        
        # Update state
        state.conversations = conversations
    except Exception as e:
        print(f"刷新历史会话出错: {str(e)}")

# ===== Sidebar Functions =====
async def new_chat():
    """Handle the new chat button click."""
    # Create new conversation
    state.current_conversation_id = create_conversation_id()
    state.messages = []
    
    # Clear chat UI
    chat_messages.clear()
    with chat_messages:
        ui.chat_message("您好! 请问有什么可以帮到您的? 😊", name="SX AI")
    
    # Update sidebar
    await rebuild_sidebar()

async def load_conversation(conversation_id: str):
    """Load a conversation when clicked in the sidebar."""
    try:
        conversations_data = await get_conversation_history(state.user_id, conversation_id)
        
        if conversations_data and len(conversations_data) > 0:
            state.messages = conversations_data[0].get("messages", [])
            state.current_conversation_id = conversation_id
            
            # Update UI
            chat_messages.clear()
            for message in state.messages:
                if "role" in message and "content" in message:
                    with chat_messages:
                        if message["role"] == "user":
                            ui.chat_message(message["content"], name="你", send=True)
                        else:
                            ui.chat_message(message["content"], name="SX AI")
        else:
            ui.notify("无法加载会话数据", color="negative")
    except Exception as e:
        ui.notify(f"加载会话出错: {str(e)}", color="negative")

async def delete_conversation_handler(idx: int, conversation_id: str):
    """Delete a conversation."""
    if await delete_conversation(state.user_id, conversation_id):
        # Remove from local list
        state.conversations.pop(idx)
        ui.notify("会话已删除!", color="positive")
        
        # If the deleted conversation is the current one, clear it
        if state.current_conversation_id == conversation_id:
            state.messages = []
            state.current_conversation_id = create_conversation_id()
            chat_messages.clear()
            with chat_messages:
                ui.chat_message("您好! 请问有什么可以帮到您的? 😊", name="SX AI")
        
        # Update sidebar
        await rebuild_sidebar()

async def rebuild_sidebar():
    """Rebuild the sidebar with updated conversations."""
    conversation_container.clear()
    
    with conversation_container:
        # Title and new chat button
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("SX AI").classes("text-xl font-bold")
        
        # New chat button
        ui.button("➕ New chat", on_click=new_chat).classes("w-full mb-4 bg-brown-700 text-white")
        
        # Conversations header
        ui.label("Recents").classes("text-lg font-medium mt-4")
        
        # Conversations list
        for idx, conversation in enumerate(state.conversations):
            with ui.row().classes("w-full items-center justify-between hover:bg-gray-100 p-2 rounded"):
                ui.button(conversation["title"], on_click=lambda c=conversation["id"]: load_conversation(c)) \
                   .classes("text-left flex-grow no-underline text-black")
                ui.button("🗑️", on_click=lambda i=idx, c=conversation["id"]: delete_conversation_handler(i, c)) \
                   .classes("text-gray-500 min-w-8 w-8")

# ===== Audio Functions =====
async def toggle_recording():
    """Toggle audio recording state."""
    state.is_recording = not state.is_recording
    
    if state.is_recording:
        ui.notify("开始录音...", color="blue")
        # In a real implementation, this would trigger the browser's recording API
        # For now, we'll simulate recording for a short time
        await asyncio.sleep(2)
        state.is_recording = False
        ui.notify("录音结束", color="green")
        
        # Simulate processing audio
        state.audio_text = await process_audio_data(b"")
        ui.notify(f"语音识别结果: {state.audio_text}", color="info")
        
        # If we got text, send it as a message
        if state.audio_text:
            await handle_new_message(state.audio_text)
            state.audio_text = ""
    else:
        ui.notify("停止录音", color="orange")

async def handle_audio_upload(file_content: bytes):
    """Handle uploaded audio file."""
    ui.notify("处理上传的音频文件...", color="blue")
    
    # Simulate audio processing
    await asyncio.sleep(1)
    
    # Store audio data
    state.audio_data = file_content
    
    # Process audio (in a real app, call speech-to-text service)
    state.audio_text = await process_audio_data(file_content)
    ui.notify(f"语音识别结果: {state.audio_text}", color="info")
    
    # If we got text, send it as a message
    if state.audio_text:
        await handle_new_message(state.audio_text)
        state.audio_text = ""

# ===== UI Layout =====
# CSS Theme
with ui.header().classes("bg-blue-500 text-white p-4 shadow-md"):
    ui.label("SX AI").classes("text-2xl font-bold")

# Left sidebar for conversations
with ui.left_drawer(value=True).classes("bg-gray-50 p-4").style("width: 300px"):
    conversation_container = ui.column().classes("w-full gap-2")

# Main content area
with ui.column().classes("w-full h-screen p-4"):
    # Chat header
    with ui.card().classes("w-full bg-blue-500 text-white p-4 text-center rounded-t-lg"):
        ui.image("https://i.imgur.com/gjzXgY7.png").classes("w-16 h-16 mx-auto rounded-full")
        ui.label("有疑问吗? 联系我们!").classes("text-lg mt-2")
        with ui.row().classes("justify-center items-center gap-1 text-sm"):
            ui.icon("circle").classes("text-green-400")
            ui.label("客服在线")
    
    # Chat messages container
    chat_container = ui.card().classes("w-full flex-grow overflow-auto p-4 bg-gray-100 rounded-none")
    with chat_container:
        chat_messages = ui.element("div").classes("flex flex-col gap-4")
    
    # Add initial message
    with chat_messages:
        with ui.element("div").classes("bg-white p-3 rounded-lg shadow-sm max-w-[80%] self-start"):
            ui.label("您好! 请问有什么可以帮到您的? 😊").classes("text-gray-800")
    
    # Input container
    with ui.card().classes("w-full p-4 rounded-b-lg flex items-center gap-2"):
        # Mic button
        ui.button(icon="mic", on_click=toggle_recording).props("round").classes("bg-blue-500 text-white")
        
        # File upload button
        file_upload = ui.upload(
            label="",
            auto_upload=True,
            on_upload=lambda e: handle_audio_upload(e.content),
            multiple=False
        ).props('accept=".mp3,.wav" hide-upload-btn').classes("max-w-40")
        
        # Text input
        input_field = ui.input(placeholder="输入你的信息...").classes("flex-grow")
        
        # Send button
        ui.button(icon="send", on_click=lambda: handle_new_message(input_field.value)) \
            .props("round").classes("bg-blue-500 text-white")

# Initial load of conversations
@app.on_startup
async def startup():
    await refresh_conversation_list()
    await rebuild_sidebar()

ui.run(
    title="SX AI", 
    port=8889,
    host="0.0.0.0",
    # base_url="/ai/chat_sys/chat_health_report"
)