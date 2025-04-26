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
import pyaudio
import asyncio
import aiohttp
import json
import io
import wave
import base64


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
    
    
    
def generate_tts_bake(text: str) -> Optional[str]:
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
    



def generate_tts(text: str) -> Optional[str]:
    """
    生成文本的语音合成，将所有音频片段拼接起来后一次性播放。
    只负责播放音频，不显示任何额外UI元素。
    
    Args:
        text: 要转换为语音的文本
        
    Returns:
        音频URL，如果使用流式播放则返回None
    """
    try:
        # 创建一个简单的占位符只用于放置最终音频
        audio_placeholder = st.empty()
        debug_placeholder = st.empty()  # 用于调试信息
        
        # 创建payload
        tts_url = "http://1.71.15.121:3000/tts/stream"
        payload = {
            "text": text,
            "style": "希望你以后能够做的比我还好呦。",
            "speed": 1.3
        }
        
        # 设置流式请求的头部
        headers = {"Accept": "multipart/mixed; boundary=chunk"}
        
        # 创建缓冲区用于存储完整音频
        all_audio_chunks = []
        
        debug_placeholder.text("正在发送TTS请求...")
        
        # 发起流式请求
        with requests.post(
            tts_url, 
            json=payload,
            headers=headers,
            stream=True
        ) as response:
            if response.status_code != 200:
                debug_placeholder.text(f"TTS服务请求失败: {response.status_code}")
                return None
                
            # 处理多部分响应流
            boundary = b"--chunk"
            buffer = b""
            chunk_count = 0
            has_end_marker = False
            
            # 处理每个响应块
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    buffer += chunk
                    
                    # 特殊处理：检查是否有结束标记
                    end_marker_pos = buffer.find(boundary + b"--")
                    if end_marker_pos != -1:
                        has_end_marker = True
                        # 确保在结束标记前的内容也被处理
                        if end_marker_pos > 0:
                            final_part = buffer[:end_marker_pos]
                            # 处理最后一部分数据
                            if b"Content-Type: audio/wav" in final_part:
                                header_end = final_part.find(b"\r\n\r\n")
                                if header_end != -1:
                                    content = final_part[header_end + 4:]
                                    chunk_count += 1
                                    debug_placeholder.text(f"找到最终音频块 {chunk_count}!")
                                    all_audio_chunks.append(content)
                        # 清空缓冲区，因为已经找到了结束标记
                        buffer = b""
                        continue
                    
                    # 处理完整部分
                    while True:
                        # 查找边界
                        boundary_pos = buffer.find(boundary)
                        if boundary_pos == -1:
                            break
                        
                        # 提取部分
                        part = buffer[:boundary_pos]
                        buffer = buffer[boundary_pos + len(boundary):]
                        
                        # 跳过空部分和结束标记
                        if not part.strip():
                            continue
                            
                        # 如果这是结束标记，处理并退出
                        if buffer.startswith(b"--"):
                            has_end_marker = True
                            break
                        
                        # 处理头部和内容
                        if b"Content-Type: " in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                header = part[:header_end].decode('utf-8', errors='ignore')
                                content = part[header_end + 4:]
                                
                                # 处理音频块
                                if "audio/wav" in header:
                                    chunk_count += 1
                                    debug_placeholder.text(f"接收到音频块 {chunk_count}")
                                    all_audio_chunks.append(content)
            
            # 处理缓冲区中剩余的最后一个部分（可能是最后一个音频块）
            if buffer and not has_end_marker:
                if b"Content-Type: audio/wav" in buffer:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end != -1:
                        content = buffer[header_end + 4:]
                        chunk_count += 1
                        debug_placeholder.text(f"从剩余缓冲区找到最后一个音频块 {chunk_count}")
                        all_audio_chunks.append(content)
        
        debug_placeholder.text(f"总共收集到 {len(all_audio_chunks)} 个音频块")
        
        # 确保至少有一个音频块
        if not all_audio_chunks:
            debug_placeholder.text("没有收集到音频块，无法生成音频")
            return None
            
        # 如果只有一个音频块，直接使用
        if len(all_audio_chunks) == 1:
            debug_placeholder.text("只有一个音频块，直接播放")
            with audio_placeholder:
                b64 = base64.b64encode(all_audio_chunks[0]).decode()
                audio_html = f"""
                    <audio autoplay controls>
                        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    </audio>
                    """
                st.markdown(audio_html, unsafe_allow_html=True)
            return None
        
        # 合并多个音频块
        debug_placeholder.text(f"合并 {len(all_audio_chunks)} 个音频块...")
        
        try:
            # 使用wave模块合并
            import io
            import wave
            
            # 创建输出缓冲区
            output = io.BytesIO()
            
            # 处理第一个音频块，获取WAV参数
            with io.BytesIO(all_audio_chunks[0]) as first_chunk_io:
                with wave.open(first_chunk_io, 'rb') as first_wave:
                    # 获取WAV参数
                    params = first_wave.getparams()
                    
                    # 创建输出WAV文件
                    with wave.open(output, 'wb') as output_wave:
                        # 设置参数
                        output_wave.setparams(params)
                        
                        # 添加第一个块的数据
                        first_wave.rewind()
                        output_wave.writeframes(first_wave.readframes(first_wave.getnframes()))
                        
                        # 处理其余音频块
                        for i, chunk in enumerate(all_audio_chunks[1:], 1):
                            try:
                                with io.BytesIO(chunk) as chunk_io:
                                    with wave.open(chunk_io, 'rb') as chunk_wave:
                                        # 添加当前块的数据
                                        chunk_frames = chunk_wave.readframes(chunk_wave.getnframes())
                                        output_wave.writeframes(chunk_frames)
                                        debug_placeholder.text(f"成功合并音频块 {i+1}")
                            except Exception as e:
                                debug_placeholder.text(f"处理音频块 {i+1} 时出错: {e}")
                                continue
            
            # 获取完整的WAV数据
            complete_wav = output.getvalue()
            debug_placeholder.text(f"合并成功，总大小: {len(complete_wav)} 字节")
            
            # 播放完整的音频
            with audio_placeholder:
                b64 = base64.b64encode(complete_wav).decode()
                audio_html = f"""
                    <audio autoplay controls>
                        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    </audio>
                    """
                st.markdown(audio_html, unsafe_allow_html=True)
                
        except Exception as e:
            debug_placeholder.text(f"使用wave模块合并失败: {e}")
            debug_placeholder.text("尝试使用二进制合并方法...")
            
            # 备用方法：直接操作WAV文件字节
            try:
                # 提取第一个块的WAV头
                first_chunk = all_audio_chunks[0]
                data_pos = first_chunk.find(b'data')
                
                if data_pos != -1:
                    # 找到data块后的位置
                    header_end = data_pos + 8  # "data" + 4字节大小
                    
                    # 提取头部
                    wav_header = first_chunk[:header_end]
                    
                    # 创建合并数据
                    merged_data = bytearray()
                    
                    # 添加第一个块的数据
                    merged_data.extend(first_chunk[header_end:])
                    
                    # 添加其余块的数据
                    for i, chunk in enumerate(all_audio_chunks[1:], 1):
                        chunk_data_pos = chunk.find(b'data')
                        if chunk_data_pos != -1:
                            # 找到data块，跳过头部
                            chunk_header_end = chunk_data_pos + 8
                            merged_data.extend(chunk[chunk_header_end:])
                            debug_placeholder.text(f"添加音频块 {i+1} 的数据")
                        else:
                            # 未找到data块，添加整个块（不太可能发生）
                            merged_data.extend(chunk)
                            debug_placeholder.text(f"未在音频块 {i+1} 中找到data标记，添加整个块")
                    
                    # 更新数据大小
                    total_data_size = len(merged_data)
                    
                    # 更新data块大小（data标记后的4字节）
                    size_bytes = total_data_size.to_bytes(4, byteorder='little')
                    
                    # 创建修改后的头部
                    modified_header = bytearray(wav_header)
                    modified_header[data_pos + 4:data_pos + 8] = size_bytes
                    
                    # 更新RIFF大小（文件总大小 - 8）
                    total_file_size = len(modified_header) + total_data_size - 8
                    riff_size_bytes = total_file_size.to_bytes(4, byteorder='little')
                    modified_header[4:8] = riff_size_bytes
                    
                    # 合并头部和数据
                    complete_wav = bytes(modified_header) + bytes(merged_data)
                    debug_placeholder.text(f"二进制合并成功，总大小: {len(complete_wav)} 字节")
                    
                    # 播放完整的音频
                    with audio_placeholder:
                        b64 = base64.b64encode(complete_wav).decode()
                        audio_html = f"""
                            <audio autoplay controls>
                                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                            </audio>
                            """
                        st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    debug_placeholder.text("在第一个音频块中未找到data标记，无法合并")
                    # 回退到播放第一个块
                    with audio_placeholder:
                        b64 = base64.b64encode(all_audio_chunks[0]).decode()
                        audio_html = f"""
                            <audio autoplay controls>
                                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                            </audio>
                            """
                        st.markdown(audio_html, unsafe_allow_html=True)
                    
            except Exception as e:
                debug_placeholder.text(f"二进制合并也失败: {e}")
                debug_placeholder.text("播放第一个音频块作为回退")
                
                # 所有方法都失败，至少播放第一个音频块
                with audio_placeholder:
                    b64 = base64.b64encode(all_audio_chunks[0]).decode()
                    audio_html = f"""
                        <audio autoplay controls>
                            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                        </audio>
                        """
                    st.markdown(audio_html, unsafe_allow_html=True)
                
        return None  # 流式播放模式不返回URL
        
    except Exception as e:
        st.error(f"生成语音时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None



def generate_tts_(text: str) -> Optional[str]:
    """
    Generate text-to-speech audio and return a base64 encoded data URL.
    No UI elements are added, and no direct playback is performed.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Base64 encoded data URL for audio, or None if generation fails
    """
    try:
        # Create TTS request
        tts_url = "http://1.71.15.121:3000/tts/stream"
        payload = {
            "text": text,
            "style": "希望你以后能够做的比我还好呦。",
            "speed": 1.3
        }
        
        # Setup headers for streaming request
        headers = {"Accept": "multipart/mixed; boundary=chunk"}
        
        # Buffer to store complete audio data
        all_audio_chunks = []
        
        # Make streaming request
        with requests.post(
            tts_url, 
            json=payload,
            headers=headers,
            stream=True
        ) as response:
            if response.status_code != 200:
                return None
                
            # Process multipart response stream
            boundary = b"--chunk"
            buffer = b""
            has_end_marker = False
            
            # Process each response chunk
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    buffer += chunk
                    
                    # Check for end marker
                    end_marker_pos = buffer.find(boundary + b"--")
                    if end_marker_pos != -1:
                        has_end_marker = True
                        # Process content before end marker
                        if end_marker_pos > 0:
                            final_part = buffer[:end_marker_pos]
                            if b"Content-Type: audio/wav" in final_part:
                                header_end = final_part.find(b"\r\n\r\n")
                                if header_end != -1:
                                    content = final_part[header_end + 4:]
                                    all_audio_chunks.append(content)
                        buffer = b""
                        continue
                    
                    # Process complete parts
                    while True:
                        # Find boundary
                        boundary_pos = buffer.find(boundary)
                        if boundary_pos == -1:
                            break
                        
                        # Extract part
                        part = buffer[:boundary_pos]
                        buffer = buffer[boundary_pos + len(boundary):]
                        
                        # Skip empty parts
                        if not part.strip():
                            continue
                            
                        # Check for end marker
                        if buffer.startswith(b"--"):
                            has_end_marker = True
                            break
                        
                        # Process headers and content
                        if b"Content-Type: " in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                header = part[:header_end].decode('utf-8', errors='ignore')
                                content = part[header_end + 4:]
                                
                                # Process audio chunks
                                if "audio/wav" in header:
                                    all_audio_chunks.append(content)
            
            # Process remaining buffer
            if buffer and not has_end_marker:
                if b"Content-Type: audio/wav" in buffer:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end != -1:
                        content = buffer[header_end + 4:]
                        all_audio_chunks.append(content)
        
        # Return if no audio chunks collected
        if not all_audio_chunks:
            return None
            
        # If only one chunk, return it directly
        final_audio_data = None
        if len(all_audio_chunks) == 1:
            final_audio_data = all_audio_chunks[0]
        else:
            # Merge multiple audio chunks
            try:
                # Use wave module to merge
                import io
                import wave
                
                output = io.BytesIO()
                
                # Process first audio chunk to get WAV parameters
                with io.BytesIO(all_audio_chunks[0]) as first_chunk_io:
                    with wave.open(first_chunk_io, 'rb') as first_wave:
                        params = first_wave.getparams()
                        
                        # Create output WAV file
                        with wave.open(output, 'wb') as output_wave:
                            # Set parameters
                            output_wave.setparams(params)
                            
                            # Add first chunk data
                            first_wave.rewind()
                            output_wave.writeframes(first_wave.readframes(first_wave.getnframes()))
                            
                            # Process remaining audio chunks
                            for chunk in all_audio_chunks[1:]:
                                try:
                                    with io.BytesIO(chunk) as chunk_io:
                                        with wave.open(chunk_io, 'rb') as chunk_wave:
                                            # Add current chunk data
                                            chunk_frames = chunk_wave.readframes(chunk_wave.getnframes())
                                            output_wave.writeframes(chunk_frames)
                                except Exception:
                                    continue
                
                # Get complete WAV data
                final_audio_data = output.getvalue()
                    
            except Exception:
                # Fallback: direct binary merge
                try:
                    # Extract first chunk WAV header
                    first_chunk = all_audio_chunks[0]
                    data_pos = first_chunk.find(b'data')
                    
                    if data_pos != -1:
                        # Find position after data block
                        header_end = data_pos + 8  # "data" + 4-byte size
                        
                        # Extract header
                        wav_header = first_chunk[:header_end]
                        
                        # Create merged data
                        merged_data = bytearray()
                        
                        # Add first chunk data
                        merged_data.extend(first_chunk[header_end:])
                        
                        # Add remaining chunk data
                        for chunk in all_audio_chunks[1:]:
                            chunk_data_pos = chunk.find(b'data')
                            if chunk_data_pos != -1:
                                # Found data block, skip header
                                chunk_header_end = chunk_data_pos + 8
                                merged_data.extend(chunk[chunk_header_end:])
                            else:
                                # No data block found, add entire chunk
                                merged_data.extend(chunk)
                        
                        # Update data size
                        total_data_size = len(merged_data)
                        
                        # Update data block size (4 bytes after data marker)
                        size_bytes = total_data_size.to_bytes(4, byteorder='little')
                        
                        # Create modified header
                        modified_header = bytearray(wav_header)
                        modified_header[data_pos + 4:data_pos + 8] = size_bytes
                        
                        # Update RIFF size (total file size - 8)
                        total_file_size = len(modified_header) + total_data_size - 8
                        riff_size_bytes = total_file_size.to_bytes(4, byteorder='little')
                        modified_header[4:8] = riff_size_bytes
                        
                        # Merge header and data
                        final_audio_data = bytes(modified_header) + bytes(merged_data)
                    else:
                        # Return first chunk if header parsing fails
                        final_audio_data = all_audio_chunks[0]
                        
                except Exception:
                    # All methods failed, return first chunk
                    final_audio_data = all_audio_chunks[0]
        
        # Create URL from audio data
        if final_audio_data:
            import base64
            # Just return the raw base64 data without creating HTML
            return base64.b64encode(final_audio_data).decode()
        return None
        
    except Exception as e:
        print(f"TTS generation error: {str(e)}")
        return None






def generate_tts_stream(text: str) -> Optional[str]:
    """
    生成文本的语音合成，并在浏览器中播放音频流。
    只负责播放音频，不显示任何额外UI元素。
    
    Args:
        text: 要转换为语音的文本
        
    Returns:
        音频URL，如果使用流式播放则返回None
    """
    try:
        # 创建一个简单的占位符只用于放置音频
        audio_placeholder = st.empty()
        
        # 创建payload
        tts_url = "http://1.71.15.121:3000/tts/stream"
        payload = {
            "text": text,
            "style": "希望你以后能够做的比我还好呦。",
            "speed": 1.3,
            "use_batch": True
        }
        
        # 设置流式请求的头部
        headers = {"Accept": "multipart/mixed; boundary=chunk"}
        
        # 发起流式请求
        with requests.post(
            tts_url, 
            json=payload,
            headers=headers,
            stream=True
        ) as response:
            if response.status_code != 200:
                return None
                
            # 处理多部分响应流
            boundary = b"--chunk"
            buffer = b""
            
            # 处理每个响应块
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    buffer += chunk
                    
                    # 处理完整部分
                    while True:
                        # 查找边界
                        boundary_pos = buffer.find(boundary)
                        if boundary_pos == -1:
                            break
                        
                        # 提取部分
                        part = buffer[:boundary_pos]
                        buffer = buffer[boundary_pos + len(boundary):]
                        
                        # 跳过空部分和结束标记
                        if not part.strip() or buffer.startswith(b"--"):
                            continue
                        
                        # 处理头部和内容
                        if b"Content-Type: " in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                header = part[:header_end].decode('utf-8', errors='ignore')
                                content = part[header_end + 4:]
                                
                                # 处理音频块
                                if "audio/wav" in header:
                                    # 在浏览器中播放
                                    with audio_placeholder:
                                        # 使用base64编码音频数据并创建HTML audio元素
                                        b64 = base64.b64encode(content).decode()
                                        audio_html = f"""
                                            <audio autoplay>
                                                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                                            </audio>
                                            """
                                        st.markdown(audio_html, unsafe_allow_html=True)
            
            return None  # 使用流式播放时不返回URL
        
    except Exception as e:
        return None