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
    



def generate_tts_fei(text: str) -> None:
    """
    生成文本的语音合成，使用流式播放方式立即播放每个收到的音频片段。
    采用 st.components.v1.html 和自定义 JavaScript 实现无缝播放。
    
    Args:
        text: 要转换为语音的文本
    """
    try:
        # 创建一个组件占位符
        player_container = st.empty()
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
        
        # 初始化JavaScript播放器
        # 这个播放器使用Web Audio API实现流式播放
        init_player_js = """
        <div id="audio-status">等待音频流...</div>
        <div id="audio-controls" style="display:none">
            <button id="play-pause-btn" onclick="togglePlayPause()">暂停</button>
            <div style="margin-top:10px">
                <span id="current-time">0:00</span> / 
                <span id="duration">0:00</span>
            </div>
            <progress id="progress-bar" value="0" max="100" style="width:100%"></progress>
        </div>
        
        <script>
        // 创建音频上下文和播放队列
        let audioContext = null;
        let audioQueue = [];
        let isPlaying = false;
        let currentSource = null;
        let startTime = 0;
        let totalDuration = 0;
        let currentTime = 0;
        let nextStartTime = 0;
        let bufferCount = 0;
        
        // 初始化音频上下文
        function initAudioContext() {
            try {
                // 创建音频上下文
                window.AudioContext = window.AudioContext || window.webkitAudioContext;
                audioContext = new AudioContext();
                document.getElementById('audio-status').textContent = '音频播放器已准备就绪';
                isPlaying = true;
                startTime = audioContext.currentTime;
                nextStartTime = startTime;
                
                // 更新UI
                document.getElementById('audio-controls').style.display = 'block';
                
                // 开始UI更新定时器
                setInterval(updateUI, 500);
                
                return true;
            } catch (e) {
                document.getElementById('audio-status').textContent = '无法初始化音频播放器: ' + e.message;
                return false;
            }
        }
        
        // 将base64编码的WAV转换为AudioBuffer
        async function decodeAudioData(base64Data) {
            const byteString = atob(base64Data);
            const arrayBuffer = new ArrayBuffer(byteString.length);
            const uint8Array = new Uint8Array(arrayBuffer);
            
            for (let i = 0; i < byteString.length; i++) {
                uint8Array[i] = byteString.charCodeAt(i);
            }
            
            return await audioContext.decodeAudioData(arrayBuffer);
        }
        
        // 添加新音频块到队列并播放
        async function addAudioChunk(base64Data) {
            if (!audioContext && !initAudioContext()) {
                return;
            }
            
            try {
                bufferCount++;
                document.getElementById('audio-status').textContent = `正在处理音频块 #${bufferCount}`;
                
                // 解码音频数据
                const audioBuffer = await decodeAudioData(base64Data);
                
                // 将音频块加入队列并安排播放
                scheduleAudioBuffer(audioBuffer);
                
                document.getElementById('audio-status').textContent = `已排队播放音频块 #${bufferCount}`;
            } catch (e) {
                document.getElementById('audio-status').textContent = `处理音频块 #${bufferCount} 时出错: ${e.message}`;
            }
        }
        
        // 安排音频缓冲区的播放
        function scheduleAudioBuffer(audioBuffer) {
            if (!isPlaying) {
                audioQueue.push(audioBuffer);
                return;
            }
            
            // 创建音频源
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            
            // 计算当前播放时间
            const now = audioContext.currentTime;
            
            // 如果nextStartTime已经过去，则从现在开始播放
            if (nextStartTime < now) {
                nextStartTime = now;
            }
            
            // 安排在正确的时间播放
            source.start(nextStartTime);
            
            // 更新下一个开始时间
            nextStartTime += audioBuffer.duration;
            
            // 更新总持续时间
            totalDuration += audioBuffer.duration;
            
            // 存储当前播放的源
            if (!currentSource) {
                currentSource = source;
            }
            
            // 当这个源播放完毕后，移除它
            source.onended = function() {
                if (currentSource === source) {
                    currentSource = null;
                }
            };
        }
        
        // 更新UI显示
        function updateUI() {
            if (!audioContext) return;
            
            // 计算当前播放位置
            currentTime = audioContext.currentTime - startTime;
            if (currentTime > totalDuration) {
                currentTime = totalDuration;
            }
            
            // 更新进度条
            const progressBar = document.getElementById('progress-bar');
            progressBar.value = totalDuration > 0 ? (currentTime / totalDuration) * 100 : 0;
            
            // 更新时间显示
            document.getElementById('current-time').textContent = formatTime(currentTime);
            document.getElementById('duration').textContent = formatTime(totalDuration);
        }
        
        // 格式化时间为 分:秒 格式
        function formatTime(seconds) {
            const minutes = Math.floor(seconds / 60);
            seconds = Math.floor(seconds % 60);
            return `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
        
        // 播放/暂停切换
        function togglePlayPause() {
            if (isPlaying) {
                // 暂停
                audioContext.suspend();
                isPlaying = false;
                document.getElementById('play-pause-btn').textContent = '播放';
            } else {
                // 恢复播放
                audioContext.resume();
                isPlaying = true;
                document.getElementById('play-pause-btn').textContent = '暂停';
                
                // 如果队列中有待播放的音频块，开始播放它们
                while (audioQueue.length > 0 && isPlaying) {
                    const buffer = audioQueue.shift();
                    scheduleAudioBuffer(buffer);
                }
            }
        }
        
        // 音频全部播放完成的回调
        function onPlaybackComplete() {
            document.getElementById('audio-status').textContent = '播放完成';
        }
        </script>
        """
        
        # 首先渲染播放器
        player_container.components.v1.html(init_player_js, height=150)
        
        debug_placeholder.text("正在发送TTS请求...")
        
        # 创建Streamlit回调函数，用于JavaScript交互
        def add_audio_chunk_callback(chunk_b64):
            # 构造JavaScript代码调用addAudioChunk函数
            js_code = f"""
            <script>
            (function() {{
                // 调用父窗口中的addAudioChunk函数
                parent.document.querySelector('[data-testid="stHtmlFrame"]').contentWindow.addAudioChunk('{chunk_b64}');
            }})();
            </script>
            """
            # 使用一个临时组件执行JavaScript
            st.components.v1.html(js_code, height=0)
        
        # 发起流式请求
        with requests.post(
            tts_url, 
            json=payload,
            headers=headers,
            stream=True
        ) as response:
            if response.status_code != 200:
                debug_placeholder.text(f"TTS服务请求失败: {response.status_code}")
                return
                
            # 处理多部分响应流
            boundary = b"--chunk"
            buffer = b""
            chunk_count = 0
            
            # 处理每个响应块
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    buffer += chunk
                    
                    # 特殊处理：检查是否有结束标记
                    end_marker_pos = buffer.find(boundary + b"--")
                    if end_marker_pos != -1:
                        # 确保在结束标记前的内容也被处理
                        if end_marker_pos > 0:
                            final_part = buffer[:end_marker_pos]
                            # 处理最后一部分数据
                            if b"Content-Type: audio/wav" in final_part:
                                header_end = final_part.find(b"\r\n\r\n")
                                if header_end != -1:
                                    content = final_part[header_end + 4:]
                                    chunk_count += 1
                                    debug_placeholder.text(f"找到并播放最终音频块 {chunk_count}!")
                                    # 立即播放这个块
                                    chunk_b64 = base64.b64encode(content).decode()
                                    add_audio_chunk_callback(chunk_b64)
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
                                    debug_placeholder.text(f"接收并播放音频块 {chunk_count}")
                                    
                                    # 立即将此音频块发送到JavaScript播放器
                                    chunk_b64 = base64.b64encode(content).decode()
                                    add_audio_chunk_callback(chunk_b64)
            
            # 处理缓冲区中剩余的最后一个部分（可能是最后一个音频块）
            if buffer:
                if b"Content-Type: audio/wav" in buffer:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end != -1:
                        content = buffer[header_end + 4:]
                        chunk_count += 1
                        debug_placeholder.text(f"从剩余缓冲区找到并播放最后一个音频块 {chunk_count}")
                        
                        # 播放最后一个块
                        chunk_b64 = base64.b64encode(content).decode()
                        add_audio_chunk_callback(chunk_b64)
        
        debug_placeholder.text(f"完成！已播放 {chunk_count} 个音频块")
        
        # 播放完成的回调
        complete_js = """
        <script>
        (function() {
            // 调用父窗口中的完成回调函数
            parent.document.querySelector('[data-testid="stHtmlFrame"]').contentWindow.onPlaybackComplete();
        })();
        </script>
        """
        st.components.v1.html(complete_js, height=0)
                
    except Exception as e:
        st.error(f"生成流式语音时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())




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




def play_audio_chunks(audio_chunks, audio_placeholder, debug_placeholder):
    """
    播放收集到的音频块。
    使用AudioQueue模式在JavaScript中控制音频块的连续播放。
    
    Args:
        audio_chunks: 音频块列表
        audio_placeholder: Streamlit占位符用于显示音频
        debug_placeholder: Streamlit占位符用于显示调试信息
    """
    if not audio_chunks:
        return
        
    try:
        # 创建一个序列化的音频块数组供JavaScript使用
        chunks_b64 = [base64.b64encode(chunk).decode() for chunk in audio_chunks]
        chunks_json = json.dumps(chunks_b64)
        
        # 创建一个JavaScript音频队列播放器
        audio_player_html = f"""
        <div id="audio-player-container"></div>
        <script>
        (function() {{
            // 当前音频索引
            let currentIndex = 0;
            // 音频数据
            const audioChunks = {chunks_json};
            // 是否已经初始化
            let initialized = false;
            
            function playNextChunk() {{
                // 如果已经播放完所有块，则返回
                if (currentIndex >= audioChunks.length) {{
                    console.log("所有音频块播放完成");
                    return;
                }}
                
                // 获取当前音频块的base64数据
                const chunk = audioChunks[currentIndex];
                const audioElement = new Audio(`data:audio/wav;base64,${{chunk}}`);
                
                // 在音频播放结束时播放下一个块
                audioElement.onended = function() {{
                    currentIndex++;
                    playNextChunk();
                }};
                
                // 开始播放
                console.log(`播放音频块 ${{currentIndex + 1}}/${{audioChunks.length}}`);
                audioElement.play().catch(e => {{
                    console.error("播放出错:", e);
                    // 尝试自动播放失败时，创建一个可见的播放按钮
                    if (!initialized) {{
                        const container = document.getElementById("audio-player-container");
                        const button = document.createElement("button");
                        button.innerText = "点击开始播放";
                        button.onclick = function() {{
                            audioElement.play();
                            this.disabled = true;
                            initialized = true;
                        }};
                        container.appendChild(button);
                    }}
                }});
            }}
            
            // 开始播放第一个块
            playNextChunk();
        }})();
        </script>
        """
        
        with audio_placeholder:
            st.markdown(audio_player_html, unsafe_allow_html=True)
            
    except Exception as e:
        debug_placeholder.text(f"播放音频块时出错: {e}")
        
        # 回退方案：至少播放第一个块
        try:
            first_chunk = audio_chunks[0]
            b64 = base64.b64encode(first_chunk).decode()
            audio_html = f"""
                <audio autoplay controls>
                    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                </audio>
            """
            with audio_placeholder:
                st.markdown(audio_html, unsafe_allow_html=True)
            debug_placeholder.text("回退到仅播放第一个音频块")
        except Exception as e2:
            debug_placeholder.text(f"回退播放也失败: {e2}")
        
    except Exception as e:
        st.error(f"生成语音时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None







def generate_tts_stream_bake(text: str) -> Optional[str]:
    """
    生成文本的语音合成，以流式方式播放音频。
    在接收到每个音频片段时即时播放，而不是等待所有片段合并后再播放。
    
    Args:
        text: 要转换为语音的文本
        
    Returns:
        音频URL，如果使用流式播放则返回None
    """
    try:
        # 创建一个占位符用于显示当前正在播放的音频
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
        
        debug_placeholder.text("正在发送TTS请求...")
        
        # 创建一个会话变量用于跟踪当前播放状态
        if 'tts_chunk_index' not in st.session_state:
            st.session_state.tts_chunk_index = 0
        
        # 重置会话变量
        st.session_state.tts_chunk_index = 0
        
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
                        
                        # 跳过空部分
                        if not part.strip():
                            continue
                            
                        # 检查是否是结束标记
                        if buffer.startswith(b"--"):
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
                                    chunk_index = st.session_state.tts_chunk_index
                                    st.session_state.tts_chunk_index += 1
                                    debug_placeholder.text(f"接收并播放音频块 {chunk_count}")
                                    
                                    # 即时播放这个音频块
                                    b64 = base64.b64encode(content).decode()
                                    audio_html = f"""
                                        <audio autoplay id="tts_chunk_{chunk_index}">
                                            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                                        </audio>
                                        <script>
                                            // 添加监听器检测音频播放结束
                                            document.getElementById("tts_chunk_{chunk_index}").onended = function() {{
                                                // 音频播放完毕后触发的事件
                                                console.log("音频块 {chunk_index} 播放完成");
                                            }};
                                        </script>
                                    """
                                    with audio_placeholder:
                                        st.markdown(audio_html, unsafe_allow_html=True)
                                        
                                    # 为了确保流畅播放，添加一个小延迟
                                    import time
                                    time.sleep(0.1)
            
            # 处理缓冲区中剩余的最后一个部分
            if buffer:
                if b"Content-Type: audio/wav" in buffer:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end != -1:
                        content = buffer[header_end + 4:]
                        chunk_count += 1
                        chunk_index = st.session_state.tts_chunk_index
                        st.session_state.tts_chunk_index += 1
                        debug_placeholder.text(f"接收并播放最后一个音频块 {chunk_count}")
                        
                        # 播放最后一个音频块
                        b64 = base64.b64encode(content).decode()
                        audio_html = f"""
                            <audio autoplay id="tts_chunk_{chunk_index}">
                                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                            </audio>
                        """
                        with audio_placeholder:
                            st.markdown(audio_html, unsafe_allow_html=True)
        
        debug_placeholder.text(f"总共接收并播放了 {chunk_count} 个音频块")
        return None  # 流式播放模式不返回URL
        
    except Exception as e:
        st.error(f"生成语音时出错: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None





def generate_tts_stream_(text: str) -> Optional[str]:
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