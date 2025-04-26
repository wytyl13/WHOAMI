#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/19
@File    : chattts_client.py
Client example for ChatTTS FastAPI server
"""

import asyncio
import json
import aiohttp
import io
import wave
import pyaudio
import re
from typing import List, Dict, Any, Optional

# Server configuration
SERVER_URL = "http://localhost:3000"


class ChatTTSClient:
    """Client for ChatTTS FastAPI server"""
    
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
        # Initialize PyAudio for audio playback
        self.p = pyaudio.PyAudio()
    
    async def text_to_speech_simple(self, text: str) -> bytes:
        """
        Send text to TTS server and get audio data (simple approach)
        Returns the complete audio data as bytes
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/tts",
                json={"text": text}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Error from server: {error_text}")
                
                # Get complete response as bytes
                audio_data = await response.read()
                return audio_data
    
    async def text_to_speech_stream(self, text: str) -> None:
        """
        Send text to TTS server and play audio chunks as they arrive
        Uses the streaming endpoint with metadata
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/tts/stream",
                json={"text": text}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Error from server: {error_text}")
                
                # Process multipart stream
                boundary = b"--chunk"
                content_type = response.headers.get("Content-Type", "")
                if "boundary=chunk" in content_type:
                    # Read the response as it streams
                    buffer = b""
                    current_part_type = None
                    metadata = None
                    
                    # Create stream reader
                    async for chunk in response.content.iter_any():
                        buffer += chunk
                        
                        # Process complete parts
                        while True:
                            # Find boundary
                            boundary_pos = buffer.find(boundary)
                            if boundary_pos == -1:
                                break
                            
                            # Extract part
                            part = buffer[:boundary_pos]
                            buffer = buffer[boundary_pos + len(boundary):]
                            
                            # Skip empty parts and end markers
                            if not part.strip() or buffer.startswith(b"--"):
                                continue
                            
                            # Process header and content
                            if b"Content-Type: " in part:
                                header_end = part.find(b"\r\n\r\n")
                                if header_end != -1:
                                    header = part[:header_end].decode('utf-8', errors='ignore')
                                    content = part[header_end + 4:]
                                    
                                    # Determine content type
                                    if "application/json" in header:
                                        # Process metadata
                                        try:
                                            metadata = json.loads(content)
                                            print(f"Received metadata: {metadata}")
                                        except json.JSONDecodeError:
                                            print("Failed to parse metadata JSON")
                                    
                                    elif "audio/wav" in header:
                                        # Play audio chunk
                                        if metadata:
                                            print(f"Playing audio for: {metadata.get('chunk_text', '')}")
                                        
                                        # Play the audio chunk
                                        self.play_audio(content)
                else:
                    print("Response is not in multipart format")
    
    def play_audio(self, audio_data: bytes) -> None:
        """Play audio data using PyAudio"""
        try:
            # Parse WAV data
            wav_file = io.BytesIO(audio_data)
            with wave.open(wav_file, 'rb') as wf:
                # Open a stream
                stream = self.p.open(
                    format=self.p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                # Read data
                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)
                
                # Close stream
                stream.stop_stream()
                stream.close()
        except Exception as e:
            print(f"Error playing audio: {str(e)}")
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'p'):
            self.p.terminate()


async def main():
    """Main function to test the client"""
    client = ChatTTSClient()
    
    try:
        text = "当地时间4月18日，美国哥伦比亚广播公司报道称，多名消息人士透露，由于特朗普政府对中国商品加征的畸高关税将导致供应链危机，特朗普政府内部已开始讨论组建一个工作组，以便在未能与中国政府谈判取得突破的情况下紧急处理这些问题。消息人士称，目前有关成立该小组的具体情况还未敲定任何细节，但工作组可能包括副总统万斯、财政部长贝森特、商务部长卢特尼克、白宫国家经济委员会主任凯文·哈西特、白宫经济顾问委员会主席斯蒂芬·米兰和美国贸易代表贾米森·格里尔。针对美方对华加征畸高关税，我外交部已多次回应。4月17日，外交部发言人再次强调，美方对华轮番加征畸高关税已经沦为数字游戏，在经济上已无实际意义，只会更加暴露出美方将关税工具化、武器化，搞霸凌胁迫的伎俩。关税战、贸易战没有赢家，中方不愿打，但也不怕打。如果美方继续玩弄关税数字游戏，中方将不予理会。倘若美方执意继续实质性侵害中方权益，中方将坚决反制，奉陪到底。"
        print(f"Converting text to speech: {text}")
        
        # Use streaming version (with metadata)
        await client.text_to_speech_stream(text)
        
        # Alternatively, get complete audio (simple approach)
        # audio_data = await client.text_to_speech_simple(text)
        # client.play_audio(audio_data)
    
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())