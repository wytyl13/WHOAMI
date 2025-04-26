#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/19
@File    : chattts_server.py
WebSocket server for ChatTTS text-to-speech service
"""

import asyncio
import json
import logging
import re
from typing import List, Optional, Dict, Any
import base64
import websockets
from websockets.server import WebSocketServerProtocol
from typing import (
    List,
)

from whoami.tool.agent.tool import ChatTTSImpl, AudioType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChatTTSServer:
    """WebSocket server for ChatTTS text-to-speech service"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 3000,
        max_chunk_length: int = 50,
        chattts_impl: Optional[ChatTTSImpl] = None
    ):
        self.host = host
        self.port = port
        self.max_chunk_length = max_chunk_length
        self.chattts_impl = chattts_impl or ChatTTSImpl()
        self.clients = set()
    
    async def register(self, websocket: WebSocketServerProtocol) -> None:
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"New client connected. Total clients: {len(self.clients)}")
    
    
    async def unregister(self, websocket: WebSocketServerProtocol) -> None:
        """Unregister a client connection"""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    
    def split_text(self, text: str) -> List[str]:
        """
        智能分割文本为句子，确保尽可能不拆分完整句子
        
        Args:
            text: 要分割的文本
            max_size: 建议的最大字符数，但完整句子会被保留
        
        Returns:
            分割后的句子列表
        """
        # 主要标点符号分割（这些标点表示句子的结束）
        main_punct_pattern = r'([。！？\.!\?])'
        sentences = re.split(main_punct_pattern, text)
        
        # 将标点符号重新附加到句子
        complete_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and re.match(main_punct_pattern, sentences[i+1]):
                # 当前片段 + 标点符号
                complete_sentences.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                # 没有标点的片段
                if sentences[i].strip():
                    complete_sentences.append(sentences[i])
                i += 1
        
        # 次要标点符号（分号、冒号等）作为合并的考虑因素
        secondary_punct_pattern = r'([；：;:](?!\d))'  # 添加负向前瞻，避免匹配时间中的冒号
        refined_sentences = []
        
        for sentence in complete_sentences:
            if not sentence.strip():
                continue
            
            # 如果句子较短，直接添加
            if len(sentence) <= self.max_chunk_length:
                refined_sentences.append(sentence)
            else:
                # 尝试按次要标点分割长句
                subparts = re.split(secondary_punct_pattern, sentence)
                
                temp_parts = []
                i = 0
                while i < len(subparts):
                    if i + 1 < len(subparts) and re.match(secondary_punct_pattern, subparts[i+1]):
                        # 当前片段 + 标点符号
                        temp_parts.append(subparts[i] + subparts[i+1])
                        i += 2
                    else:
                        if subparts[i].strip():
                            temp_parts.append(subparts[i])
                        i += 1
                
                # 保留这些分割
                refined_sentences.extend(temp_parts)
        
        # 尝试组合短句，但保留长句
        final_result = []
        current_chunk = ""
        
        for sentence in refined_sentences:
            # 如果句子本身已经超过最大长度，直接保留
            if len(sentence) > self.max_chunk_length:
                if current_chunk:
                    final_result.append(current_chunk)
                    current_chunk = ""
                final_result.append(sentence)
            # 否则尝试组合
            elif len(current_chunk) + len(sentence) <= self.max_chunk_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    final_result.append(current_chunk)
                current_chunk = sentence
        
        # 添加最后一个组合
        if current_chunk:
            final_result.append(current_chunk)
            
        return final_result
    
    
    async def process_text(self, text: str) -> List[bytes]:
        """Process text and return audio data chunks"""
        try:
            # Split text into chunks
            text_chunks = self.split_text(text)
            logger.info(f"Split text into {len(text_chunks)} chunks")
            
            # 不再存储所有音频块，而是处理一个发送一个
            audio_chunks = []
            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
                audio_data = await self.chattts_impl.execute(
                    text=chunk,
                    output_path=None,
                    audio_type="wav_bytes"
                )
                # 直接在这里发送而不是返回所有块
                yield (i, chunk, audio_data)
            
                
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
    
    
    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle client connection and messages"""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    # Parse the incoming message
                    data = json.loads(message)
                    
                    # Check message type
                    message_type = data.get("type", "text")
                    
                    if message_type == "text":
                        # Handle text-to-speech request
                        text = data.get("text", "")
                        if not text:
                            await websocket.send(json.dumps({
                                "status": "error",
                                "message": "No text provided"
                            }))
                            continue
                        
                        logger.info(f"Received text: {text[:50]}...")
                        
                        # Send acknowledgment
                        await websocket.send(json.dumps({
                            "status": "processing",
                            "message": "Processing your text"
                        }))
                        
                        # Process text and get audio chunks
                        total_chunks = len(self.split_text(text))
                        async for i, chunk_text, audio_data in self.process_text(text):
                            # 发送每个块的元数据
                            await websocket.send(json.dumps({
                                "status": "chunk",
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_text": chunk_text
                            }))
                        
                            # 发送音频数据
                            await websocket.send(audio_data)
                        
                        
                    elif message_type == "audio_chunks":
                        # Handle pre-processed audio chunks
                        audio_chunks_b64 = data.get("audio_chunks", [])
                        chunk_texts = data.get("chunk_texts", [])
                        
                        if not audio_chunks_b64:
                            await websocket.send(json.dumps({
                                "status": "error",
                                "message": "No audio chunks provided"
                            }))
                            continue
                        
                        logger.info(f"Received {len(audio_chunks_b64)} pre-processed audio chunks")
                        
                        # Convert base64 audio data to bytes
                        audio_chunks = [base64.b64decode(chunk) for chunk in audio_chunks_b64]
                        
                        # Send the chunks back to the client
                        await self.handle_audio_chunks(websocket, audio_chunks, chunk_texts)
                    
                    else:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "message": f"Unknown message type: {message_type}"
                        }))
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": f"Error processing request: {str(e)}"
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        finally:
            await self.unregister(websocket)
    
    
    async def start_server(self) -> None:
        """Start the WebSocket server"""
        logger.info(f"Starting ChatTTS WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port
        ):
            # Keep the server running
            await asyncio.Future()


async def main():
    """Main function to start the server"""
    server = ChatTTSServer()
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())