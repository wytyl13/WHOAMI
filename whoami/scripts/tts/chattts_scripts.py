#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/19
@File    : chattts_fastapi_server.py
FastAPI server for ChatTTS text-to-speech service with streaming response
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any, Generator
import asyncio
import base64
import cn2an
import io
import wave
import traceback
import os
from datetime import datetime
import shutil
import uuid
from pydub import AudioSegment
import numpy as np
from scipy import signal


from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from whoami.tool.agent.tool import ChatTTSImpl, AudioType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define request models
class TTSRequest(BaseModel):
    text: str
    pcm_flag: Optional[int] = 0


class AudioChunkRequest(BaseModel):
    audio_chunks: List[str]  # base64 encoded audio chunks
    chunk_texts: Optional[List[str]] = None


class ChatTTSHTTPServer:
    """FastAPI server for ChatTTS text-to-speech service with streaming response"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 3000,
        max_chunk_length: int = 20,
        chattts_impl: Optional[ChatTTSImpl] = None
    ):
        self.host = host
        self.port = port
        self.max_chunk_length = max_chunk_length
        self.chattts_impl = chattts_impl or ChatTTSImpl()
        self.app = FastAPI(title="ChatTTS HTTP API", description="Stream TTS audio chunks via HTTP")
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )
        
        # Register routes
        self.register_routes()
    
    def register_routes(self):
        """Register FastAPI routes"""
        
        from fastapi.staticfiles import StaticFiles
        # 定义静态文件目录路径
        output_dir = "/work/ai/WHOAMI/whoami/tool/tts/output"
        # 添加静态文件路由
        self.app.mount("/out", StaticFiles(directory=output_dir), name="out")
        
        
        @self.app.post("/tts")
        async def text_to_speech(request: TTSRequest):
            """
            Process text-to-speech request and stream audio chunks
            """
            text = request.text
            pcm_flag = request.pcm_flag
            
            if not text:
                raise HTTPException(status_code=400, detail="No text provided")
            
            logger.info(f"Received text: {text[:50]}...")
            
            # Use StreamingResponse to stream audio chunks
            return StreamingResponse(
                self.stream_audio(text, pcm_flag),
                media_type="application/octet-stream"
            )
            
        @self.app.post("/tts_url")
        async def tts_url(request: TTSRequest):
            """
            Process text-to-speech request and return audio file URL
            """
            text = request.text
            
            if not text:
                raise HTTPException(status_code=400, detail="No text provided")
            
            logger.info(f"Received text: {text[:50]}...")
            
            # 获取音频文件路径
            file_path = await self.stream_audio_url(text)
            
            if not file_path:
                raise HTTPException(status_code=500, detail="Failed to generate audio")
            
            # 从文件路径创建URL（根据实际部署情况调整）
            # 假设文件路径格式为: /work/ai/WHOAMI/whoami/tool/tts/output/tts_xxx.wav
            file_name = os.path.basename(file_path)
            url = f"http://1.71.15.121:3000/out/{file_name}"
            response = {"url": url, "file_path": file_path}
            logger.info(f"response: {response}")
            # 返回JSON响应，包含音频文件URL
            return response
        
        
        
        @self.app.post("/tts/stream")
        async def text_to_speech_with_metadata(request: TTSRequest):
            """
            Process text-to-speech request and stream audio chunks with metadata
            Returns a multipart stream with JSON metadata and binary audio chunks
            """
            text = request.text
            pcm_flag = request.pcm_flag
            
            if not text:
                raise HTTPException(status_code=400, detail="No text provided")
            
            logger.info(f"Received text: {text[:50]}...")
            
            return StreamingResponse(
                self.stream_audio_with_metadata(text, pcm_flag),
                media_type="multipart/mixed; boundary=chunk"
            )
    
    
    def split_text_bake(self, text: str, max_size: int = 50) -> List[str]:
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
            if len(sentence) <= max_size:
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
            if len(sentence) > max_size:
                if current_chunk:
                    final_result.append(current_chunk)
                    current_chunk = ""
                final_result.append(sentence)
            # 否则尝试组合
            elif len(current_chunk) + len(sentence) <= max_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    final_result.append(current_chunk)
                current_chunk = sentence
        
        # 添加最后一个组合
        if current_chunk:
            final_result.append(current_chunk)
        
        return final_result
    
    
    def split_text(self, text: str) -> List[str]:
        """
        智能分割文本为句子，确保尽可能不拆分完整句子
        
        Args:
            text: 要分割的文本
        
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
    
    

    def wav_to_pcm(self, wav_data: bytes) -> bytes:
        """Convert WAV audio data to PCM format with fixed 8000 Hz sample rate
        optimized for ESP32 I2S compatibility"""
        
        logger.info("wav_to_pcm ---------------------------------------------")
        try:
            import numpy as np
            from scipy import signal
            
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    # Get WAV parameters
                    channels = wav_file.getnchannels() 
                    width = wav_file.getsampwidth()
                    original_rate = wav_file.getframerate()
                    
                    logger.info(f"Original WAV: channels={channels}, width={width}, rate={original_rate}")
                    
                    # Get PCM data
                    pcm_data = wav_file.readframes(wav_file.getnframes())
                    
                    # Convert to numpy array for resampling (16-bit PCM)
                    audio = np.frombuffer(pcm_data, dtype=np.int16)
                    
                    
                    if channels == 2:
                        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    logger.info("Converted stereo to mono")
                    
                    
                    
                    # Target rate is fixed at 8000 Hz
                    target_rate = 16000
                    
                    # Calculate resampling ratio and new length
                    ratio = target_rate / original_rate
                    new_length = int(len(audio) * ratio)
                    
                    logger.info(f"Resampling from {original_rate}Hz to {target_rate}Hz (ratio: {ratio:.3f})")
                    
                    # Resample using scipy.signal.resample
                    resampled = signal.resample(audio, new_length)
                    
                    # Ensure the data is aligned to the proper boundaries (important for ESP32 I2S)
                    # ESP32 I2S can be sensitive to buffer alignment
                    resampled = resampled.astype(np.int16)
                    
                    # Some ESP32 I2S implementations perform better when the data size is a multiple of 4
                    # Add padding if necessary
                    if len(resampled.tobytes()) % 4 != 0:
                        pad_size = 4 - (len(resampled.tobytes()) % 4)
                        padding = np.zeros(pad_size // 2, dtype=np.int16)
                        resampled = np.concatenate([resampled, padding])
                        logger.info(f"Added {pad_size} bytes of padding for alignment")
                    
                    # Convert back to bytes
                    pcm_bytes = resampled.tobytes()
                    logger.info(f"PCM data size: {len(pcm_bytes)} bytes")
                    
                    return pcm_bytes
        except ImportError:
            logger.error("Resampling requires numpy and scipy. Please install these packages.")
            raise
        except Exception as e:
            logger.error(f"Error converting WAV to PCM: {str(e)}")
            raise
    
    
    
    def digits_to_chinese(self, s, use_yao=False):
        num_map = {
            '0': '零',
            '1': '幺' if use_yao else '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
            '7': '七',
            '8': '八',
            '9': '九',
        }
        return ''.join(num_map[c] for c in s)


    def minute_to_chinese(self, mm):
        if mm == '00':
            return '整'
        elif mm.startswith('0'):
            return '零' + cn2an.an2cn(int(mm[1])) + '分'
        elif mm == '10':
            return '十分'
        else:
            return cn2an.an2cn(int(mm)) + '分'


    def time_range_repl(self, m):
        h1, m1 = m.group(1), m.group(2)
        h2, m2 = m.group(3), m.group(4)
        left = cn2an.an2cn(int(h1)) + '点' + self.minute_to_chinese(m1)
        right = cn2an.an2cn(int(h2)) + '点' + self.minute_to_chinese(m2)
        return f"{left}至{right}"


    def time_repl(self, m):
        h, mi = m.group(1), m.group(2)
        h_txt = cn2an.an2cn(int(h))
        mi_txt = self.minute_to_chinese(mi)
        return f"{h_txt}点{mi_txt}"
    
    
    def preprocess_chattts_text(self, sentence: str):
        if not sentence:
            return ""
    
        units = ["分钟", "小时", "秒", "天", "周", "月", "年", "次", "千米", "米", "厘米", 
                "毫米", "公里", "千克", "克", "毫克", "吨", "升", "毫升", "度", "摄氏度", "华氏度", "%"]

        # 1. 区号-手机号：加特殊TAG防止二次转换
        def phone_with_code_repl(m):
            code = m.group(1)
            number = m.group(2)
            # 用<tag>包裹区号部分，防止后续被改写
            code_cn = self.digits_to_chinese(code)
            number_cn = self.digits_to_chinese(number, use_yao=True)
            return f"<code>{code_cn}</code>-{number_cn}"
        sentence = re.sub(r'\b(\d{2,3})[-—](1\d{10})\b', phone_with_code_repl, sentence)
        sentence = re.sub(r'\+\s?(\d{2,3})[-—](1\d{10})\b', phone_with_code_repl, sentence)

        # 2. 处理11位手机号（非区号场景）
        def phone_repl(m):
            return self.digits_to_chinese(m.group(0), use_yao=True)
        sentence = re.sub(r'(?<!\d)(1\d{10})(?!\d)', phone_repl, sentence)

        # 3. 处理年份，逐位读
        sentence = re.sub(
            r'(\d{4})年',
            lambda m: self.digits_to_chinese(m.group(1)) + '年',
            sentence,
        )

        # 4. 处理时间区间
        sentence = re.sub(
            r'(\d{1,2}):(\d{2})\s*[-~—]\s*(\d{1,2}):(\d{2})',
            self.time_range_repl, sentence
        )

        # 5. 处理单个时间
        sentence = re.sub(
            r'(\d{1,2}):(\d{2})(?!\d)', self.time_repl, sentence
        )

        # 6. 百分比
        sentence = re.sub(r'(\d+)%', lambda m: '百分之' + cn2an.an2cn(int(m.group(1))), sentence)

        # 7. 比例表达
        sentence = re.sub(r'(\d+)\s*:\s*(\d+)', lambda m: cn2an.an2cn(int(m.group(1))) + '比' + cn2an.an2cn(int(m.group(2))), sentence)

        # 8. 单位表达
        for unit1 in units:
            for unit2 in units:
                pattern = rf'(\d+)\s*{unit1}/({unit2})'
                sentence = re.sub(pattern, lambda m: cn2an.an2cn(int(m.group(1))) + unit1 + '每' + m.group(2), sentence)

        sentence = re.sub(r'(\d+)\s*[°℃]\s*C?', lambda m: cn2an.an2cn(int(m.group(1))) + '摄氏度', sentence)
        sentence = re.sub(r'(\d+)\s*[°℉]\s*F?', lambda m: cn2an.an2cn(int(m.group(1))) + '华氏度', sentence)

        for unit in units:
            pattern = rf'(\d+)\s*{unit}'
            sentence = re.sub(pattern, lambda m: cn2an.an2cn(int(m.group(1))) + unit, sentence)

        # 9. 日期处理
        sentence = re.sub(
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            lambda m: self.digits_to_chinese(m.group(1)) + '年' + cn2an.an2cn(int(m.group(2))) + '月' + cn2an.an2cn(int(m.group(3))) + '日',
            sentence
        )
        sentence = re.sub(
            r'(\d{4})/(\d{1,2})/(\d{1,2})',
            lambda m: self.digits_to_chinese(m.group(1)) + '年' + cn2an.an2cn(int(m.group(2))) + '月' + cn2an.an2cn(int(m.group(3))) + '日',
            sentence
        )
        sentence = re.sub(
            r'(\d{1,2})/(\d{1,2})(?!/)',
            lambda m: cn2an.an2cn(int(m.group(1))) + '月' + cn2an.an2cn(int(m.group(2))) + '日',
            sentence
        )

        # 10. 标点和特殊符号转换为空格
        pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9<>/-]'
        def replace_with_space(match):
            matched_text = match.group(0)
            for unit in units:
                if matched_text in unit:
                    return matched_text
            return " "
        sentence = re.sub(pattern, replace_with_space, sentence)
        
        # 11. 剩余数字转为汉字（保护带有<tag>的区号）
        # 只替换未被<tag>包裹的数字
        def safe_cn2an(m):
            # 如果在<code>标签内则跳过
            if m.group(0).startswith('<code>') and m.group(0).endswith('</code>'):
                return m.group(0)
            else:
                return cn2an.an2cn(int(m.group(0)))
        # 先把<code>部分临时替换为特殊字符，避免被正则命中
        code_list = []
        def code_save(m):
            code_list.append(m.group(0))
            return f'[[[CODE{len(code_list)-1}]]]'
        sentence = re.sub(r'<code>.*?</code>', code_save, sentence)
        # 替换剩余数字
        sentence = re.sub(r'\d+', lambda m: cn2an.an2cn(int(m.group(0))), sentence)
        # 恢复<code>部分
        def code_restore(m):
            idx = int(m.group(1))
            # 去掉<code>标签
            return code_list[idx][6:-7]
        sentence = re.sub(r'\[\[\[CODE(\d+)\]\]\]', code_restore, sentence)
        
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence.strip()
    
    
    def _remove_trailing_noise_pydub(self, audio_data: bytes, silence_threshold: float = -40, 
                                min_silence_ms: int = 100, fade_duration_ms: int = 50) -> bytes:
        """
        使用pydub库处理音频末尾的杂音
        
        参数:
            audio_data: WAV格式的音频字节数据
            silence_threshold: 静音阈值(dBFS)，低于此值被视为静音
            min_silence_ms: 检测静音的最小长度(毫秒)
            fade_duration_ms: 淡出效果的持续时间(毫秒)
            
        返回:
            处理后的音频字节数据
        """
        # 加载音频数据
        audio = AudioSegment.from_wav(io.BytesIO(audio_data))
        
        # 计算分析窗口大小
        chunk_length_ms = min_silence_ms
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        if len(chunks) <= 2:  # 音频太短，不处理
            return audio_data
        
        # 从后向前查找第一个非静音区域
        trim_index = len(chunks) - 1
        
        # 跳过最后一个chunk，因为它可能包含部分杂音
        for i in range(len(chunks) - 2, 0, -1):
            if chunks[i].dBFS >= silence_threshold:
                # 找到了最后的非静音区域
                # 保留到这个非静音区域后一个chunk
                trim_index = i + 1
                break
        
        # 截取到指定位置，保留一些余量以防止截断过度
        safe_margin_chunks = 1  # 保留额外的chunk数量
        trim_ms = min((trim_index + safe_margin_chunks) * chunk_length_ms, len(audio))
        trimmed_audio = audio[:trim_ms]
        
        # 添加淡出效果以平滑结尾
        fade_ms = min(fade_duration_ms, len(trimmed_audio) // 5)  # 确保淡出不超过音频长度的1/5
        if fade_ms > 10:  # 只有当有足够长度时才应用淡出
            trimmed_audio = trimmed_audio.fade_out(fade_ms)
        
        # 转回字节
        buffer = io.BytesIO()
        trimmed_audio.export(buffer, format="wav")
        buffer.seek(0)
        return buffer.read()
    
    
    async def process_text_chunk(self, chunk: str, audio_type: str = 'wav_bytes') -> bytes:
        """Process a single text chunk and return audio data"""
        try:
            audio_data = await self.chattts_impl.execute(
                text=chunk,
                output_path=None,
                audio_type="wav_bytes"
            )
            # audio_data = self._remove_trailing_noise_pydub(audio_data)
            return audio_data
        except Exception as e:
            logger.error(f"Error processing text chunk: {str(e)}")
            raise
    
    
    def resample(self, audio_data, target_sample_rate, original_sample_rate):
        try:
            # 将PCM数据转换为numpy数组以便处理
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            
            # 计算重采样比例
            resampling_ratio = target_sample_rate / original_sample_rate
            
            # 计算新的采样点数
            new_num_samples = int(len(audio_data) * resampling_ratio)
            
            # 使用scipy的resample函数重采样
            resampled_audio = signal.resample(audio_data, new_num_samples)
            
            # 将重采样后的数据转换回int16格式
            resampled_audio = resampled_audio.astype(np.int16)
            
            # 转换回bytes
            resampled_pcm_data = resampled_audio.tobytes()
        except Exception as e:
            raise ValueError("resample error: {str(e)}") from e
        
        return resampled_pcm_data
    
    
    async def stream_audio_url(self, text: str) -> str:
        """
        处理文本并合并PCM音频数据为WAV文件
        """
        try:
            # 1. 分割文本
            text_chunks = self.split_text(text)
            logger.info(f"Split text into {len(text_chunks)} chunks")
            
            # 2. 收集所有PCM数据
            all_pcm_data = bytearray()
            original_sample_rate = 24000  # 根据你的实际采样率设置
            target_sample_rate = 8000
            channels = 1         # 单声道
            sampwidth = 2        # 16-bit = 2字节
            
            for i, chunk in enumerate(text_chunks):
                chunk = self.preprocess_chattts_text(chunk)
                logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")
                pcm_data = await self.process_text_chunk(chunk, audio_type='pcm_type')  # 确保返回的是PCM数据
                
                if pcm_data and isinstance(pcm_data, bytes):
                    all_pcm_data.extend(pcm_data)
                    logger.info(f"Added {len(pcm_data)} bytes, total: {len(all_pcm_data)}")
                else:
                    logger.warning(f"Chunk {i+1} returned invalid data")
            
            if not all_pcm_data:
                logger.error("No PCM data collected")
                return ""
            
            
            # 3. 重采样处理
            resampled_pcm_data = self.resample(all_pcm_data, target_sample_rate, original_sample_rate)
            
            # 3. 创建输出目录
            output_dir = "/work/ai/WHOAMI/whoami/tool/tts/output"
            os.makedirs(output_dir, exist_ok=True)
            
            # 4. 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"tts_{timestamp}_{unique_id}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # 5. 写入WAV文件
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setparams((
                    channels,         # 声道数
                    sampwidth,       # 采样字节宽度
                    target_sample_rate,     # 采样率
                    len(all_pcm_data) // (channels * sampwidth),  # 总帧数
                    'NONE',          # 压缩类型
                    'not compressed' # 压缩描述
                ))
                wav_file.writeframes(resampled_pcm_data)
            
            logger.info(f"Successfully saved WAV file: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    
    
    
    async def stream_audio(self, text: str, pcm_flag: int = 0) -> Generator[bytes, None, None]:
        """
        Stream audio chunks for the given text
        Simple binary stream
        """
        # Split text into chunks
        text_chunks = self.split_text(text)
        logger.info(f"Split text into {len(text_chunks)} chunks")
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk}")
            audio_data = await self.process_text_chunk(chunk)
            
            # Convert to PCM if flag is set
            if pcm_flag:
                audio_data = self.wav_to_pcm(audio_data)
            
            yield audio_data
    
    
    async def stream_audio_with_metadata(self, text: str, pcm_flag: int = 0) -> Generator[bytes, None, None]:
        """
        Stream audio chunks with metadata for the given text
        Returns a multipart stream with JSON metadata and binary audio chunks
        """
        # Split text into chunks
        text_chunks = self.split_text(text)
        total_chunks = len(text_chunks)
        logger.info(f"Split text into {total_chunks} chunks")
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{total_chunks}: {chunk}")
            chunk_text = self.preprocess_chattts_text(chunk)
            # Create metadata
            metadata = {
                "status": "chunk",
                "chunk_index": i,
                "total_chunks": total_chunks,
                "chunk_text": chunk,
                "format": "pcm" if pcm_flag else "wav"
            }
            
            # Send metadata
            metadata_json = json.dumps(metadata)
            yield b"--chunk\r\n"
            yield b"Content-Type: application/json\r\n\r\n"
            yield metadata_json.encode('utf-8')
            yield b"\r\n"

            audio_type = 'pcm_data' if pcm_flag != 0 else 'wav_data'
            # Process text and get audio
            audio_data = await self.process_text_chunk(chunk_text, audio_type='pcm_data')
            
            audio_data = self.resample(audio_data, 8000, 24000) if audio_data == 'pcm_data' else audio_data
            
            # Convert to PCM if flag is set
            if pcm_flag:
                audio_data = self.wav_to_pcm(audio_data)
            
            
            # Send audio data
            yield b"--chunk\r\n"
            yield b"Content-Type: audio/wav\r\n\r\n"
            yield audio_data
            yield b"\r\n"
        
        # End of multipart stream
        yield b"--chunk--\r\n"
    
    def start(self):
        """Start the HTTP server using uvicorn"""
        import uvicorn
        logger.info(f"Starting ChatTTS HTTP server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


def main():
    """Main function to start the server"""
    server = ChatTTSHTTPServer()
    server.start()


if __name__ == "__main__":
    main()