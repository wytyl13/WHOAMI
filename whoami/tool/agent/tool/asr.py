#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/18 15:49
@Author  : weiyutao
@File    : asr.py
"""

import io
import logging
from pydub import AudioSegment
import tempfile
import os
from abc import ABC, abstractmethod
from typing import (
    Optional,
    List,
    Type
    
)
from pydantic import BaseModel, Field
import aiohttp


from whoami.tool.agent.base_tool import tool



class ASRSchema(BaseModel):
    question: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="用户需要转录的音频数据，可以是原始音频数据也可以是上传的音频文件数据"
    )
    file_path: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="用户需要转录的音频路径，可以是一个绝对路径，也可以是一个url"
    )



@tool
class ASR:
    """
    Automatic Speech Recognition.
    """
    end_flag: int = 0
    args_schema: Type[BaseModel] = ASRSchema
    model_path: Optional[str] = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model_path' in kwargs:
            self.model_path = kwargs.pop('model_path')

    @abstractmethod
    async def process_audio(self, audio_data: io.BytesIO) -> str:
        """
        Process audio data need to implement in inherited class.       
        """
    
    
    async def check_and_prepare_audio_data(self, audio_data=None):
        """
        检查音频数据是否为空，并转换UploadFile为BytesIO对象
        
        Args:
            audio_data: 可以是UploadFile、字节数据或文件类对象
            
        Returns:
            处理后的音频数据（如果为UploadFile，返回BytesIO）
            
        Raises:
            ValueError: 如果数据为空或无法读取
        """
        data_type = type(audio_data).__name__
        self.logger.info(f"检查音频数据 - 类型: {data_type}")
        
        # 检查数据是否为None
        if audio_data is None:
            self.logger.info("音频数据为None")
            raise ValueError("没有提供音频数据")
        
        # 处理UploadFile对象
        if hasattr(audio_data, "filename"):
            self.logger.info(f"处理UploadFile对象 - 文件名: {audio_data.filename}")
            
            # 检查是否有文件名
            if not audio_data.filename:
                self.logger.info("UploadFile没有文件名")
                raise ValueError("上传的文件没有文件名")
            
            # 读取文件内容
            try:
                if hasattr(audio_data.read, "__await__"):
                    self.logger.info("使用异步read方法")
                    content = await audio_data.read()
                else:
                    self.logger.info("使用同步read方法")
                    content = audio_data.read()
                
                
                # Now check if content is a coroutine (this should not happen after awaiting)
                if hasattr(content, "__await__"):
                    self.logger.warning("读取的内容是一个协程，尝试await它")
                    content = await content
                
                self.logger.info(f"读取的数据大小: {len(content) if content else 0} 字节")
                
                # 重置文件指针
                try:
                    if hasattr(audio_data, "seek"):
                        if hasattr(audio_data.seek, "__await__"):
                            await audio_data.seek(0)
                        else:
                            audio_data.seek(0)
                except Exception as seek_error:
                    self.logger.warning(f"重置文件指针时出错: {str(seek_error)}")
                
                # 检查内容是否为空
                if not content or len(content) == 0:
                    self.logger.info("文件内容为空")
                    raise ValueError("上传的文件内容为空")
                
                # 将内容转换为BytesIO对象
                return io.BytesIO(content)
                
            except ValueError:
                # 重新抛出ValueError
                raise
            except Exception as e:
                self.logger.error(f"读取UploadFile内容时出错: {str(e)}")
                raise ValueError(f"无法读取上传的文件内容: {str(e)}")
        
        # 处理bytes对象
        elif isinstance(audio_data, bytes):
            self.logger.info(f"处理bytes对象 - 大小: {len(audio_data)} 字节")
            if len(audio_data) == 0:
                raise ValueError("提供的字节数据为空")
            return io.BytesIO(audio_data)
        
        # 处理BytesIO对象
        elif isinstance(audio_data, io.BytesIO):
            self.logger.info("处理BytesIO对象")
            current_pos = audio_data.tell()
            audio_data.seek(0, io.SEEK_END)
            size = audio_data.tell()
            audio_data.seek(current_pos)  # 恢复原来的位置
            
            self.logger.info(f"BytesIO大小: {size} 字节")
            if size == 0:
                raise ValueError("提供的BytesIO对象为空")
            return audio_data
        
        # 处理一般的文件类对象
        elif hasattr(audio_data, "read"):
            self.logger.info("处理一般文件对象")
            try:
                # 尝试读取整个文件内容
                current_pos = audio_data.tell() if hasattr(audio_data, "tell") else 0
                content = audio_data.read()
                
                if hasattr(audio_data, "seek"):
                    audio_data.seek(current_pos)  # 恢复位置
                
                self.logger.info(f"读取的数据大小: {len(content) if content else 0} 字节")
                if not content or len(content) == 0:
                    raise ValueError("提供的文件对象内容为空")
                
                # 返回原始文件对象
                return audio_data
                
            except ValueError:
                # 重新抛出ValueError
                raise
            except Exception as e:
                self.logger.error(f"读取文件对象内容时出错: {str(e)}")
                raise ValueError(f"无法读取文件对象内容: {str(e)}")
        
        # 未知类型
        else:
            error_msg = f"不支持的音频数据类型: {data_type}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)


    async def execute(self, audio_data=None, file_path=None) -> str:
        """
        执行语音识别
        
        Args:
            audio_data: 可以是文件对象或原始音频数据字节
            file_path: 音频文件路径，可以是本地路径或URL（更高的优先级）
                
        Returns:
            str: 转录的文本结果
        """
        # 如果提供了文件路径，从文件路径获取音频数据
        # 如果提供了文件路径，从文件路径获取音频数据
        if file_path is not None:
            audio_data = await self._load_from_path(file_path)
        
        # 检查并处理音频数据
        try:
            audio_data = await self.check_and_prepare_audio_data(audio_data)
        except ValueError as e:
            error_info = f"音频数据检查失败: {str(e)}"
            self.logger.error(error_info)
            raise ValueError(error_info) from e
        
        
        # 统一处理音频数据为字节流
        audio_bytes = await self._get_audio_bytes(audio_data)
        
        
        # 检测并转换音频格式为wav
        wav_audio = self._convert_to_wav(audio_bytes)
        
        # 调用process_audio处理音频
        transcription = await self.process_audio(wav_audio)
        
        return transcription
    
    
    async def _load_from_path(self, file_path):
        """
        从文件路径加载音频数据
        
        Args:
            file_path: 本地文件路径或URL
            
        Returns:
            bytes: 音频数据
        """
        # 检查是否是URL
        if file_path.startswith(('http://', 'https://', 'ftp://')):
            self.logger.info(f"从URL加载音频: {file_path}")
            # 使用aiohttp异步获取URL内容
            
            async with aiohttp.ClientSession() as session:
                async with session.get(file_path) as response:
                    if response.status != 200:
                        raise ValueError(f"无法下载音频文件，HTTP状态码: {response.status}")
                    return await response.read()
        else:
            # 本地文件路径
            self.logger.info(f"从本地路径加载音频: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到音频文件: {file_path}")
                
            with open(file_path, 'rb') as f:
                return f.read()
    
    
    async def _get_audio_bytes(self, audio_data):
        """将不同形式的音频数据转换为字节流"""
        if hasattr(audio_data, "read"):
            try:
                # Try to use it as a regular file object first
                return io.BytesIO(audio_data.read())
            except TypeError as e:
                # If we get TypeError, it might be because read() is a coroutine
                if "coroutine" in str(e):
                    # It's an async read, await it
                    content = await audio_data.read()
                    return io.BytesIO(content)
                else:
                    # It's some other TypeError, re-raise
                    raise
        elif isinstance(audio_data, bytes):
            # Raw bytes data
            return io.BytesIO(audio_data)
        elif isinstance(audio_data, io.BytesIO):
            # Already a BytesIO object
            return audio_data
        else:
            raise ValueError(f"不支持的音频数据类型: {type(audio_data)}")
    
    
    
    
    def _convert_to_wav(self, audio_bytes):
        """
        检测音频格式并转换为wav格式
        
        Args:
            audio_bytes: 音频数据的BytesIO对象
                
        Returns:
            BytesIO: wav格式的音频数据
        """
        self.logger.info("开始音频格式检测和转换")
        
        # 记录输入数据类型
        self.logger.info(f"输入数据类型: {type(audio_bytes)}")
        
        # 检查是否为BytesIO对象
        if not isinstance(audio_bytes, io.BytesIO):
            self.logger.info(f"将{type(audio_bytes)}转换为BytesIO对象")
            if isinstance(audio_bytes, bytes):
                temp_bytes = io.BytesIO(audio_bytes)
                audio_bytes = temp_bytes
            else:
                self.logger.error(f"无法处理的音频数据类型: {type(audio_bytes)}")
                raise ValueError(f"音频数据必须是BytesIO对象或bytes，实际类型: {type(audio_bytes)}")
        
        # 记录数据大小
        audio_bytes.seek(0, io.SEEK_END)
        data_size = audio_bytes.tell()
        audio_bytes.seek(0)
        self.logger.info(f"音频数据大小: {data_size} 字节")
        
        # 检查是否有足够的数据
        if data_size < 12:
            self.logger.warning(f"音频数据太小 ({data_size} 字节)，可能不是有效音频")
        
        # 保存到临时文件以检测格式
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_bytes.getvalue())
            temp_path = temp_file.name
            self.logger.info(f"音频数据保存到临时文件: {temp_path}")
        
        try:
            # 首先检查文件头部是否为WAV
            with open(temp_path, 'rb') as f:
                header = f.read(12)
                header_hex = ' '.join([f'{b:02X}' for b in header])
                self.logger.info(f"文件头12字节: {header_hex}")
                
                is_wav = header.startswith(b'RIFF') and b'WAVE' in header
                if is_wav:
                    self.logger.info("文件头部识别为WAV格式，无需转换")
                    
                    # 进一步验证WAV格式的正确性
                    try:
                        import wave
                        with wave.open(temp_path, 'rb') as wav_file:
                            channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()
                            frame_rate = wav_file.getframerate()
                            frames = wav_file.getnframes()
                            self.logger.info(f"WAV文件参数: 通道数={channels}, 采样宽度={sample_width}字节, "
                                            f"采样率={frame_rate}Hz, 帧数={frames}, 时长={frames/frame_rate:.2f}秒")
                    except Exception as e:
                        self.logger.warning(f"虽然头部像WAV，但验证WAV格式失败: {str(e)}")
                        is_wav = False
                        
                if is_wav:
                    audio_bytes.seek(0)
                    return audio_bytes
                else:
                    self.logger.info("不是WAV格式，需要转换")
            
            # 尝试检测其他音频格式
            try:
                # 读取更多字节来识别其他格式
                with open(temp_path, 'rb') as f:
                    header = f.read(32)  # 读取更多字节以识别其他格式
                
                format_name = "unknown"
                if header.startswith(b'\xFF\xFB') or header.startswith(b'ID3'):
                    format_name = "mp3"
                    self.logger.info("检测到MP3格式")
                elif header.startswith(b'OggS'):
                    format_name = "ogg"
                    self.logger.info("检测到OGG格式")
                elif header.startswith(b'fLaC'):
                    format_name = "flac"
                    self.logger.info("检测到FLAC格式")
                else:
                    # 检查是否可能是PCM数据
                    import numpy as np
                    try:
                        # 尝试将数据加载为16位PCM样本
                        with open(temp_path, 'rb') as f:
                            pcm_data = np.fromfile(f, dtype=np.int16)
                        
                        # 计算一些统计数据来判断是否像PCM
                        mean = np.mean(pcm_data)
                        std = np.std(pcm_data)
                        max_val = np.max(np.abs(pcm_data))
                        
                        self.logger.info(f"PCM数据统计: 均值={mean:.2f}, 标准差={std:.2f}, 最大绝对值={max_val}")
                        
                        # 如果数据看起来像音频PCM（有合理的均值、标准差和峰值）
                        if 0 <= max_val <= 32767 and std > 100:
                            format_name = "pcm"
                            self.logger.info("数据统计特征符合PCM音频，假设为PCM格式")
                        else:
                            self.logger.warning("数据统计特征不符合典型PCM音频")
                    except Exception as e:
                        self.logger.warning(f"PCM检测失败: {str(e)}")
                
                # 尝试使用PyDub加载音频
                self.logger.info(f"尝试使用PyDub加载音频，假设格式为: {format_name}")
                
                if format_name != "pcm":
                    try:
                        audio = AudioSegment.from_file(temp_path, format=format_name if format_name != "unknown" else None)
                        self.logger.info(f"成功加载为{format_name}格式, 采样率={audio.frame_rate}Hz, 通道数={audio.channels}, "
                                        f"采样宽度={audio.sample_width}字节, 时长={len(audio)/1000:.2f}秒")
                    except Exception as e:
                        self.logger.warning(f"使用PyDub加载{format_name}格式失败: {str(e)}")
                        # 如果其他格式加载失败，尝试PCM
                        format_name = "pcm"
                
                if format_name == "pcm":
                    # 尝试不同的PCM参数
                    pcm_configs = [
                        {"sample_width": 2, "frame_rate": 16000, "channels": 1},  # 常见ESP32配置
                        {"sample_width": 2, "frame_rate": 8000, "channels": 1},   # 低采样率
                        {"sample_width": 2, "frame_rate": 44100, "channels": 1},  # CD音质
                        {"sample_width": 2, "frame_rate": 22050, "channels": 1},  # 中等质量
                    ]
                    
                    # 尝试所有配置
                    for config in pcm_configs:
                        try:
                            self.logger.info(f"尝试PCM参数: {config}")
                            audio = AudioSegment.from_raw(
                                temp_path, 
                                sample_width=config["sample_width"], 
                                frame_rate=config["frame_rate"], 
                                channels=config["channels"]
                            )
                            self.logger.info(f"成功加载PCM, 配置: {config}, 时长={len(audio)/1000:.2f}秒")
                            break
                        except Exception as e:
                            self.logger.warning(f"使用配置{config}加载PCM失败: {str(e)}")
                    else:
                        # 所有尝试都失败，使用默认参数
                        self.logger.warning("所有PCM参数尝试均失败，使用默认参数")
                        audio = AudioSegment.from_raw(temp_path, sample_width=2, frame_rate=16000, channels=1)
                
                # 转换为wav
                self.logger.info("开始转换为WAV格式")
                wav_bytes = io.BytesIO()
                audio.export(wav_bytes, format="wav")
                wav_bytes.seek(0)
                
                # 检查转换后的WAV文件大小
                wav_bytes.seek(0, io.SEEK_END)
                wav_size = wav_bytes.tell()
                wav_bytes.seek(0)
                self.logger.info(f"转换后WAV大小: {wav_size} 字节")
                
                # 验证WAV格式
                try:
                    import wave
                    with io.BytesIO(wav_bytes.getvalue()) as temp_wav:
                        wav_file = wave.open(temp_wav, 'rb')
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        frame_rate = wav_file.getframerate()
                        frames = wav_file.getnframes()
                        self.logger.info(f"转换后WAV参数: 通道数={channels}, 采样宽度={sample_width}字节, "
                                        f"采样率={frame_rate}Hz, 帧数={frames}, 时长={frames/frame_rate:.2f}秒")
                except Exception as e:
                    self.logger.warning(f"验证转换后的WAV失败: {str(e)}")
                
                return wav_bytes
                    
            except Exception as e:
                # 如果所有尝试都失败，记录错误并尝试基本PCM转换
                self.logger.error(f"所有格式检测和转换尝试失败: {str(e)}")
                self.logger.info("最后尝试作为16kHz单声道PCM处理")
                
                try:
                    audio = AudioSegment.from_raw(temp_path, sample_width=2, frame_rate=16000, channels=1)
                    wav_bytes = io.BytesIO()
                    audio.export(wav_bytes, format="wav")
                    wav_bytes.seek(0)
                    self.logger.info("基本PCM到WAV转换完成")
                    return wav_bytes
                except Exception as final_e:
                    self.logger.error(f"基本PCM转换也失败: {str(final_e)}")
                    raise ValueError(f"无法处理音频数据: {str(final_e)}")
                    
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                self.logger.info(f"临时文件已删除: {temp_path}")
    
    
    
    
    def _convert_to_wav_bake(self, audio_bytes):
        """
        检测音频格式并转换为wav格式
        
        Args:
            audio_bytes: 音频数据的BytesIO对象
                
        Returns:
            BytesIO: wav格式的音频数据
        """
        # 保存到临时文件以检测格式
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_bytes.getvalue())
            temp_path = temp_file.name
        
        try:
            # 首先检查文件头部是否为WAV
            with open(temp_path, 'rb') as f:
                header = f.read(12)
                is_wav = header.startswith(b'RIFF') and b'WAVE' in header
                if is_wav:
                    self.logger.info("文件头部识别为WAV格式")
                    audio_bytes.seek(0)
                    return audio_bytes
            
            # 尝试加载音频以检测格式
            try:
                # PyDub doesn't provide a direct way to get the format
                # Instead, we'll try to open it as a general file and check the file extension
                audio = AudioSegment.from_file(temp_path)
                # Get the format from the file extension or assume mp3 if unknown
                format_name = os.path.splitext(temp_path)[1]
                if format_name.startswith('.'):
                    format_name = format_name[1:]  # Remove leading dot
                if not format_name:
                    format_name = "unknown"
                self.logger.info(f"检测到音频格式: {format_name}")
                
                # If we got here, we have a valid audio file, convert to WAV
                wav_bytes = io.BytesIO()
                audio.export(wav_bytes, format="wav")
                wav_bytes.seek(0)
                return wav_bytes
                
            except Exception as e:
                # 如果无法检测格式，假设为pcm并尝试转换
                self.logger.info(f"无法检测音频格式，原因: {str(e)}，假设为pcm")
                audio = AudioSegment.from_raw(temp_path, sample_width=2, frame_rate=16000, channels=1)
                
                # 转换为wav
                wav_bytes = io.BytesIO()
                audio.export(wav_bytes, format="wav")
                wav_bytes.seek(0)
                return wav_bytes
                
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)