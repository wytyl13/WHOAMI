#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/18 15:44
@Author  : weiyutao
@File    : tts.py
"""
from typing import (
    Type,
    Optional,
    Union
)
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import io
import torchaudio
import torch
from enum import Enum
import numpy as np
import tempfile
import os


from whoami.tool.agent.base_tool import tool


class StrEnum(str, Enum):
    def __str__(self) -> str:
        # overwrite the __str__ method to implement enum_instance.attribution == enum_instance.attribution.value
        return self.value
    
    def __repr__(self) -> str:
        return f"'{str(self)}'"


class TTSSchema(BaseModel):
    text: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="需要进行语音合成的文本"
    )
    


class AudioType(StrEnum):
    """Audio type"""
    wav_bytes = "wav_bytes"
    pcm_bytes = "pcm_bytes"
    


@tool
class TTS:
    """
    text to speech
    """
    end_flag: int = 0
    args_schema: Type[BaseModel] = TTSSchema
    audio_type: Optional[AudioType] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'audio_type' in kwargs:
            self.audio_type = kwargs.pop('audio_type')
        self.audio_type = AudioType if self.audio_type is None else self.audio_type
    
    
    
    @abstractmethod
    async def text_audio(self, text=None) -> str:
        """
        Process text data to audio data need to implement in inherited class.       
        """
    
    
    async def execute(
        self, 
        text=None, 
        audio_type: AudioType = None,
        output_path: str = None
    ) -> Union[str, bytes, np.ndarray]:
        """
        执行语音合成
        
        Args:
            text: 需要合成的文本
                
        Returns:
            str: 语音合成结果
            bytes: 原始音频
        """
        if text is None:
            raise ValueError("text must not be null!")

        
        result = await self.text_audio(text=text)
        
        if audio_type == AudioType.pcm_bytes:
            # 转换为PCM字节格式 (16位有符号整数)
            pcm_data = (result * 32767).astype(np.int16).tobytes()
            return pcm_data
        
        elif audio_type == AudioType.wav_bytes:
            buffer = io.BytesIO()
            # 将numpy数组转换为PyTorch张量并保存为WAV
            torchaudio.save(buffer, torch.from_numpy(result).unsqueeze(0), 24000, format="wav")
            buffer.seek(0)
            return buffer.read()

        if output_path is None:
            raise ValueError("OUTPUT PATH MUST NOT BE NULL IF YOU NOT SET THE AUDIO TYPE")
        

        try:
            torchaudio.save(output_path, torch.from_numpy(result).unsqueeze(0), 24000)
        except Exception as e:
            raise ValueError("FAIL TO SAVE THE AUDIO DATA TO WAV FILE!")
        return output_path
    
    
        
        