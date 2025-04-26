#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/18 17:04
@Author  : weiyutao
@File    : sensevoice_asr.py
"""

import io
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import os
import tempfile
import asyncio
from typing import (
    Optional
)
from funasr import AutoModel

from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool import ASR



@tool
class SenseVoiceAsr(ASR):
    
    model: Optional[AutoModel] = None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_sensevoice_model(self):
        """
        加载SenseVoice模型，处理可能的错误
        """
        try:
            
            
            # 有两种方式：
            # 1. 如果模型已经下载到本地，直接使用本地路径
            model_dir = '/root/.cache/modelscope/hub/models/iic/SenseVoiceSmall' if self.model_path is None else self.model_path
            if os.path.exists(model_dir):
                self.logger.info(f"使用本地模型路径: {model_dir}")
            else:
                # 2. 如果本地没有，从modelscope下载模型
                self.logger.info("本地未找到模型，从modelscope下载")
                model_dir = snapshot_download('iic/SenseVoiceSmall')
                self.logger.info(f"模型已下载到: {model_dir}")
            
            model = AutoModel(
                model=model_dir,
                trust_remote_code=True,
                disable_update=True,  # 禁用更新检查
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",
            )
            
            self.logger.info("SenseVoice模型加载成功")
            return model
        except Exception as e:
            error_info = f"加载SenseVoice模型失败: {str(e)}"
            self.logger.error(error_info)
            raise ValueError(error_info) from e
    
    
    async def process_audio(self, audio_data: io.BytesIO) -> str:
        if self.model is None:
            self.model = self.load_sensevoice_model()
        try:
            # 保存音频数据到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data.getvalue())
            
            try:

                # 使用SenseVoice处理音频
                res = self.model.generate(
                    input=temp_path,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                )
                
                transcription = rich_transcription_postprocess(res[0]["text"])
                self.logger.info(f"转录成功: {transcription[:50]}...")
                return transcription
            finally:
                # 删除临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            error_info = f"语音处理错误: {str(e)}"
            self.logger.error(error_info)
            raise ValueError(error_info) from e



if __name__ == '__main__':
    sensevoice = SenseVoiceAsr()
    async def main():
        result = await sensevoice.execute(file_path="http://1.71.15.121:3000/audio/5da37b1d-84bc-40ad-b4c9-ba84e2f1f991.wav")
        print(result)
        
    asyncio.run(main())
    