#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/05 11:22
@Author  : weiyutao
@File    : llm_config.py
"""
from enum import Enum
from whoami.utils.yaml_model import YamlModel
from typing import Optional

class LLMType(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

    def __missing__(cls, value):
        return cls.OPENAI

class LLMConfig(YamlModel):
    
    api_key: str = ""
    api_type: LLMType = LLMType.OPENAI
    base_url: str = "https://api.openai.com/v1"
    model: Optional[str] = None
    timeout: int = 600
    
    
