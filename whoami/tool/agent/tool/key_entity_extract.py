#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/05 18:17
@Author  : weiyutao
@File    : key_entity_extract.py
"""

from typing import (
    Any,
    Optional,
    overload
)
from datetime import datetime
from pathlib import Path
import asyncio

from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool import InfoExtract
from whoami.llm_api.base_llm import BaseLLM


@tool
class KeyEntityExtract:
    
    @overload
    def __init__(
        self, 
        llm: Optional[BaseLLM] = None
    ):
        ...
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'llm' in kwargs:
            self.llm = kwargs.get('llm')
    
    
    async def execute(
        self, 
        input
    ):
        prompt = self.system_prompt.format(
            input=input,
        )
        
        messages = [{"role": "user", "content": prompt}]
        result = await self.llm._whoami_text(messages=messages, timeout=30, user_stop_words=[])
        return result
    


if __name__ == '__main__':
    from whoami.llm_api.ollama_llm import OllamaLLM
    from whoami.configs.llm_config import LLMConfig
    llm = OllamaLLM(
        config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')),
        temperature=0.0
    )
    key_entity_extract = KeyEntityExtract(llm=llm)
    
    async def main():
        result = await key_entity_extract.execute("北辰集团与北京银行合作围绕养老金融")
        return result
    result = asyncio.run(main())
    print(result)
    