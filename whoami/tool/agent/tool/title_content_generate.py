#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/05 17:11
@Author  : weiyutao
@File    : title_content_generate.py
"""


from typing import (
    Any,
    Optional,
    overload
)
from datetime import datetime
from pathlib import Path


from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool import InfoExtract
from whoami.llm_api.base_llm import BaseLLM


@tool
class TitleContentGenerate:
    
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
            
            
    async def execute(self, title1, title2, subject, abstract, search_content):
        # prompt = self.system_prompt.format(
        #     title1=title1,
        #     title2=title2,
        #     subject=subject,
        #     abstract=abstract,
        #     search_content=search_content
        # )
        
        prompt = self.system_prompt.replace("title1", title1)
        prompt = prompt.replace("title2", title2)
        prompt = prompt.replace("subject", subject)
        prompt = prompt.replace("abstract", abstract)
        prompt = prompt.replace("search_content", str(search_content))
        
        messages = [{"role": "user", "content": prompt}]
        result = await self.llm._whoami_text(messages=messages, timeout=30, user_stop_words=[])
        
        return result
        
            