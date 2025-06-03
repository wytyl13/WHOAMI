#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/07 10:44
@Author  : weiyutao
@File    : sleep_indices_extract.py
"""

from typing import (
    Any
)
from datetime import datetime
from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool import InfoExtract

@tool
class SleepIndicesExtract(InfoExtract):
    """睡眠指标关键字提取
    从用户输入的内容中提取需要查询的睡眠报告相关的关键字
    Args:
        InfoExtract (_type_): _description_
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'llm' in kwargs:
            self.llm = kwargs.get('llm')
            
        if 'default_extract_result' in kwargs:
            self.default_extract_result = kwargs.get('default_extract_result')
            
        if 'field_description' in kwargs:
            self.field_description = kwargs.get('field_description')
            
        self.system_prompt = self.system_prompt.replace("{field_description}", str(self.field_description))


if __name__ == '__main__':
    from whoami.provider.sql_provider import SqlProvider
    from whoami.tool.health_report.sleep_indices import SleepIndices
    from whoami.llm_api.ollama_llm import OllamaLLM
    from whoami.configs.llm_config import LLMConfig
    from pathlib import Path
    
    llm_qwen = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))
    sql_provider = SqlProvider(model=SleepIndices, sql_config_path='/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml')
    field_description = sql_provider.get_field_names_and_descriptions()
    print(field_description)
    sleep_indice_extract = SleepIndicesExtract(llm=llm_qwen, field_description=field_description)
    
    import asyncio
    async def main():
        print(sleep_indice_extract.system_prompt)
        result = await sleep_indice_extract.execute(question="汇报下昨天的心率情况")
        print(result)
    asyncio.run(main())
