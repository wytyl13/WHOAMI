#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/31 09:03
@Author  : weiyutao
@File    : test_ollama_api.py
"""

import asyncio

from pathlib import Path
from whoami.llm_api.ollama_llm import OllamLLM
from whoami.configs.llm_config import LLMConfig
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sleep_indices import SleepIndices


async def main():
    """异步调用失败！同步调用正常"""
    llm = OllamLLM(LLMConfig.from_file(Path("/home/weiyutao/work/WHOAMI/whoami/scripts/test/ollama_config.yaml")))
    content = await llm.whoami("我是谁")  # 现在可以正确await
    return content


if __name__ == '__main__':
    # asyncio.run(main())  # 使用 asyncio.run 异步调用失败！
    
    """同步调用正常"""
    llm = OllamLLM(
        LLMConfig.from_file(Path("/home/weiyutao/work/WHOAMI/whoami/scripts/test/ollama_config.yaml")), 
        temperature=0.0
    )
    
    sql_provider = SqlProvider(sql_config_path="/home/weiyutao/work/WHOAMI/whoami/scripts/health_report/sql_config.yaml", model=SleepIndices)
    filed_description = sql_provider.get_field_names_and_descriptions()
    health_data = sql_provider.get_record_by_condition({"device_sn": "13D6F349200080712111952D07", "query_date": "2024-12-28"})
    if health_data:
        del health_data[0]['breath_bpm_image_x_y']
        del health_data[0]['heart_bpm_image_x_y']
    
    health_prompt = f"""
    You are a professional health doctor, please give professional advice based on the user's health data, The health data fields correspond as follows:
    {filed_description}
    
    The breakdown of the user's health data is as follows:
    {health_data}
    
    note:
    - Focus on recommendations
    - Output as plain Chinese text
    """
    
    content = llm.whoami(health_prompt, stream=False, user_stop_words=[])
    print(content)