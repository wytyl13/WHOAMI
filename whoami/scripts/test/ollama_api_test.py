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
    health_data = sql_provider.get_record_by_condition({"device_sn": "13D6F349200080712111952D07", "query_date": "2025-1-16"})
    if health_data:
        del health_data[0]['breath_bpm_image_x_y']
        del health_data[0]['heart_bpm_image_x_y']
    
    # health_prompt = f"""
    # You are a professional health doctor, please give professional description based on the user's health data, The health data fields correspond standard as follows:
    # {filed_description}
    
    # The breakdown of the user's health data is as follows:
    # {health_data}
    
    # note:
    # - Focus on recommendations
    # - Output as plain Chinese text
    # """
    
    
    health_prompt = f"""
    请您作为一位专业的睡眠健康医生，基于以下睡眠监测数据生成一段专业的健康分析描述。
    
    睡眠数据：
    {filed_description}
    
    健康标准区间：
    {health_data}
    
    分析要求：
    1. 严格对照实际数据和标准区间进行分析，确保数值完全准确
    2. 以一段流畅的文字呈现，不使用特殊符号或分段
    3. 优先分析异常指标，重点说明其偏离标准范围的程度
    4. 结合所有指标给出专业的整体诊断判断
    5. 使用专业医学视角，但确保描述通俗易懂
    6. 在描述最后给出针对性的改善建议
    7. 控制总体描述在150字以内
    """
    
    
    # health_prompt = """
    # 帮我颠倒输出这句话单词
    # Yesterday, my TV stopped working. Now, I can’t turn it on at all.
    # """
    # health_prompt = """
    # 帮我颠倒输出这句话
    # 我是谁，我来自哪里，我要到哪里去？
    # """
    
    content = llm.whoami(health_prompt, stream=False, user_stop_words=[])
    print(content)