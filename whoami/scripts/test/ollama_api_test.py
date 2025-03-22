#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/31 09:03
@Author  : weiyutao
@File    : test_ollama_api.py
"""

import asyncio
import re
import json

from pathlib import Path
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sleep_indices import SleepIndices
from whoami.tool.disease_predict.sx_disease_predict import SxDiseasePredict


async def main():
    """异步调用失败！同步调用正常"""
    llm = OllamaLLM(LLMConfig.from_file(Path("/work/ai/WHOAMI/whoami/scripts/test/ollama_config_deepseek.yaml")))
    content = await llm.whoami("我是谁")  # 现在可以正确await
    return content


sx_disease_predict_sql_provider = SqlProvider(sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml", model=SxDiseasePredict)

async def main():
    """异步主函数"""
    llm = OllamaLLM(
        LLMConfig.from_file(Path("/work/ai/WHOAMI/whoami/scripts/test/ollama_config_deepseek.yaml")), 
        temperature=0.0
    )
    
    sql_provider = SqlProvider(sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml", model=SleepIndices)
    filed_description = sql_provider.get_field_names_and_descriptions()
    
    health_standard_reference = {
        "waking_second": [0, 1860],
        "sleep_efficiency": [0.8, 1],
        "sleep_second": [21600, 36000],
        "deep_sleep_efficiency": [0.2, 0.6],
        "leave_count": [0, 2],
        "to_sleep_second": [0, 1800],
        "body_move_exponent": [1.25, 15],
        "breath_bpm": [8, 22],
        "heart_bpm": [40, 100],
        "breath_exception_exponent": [0, 5]
    }
    
    health_data = sql_provider.get_record_by_condition(
        condition={"query_date": "2025-3-19"}, 
        exclude_fields=[
            'health_advice',
            'sleep_stage_image_x_y',
            'body_move_image_x_y',
            'breath_exception_image_sixty_x_y',
            'heart_bpm_image_x_y',
            'breath_bpm_image_x_y',
            'breath_exception_count',
            'breath_exception_image_x_y'
        ]
    )
    
    description = sx_disease_predict_sql_provider.get_field_names_and_descriptions()    
    reverse_description = {v: k for k, v in description.items()}
    
    for item in health_data:
        device_sn = item["device_sn"]
        health_prompt = f"""
        请根据下面的睡眠数据得到该患者患每一种疾病的概率。
        
        疾病描述：
        心血管疾病, 呼吸系统疾病, 神经系统疾病, 睡眠障碍, 代谢性疾病, 心理健康问题, 感染性疾病
        
        健康标准区间：
        {health_standard_reference}
        
        实际睡眠数据：
        {item}
        
        分析要求：
        1. 仅输出每种疾病的概率值即可
        2. 以字典的形式输出，键是疾病名称，值是概率值，值不能为空，不能为none，必须大于0小于1
        """
        
        messages = [{"role": "user", "content": health_prompt}]
        # 在这里使用await，因为_whoami_text是异步函数
        content = await llm._whoami_text(messages=messages, timeout=30, user_stop_words=[])
        cleaned_response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        json_match = re.search(r'\{[\s\S]*\}', cleaned_response)
        if json_match:
            json_string = json_match.group(0)
            try:
                # 将提取的 JSON 字符串解析为字典
                data = json.loads(json_string)
                save_sql_data = {}
                for key, value in data.items():
                    save_sql_data[reverse_description[key]] = value
                    
                save_sql_data["device_sn"] = device_sn
                print(save_sql_data)
                # 如果add_record是异步函数，需要加await
                # await sx_disease_predict_sql_provider.add_record(save_sql_data)
                # 如果是同步函数，则直接调用
                sx_disease_predict_sql_provider.add_record(save_sql_data)
                
            except json.JSONDecodeError as e:
                print(f"JSON 解析出错: {e}")
        else:
            print("未找到json")

if __name__ == '__main__':
    # 使用asyncio.run来运行异步主函数
    asyncio.run(main())