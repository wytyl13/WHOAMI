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
from whoami.llm_api.ollama_llm import OllamLLM
from whoami.configs.llm_config import LLMConfig
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sleep_indices import SleepIndices


async def main():
    """异步调用失败！同步调用正常"""
    llm = OllamLLM(LLMConfig.from_file(Path("/home/weiyutao/work/WHOAMI/whoami/scripts/test/ollama_config.yaml")))
    content = await llm.whoami("我是谁")  # 现在可以正确await
    return content

from whoami.tool.disease_predict.sx_disease_predict import SxDiseasePredict

sx_disease_predict_sql_provider = SqlProvider(sql_config_path="/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml", model=SxDiseasePredict)

if __name__ == '__main__':
    # asyncio.run(main())  # 使用 asyncio.run 异步调用失败！
    
    """同步调用正常"""
    llm = OllamLLM(
        LLMConfig.from_file(Path("/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml")), 
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
    
    
    # health_data = sql_provider.get_record_by_condition(
    #     condition={"device_sn": "13D6F349200080712111952D07", "query_date": "2025-1-19"}, 
    #     exclude_fields=
    #     [
    #         'health_advice',
    #         'sleep_stage_image_x_y',
    #         'body_move_image_x_y',
    #         'breath_exception_image_sixty_x_y',
    #         'heart_bpm_image_x_y',
    #         'breath_bpm_image_x_y',
    #         'breath_exception_count',
    #         'breath_exception_image_x_y'
    #     ]
    # )
    health_data = sql_provider.get_record_by_condition(
        condition={"query_date": "2025-2-20"}, 
        exclude_fields=
        [
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
    请您作为一位专业的睡眠健康医生，基于以下睡眠监测数据和标准去见对异常数据进行分析并给出建议
    
    字段描述：
    {filed_description}
    
    健康标准区间：
    {health_standard_reference}
    
    实际睡眠数据：
    {health_data}
    
    分析要求：
    1. 严格对照实际数据和标准区间进行分析，确保数值完全准确，内容简洁，不要出现患者等第三人称字眼。
    2. 仅分析异常指标，重点说明其偏离标准范围的程度，并给出异常指标可能造成的身体健康隐患。
    3. 使用专业医学视角，但确保描述通俗易懂，必须在最后给出针对性的改善建议。
    4. 控制总体描述在150字以内，以一段流畅的文字呈现，不使用特殊符号或分段。
    """
    
    data = {
		"id": "主键id",
		"device_sn": "设备SN码",
		"cardiovascular_disease": "心血管疾病",
		"respiratory_disease": "呼吸系统疾病",
		"neurological_disease": "神经系统疾病",
		"sleep_disorder": "睡眠障碍",
		"metabolic_disease": "代谢性疾病",
		"mental_health_disease": "心理健康问题",
		"infectious_disease": "感染性疾病",
		"creator": "创建者",
		"create_time": "创建时间",
		"deleted": "是否删除"
	}
    
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
    
    
    # health_prompt = """
    # 帮我颠倒输出这句话单词
    # Yesterday, my TV stopped working. Now, I can’t turn it on at all.
    # """
    # health_prompt = """
    # 帮我颠倒输出这句话
    # 我是谁，我来自哪里，我要到哪里去？
    # """
    # print(health_prompt)
        content = llm.whoami(health_prompt, stream=False, user_stop_words=[])
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
                sx_disease_predict_sql_provider.add_record(save_sql_data)
                
            except json.JSONDecodeError as e:
                print(f"JSON 解析出错: {e}")
        else:
            print("未找到json")