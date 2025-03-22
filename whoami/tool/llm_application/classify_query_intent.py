
import json
from pathlib import Path
import asyncio
from typing import (
    List,
    Dict,
    Optional
)

from whoami.configs.llm_config import LLMConfig
from whoami.llm_api.ollama_llm import OllamaLLM

llm=OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml')))
from pathlib import Path
import asyncio

from whoami.configs.llm_config import LLMConfig
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.tool.base.base_tool import BaseTool

llm=OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml')))

class ClassifyQueryIntent(BaseTool):
    query: Optional[str] = None
    classify_topic: Optional[str] = None
    classify_topic_info: Optional[str] = None
    classify_topic_info_standard: Optional[str] = None
    sleep_keywords: Optional[List[str]] = None
    def __init__(
        self, 
        query: Optional[str] = None,
        classify_topic: Optional[str] = None,
        classify_topic_info: Optional[str] = None,
        classify_topic_info_standard: Optional[str] = None,
        sleep_keywords: Optional[List[str]] = None
    ):
        super().__init__()
        self.query = query
        self.classify_topic = classify_topic
        self.classify_topic_info = classify_topic_info
        self.classify_topic_info_standard = classify_topic_info_standard
        self.sleep_keywords = sleep_keywords
    """
    - 睡眠时长：总监测时长、总在床时长、睡眠时长、深度睡眠时长、浅睡时长、清醒时长、入睡时长、离床时间
    - 睡眠效率：睡眠效率、深睡效率
    - 睡眠评分：评分、评分归类、超越人数百分比
    - 睡眠状态：上床时间、入睡时间、醒来时间、离床时间、离床次数、夜醒次数
    - 生理指标：平均心率、最大心率、最小心率、平均呼吸率、最大呼吸率、最小呼吸率、心率状态、呼吸率状态
    - 体动数据：体动次数、体动指数、平均体动次数、最大体动次数、最小体动次数、体动状态
    - 呼吸异常：呼吸异常次数、呼吸异常指数、典型呼吸异常事件
    - 图表数据：睡眠阶段划分图、心率折线图、体动图、呼吸异常图、呼吸率图、心率图
    - 健康建议：健康建议内容
    
    1. 如果查询明确提及睡眠报告的任何指标或数据，应判断为相关
    2. 如果查询间接询问用户睡眠质量、睡眠情况等，应判断为相关
    3. 如果查询与睡眠报告完全无关，应判断为不相关

    Args:
        BaseTool (_type_): _description_
    """
    
    async def _run(
        self, 
        query: Optional[str] = None,
        classify_topic: Optional[str] = None,
        classify_topic_info: Optional[str] = None,
        classify_topic_info_standard: Optional[str] = None,
        sleep_keywords: Optional[List[str]] = None
    ):
        """
        判断用户问题是否与classify_topic相关
        
        返回:
            dict: {"is_report_related": True/False, "confidence": 0.0-1.0}
        """
        query = query if query is not None else self.query
        classify_topic = classify_topic if classify_topic is not None else self.classify_topic
        classify_topic_info = classify_topic_info if classify_topic_info is not None else self.classify_topic_info
        classify_topic_info_standard = classify_topic_info_standard if classify_topic_info_standard is not None else self.classify_topic_info_standard
        sleep_keywords = sleep_keywords if sleep_keywords is not None else self.sleep_keywords

        if query is None or query == "":
            raise ValueError('query must not be null!')
        
        if classify_topic is None or classify_topic == "":
            raise ValueError('classify_topic must not be null!')
        
        classify_topic_info_prompt = f"""
        \n{classify_topic}包含以下指标和数据:
        {classify_topic_info}
        """ if classify_topic_info is not None else """"""
        
        classify_topic_info_standard = f"""
        \n判断标准：
        {classify_topic_info_standard}
        """ if classify_topic_info is not None else """"""
        
        intent_prompt = f"""
        请分析以下用户查询，判断它是否与用户{classify_topic}相关:
        {classify_topic_info_prompt}
        
        
        用户查询："{query}"
        
        评估该查询是否与{classify_topic}相关，返回JSON格式:
        {{
            "is_related": true/false,  // true表示与{classify_topic}相关，false表示不相关
            "confidence": 0.0-1.0,  // 置信度，0.0表示完全不确定，1.0表示完全确定
            "reasoning": "简短解释您的判断理由"
        }}
        {classify_topic_info_standard}
        """
        return intent_prompt
        messages = [{"role": "user", "content": intent_prompt}]
        response = await llm._whoami_text(messages=messages, timeout=30, user_stop_words=[])
        try:
            intent_data = json.loads(response)
            result = {
                "is_related": intent_data.get("is_related", False),
                "confidence": intent_data.get("confidence", 0.5),
                "reasoning": intent_data.get("reasoning", "未提供理由")
            }
            return result
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试从文本中提取JSON部分
            response = response.strip()
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                try:
                    json_str = response[json_start:json_end]
                    intent_data = json.loads(json_str)
                    result = {
                        "is_report_related": intent_data.get("is_related", False),
                        "confidence": intent_data.get("confidence", 0.5),
                        "reasoning": intent_data.get("reasoning", "未提供理由")
                    }
                    return result
                except json.JSONDecodeError:
                    pass  # 继续到关键词匹配备选方案
                
            # 尝试基于关键词的简单判断作为备选方案
            if sleep_keywords is None or not sleep_keywords:
                raise ValueError('sleep_keywords is none! return default relation status!')
            
            keyword_match = any(keyword in query for keyword in sleep_keywords)
            confidence = 0.7 if keyword_match else 0.5
            return {
                "is_related": keyword_match,
                "confidence": confidence,
                "reasoning": "基于关键词匹配的备选判断" if keyword_match else "JSON解析失败，无法确定相关性"
            }
        except Exception as e:
            # 处理其他所有异常
            return {
                "is_related": False,
                "confidence": 0.5,
                "reasoning": f"处理失败: {str(e)}"
            }
            
            
    def get_relation_status(self, result):
        
        if result is None:
            raise ValueError('Result must not be null!')
        try:
            status = result["is_related"]
        except Exception as e:
            raise ValueError(f"Fail to get relation status, Error: {str(e)}")
        return status

classify_query_intent = ClassifyQueryIntent()

async def main():
    result = await classify_query_intent._run(
        query="",
        classify_topic="睡眠报告",
        classify_topic_info="""
        - 睡眠时长：总监测时长、总在床时长、睡眠时长、深度睡眠时长、浅睡时长、清醒时长、入睡时长、离床时间
        - 睡眠效率：睡眠效率、深睡效率
        - 睡眠评分：评分、评分归类、超越人数百分比
        - 睡眠状态：上床时间、入睡时间、醒来时间、离床时间、离床次数、夜醒次数
        - 生理指标：平均心率、最大心率、最小心率、平均呼吸率、最大呼吸率、最小呼吸率、心率状态、呼吸率状态
        - 体动数据：体动次数、体动指数、平均体动次数、最大体动次数、最小体动次数、体动状态
        - 呼吸异常：呼吸异常次数、呼吸异常指数、典型呼吸异常事件
        - 图表数据：睡眠阶段划分图、心率折线图、体动图、呼吸异常图、呼吸率图、心率图
        - 健康建议：健康建议内容
        """,
        classify_topic_info_standard="""
        1. 相关（is_report_related=true）：
        - 查询明确要获取睡眠报告中的具体数据或指标值（如"昨晚深睡时长是多少"）
        - 查询具体询问用户的睡眠数据表现（如"用户睡眠质量怎么样"）
        - 查询包含对睡眠报告数据的统计或分析（如"上周平均心率是多少"）
        
        2. 不相关（is_report_related=false）：
        - 查询是关于如何操作、如何查看、如何使用、如何获取睡眠报告的操作性问题（如"如何查看睡眠报告"）
        - 查询是关于睡眠报告功能、使用方法、界面等的问题（如"睡眠报告在哪个页面"）
        - 查询是关于系统、设备或其他与睡眠报告数据无关的问题
        - 查询虽然提到"睡眠报告"，但不是询问其中的数据内容
        """,
        sleep_keywords = ["睡眠", "心率", "呼吸", "体动", "深睡", "浅睡", "夜醒", "入睡", "报告", 
                    "效率", "评分", "清醒", "床", "监测", "异常"]
    )
    status = classify_query_intent.get_relation_status(result=result)
    return status

        
if __name__ == '__main__':
    status = asyncio.run(main()) 
    print(status)
    