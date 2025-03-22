
import json
from pathlib import Path
import asyncio
from typing import (
    List,
    Dict,
    Optional
)

from pathlib import Path
import asyncio

from whoami.tool.base.base_tool import BaseTool


class ClassifyQueryIntentTemplate(BaseTool):
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
    
    def _run(
        self, 
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
        try:
            classify_topic = classify_topic if classify_topic is not None else self.classify_topic
            classify_topic_info = classify_topic_info if classify_topic_info is not None else self.classify_topic_info
            classify_topic_info_standard = classify_topic_info_standard if classify_topic_info_standard is not None else self.classify_topic_info_standard
            sleep_keywords = sleep_keywords if sleep_keywords is not None else self.sleep_keywords

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
            
            评估该查询是否与{classify_topic}相关，返回JSON格式:
            {{
                "is_related": true/false,  // true表示与{classify_topic}相关，false表示不相关
                "confidence": 0.0-1.0,  // 置信度，0.0表示完全不确定，1.0表示完全确定
                "reasoning": "简短解释您的判断理由"
            }}
            {classify_topic_info_standard}
            """
            return intent_prompt
        except Exception as e:
            raise ValueError(f"fail to customer classify query intent template function! {str(e)}") from e
            