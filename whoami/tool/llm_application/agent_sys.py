#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11 09:45
@Author  : weiyutao
@File    : chat.py
"""

from typing import (
    Optional,
    Dict,
    Type,
    Union
)

from whoami.tool.base.base_tool import BaseTool
from whoami.tool.llm_application.query_classifier import QueryClassifier
from whoami.utils.utils import StrEnum
from whoami.tool.llm_application.database_search_engine import DatabaseSearchEngine
from whoami.tool.search.google_search_provider import GoogleSearch
from whoami.tool.llm_application.direct import Direct


class Tool(StrEnum):
    DATABASE = "DatabaseSearchEngine"
    
    WEB = "GoogleSearch"
    
    DIRECT = "Direct"


TOOL_CLASSES: Dict[Tool, Type] = {
    Tool.DATABASE: DatabaseSearchEngine,
    Tool.WEB: GoogleSearch,
    Tool.DIRECT: Direct
}


class AgentSys(BaseTool):
    query_classifier: Optional[QueryClassifier] = None
    
    
    def __init__(self, llm):
        super().__init__()
        self.query_classifier = QueryClassifier(llm=llm)
    
    def _run(self, question, device_sn):
        pass
    
    
    def _forward(self, question):
        try:
            tool_ = self.query_classifier.classifier_query(query_str=question)
        except Exception as e:
            self.logger.error(e) 
        
        if not getattr(Tool, tool_):
            raise ValueError(f"未知的工具类型: {tool_}")

        try:
            tool_class: Union[DatabaseSearchEngine, GoogleSearch, Direct] = TOOL_CLASSES[getattr(Tool, tool_)]
            result = tool_class._run(conditions={"device_sn": device_sn, "query_date": "2025-03-11"}, exclude_fields=['breath_bpm_image_x_y', 'heart_bpm_image_x_y', 'sleep_stage_image_x_y'])
            return result
        except Exception as e:
            self.logger.error(str(e))
            return str(e)
    
    
    