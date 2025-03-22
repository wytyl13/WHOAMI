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
    Union,
    List,
    ClassVar
)
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
import asyncio


from whoami.tool.base.base_tool import BaseTool
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.provider.sql_provider import SqlProvider
from whoami.configs.sql_config import SqlConfig
from whoami.tool.llm_application.sx_coversation_history import SxConversationHistory
from whoami.tool.search.google_search import GoogleSearch
from whoami.utils.utils import Utils
from whoami.tool.llm_application.enhance_retrieval import EnhanceRetrieval
from whoami.tool.llm_application.health_report_chat import HealthReportChat



google_search = GoogleSearch(snippet_flag=1, search_config_path='/work/ai/WHOAMI/whoami/scripts/test/search_config.yaml')


class ChatSys(BaseTool):

    llm: Optional[Union[OllamaLLM, Ollama]] = None
    sql_provider: Optional[SqlProvider] = None
    full_response: Optional[list[Dict[str, str]]] = None
    enhance_ : Optional[EnhanceRetrieval] = None
    health_report_chat: Optional[HealthReportChat] = None
    
    def __init__(
        self, 
        llm: Optional[Union[OllamaLLM, Ollama]] = None, 
        sql_provider: Optional[SqlProvider] = None,
        sql_config: Optional[SqlConfig] = None,
        sql_config_path: Optional[str] = None
    ):
        
        super().__init__()
        self.llm = llm if llm is not None else self.llm
        self.sql_provider = sql_provider if sql_provider is not None else self.sql_provider
        if self.llm is None:
            raise ValueError("Attribution llm must not be null!")
        
        # 初始化数据库提供者
        if sql_config or sql_config_path:
            self.sql_provider = SqlProvider(
                model=SxConversationHistory,
                sql_config=sql_config,
                sql_config_path=sql_config_path
            )
        self.enhance_ = EnhanceRetrieval(llm=llm)
        self.health_report_chat = HealthReportChat()
        # if self.sql_provider is None:
        #     raise ValueError("Attribution sql_provide must not be null!")


    def _convert_to_llama_index_chat_messages(self, messages):
        """将字典列表转换为ChatMessage对象列表"""
        chat_messages = []
        for msg in messages:
            role_str = msg['role']
            content = msg['content']
            
            role_map = {
                'user': MessageRole.USER,
                'assistant': MessageRole.ASSISTANT,
                'system': MessageRole.SYSTEM
            }
            
            if role_str not in role_map:
                raise ValueError(f"不支持的角色: {role_str}")
                
            chat_messages.append(ChatMessage(role=role_map[role_str], content=content))
        
        return chat_messages


    async def _run(self, messages_history: list[Dict[str, str]], question: str = None, direct_flag: int = 0):
        self.logger.info(f"messages_history ---------------------------------------------------  {messages_history}")
        # 清空之前的响应收集
        self.full_response = []
        
        # web search and preprocess
        """
        status, web_content = google_search(query=question)
        if not status:
            self.logger.warning(f"Fail to web search! query: {question}")
            web_content = []
        self.logger.info(f"web_content ---------------------------------------------------  {web_content}")
        handle_web_content = []
        for item in web_content:
            web_content_item = item["fetch_url_content"] if "fetch_url_content" in item else item["html_snippet"]
            if web_content_item == "":
                continue
            link = item["link"]
            handle_web_content.append({link: web_content_item})
        
        self.logger.info(handle_web_content)
        """
        handle_web_content = []
        
        chat_stream = self.enhance_._run(
            message_history=messages_history, 
            query=question
        ) if direct_flag else self.health_report_chat._run(
            message_history=messages_history, 
            query=question
        )
        async for chunk in chat_stream:
            # 收集完整响应
            self.full_response.append(chunk)
            # 返回当前块
            yield chunk
            
        # 记录完整响应
        complete_response = "".join(self.full_response)
        self.logger.info(f"Complete response length: {len(complete_response)}")
        self.logger.info(f"First 100 chars: {complete_response[:100]}")


    async def save_qa_to_db(self, conversation_id, user_id, question):
        """异步保存问题和回答到数据库"""
        if self.sql_provider is None:
            self.logger.info("数据库提供者未初始化，无法保存对话")
            return False
            
        # 获取完整回答
        answer = "".join(self.full_response)
        self.logger.info(f"准备保存数据 - 问题长度: {len(question)}, 回答长度: {len(answer)}")
        
        if not answer:
            self.logger.error("回答为空，不保存到数据库")
            return False
        
        try:
            # 使用事件循环在后台执行数据库操作
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # 使用默认执行器
                lambda: self._save_qa_records(conversation_id, user_id, question, answer)
            )
            self.logger.info(f"保存结果: {result}")
            return result
        except Exception as e:
            print(f"异步保存Q&A失败: {str(e)}")
            return False
    
    
    def _save_qa_records(self, conversation_id, user_id, question, answer):
        """同步执行实际的数据库保存操作"""
        try:
            # 保存用户问题
            user_record = {
                'conversation_id': conversation_id,
                'user_id': user_id,
                'role': 'user',
                'content': question
            }
            self.sql_provider.add_record(user_record)
            
            # 保存AI回答
            assistant_record = {
                'conversation_id': conversation_id,
                'user_id': user_id,
                'role': 'assistant',
                'content': answer
            }
            self.sql_provider.add_record(assistant_record)
            
            print(f"成功保存Q&A: 问题长度={len(question)}, 回答长度={len(answer)}")
            return True
            
        except Exception as e:
            print(f"保存Q&A记录失败: {str(e)}")
            return False