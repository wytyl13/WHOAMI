#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/07/17 18:19
@Author  : weiyutao
@File    : handle_shixun_tonggao.py
"""

from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Any
)
import aiohttp
import asyncio



from whoami.tool.agent.base_tool import tool
# from whoami.tool.llm_application.enhance_retrieval import EnhanceRetrieval
from whoami.tool.agent.tool import EnhanceRetrieval
from whoami.utils.utils import Utils


utils = Utils()

class TongzhiTonggaoSchema(BaseModel):
    operation: str = Field(
        ...,
        description="用户对时讯消息或通告的操作方式，从列表中选择其中一个：['ADD', 'DELET', 'UPDATE', 'LIST']"
    )
    type: str = Field(
        ...,
        description="用户上传消息的类型，从列表中选择其中一个：[\"时讯消息\", \"通告\"]"
    )
    content: str = Field(
        description="用户上传的内容，如果没有匹配到，置为空"
    )


@tool
class HandleTongzhiTonggao:
    args_schema: Type[BaseModel] = TongzhiTonggaoSchema
    # 如果希望算法更加高效，设置end_flag为1，但是不能保证回复的质量
    end_flag: int = 1
    session: Optional[aiohttp.ClientSession] = None
    enhance_llm: Optional[EnhanceRetrieval] = None
    def __init__(self, **kwargs):
        
        # you should implement any private attribute here first. 
        super().__init__(**kwargs)
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.pop('enhance_llm')
    
    
    # async def _get_session(self):
    #     """获取或创建HTTP会话"""
    #     if self.session is None:
    #         self.session = aiohttp.ClientSession()
    #     return self.session

    
    async def _get_session(self):
        """获取或创建HTTP会话"""
        if self.session is None:
            import ssl
            # 创建不验证SSL证书的上下文
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
    
    
    
    async def _make_request(self, url: str, method: str = "POST", data: Dict[str, Any] = None):
        """发送HTTP请求"""
        session = await self._get_session()
        try:
            if method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    return await response.json()
            elif method.upper() == "GET":
                async with session.get(url, params=data) as response:
                    return await response.json()
        except Exception as e:
            return {"error": str(e)}
    
    
    async def execute(
        self, 
        operation: str,
        type: str, 
        content: Optional[str] = "",
        content_id: Optional[str] = "",
        username: Optional[str] = "shunxikeji",
        location: Optional[str] = None,
        role: Optional[str] = None
    ):
        if role == "user":
            response = f"收到{type}！但是您没有相应的权限！请联系社区管理人员！"
            for item in response:
                yield item
            return
        operation = "ADD" if operation in ["添加", "新增", "增加"] else operation
        operation = "LIST" if operation in ["查看", "列出", "历史"] else operation
         
        operation_url = {
            "ADD": "https://1.71.15.121:8889/api/save_community_real_time_data",
            "LIST": "https://1.71.15.121:8889/api/list_community_real_time_data"
        }
        
        if operation == "":
            response = f"收到{type}. 请告知您具体的操作类型: 【新增/修改/删除/查看】"
            for item in response:
                yield item
            return
        
        if operation not in operation_url:
            response = f"不支持的操作类型: {operation}. 支持的操作: {list(operation_url.keys())}"
            for item in response:
                yield item
            return
        
        url = operation_url[operation] if operation != "" else "LIST"
        if operation == "ADD":
            request_data = {
                "type": type,
                "content": content,
                "username": username
            }
            method = "POST"
            
        elif operation == "LIST":
            request_data = {
                "type": type,
                "username": username
            }
            method = "POST"
        
        else:
            request_data = {
                "type": type,
                "username": username
            }
            method = "POST"
        
        # 发送请求
        result = await self._make_request(url, method, request_data)
        if result["success"]:
            result_data = result["data"]
            if isinstance(result_data, str):
                for item in str(result["message"]):
                    yield item
                    await asyncio.sleep(0.01)
                # 更新向量数据库
                if content and result["data"]:
                    self.enhance_llm.retrieval.add_text(text=content, text_id=result_data)
            elif isinstance(result_data, list):
                self.logger.info(result["data"])
                formatted_text = utils.format_notices_data_markdown(type=type, data_list=result["data"])
                for char in formatted_text:
                    yield char
                    await asyncio.sleep(0.01)
            else:
                for item in str(result["message"]):
                    yield item
                    await asyncio.sleep(0.01)
            return
        else:
            for item in result["message"]:
                yield item
                await asyncio.sleep(0.01)
            return
        
        