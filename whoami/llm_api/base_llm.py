
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/05 17:48
@Author  : weiyutao
@File    : base_llm.py
"""

from typing import Optional, Union
from pydantic import BaseModel
from abc import ABC, abstractmethod
import asyncio
import inspect
import nest_asyncio
from typing import Optional, Union, Awaitable

from whoami.configs.llm_config import LLMConfig
from whoami.utils.log import Logger

LLM_API_TIMEOUT = 300
USE_CONFIG_TIMEOUT = 0

class BaseLLM(ABC):
    config: LLMConfig
    system_prompt = "You are a helpful assistant"
    use_system_prompt: bool = True
    logger = Logger("BaseLLM")
    
    @abstractmethod
    def __init__(self, config: LLMConfig):
        pass
    
    def _default_sys_msg(self):
        return self._sys_msg(self.system_prompt)
    
    def _sys_msg(self, msg: str) -> dict[str, str]:
        return {"role": "system", "content": msg}
    
    def _sys_msgs(self, msgs: list[str]) -> list[dict[str, str]]:
        return [self._sys_msg(msg) for msg in msgs]
    
    @abstractmethod
    def _whoami_text(self, messages: list[dict[str, str]], timeout: int, user_stop_words: list = []):
        """_whoami_text implemented by inherited class"""
    
    @abstractmethod
    def _whoami_text_stream(self, messages: list[dict[str, str]], timeout: int, user_stop_words: list = []):
        """_whoami_text_stream implemented by inherited class"""
    
    @abstractmethod
    async def _whoami(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """Asynchronous version of completion
        All GPTAPIs are required to provide the standard OpenAI completion interface
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello, show me python hello world code"},
            # {"role": "assistant", "content": ...}, # If there is an answer in the history, also include it
        ]
        """

    def get_choice_text(self, res: dict) -> str:
        """Required to provide the first text of choice"""
        return res.get("choices")[0]["message"]["content"]
    
    async def whoami_text(self, messages: list[dict[str, str]], stream: bool, timeout: int = USE_CONFIG_TIMEOUT, user_stop_words: list = []) -> str:
        if stream:
            return await self._whoami_text_stream(messages, timeout=self.get_timeout(timeout), user_stop_words=user_stop_words)
        res = await self._whoami_text(messages, timeout=self.get_timeout(timeout), user_stop_words=user_stop_words)
        return self.get_choice_text(res)
        
    def get_timeout(self, timeout: int) -> int:
        return timeout or self.config.timeout or LLM_API_TIMEOUT
    
    def format_messages(self, msg: Union[str, list[dict[str, str]]]) -> list[dict[str, str]]:
        """将输入消息格式化为标准格式"""
        if isinstance(msg, str):
            return [{"role": "user", "content": msg}]
        elif isinstance(msg, list):
            return msg
        else:
            raise ValueError(f"Unsupported message format: {type(msg)}")
    
    def whoami(
        self, 
        msg: Union[str, list[dict[str, str]]],
        sys_msgs: Optional[list[str]] = None,
        stream=True,
        timeout=USE_CONFIG_TIMEOUT,
        user_stop_words: list = []
    ) -> Union[str, Awaitable[str]]:
        """统一的 whoami 接口，支持同步和异步调用"""
        """仅同步调用成功，异步调用失败"""
        
        # 准备消息
        messages = self._sys_msgs(sys_msgs) if sys_msgs else ([self._default_sys_msg()] if self.use_system_prompt else [])
        messages.extend(self.format_messages(msg))
        self.logger.debug(messages)

        # 检测调用者是否是异步函数
        caller_frame = inspect.currentframe().f_back
        if caller_frame is not None and asyncio.iscoroutinefunction(caller_frame.f_code):
            # 在异步环境中，使用 await 调用 whoami_text
            return self._async_whoami(messages, stream, timeout, user_stop_words=user_stop_words)
        
        # 在同步环境中，执行同步版本
        return self._sync_whoami(messages, stream, timeout, user_stop_words=user_stop_words)
    
    async def _async_whoami(
        self,
        messages: list[dict[str, str]],
        stream=True,
        timeout=USE_CONFIG_TIMEOUT,
        user_stop_words: list = []
    ) -> str:
        """异步实现"""
        return await self.whoami_text(messages, stream, timeout, user_stop_words=user_stop_words)

    def _sync_whoami(
        self,
        messages: list[dict[str, str]],
        stream=True,
        timeout=USE_CONFIG_TIMEOUT,
        user_stop_words: list = []
    ) -> str:
        
        nest_asyncio.apply()  # 允许在已有事件循环中嵌套使用
        
        loop = asyncio.get_event_loop()
        
        if loop.is_running():
            # 如果事件循环正在运行，使用 run_until_complete
            return loop.run_until_complete(self._async_whoami(messages, stream, timeout, user_stop_words=user_stop_words))
        
        # 如果没有运行中的事件循环，直接执行
        return asyncio.run(self._async_whoami(messages, stream, timeout, user_stop_words=user_stop_words))
    
    """
    def whoami(
        self, 
        msg: Union[str, list[dict[str, str]]],
        sys_msgs: Optional[list[str]] = None,
        stream = True,
        timeout = USE_CONFIG_TIMEOUT,
    ) -> str:
        if sys_msgs:
            messages = self._sys_msgs(sys_msgs)
        else:
            messages = [self._default_sys_msg()] if self.use_system_prompt else []
        messages.extend(self.format_messages(msg))
        self.logger.debug(messages)
        return self.whoami_text(messages, stream, timeout)
    """
