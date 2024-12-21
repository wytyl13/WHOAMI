
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
    def _whoami_text(self, messages: list[dict[str, str]], timeout: int):
        """_whoami_text implemented by inherited class"""
    
    @abstractmethod
    def _whoami_text_stream(self, messages: list[dict[str, str]], timeout: int):
        """_whoami_text_stream implemented by inherited class"""
    
    @abstractmethod
    async def whoami_text(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
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
    
    async def whoami_text(self, messages: list[dict[str, str]], stream: bool, timeout: int = USE_CONFIG_TIMEOUT) -> str:
        if stream:
            return await self._whoami_text_stream(messages, timeout=self.get_timeout(timeout))
        res = await self._whoami_text(messages, timeout=self.get_timeout(timeout))
        return self.get_choice_text(res)
        
    def get_timeout(self, timeout: int) -> int:
        return timeout or self.config.timeout or LLM_API_TIMEOUT
    
    async def whoami(
        self, 
        msg: Union[str, list[dict[str, str]]],
        sys_msgs: Optional[list[str]] = None,
        stream = True,
        timeout = USE_CONFIG_TIMEOUT,
    ) -> str:
        if sys_msgs:
            message = self._sys_msgs(sys_msgs)
        else:
            message = [self._default_sys_msg()]
        if not self.use_system_prompt:
            message = []
        message.extend(msg)
        self.logger.debug(message)
        res = await self.whoami_text(message, stream, timeout)