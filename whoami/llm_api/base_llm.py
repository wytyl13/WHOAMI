#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/05 17:48
@Author  : weiyutao
@File    : base_llm.py
"""

import time
from typing import Optional, Union, List, Dict, Any, Awaitable, AsyncGenerator
from pydantic import BaseModel
from abc import ABC, abstractmethod
import asyncio
import inspect
import nest_asyncio
import concurrent.futures

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
    async def _whoami_text(self, messages: list[dict[str, str]], timeout: int, user_stop_words: list = []):
        """_whoami_text implemented by inherited class"""
    
    @abstractmethod
    async def _whoami_text_stream(self, messages: list[dict[str, str]], timeout: int, user_stop_words: list = []):
        """_whoami_text_stream implemented by inherited class"""
    
    @abstractmethod
    async def _whoami(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT):
        """Asynchronous version of completion
        All LLM APIs are required to provide the standard completion interface
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello, show me python hello world code"},
            # {"role": "assistant", "content": ...}, # If there is an answer in the history, also include it
        ]
        """

    def get_choice_text(self, res: dict) -> str:
        """Required to provide the first text of choice"""
        self.logger.debug(f"Getting choice text from: {res}")
        if not res or "choices" not in res or not res.get("choices"):
            self.logger.warning("Invalid response format: no choices found")
            return "Error: Invalid response format"
        
        try:
            return res.get("choices")[0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error extracting choice text: {e}")
            return f"Error extracting response: {str(e)}"
    
    async def whoami_text(self, messages: list[dict[str, str]], stream: bool, timeout: int = USE_CONFIG_TIMEOUT, user_stop_words: list = []) -> Union[str, AsyncGenerator]:
        """Unified async method for text completion with or without streaming"""
        try:
            start_time = time.time()
            self.logger.debug(f"Starting whoami_text: stream={stream}, timeout={timeout}")
            
            if stream:
                # For streaming, we return the generator directly
                self.logger.debug("Returning streaming generator")
                return self._whoami_text_stream(messages, timeout=self.get_timeout(timeout), user_stop_words=user_stop_words)
            else:
                # For non-streaming, we get the response and extract text
                self.logger.debug("Calling _whoami_text for non-streaming response")
                res = await self._whoami_text(messages, timeout=self.get_timeout(timeout), user_stop_words=user_stop_words)
                
                elapsed = time.time() - start_time
                self.logger.debug(f"whoami_text completed in {elapsed:.2f} seconds")
                
                result = self.get_choice_text(res)
                self.logger.debug(f"Extracted result: {result[:100]}..." if len(result) > 100 else f"Extracted result: {result}")
                return result
        except Exception as e:
            self.logger.error(f"Error in whoami_text: {e}")
            raise
        
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
        
        start_time = time.time()
        self.logger.debug(f"Starting whoami: stream={stream}, timeout={timeout}")
        
        # 准备消息
        messages = self._sys_msgs(sys_msgs) if sys_msgs else ([self._default_sys_msg()] if self.use_system_prompt else [])
        messages.extend(self.format_messages(msg))
        self.logger.debug(f"Prepared messages: {messages}")

        # 检测调用者是否是异步函数
        caller_frame = inspect.currentframe().f_back
        is_async_caller = caller_frame is not None and asyncio.iscoroutinefunction(caller_frame.f_code)
        self.logger.debug(f"Detected async caller: {is_async_caller}")
        
        if is_async_caller:
            # 异步环境直接返回coroutine对象
            self.logger.debug("Returning async coroutine")
            return self._async_whoami(messages, stream, timeout, user_stop_words)
        else:
            # 同步环境下，执行同步版本
            self.logger.debug("Executing sync version")
            result = self._sync_whoami(messages, stream, timeout, user_stop_words)
            
            elapsed = time.time() - start_time
            self.logger.debug(f"whoami completed in {elapsed:.2f} seconds")
            
            return result
    
    async def _async_whoami(
        self,
        messages: list[dict[str, str]],
        stream=True,
        timeout=USE_CONFIG_TIMEOUT,
        user_stop_words: list = []
    ) -> str:
        """异步实现"""
        try:
            start_time = time.time()
            self.logger.debug(f"Starting _async_whoami: stream={stream}, timeout={timeout}")
            
            if stream:
                # For streaming, collect all chunks into a single string
                result = []
                self.logger.debug("Processing streaming response")
                async_gen = await self.whoami_text(messages, stream, timeout, user_stop_words)
                async for chunk in async_gen:
                    self.logger.debug(f"Received chunk: {chunk[:50]}..." if len(chunk) > 50 else f"Received chunk: {chunk}")
                    result.append(chunk)
                
                full_result = "".join(result)
                self.logger.debug(f"Collected full result: {full_result[:100]}..." if len(full_result) > 100 else f"Collected full result: {full_result}")
                
                elapsed = time.time() - start_time
                self.logger.debug(f"_async_whoami completed in {elapsed:.2f} seconds")
                
                return full_result
            else:
                # For non-streaming, just return the result
                self.logger.debug("Getting non-streaming response")
                result = await self.whoami_text(messages, stream, timeout, user_stop_words)
                
                elapsed = time.time() - start_time
                self.logger.debug(f"_async_whoami completed in {elapsed:.2f} seconds")
                
                return result
        except Exception as e:
            self.logger.error(f"Error in async whoami: {e}")
            raise

    def _sync_whoami(
        self,
        messages: list[dict[str, str]],
        stream=True,
        timeout=USE_CONFIG_TIMEOUT,
        user_stop_words: list = []
    ) -> str:
        """同步实现 - 使用事件循环运行异步代码"""
        try:
            start_time = time.time()
            self.logger.debug(f"Starting _sync_whoami with timeout: {timeout}")
            
            # 尝试直接使用线程池，避免事件循环问题
            with concurrent.futures.ThreadPoolExecutor() as executor:
                def run_async_in_new_loop():
                    # 创建新的事件循环
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        return loop.run_until_complete(
                            self._async_whoami(messages, stream, timeout, user_stop_words)
                        )
                    finally:
                        loop.close()
                
                # 在单独的线程中运行异步代码
                self.logger.debug("Submitting async task to thread pool")
                future = executor.submit(run_async_in_new_loop)
                
                # 设置超时
                try:
                    result = future.result(timeout=timeout + 5)  # 给线程多5秒的余地
                    
                    elapsed = time.time() - start_time
                    self.logger.debug(f"_sync_whoami completed in {elapsed:.2f} seconds")
                    
                    return result
                except concurrent.futures.TimeoutError:
                    self.logger.error(f"Execution timed out after {timeout + 5} seconds")
                    return "Error: Request timed out"
                
        except Exception as e:
            self.logger.error(f"Error in sync whoami: {e}", exc_info=True)
            return f"Error: {str(e)}"