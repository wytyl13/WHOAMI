#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/05 17:48
@Author  : weiyutao
@File    : ollama_llm.py
"""
import json
import time
from typing import Optional, List, Dict, Any, Union, Awaitable, AsyncGenerator

from whoami.llm_api.base_llm import BaseLLM
from whoami.configs.llm_config import LLMConfig, LLMType
from whoami.llm_api.general_api_requestor import GeneralAPIRequestor
from whoami.utils.log import Logger

USE_CONFIG_TIMEOUT = 0

class OllamaLLM(BaseLLM):
    def __init__(
        self, 
        config: LLMConfig,
        temperature: float = 0.3
    ):
        self.__init__ollama__(config)
        self.config = config
        self.use_system_prompt = True
        self.suffix_url = "/chat"
        self.http_method = "post"
        self.client = GeneralAPIRequestor(base_url=config.base_url)
        self.logger = Logger('OllamLLM')
        self.temperature = temperature

    def __init__ollama__(self, config: LLMConfig):
        assert config.base_url, "ollama base url is required!"
        self.model = config.model

    def get_choice_text(self, res: dict) -> str:
        """提取响应中的文本内容"""
        # self.logger.info(f"尝试从以下响应中提取文本: {res}")
        
        # 检查各种可能的响应格式
        try:
            if not res:
                self.logger.error("响应为空")
                return "响应为空"
                
            # 检查是否是标准OpenAI格式
            if isinstance(res, dict) and "choices" in res and len(res["choices"]) > 0:
                if "message" in res["choices"][0] and "content" in res["choices"][0]["message"]:
                    return res["choices"][0]["message"]["content"]
            
            # 检查是否是Ollama特有格式
            if isinstance(res, dict):
                if "message" in res and "content" in res["message"]:
                    return res["message"]["content"]
                elif "response" in res:
                    return res["response"]
                elif "content" in res:
                    return res["content"]
            
            # 如果是字符串，直接返回
            if isinstance(res, str):
                return res
                
            # 如果是其他格式，尝试转换为字符串
            self.logger.warning(f"未知的响应格式: {type(res)}, 尝试转换为字符串")
            return str(res)
            
        except Exception as e:
            self.logger.error(f"提取内容时出错: {e}, 响应类型: {type(res)}")
            if isinstance(res, dict):
                self.logger.error(f"响应键: {list(res.keys())}")
            return "无法提取响应内容"
        
    
    def _decode_and_load(self, chunk: bytes, encoding: str = "utf-8") -> dict:
        try:
            chunk_str = chunk.decode(encoding)
            self.logger.debug(f"Decoded chunk: {chunk_str[:100]}..." if len(chunk_str) > 100 else f"Decoded chunk: {chunk_str}")
            return json.loads(chunk_str)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to decode JSON from: {chunk_str if 'chunk_str' in locals() else chunk}")
            return {"content": chunk_str if 'chunk_str' in locals() else "", "done": False}
        except Exception as e:
            self.logger.error(f"Error decoding chunk: {e}")
            return {"content": "", "done": False}

    def _const_kwargs(self, messages: list[dict], stream: bool = False, user_stop_words: list = []) -> dict:
        
        # 合并默认终止符和用户提供的终止符
        # 提取system消息
        kwargs = {
            "model": self.model, 
            "messages": messages, 
            "options": 
                {
                    "temperature": self.temperature,
                }, 
            "stream": stream
        }
        
        if user_stop_words:
            kwargs["options"]["stop"] = user_stop_words
        
        self.logger.debug(f"Constructed kwargs: {kwargs}")
        return kwargs

    async def _whoami(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        """Asynchronous version of completion"""
        start_time = time.time()
        self.logger.debug(f"Starting _whoami with timeout {timeout}")
        
        resp = await self._whoami_text(messages, timeout=self.get_timeout(timeout), user_stop_words=[])
        
        elapsed = time.time() - start_time
        self.logger.debug(f"_whoami completed in {elapsed:.2f} seconds")
        
        return resp

    async def _whoami_text(self, messages: List[Dict[str, str]], timeout: int, user_stop_words: List[str]) -> dict:
        """Non-streaming text completion"""
        try:
            start_time = time.time()
            self.logger.debug(f"Starting _whoami_text with timeout {timeout}")
            self.logger.debug(f"Request params: {self._const_kwargs(messages=messages, user_stop_words=user_stop_words)}")
            resp, _, _ = await self.client.arequest(
                method=self.http_method,
                url=self.suffix_url,
                params=self._const_kwargs(messages=messages, user_stop_words=user_stop_words),
                request_timeout=timeout,
            )
            
            elapsed = time.time() - start_time
            self.logger.debug(f"API request completed in {elapsed:.2f} seconds")
            
            # Make a fake response that matches OpenAI format if empty or None
            if not resp:
                self.logger.warning("Empty response from API")
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "Error: Empty response from model"
                            }
                        }
                    ]
                }
                
            resp_dict = self._decode_and_load(resp)
            # Convert to OpenAI-like format for compatibility
            if "choices" not in resp_dict:
                content = self.get_choice_text(resp_dict)
                resp_dict = {
                    "choices": [
                        {
                            "message": {
                                "content": content
                            }
                        }
                    ]
                }
            
            self.logger.debug(f"Processed response: {resp_dict}")
            # return resp_dict
            return content
        except Exception as e:
            self.logger.error(f"Error in _whoami_text: {e}")
            # Return a formatted error response
            return f"Error: {str(e)}"
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"Error: {str(e)}"
                        }
                    }
                ]
            }

    async def _whoami_text_stream(self, messages: List[Dict[str, str]], timeout: int, user_stop_words: List[str]) -> AsyncGenerator[str, None]:
        """Streaming text completion that yields chunks of the response"""
        try:
            
            # 记录请求开始和参数
            self.logger.info(f"Starting streaming request with timeout {timeout}")
            self.logger.info(f"Messages: {messages}")
            
            # 记录请求参数
            params = self._const_kwargs(messages=messages, user_stop_words=user_stop_words, stream=True)
            self.logger.info(f"Request params: {params}")
            
            
            start_time = time.time()
            self.logger.debug(f"Starting streaming request with timeout {timeout}")
            
            stream_resp, _, _ = await self.client.arequest(
                method=self.http_method,
                url=self.suffix_url,
                stream=True,
                params=params,
                request_timeout=timeout,
            )
            
            # 检查响应类型
            # self.logger.info(f"Received response of type: {type(stream_resp)}")
            # self.logger.info(f"Response has __aiter__: {hasattr(stream_resp, '__aiter__')}")
            
            self.logger.debug("Stream response object received, starting to process chunks")
            
            # Process each chunk from the stream
            if hasattr(stream_resp, '__aiter__'):
                async for raw_chunk in stream_resp:
                    # self.logger.info(f"Raw first chunk sample: {str(raw_chunk)[:1000]}")
                    chunk = self._decode_and_load(raw_chunk)
                    # self.logger.info(f"Decoded chunk structure: {json.dumps(chunk, default=str)[:200]}")
                    
                    if not chunk.get("done", False):
                        content = self.get_choice_text(chunk)
                        yield content if content else ""
                        # if content:  # Only yield non-empty content
                        #     self.logger.debug(f"Yielding chunk: {content[:50]}..." if len(content) > 50 else f"Yielding chunk: {content}")
                        #     yield content
            else:
                # Handle case where stream_resp is bytes or another non-iterable
                self.logger.warning(f"stream_resp is not an async iterable, got {type(stream_resp)}")
                if isinstance(stream_resp, bytes):
                    chunk = self._decode_and_load(stream_resp)
                    content = self.get_choice_text(chunk)
                    if content:
                        yield content
                else:
                    yield f"Error: Unexpected response type {type(stream_resp)}"
            elapsed = time.time() - start_time
            self.logger.debug(f"Streaming completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in streaming response: {e}")
            # Yield an error message in the stream
            yield f"Error occurred during streaming: {str(e)}"

    # 添加一个直接同步调用的备用方法
    def sync_whoami(self, messages: List[Dict[str, str]], timeout: int = 30) -> str:
        """
        直接同步调用，不使用异步机制，用于调试或紧急情况
        """
        import requests
        
        self.logger.debug("Using direct sync_whoami method")
        
        try:
            url = f"{self.config.base_url}{self.suffix_url}"
            params = self._const_kwargs(messages=messages)
            
            self.logger.debug(f"Direct API call to: {url}")
            self.logger.debug(f"With params: {params}")
            
            response = requests.post(
                url=url,
                json=params,
                timeout=timeout
            )
            
            if response.status_code != 200:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code} - {response.text}"
            
            resp_json = response.json()
            self.logger.debug(f"Direct response: {resp_json}")
            
            # 提取内容
            if "message" in resp_json:
                return resp_json["message"].get("content", "")
            elif "response" in resp_json:
                return resp_json["response"]
            elif "content" in resp_json:
                return resp_json["content"]
            else:
                return str(resp_json)
                
        except Exception as e:
            self.logger.error(f"Error in sync_whoami: {e}")
            return f"Error: {str(e)}"