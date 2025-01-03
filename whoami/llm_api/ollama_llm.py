#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/05 17:48
@Author  : weiyutao
@File    : ollama_llm.py
"""
import json


from whoami.llm_api.base_llm import BaseLLM
from whoami.configs.llm_config import LLMConfig, LLMType
from whoami.llm_api.general_api_requestor import GeneralAPIRequestor
from whoami.utils.log import Logger

USE_CONFIG_TIMEOUT = 0

class OllamLLM(BaseLLM):
    def __init__(
        self, 
        config: LLMConfig,
        temperature: float = 0.3
    ):
        self.__init__ollama__(config)
        self.config = config
        self.use_system_prompt = False
        self.suffix_url = "/chat"
        self.http_method = "post"
        self.client = GeneralAPIRequestor(base_url=config.base_url)
        self.logger = Logger('OllamLLM')
        self.temperature = temperature

    def __init__ollama__(self, config: LLMConfig):
        assert config.base_url, "ollama base url is required!"
        self.model = config.model

    def get_choice_text(self, res: dict) -> str:
        """处理Ollama的响应格式"""
        # Ollama的响应格式与OpenAI不同，需要适配
        if "message" in res:
            return res["message"].get("content", "")
        elif "response" in res:
            return res["response"]
        elif "content" in res:
            return res["content"]
        else:
            self.logger.warning(f"Unexpected response format: {res}")
            return str(res)  # 作为后备方案返回整个响应的字符串形式
    
    def _decode_and_load(self, chunk: bytes, encoding: str = "utf-8") -> dict:
        chunk = chunk.decode(encoding)
        return json.loads(chunk)

    def _const_kwargs(self, messages: list[dict], stream: bool = False, user_stop_words: list = []) -> dict:
        kwargs = {
            "model": self.model, 
            "messages": messages, 
            "options": 
                {
                    "temperature": self.temperature,
                    "stop": user_stop_words,
                }, 
            "stream": stream
        }
        return kwargs

    async def _whoami(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        return await self._whoami_text(messages, timeout=self.get_timeout(timeout))

    async def _whoami_text(self, messages, timeout, user_stop_words):
        resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.suffix_url,
            params=self._const_kwargs(messages=messages, user_stop_words=user_stop_words),
            request_timeout=self.get_timeout(timeout),
        )
        resp = self._decode_and_load(resp)
        return resp

    async def _whoami_text_stream(self, messages, timeout, user_stop_words) -> str:
        stream_resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.suffix_url,
            stream=True,
            params=self._const_kwargs(messages=messages, user_stop_words=user_stop_words, stream=True),
            request_timeout=self.get_timeout(timeout),
        )
        collected_content = []
        async for raw_chunk in stream_resp:
            chunk = self._decode_and_load(raw_chunk)

            if not chunk.get("done", False):
                content = self.get_choice_text(chunk)
                collected_content.append(content)
                self.logger.debug(content)
        self.logger.debug("\n")

        full_content = "".join(collected_content)
        return full_content

    
    