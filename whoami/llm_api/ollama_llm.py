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
    def __init__(self, config: LLMConfig):
        self.__init__ollama__(config)
        self.config = config
        self.use_system_prompt = False
        self.suffix_url = "/chat"
        self.http_method = "post"
        self.client = GeneralAPIRequestor(base_url=config.base_url)
        self.logger = Logger('OllamLLM')

    def __init__ollama__(self, config: LLMConfig):
        assert config.base_url, "ollama base url is required!"
        self.model = config.model

    def _decode_and_load(self, chunk: bytes, encoding: str = "utf-8") -> dict:
        chunk = chunk.decode(encoding)
        return json.loads(chunk)


    async def whoami_text(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    async def _whoami_text(self, messages, timeout):
        resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.suffix_url,
            params=self._const_kwargs(messages),
            request_timeout=self.get_timeout(timeout),
        )
        resp = self._decode_and_load(resp)
        return resp

    async def _whoami_text_stream(self, messages, timeout) -> str:
        stream_resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.suffix_url,
            stream=True,
            params=self._const_kwargs(messages, stream=True),
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

    
    