#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/24 17:46
@Author  : weiyutao
@File    : api_tool.py
"""

from typing import (
    Type,
    Optional,
    Union
)
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import io
import torchaudio
import torch
from enum import Enum
import numpy as np
import tempfile
import os


from whoami.tool.agent.base_tool import tool

@tool
class ApiTool:
    """
    text to speech
    """
    end_flag: int = 0
    ak: Optional[str]= None
    url: Optional[str] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'ak' in kwargs:
            self.ak = kwargs.pop('ak')
        if 'url' in kwargs:
            self.url = kwargs.pop('url')
    
    
    @abstractmethod
    async def request_url(self, url, ak, query) -> str:
        """
        Request_url function need to implement in inherited class.  
        """
    
    
    async def execute(
        self, 
        query_key: Optional[str] = None, 
    ) -> str:
        
        if query_key is None or query_key == "":
            raise ValueError("query must not be null!")
        
        if self.ak is None:
            raise ValueError("ak must not be null!")
        
        if self.url is None:
            raise ValueError("url must not be null!")
        
        return await self.request_url(self.url, self.ak, query_key)
        
        