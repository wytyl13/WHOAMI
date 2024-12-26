#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/24 09:54
@Author  : weiyutao
@File    : data_loader.py
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator, ValidationError
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
    Type,
)

import torch
from torch.utils.data import Dataset
import numpy as np

from whoami.provider.base_ import ModelType
from whoami.utils.log import Logger
from whoami.configs.sql_config import SqlConfig
from whoami.provider.base_provider import BaseProvider
from whoami.provider.sql_provider import SqlProvider

class DataProvider(BaseProvider, Dataset):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    data: Optional[np.ndarray] = None
    sql_provider: Optional[SqlProvider] = None
    sql_query: Optional[str] = None
    model: Type[ModelType] = None
    def __init__(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        data: Optional[np.ndarray] = None,
        sql_provider: Optional[SqlProvider] = None,
        sql_query: Optional[str] = None,
        model: Type[ModelType] = None
    ) -> None:
        super().__init__()
        self._init_param(sql_config_path, sql_config, data, sql_provider, sql_query, model)
    
    def _init_param(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        data: Optional[np.ndarray] = None, 
        sql_provider: Optional[SqlProvider] = None,
        sql_query: Optional[str] = None,
        model: Type[ModelType] = None
    ):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.data = data
        self.sql_query = sql_query
        self.sql_provider = sql_provider
        self.sql_config = SqlConfig.from_file(self.sql_config_path) if self.sql_config is None and self.sql_config_path is not None else self.sql_config

        self.sql_provider = SqlProvider(model=model, sql_config=self.sql_config) if self.sql_provider is None and self.sql_config is not None else self.sql_provider

        if self.sql_provider is None and self.data is None:
            raise ValueError("fail to init the data! sql_config, data, sql_provider and sql_config_path must not be null!")
        
        if self.data is not None and not isinstance(self.data, np.ndarray):
            raise ValueError("the type of data must be numpy!")
        
        if self.data is None and self.sql_query is None:
            raise ValueError("the data and sql query must not be none!")
        self.data = self.get_data() if self.data is None else self.data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.get_item(index)
    
    @abstractmethod
    def get_data(self):
        """get data function implemented by inherited class."""
        
    def set_sql_query(self, sql_query):
        """get data function implemented by inherited class."""
        self.sql_query = sql_query
        self.data = self.get_data()

    def get_item(self, index):
        """you should overwrite this method if you want to change it."""
        return self.data[index]
        
    