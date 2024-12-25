#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/24 11:35
@Author  : weiyutao
@File    : sql_provider.py
"""
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
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import numpy as np

from whoami.provider.base_provider import BaseProvider
from whoami.configs.sql_config import SqlConfig


class SqlProvider(BaseProvider):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    sql_connection: Optional[sessionmaker] = None
    
    def __init__(self, sql_config_path: Optional[str] = None, sql_config: Optional[SqlConfig] = None) -> None:
        super().__init__()
        self._init_param(sql_config_path, sql_config)
    
    def _init_param(self, sql_config_path: Optional[str] = None, sql_config: Optional[SqlConfig] = None):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.sql_config = SqlConfig.from_file(self.sql_config_path) if self.sql_config is None and self.sql_config_path is not None else self.sql_config
        # if self.sql_config is None and self.data is None:
        #     raise ValueError("config config_path and data must not be null!")
        self.sql_connection = self.get_sql_connection() if self.sql_config is not None else self.sql_connection
        
        
    def get_sql_connection(self):
        try:
            sql_info = self.sql_config
            username = sql_info.username
            database = sql_info.database
            password = sql_info.password
            host = sql_info.host
            port = sql_info.port
        except Exception as e:
            raise ValueError(f"fail to init the sql connect information!\n{self.sql_config}") from e
        database_url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
        try:
            engine = create_engine(database_url, pool_size=10, max_overflow=20)
            SessionLocal = sessionmaker(bind=engine)
        except Exception as e:
            raise ValueError("fail to create the sql connector engine!") from e
        return SessionLocal
    
    def exec_sql(self, query: Optional[str] = None):
        with self.sql_connection() as db:
            try:
                result = db.execute(text(query)).fetchall()
                db.commit()
            except Exception as e:
                db.rollback()        
                error_info = f"Failed to execute SQL query: {query}!"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
            if result is not None:
                # 将 RowProxy 转换为列表，然后再转换为 NumPy 数组
                numpy_array = np.array(result)
                return numpy_array
            return result