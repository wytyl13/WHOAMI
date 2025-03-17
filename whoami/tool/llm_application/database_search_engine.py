#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/11 09:19
@Author  : weiyutao
@File    : database_search_engine.py
"""
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
    Generic,
    TypeVar,
    Any,
    Type,
    List
)



from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sleep_indices import SleepIndices
from whoami.tool.base.base_tool import BaseTool
from whoami.provider.base_ import ModelType


sql_config_path = '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml'


class DatabaseSearchEngine(BaseTool):
    
    sql_provider: SqlProvider = SqlProvider(model=SleepIndices, sql_config_path=sql_config_path)
    conditions: Optional[dict] = None
    
    def __init__(
        self, 
        model: Type[ModelType] = None,
        conditions: Optional[dict] = None,
    ):
        super().__init__()
        self.conditions = conditions

        if model is not None:
            self.sql_provider.set_model(model=model)


    def _run(
            self,
            sql_provider: Optional[SqlProvider] = None,
            model: Type[ModelType] = None,
            conditions: Optional[dict] = None,
            fields: Optional[list[str]] = None,
            exclude_fields: Optional[list[str]] = None
        ):
        try:
            self.sql_provider = sql_provider if sql_provider is not None else self.sql_provider
            if model is not None:
                self.sql_provider.set_model(model=model)
            self.conditions = conditions if conditions is not None else self.conditions
        except Exception as e:
            raise ValueError(f"Fail to init the parameters! {str(e)}") from e
        
        try:
            result = self.sql_provider.get_record_by_condition(condition=self.conditions, fields=fields, exclude_fields=exclude_fields)
        except Exception as e:
            raise ValueError(f"Fail to exec get_record_by_condition {str(e)}") from e
        return result

    
    