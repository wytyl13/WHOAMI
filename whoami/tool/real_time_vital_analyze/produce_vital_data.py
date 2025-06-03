#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/20 09:21
@Author  : weiyutao
@File    : produce_vital_data.py
"""
from typing import (
    Optional,
    Any,
    overload
)


from whoami.tool.agent.base_tool import tool


@tool
class ProduceVitalData:
    port: Optional[int] = 8000
    
    
    @overload
    def __init__(
        self
    ):
        ...
        
    
    @overload
    def __init__(self, *args, **Kwargs):
        ...
        
    
    def __init__(self, *args, **kwargs):
        if 'port' in kwargs:
            self.port = kwargs.pop('port', '8000')
        
        super().__init__(*args, **kwargs)
        
    
    
    def execute(
        self,
        port: Optional[int] = 8000
    ):
        self.port = port if port is not None else self.port
        
        if self.port is None:
            raise ValueError("port must not be null!")
        
        
        pass
    
    

if __name__ == '__main__':
    produce_vital_data = ProduceVitalData()
    print(produce_vital_data.port)
    
    
        
    
