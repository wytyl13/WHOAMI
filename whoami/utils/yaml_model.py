#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/05 11:46
@Author  : weiyutao
@File    : yaml_model.py
"""

from pydantic import BaseModel, model_validator
from pathlib import Path
from typing import Dict, Optional, Union
import yaml


class YamlModel(BaseModel):
    
    # def __init__(self):
        # super().__init__()
    
    
    @classmethod
    def read(cls, file_path: Optional[Union[Path, str]], encoding: str = "utf-8") -> Dict:
        file_path = Path(file_path) if not isinstance(file_path, Path) and file_path is not None else file_path
        if file_path is None or not file_path.exists():
            return {k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(v)}
        with open(file_path, "r", encoding=encoding) as file:
            return yaml.safe_load(file)

    @classmethod
    def from_file(cls, file_path: Optional[Union[Path, str]] = None) -> "YamlModel":
        return cls(**cls.read(file_path))

        