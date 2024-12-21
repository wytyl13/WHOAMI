#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/05 11:46
@Author  : weiyutao
@File    : yaml_model.py
"""

from pydantic import BaseModel, model_validator
from pathlib import Path
from typing import Dict, Optional
import yaml


class YamlModel(BaseModel):
    
    @classmethod
    def read(cls, file_path: Path, encoding: str = "utf-8") -> Dict:
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding=encoding) as file:
            return yaml.safe_load(file)

    @classmethod
    def from_file(cls, file_path: Path) -> "YamlModel":
        return cls(**cls.read(file_path))

        