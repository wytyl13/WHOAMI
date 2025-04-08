#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2025/04/06 20:10:10
@Author : weiyutao
@File : base.py
"""

from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import ClassVar, Dict, Optional, Union, Any, Type, TypeVar
import yaml

T = TypeVar('T', bound='YamlModel')

class YamlModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra='allow',  # Allow extra fields
        validate_assignment=True  # Validate values on assignment
    )
    
    @classmethod
    def read(cls, file_path: Optional[Union[Path, str]] = None, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read YAML file and return as dictionary, or default values if file doesn't exist."""
        if file_path is not None:
            file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
            if file_path.exists():
                with open(file_path, "r", encoding=encoding) as file:
                    return yaml.safe_load(file) or {}
        
        # Return default values from class annotations
        return {
            k: v.default
            for k, v in cls.model_fields.items()
            if v.default is not None and v.default is not ...
        }

    @classmethod
    def from_file(cls: Type[T], file_path: Optional[Union[Path, str]] = None, **kwargs) -> T:
        """Create instance from YAML file with optional overrides."""
        config_dict = cls.read(file_path)
        # Allow additional kwargs to override file values
        config_dict.update(kwargs)
        return cls(**config_dict)
    
    def save(self, file_path: Union[Path, str], encoding: str = "utf-8") -> None:
        """Save model to YAML file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding=encoding) as file:
            yaml.dump(self.model_dump(), file, default_flow_style=False)


class ModelConfig(YamlModel):
    """Configuration model for ML models."""
    # Define your model config fields here
    # example: learning_rate: float = 0.001
    pass