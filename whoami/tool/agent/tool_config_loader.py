from typing import Dict, Optional, Union, Any
from pathlib import Path
import os

from whoami.utils.yaml_model import YamlModel

class ToolConfigLoader(YamlModel):
    """工具配置加载器，用于加载特定工具的配置"""
    
    @classmethod
    def get_tool_config(cls, tool_name: str, config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
        """
        获取特定工具的配置
        
        Args:
            tool_name: 工具名称（类名）
            config_path: 配置文件路径，如果为None，则使用默认路径
            
        Returns:
            工具配置字典，如果不存在则返回空字典
        """
        # 使用默认配置路径（如果未指定）
        if config_path is None:
            config_path = os.environ.get('WHOAMI_TOOL_CONFIG_PATH', '/work/ai/WHOAMI/whoami/tool/agent/tool_config.yaml')
        
        # 使用父类的read方法读取整个配置文件
        all_configs = cls.read(config_path)
        
        # 获取特定工具的配置
        if isinstance(all_configs, dict):
            return all_configs.get(tool_name, {})
        
        return {}