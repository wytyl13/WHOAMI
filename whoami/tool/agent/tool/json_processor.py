import json
from typing import (
    Optional,
    Dict,
    Any
)

from whoami.tool.agent.base_tool import tool


@tool
class JsonProcessor:
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    
    def parse_json_response(
        self,
        json_response: str,
        default_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        
        # 处理默认值
        if default_result is None:
            default_result = {}
        
        # 处理空响应
        if not json_response or not isinstance(json_response, str):
            return default_result
        
        response = json_response.strip()
        if not json_response:
            return default_result
        
        
         # 首先尝试直接解析完整响应
        try:
            return json.loads(json_response)
        except json.JSONDecodeError:
            pass
        
        # 接下来尝试提取JSON对象
        json_obj_start = json_response.find('{')
        json_obj_end = json_response.rfind('}') + 1
        if json_obj_start >= 0 and json_obj_end > json_obj_start:
            try:
                json_str = json_response[json_obj_start:json_obj_end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
        # 尝试提取JSON数组
        json_arr_start = json_response.find('[')
        json_arr_end = json_response.rfind(']') + 1
        if json_arr_start >= 0 and json_arr_end > json_arr_start:
            try:
                json_str = json_response[json_arr_start:json_arr_end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # 尝试修复常见JSON语法错误
        try:
            # 修复单引号替代双引号的问题
            fixed_response = json_response.replace("'", '"')
            return json.loads(fixed_response)
        except json.JSONDecodeError:
            pass
        
        try:
            # 修复缺少引号的键
            import re
            # 寻找没有引号的键
            pattern = r'(\s*?)(\w+)(\s*?):(\s*?)(?:"|\'|[\d\[\{])'
            fixed_response = re.sub(pattern, r'\1"\2"\3:\4', json_response)
            return json.loads(fixed_response)
        except (json.JSONDecodeError, re.error):
            pass
        
        # 尝试处理键值对格式但非严格JSON的情况
        try:
            result = {}
            # 匹配简单的键值对，如 "key: value" 或 "key: value,"
            pattern = r'[\"\']?(\w+)[\"\']?\s*:\s*[\"\']?([\w\s\.]+?)[\"\']?(?=,|\n|$)'
            matches = re.findall(pattern, response)
            if matches:
                for key, value in matches:
                    # 尝试转换数值
                    try:
                        if '.' in value and value.replace('.', '', 1).isdigit():
                            value = float(value)
                        elif value.isdigit():
                            value = int(value)
                        elif value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.lower() == 'null' or value.lower() == 'none':
                            value = None
                    except ValueError:
                        pass
                    
                    result[key] = value
                
                if result:  # 如果找到至少一个键值对
                    return result
        except Exception:
            pass
        
        # 所有尝试都失败，返回默认值
        return default_result
    
    
    def get_nested_value(self, data, key_path, default=None):
        """
        安全地获取嵌套JSON中的值
        
        参数:
            data: JSON数据
            key_path: 点分隔的键路径，如 "user.profile.name"
            default: 如果找不到值，返回的默认值
            
        返回:
            找到的值或默认值
        """
        if not data or not isinstance(data, dict):
            return default
        
        if not key_path:
            return default
        
        keys = key_path.split('.')
        current = data
        
        try:
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current
        except Exception:
            return default


    def map_json_to_schema(self, data, schema, default_values=None):
        """
        将解析的JSON映射到指定的模式结构
        
        参数:
            data: 解析后的JSON数据
            schema: 目标结构，如 {"is_related": "relation.isRelated", "confidence": "scores.confidence"}
                    键是目标键，值是原始数据中的键路径
            default_values: 默认值字典，用于未找到的键
            
        返回:
            根据模式映射后的新字典
        """
        if default_values is None:
            default_values = {}
        
        if not data or not schema:
            return {}
        
        result = {}
        
        for target_key, source_path in schema.items():
            default = default_values.get(target_key)
            result[target_key] = self.get_nested_value(data, source_path, default)
        
        return result
        
    
    def extract_intent_info(self, response):
        """
        从LLM响应中提取意图信息
        
        参数:
            response: LLM的响应文本
            
        返回:
            意图信息字典
        """
        # 定义默认结果
        default_result = {
            "is_related": False,
            "confidence": 0.5,
            "reasoning": "未提供理由"
        }
        
        # 解析JSON
        parsed_json = self.parse_json_response(response, default={})
        
        # 定义字段映射模式
        schema = {
            "is_related": "is_related",
            "confidence": "confidence",
            "reasoning": "reasoning"
        }
        
        # 映射到目标结构
        return self.map_json_to_schema(parsed_json, schema, default_result)


    def _run(self, *args, **kwargs):
        raise NotImplementedError


    async def execute(self, **kwargs: Any) -> Any:
        """
        执行工具逻辑，需要子类实现
        
        子类应该重写这个方法，实现实际的工具功能。
        框架会自动处理参数验证和类型检查。
        """
        raise NotImplementedError("Tool subclasses must implement execute method")