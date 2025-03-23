from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Type, Literal, Set, Union
from pydantic import BaseModel, model_validator
import inspect


from whoami.utils.log import Logger



class BaseTool(ABC, BaseModel):
    """工具基类，结合了自动参数解析和Pydantic模型验证"""
    name: Optional[str] = None
    description: Optional[str] = None
    args_schema: Optional[Type[BaseModel]] = None
    logger: Optional[Logger] = None
    
    
    class Config:
        arbitrary_types_allowed = True  # 允许任意类型
        extra = "allow" # 允许设置额外属性，默认不允许
    
    def __init__(self, **data):
        super().__init__(**data)
        self.inputs = {}  # 输入参数名和类型的映射，注意如果要在这里设置额外属性，需要在Config中允许
        self._parse_input_signature()  # 自动解析输入参数
    
      
    @model_validator(mode="before")
    @classmethod
    def set_name_if_empty(cls, values):
        """如果名称为空，则使用类名作为工具名称"""
        if "name" not in values or not values["name"]:
            values["name"] = cls.__name__
        return values
    
    
    @model_validator(mode="before")
    @classmethod
    def set_logger_if_empty(cls, values):
        """如果日志记录器为空，则创建一个新的记录器"""
        if "logger" not in values or not values["logger"]:
            values["logger"] = Logger(cls.__name__)
        return values
    
    
    def _parse_input_signature(self):
        """解析方法签名，自动获取输入参数"""
        sig = inspect.signature(self.execute)
        for param_name, param in sig.parameters.items():
            if param_name != 'self':
                self.inputs[param_name] = param.annotation
                
                
    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """
        执行工具逻辑，需要子类实现
        
        子类应该重写这个方法，实现实际的工具功能。
        框架会自动处理参数验证和类型检查。
        """
        raise NotImplementedError("Tool subclasses must implement execute method")
    
    
    def __call__(self, **kwargs: Any) -> Any:
        """使工具可调用，提供与execute相同的接口但增加验证"""
        # 验证输入参数
        # 并且可以使用多余的参数（最终不被execute执行，但是对开发人员有用，比如日志输出等等）去做逻辑验证
        if not self.validate_inputs(kwargs):
            missing = set(self.get_input_names()) - set(kwargs.keys())
            raise ValueError(f"Missing required inputs for {self.name}: {missing}")
        
        # 如果有 args_schema，使用它进行额外验证
        if self.args_schema:
            # 只保留 args_schema 中定义的字段
            schema_fields = set(self.args_schema.__annotations__.keys())
            schema_args = {k: v for k, v in kwargs.items() if k in schema_fields}
            validated_args = self.args_schema(**schema_args)
            
            # 使用验证后的参数，但保留其他非架构参数
            kwargs = {k: getattr(validated_args, k) for k in schema_fields}
            # kwargs.update({k: getattr(validated_args, k) for k in schema_fields})
        
        # 调用实际的实现
        # 只保留 execute 方法需要的参数
        execute_params = set(self.get_input_names())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in execute_params}
        return self.execute(**filtered_kwargs)
    
    
    def get_input_names(self) -> List[str]:
        """获取输入参数名列表"""
        return list(self.inputs.keys())
    
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """验证输入参数是否满足要求"""
        required_inputs = set(self.get_input_names())
        provided_inputs = set(inputs.keys())
        return required_inputs.issubset(provided_inputs)
    
    
    def args(self) -> Dict[str, Any]:
        """获取参数架构信息"""
        if self.args_schema:
            return self.model_json_schema(self.args_schema)['properties']
        else:
            # 从方法签名生成简单的架构
            schema = {'properties': {}}
            for param_name, param_type in self.inputs.items():
                schema['properties'][param_name] = self._get_field_schema(param_type)
            return schema['properties']
        

    @property
    def tool_schema(self) -> Dict[str, Any]:
        """工具的完整描述信息，包括名称、描述和参数详情"""
        tool_info = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        # 获取参数信息
        if self.args_schema:
            # 使用args_schema获取参数信息
            schema = self.args_schema.model_json_schema()
            tool_info["parameters"]["properties"] = schema.get("properties", {})
            tool_info["parameters"]["required"] = schema.get("required", [])
        else:
            # 从execute方法签名获取参数信息
            for param_name, param_type in self.inputs.items():
                tool_info["parameters"]["properties"][param_name] = self._get_field_schema(param_type)
                # 添加到必需参数列表
                tool_info["parameters"]["required"].append(param_name)
        
        return tool_info
    
    
    def model_json_schema(
        self, 
        cls: Type[BaseModel],
        mode: Literal['validation', 'serialization'] = 'validation'
    ) -> Dict[str, Any]:
        """
        为模型生成 JSON Schema，包含 Field 的 description。
        """
        if cls is BaseModel:
            raise AttributeError('不能直接在 BaseModel 上调用，必须使用其子类')

        schema = {
            'type': 'object',
            'properties': {},
            'required': []
        }

        # 获取所有字段
        for field_name, field_type in cls.__annotations__.items():
            # 获取字段基本信息
            field_schema = self._get_field_schema(field_type)
            
            # 获取字段的 Field 元数据
            field_metadata = None
            
            # 尝试获取字段的 Field 对象
            if hasattr(cls, field_name):
                field_value = getattr(cls, field_name)
                if hasattr(field_value, 'default') and hasattr(field_value, 'description'):
                    # 这是 Pydantic v2 的 Field 对象
                    field_metadata = field_value
                elif hasattr(field_value, 'annotation') and hasattr(field_value, 'default'):
                    # 可能是 Pydantic v1 的 FieldInfo 对象
                    field_metadata = field_value
            
            # 添加描述
            if field_metadata and hasattr(field_metadata, 'description'):
                field_schema['description'] = field_metadata.description
            
            schema['properties'][field_name] = field_schema
            
            # 确定字段是否必需
            is_optional = False
            
            # 检查类型是否为 Optional
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Union and type(None) in field_type.__args__:
                is_optional = True
            
            # 检查是否有默认值
            if field_metadata and hasattr(field_metadata, 'default') and field_metadata.default is not ...:
                is_optional = True
            
            # 将必需字段添加到 required 列表
            if not is_optional:
                schema['required'].append(field_name)

        return schema
    
    def _get_field_schema(self, field_type: Type) -> Dict[str, Any]:
        """生成字段的 schema"""
        # 处理基本类型
        if field_type is str:
            return {'type': 'string'}
        elif field_type is int:
            return {'type': 'integer'}
        elif field_type is float:
            return {'type': 'number'}
        elif field_type is bool:
            return {'type': 'boolean'}
        
        # 处理 List 类型
        if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            item_type = field_type.__args__[0] if hasattr(field_type, "__args__") else Any
            items_schema = self._get_field_schema(item_type)
            return {'type': 'array', 'items': items_schema}
        
        # 处理 Optional 类型 (Union[T, None])
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            types = [t for t in field_type.__args__ if t is not type(None)]
            if len(types) == 1:
                return self._get_field_schema(types[0])
        
        # 处理其他复杂类型或未知类型
        return {'type': 'object'}
    
    
    