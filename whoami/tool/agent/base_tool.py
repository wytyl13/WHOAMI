from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Type, Literal, Set, Union
from pydantic import BaseModel, model_validator
import inspect


from whoami.utils.log import Logger
from whoami.tool.agent.tool_config_loader import ToolConfigLoader
import functools

class BaseTool(ABC, BaseModel):
    """工具基类，结合了自动参数解析和Pydantic模型验证"""
    name: Optional[str] = None
    description: Optional[str] = None
    args_schema: Optional[Type[BaseModel]] = None
    logger: Optional[Logger] = None
    end_flag: Optional[int] = None
    stream_flag: int = 0
    system_prompt: Optional[str] = None
    
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
    async def execute(self, **kwargs: Any) -> Any:
        """
        执行工具逻辑，需要子类实现
        
        子类应该重写这个方法，实现实际的工具功能。
        框架会自动处理参数验证和类型检查。
        """
        raise NotImplementedError("Tool subclasses must implement execute method")
    
    
    async def __call__(self, **kwargs: Any) -> Any:
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
    
    
    def get_simple_tool_description(self) -> str:
        """
        返回工具的简化描述，格式更友好且易于理解
        """
        # 获取工具参数的简单描述
        params_desc = ""
        if hasattr(self, 'tool_schema'):
            # 从工具的schema中获取参数信息
            schema = self.tool_schema
            properties = schema.get('parameters', {}).get('properties', {})
            
            params = []
            for param_name, param_info in properties.items():
                # 尝试从不同的位置获取description
                description = param_info.get('description', 
                            param_info.get('title', f'参数 {param_name}'))
                params.append(f'"{param_name}": "{description}"')
            
            if params:
                params_desc = "{" + ", ".join(params) + "}"
        
        # 构建简化的工具描述
        description = f"{self.name}: {self.description}"
        if params_desc:
            description += f"\n参数: {params_desc}"

        return description
    
    
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



def tool(cls=None, **decorator_kwargs):
    def decorator(cls_):
        # 获取工具配置
        tool_config = ToolConfigLoader.get_tool_config(cls_.__name__)
        if not tool_config and not cls_.__name__.endswith("Tool"):
            tool_config = ToolConfigLoader.get_tool_config(cls_.__name__ + "Tool")
            
        # 获取类注解
        cls_annotations = getattr(cls_, '__annotations__', {})
        
        # 获取类级别的属性
        class_attributes = {}
        for attr_name in dir(cls_):
            if not attr_name.startswith('__') and not callable(getattr(cls_, attr_name)):
                class_attributes[attr_name] = getattr(cls_, attr_name)
        
        # 原始初始化方法
        original_init = getattr(cls_, '__init__', lambda self, **kwargs: None)
        
        @functools.wraps(original_init)
        def wrapped_init(self, **kwargs):
            # 创建配置字典但不包含 args_schema
            config_kwargs = {}
            
            # 设置优先级顺序: 默认值 < 配置文件 < 类级别属性 < 传入参数
            
            # 1. 基本默认值
            config_kwargs['name'] = cls_.__name__
            config_kwargs['end_flag'] = 0
            config_kwargs['stream_flag'] = 0
            config_kwargs['description'] = ''
            config_kwargs['system_prompt'] = None
            
            # 2. 从配置文件加载属性，覆盖默认值
            for key, value in tool_config.items():
                if value is not None and key != 'args_schema':
                    config_kwargs[key] = value
            
            # 3. 应用类级别属性，覆盖配置文件的值
            for key, value in class_attributes.items():
                if key not in ['__annotations__', 'args_schema'] and value is not None:
                    config_kwargs[key] = value
            
            # 4. 用传入的参数更新配置，最高优先级
            for key, value in kwargs.items():
                if value is not None and key != 'args_schema':
                    config_kwargs[key] = value
            
            # 初始化 BaseTool
            BaseTool.__init__(self, **config_kwargs)
            
            # 单独处理 args_schema
            if hasattr(cls_, 'args_schema'):
                self.args_schema = getattr(cls_, 'args_schema')
            
            # 调用原始初始化方法，但要先检查参数
            if original_init is not None:
                # 获取原始init方法的签名
                params = inspect.signature(original_init).parameters
                
                # 过滤kwargs，只保留原始init可接受的参数
                filtered_kwargs = {}
                for k, v in config_kwargs.items():
                    # 如果参数在原始init的参数列表中或原始init接受**kwargs
                    if k in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                        filtered_kwargs[k] = v
                
                # 调用原始init
                original_init(self, **filtered_kwargs)
        
        # 创建包装类
        WrappedTool = type(
            cls_.__name__,
            (cls_, BaseTool),
            {
                '__init__': wrapped_init,
                '__module__': cls_.__module__,
                '__doc__': cls_.__doc__,
            }
        )
        
        return WrappedTool
    
    if cls is None:
        return decorator
    return decorator(cls)

    
    
# def tool(cls=None, **decorator_kwargs):
#     def decorator(cls_):
#         # Get tool configuration
#         tool_config = ToolConfigLoader.get_tool_config(cls_.__name__)
        
#         # Get annotations from the class
#         cls_annotations = getattr(cls_, '__annotations__', {})
        
#         # Original init
#         original_init = getattr(cls_, '__init__', lambda self, **kwargs: None)
        
#         @functools.wraps(original_init)
#         def wrapped_init(self, **kwargs):
#             # Base attributes
#             config_kwargs = {
#                 'name': tool_config.get('name', cls_.__name__),
#                 'description': tool_config.get('description', ''),
#                 'end_flag': tool_config.get('end_flag', 0),
#                 'stream_flag': tool_config.get('stream_flag', 0)
#             }
            
#             # Handle additional attributes
#             for attr_name, attr_type in cls_annotations.items():
#                 if attr_name not in ['name', 'description', 'end_flag', 'stream_flag']:
#                     config_value = tool_config.get(attr_name)
#                     if config_value is not None:
#                         config_kwargs[attr_name] = config_value
            
#             # Update with passed kwargs
#             config_kwargs.update(kwargs)
            
#             # Initialize the BaseTool part first
#             BaseTool.__init__(self, **config_kwargs)
            
#             # Then call the original init if it exists
#             if original_init is not None and original_init.__code__.co_argcount > 1:
#                 original_init(self, **config_kwargs)
        
#         # Create the wrapped class
#         WrappedTool = type(
#             cls_.__name__,
#             (cls_, BaseTool),
#             {
#                 '__init__': wrapped_init,
#                 '__module__': cls_.__module__,
#                 '__doc__': cls_.__doc__,
#             }
#         )
        
#         return WrappedTool
    
#     if cls is None:
#         return decorator
#     return decorator(cls)
    
    