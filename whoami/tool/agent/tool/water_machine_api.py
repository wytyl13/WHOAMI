
from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Any
)
import requests
import json

from whoami.tool.agent.base_tool import tool
# from whoami.tool.llm_application.enhance_retrieval import EnhanceRetrieval


class WaterMachineApiSchema(BaseModel):
    operation: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="""从用户关于饮水机操作的问题中提取相关的操作指令，可选择的指令包含：[开机, 关机, 水壶取水, 水壶停水, 开始加热, 停止加热, 开始保温, 停止保温, 打开语音, 关闭语音]
        """
    )

@tool
class WaterMachineApi:
    args_schema: Type[BaseModel] = WaterMachineApiSchema
    # 如果希望算法更加高效，设置end_flag为1，但是不能保证回复的质量
    end_flag: int = 1
    
    def __init__(self, **kwargs):
        
        # you should implement any private attribute here first. 
        super().__init__(**kwargs)
        
    async def execute(
        self, 
        operation: str, 
        message_history: List[Dict[str, Any]] = None, 
        username: Optional[str] = None, 
        location: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs,
    ):
        url = "http://1.71.15.102:48080/admin-api/device/device-info/deviceWaterMqtt"
        operation_mapping = {
            "开机": 1, 
            "关机": 2, 
            "水壶取水": 3, 
            "水壶停水": 4, 
            "开始加热": 5, 
            "停止加热": 6, 
            "开始保温": 7, 
            "停止保温": 8, 
            "打开语音": 9, 
            "关闭语音": "a", 
            # "水杯加水": "b", 
            # "水杯停水": "c"
        }
        
        params = {"content": operation_mapping.get(operation, 3)}
        try:
            # 发送HTTP请求
            response = requests.get(url, params=params, timeout=10)
            
            # 检查响应状态
            if response.status_code == 200:
                try:
                    result = response.json()
                    return_data = "成功执行操作！{result}"
                    for i in return_data:
                            yield i
                    return
                except json.JSONDecodeError:
                    return_data = f"操作失败！"
                    for i in return_data:
                        yield i
                    return
            else:
                error_info = f"操作失败！{str(e)}"
                for i in error_info:
                    yield i
                return
        
        except Exception as e:
            error_info = f"操作失败！{str(e)}"
            for i in error_info:
                yield i
            return