

from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Any
)
import aiohttp
import asyncio
import datetime
import asyncio
from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool.enhance_retrieval import EnhanceRetrieval
from whoami.tool.agent.tool.sleep_indices_sql_data import SleepIndicesSqlData
from whoami.tool.agent.tool import TimeExtract
from whoami.tool.agent.tool import WeatherApi
from whoami.tool.agent.tool.zhoubian import ZhouBian
from whoami.tool.agent.tool.health_advice import HealthAdvice
from whoami.tool.agent.tool.sleep_indices_extract import SleepIndicesExtract

weather_api = WeatherApi()

class HealthReportSimpleSchema(BaseModel):
    health_report_question: str = Field(
        ...,  # 使用 ... 表示必填字段
        # description="用户咨询的睡眠报告相关的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结"
        description="用户关于睡眠健康报告的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结。系统将分析用户的睡眠监测数据并提供专业解读。问题可涉及特定日期或时间段的睡眠质量分析、睡眠趋势比较、睡眠异常解释、健康建议等。系统将根据问题自动检索相关的睡眠数据记录，并给出专业的分析和建议。"
    )
    
    
zhoubian = ZhouBian()

@tool
class HealthReportSimple:
    """回复的内容不详细，比如对于时间跨度较大的，比如一周，回复的内容不详细
    并没有精确回复每一天的，而且对要回复的时间范围把控不仔细

    Returns:
        _type_: _description_
    """
    args_schema: BaseModel = HealthReportSimpleSchema
    # 如果希望算法更加高效，设置end_flag为1
    end_flag: int = 1
    username: Optional[str] = None
    enhance_llm: Optional[EnhanceRetrieval] = None
    session: Optional[aiohttp.ClientSession] = None
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.get('enhance_llm')
        if 'username' in kwargs:
            self.username = kwargs.get('username')
            
        # 验证 enhance_llm 是否设置
        if self.enhance_llm is None:
            self.logger.error("enhance_llm 未设置，执行方法将无法正常工作")
    
    
    async def _get_session(self):
        """获取或创建HTTP会话"""
        if self.session is None:
            import ssl
            # 创建不验证SSL证书的上下文
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
    
    
    
    async def _make_request(self, url: str, method: str = "POST", data: Dict[str, Any] = None):
        """发送HTTP请求"""
        session = await self._get_session()
        try:
            if method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    return await response.json()
            elif method.upper() == "GET":
                async with session.get(url, params=data) as response:
                    return await response.json()
        except Exception as e:
            return {"error": str(e)}
       
    
    async def execute(
        self, 
        health_report_question: str,
        username: Optional[str] = None,
        location: Optional[str] = None,
        role: Optional[str] = None
    ):
        self.logger.info(username)
        self.logger.info(location)
        self.username = username if username is not None else self.username
        request_data = {
            "username": self.username
        }
        method = "POST"
        url = "https://1.71.15.121:8889/api/list_sleep_statistics"
        # 发送请求
        result = await self._make_request(url, method, request_data)
        sql_data_ = None
        if result["success"]:
            sql_data_ = result["data"]

        user_info = self.username + location
        current_time = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = self.system_prompt.replace("user_info", user_info)
        prompt = prompt.replace("current_time", current_time)
        async for chunk in self.enhance_llm.execute(
            text_list=[], 
            message_history=[], 
            question=health_report_question,
            prompt=prompt,
            database_retrieval_data=sql_data_,
            top_k=3,
            retrieval_flag=0,
            stream_flag=1
        ):
            yield chunk
        return
    
    
    
if __name__ == '__main__':
    from whoami.configs.llm_config import LLMConfig
    from whoami.llm_api.ollama_llm import OllamaLLM
    from pathlib import Path
    
    llm_qwen = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))
    enhance_qwen = EnhanceRetrieval(llm=llm_qwen)
    health_report_tool = HealthReportSimple(enhance_llm=enhance_qwen)
    import asyncio
    async def main():
        async for chunk in health_report_tool.execute(
            health_report_question="报告下我的心率情况", 
            username="weiyutao",
            location="舜熙科技智慧养老社区"
        ):
            print(chunk)
    asyncio.run(main())