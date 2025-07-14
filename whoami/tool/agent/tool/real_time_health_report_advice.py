

from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Any
)
import datetime
import asyncio
from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool.enhance_retrieval import EnhanceRetrieval


class RealTimeHealthReportAdviceSchema(BaseModel):
    health_report_statistics: str = Field(
        ...,  # 使用 ... 表示必填字段
        # description="用户咨询的睡眠报告相关的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结"
        description="根据用户的睡眠统计数据生成专业的睡眠报告"
    )
    
    
@tool
class RealTimeHealthReportAdvice:
    """
    根据用户的睡眠统计数据生成专业的睡眠报告
    Returns:
        _type_: _description_
    """
    args_schema: BaseModel = RealTimeHealthReportAdviceSchema
    end_flag: int = 1
    enhance_llm: Optional[EnhanceRetrieval] = None
    
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.get('enhance_llm')
        if self.enhance_llm is None:
            self.logger.error("enhance_llm 未设置，执行方法将无法正常工作")
            
        self.logger.info(f"初始化 HealthReport: enhance_llm={self.enhance_llm}")
    
    
    async def execute(
        self, 
        health_report_statistics: str,
        device_sn: Optional[str] = None,
    ) -> str:
        
        date_string = datetime.datetime.now().strftime("%Y-%m-%d")
        try:
            prompt = self.system_prompt.replace("current_time", date_string)
            prompt = self.system_prompt.replace("sleep_data_state", health_report_statistics)
        except Exception as e:
            raise ValueError(f"fail to init prompt {str(e)}") from e
        
        # database sql_data_, need to filter used time_range and keywords.
        full_response = ""
        async for chunk in self.enhance_llm.execute(
            text_list=[], 
            message_history=[], 
            question="生成专业的睡眠报告",
            prompt=prompt,
            database_retrieval_data=None,
            top_k=3,
            retrieval_flag=0,
            stream_flag=1
        ):
            full_response += chunk
        
        return full_response
    
    
    
if __name__ == '__main__':
    from whoami.configs.llm_config import LLMConfig
    from whoami.llm_api.ollama_llm import OllamaLLM
    from pathlib import Path
    
    llm_qwen = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))
    enhance_qwen = EnhanceRetrieval(llm=llm_qwen)
    health_report_tool = RealTimeHealthReportAdvice(enhance_llm=enhance_qwen)
    import asyncio
    async def main():
        result = await health_report_tool.execute(
            health_report_statistics="{'avg_breath_rate': 14, 'avg_heart_rate': 79, 'heart_rate_variability': 0.1932, 'body_movement_count': 2, 'apnea_count': 2, 'rapid_breathing_count': 44, 'leave_bed_count': 3, 'total_duration': '9小时17分35秒', 'in_bed_duration': '9小时14分0秒', 'out_bed_duration': '0小时3分35秒', 'deep_sleep_duration': '0小时47分29秒', 'light_sleep_duration': '5小时12分1秒', 'awake_duration': '2小时49分7秒', 'bed_time': '2025-07-03 21:00:00', 'sleep_time': '2025-07-03 21:40:57', 'wake_time': '2025-07-04 04:32:55', 'leave_bed_time': '2025-07-04 07:01:00', 'device_sn': '13D2F34920008071211195A907', 'sleep_start_time': '2025-7-3 21:00:00', 'sleep_end_time': '2025-7-4 07:00:00'}",
            device_sn="13D6F349200080712111957107"
        )
        print(result)
    asyncio.run(main())