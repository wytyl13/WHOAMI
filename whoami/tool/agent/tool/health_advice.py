

from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Any
)
from datetime import datetime
import asyncio
from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool.enhance_retrieval import EnhanceRetrieval
from whoami.tool.agent.tool.sleep_indices_sql_data import SleepIndicesSqlData
from whoami.tool.agent.tool import TimeExtract
from whoami.tool.agent.tool import WeatherApi
from whoami.tool.agent.tool.zhoubian import ZhouBian

weather_api = WeatherApi()

class HealthAdviceSchema(BaseModel):
    health_report: str = Field(
        ...,  # 使用 ... 表示必填字段
        # description="用户咨询的睡眠报告相关的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结"
        description="用户的睡眠报告情况。"
    )
    
    
zhoubian = ZhouBian()

@tool
class HealthAdvice:
    """回复的内容不详细，比如对于时间跨度较大的，比如一周，回复的内容不详细
    并没有精确回复每一天的，而且对要回复的时间范围把控不仔细

    Returns:
        _type_: _description_
    """
    args_schema: BaseModel = HealthAdviceSchema
    # 如果希望算法更加高效，设置end_flag为1
    end_flag: int = 0
    device_sn: Optional[str] = None
    enhance_llm: Optional[EnhanceRetrieval] = None
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.get('enhance_llm')
        if 'device_sn' in kwargs:
            self.device_sn = kwargs.get('device_sn')
            
        # 验证 enhance_llm 是否设置
        if self.enhance_llm is None:
            self.logger.error("enhance_llm 未设置，执行方法将无法正常工作")
            
        self.logger.info(f"初始化 HealthReport: enhance_llm={self.enhance_llm}, device_sn={self.device_sn}")
                
                    
    async def execute(
        self, 
        health_report: str,
        device_sn: Optional[str] = None,
        elder_info: str = None
    ) -> str:
        self.logger.info(elder_info)
        self.device_sn = device_sn if device_sn is not None else self.device_sn
        address = elder_info[0]["elderly_address"] if "elderly_address" in elder_info[0] else "山西省运城市盐湖区黄河金三角(运城)创新生态集聚区科创城"
        device_sn_ = [self.device_sn]
        # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today = datetime.now().strftime('%Y-%m-%d')
        weekday_num = datetime.now().weekday()
        # 中文星期名称列表，Monday对应“星期一”
        weekday_cn = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
        weekday = weekday_cn[weekday_num]
        prompt = ""
        weather_task = weather_api.execute(query_key="who")
        sport_task = zhoubian.execute(query_district=address, query_type="公园$大学")
        shopping_task = zhoubian.execute(query_district=address, query_type="超市$商场")
        hospital_task = zhoubian.execute(query_district=address, query_type="医院$药房$诊所")
        
        weather_info, sport_info, shopping_info, hospital_info = await asyncio.gather(
            weather_task,
            sport_task,
            shopping_task,
            hospital_task
        )
        weather_info = "未查询到结果！" if weather_info is None else weather_info 
        sport_info = "未查询到结果！" if sport_info is None else sport_info 
        shopping_info = "未查询到结果！" if shopping_info is None else shopping_info 
        hospital_info = "未查询到结果！" if hospital_info is None else hospital_info 
        try:
            prompt = self.system_prompt.replace("today", today)
            prompt = self.system_prompt.replace("weekday", weekday)
            prompt = self.system_prompt.replace("health_report", health_report)
            prompt = prompt.replace("elder_info", str(elder_info))
            prompt = prompt.replace("weather_info", str(weather_info))
            prompt = prompt.replace("location_environment_factors", "山西省，运城市，盐湖区，黄河金三角(运城)创新生态集聚区科创城，附近没有大型施工情况")
            prompt = prompt.replace("nearby_parks_activity_centers", str(sport_info))
            prompt = prompt.replace("nearby_markets", str(shopping_info))
            prompt = prompt.replace("seasonal_foods", "春玉米、香椿、荠菜、苦菜、蒲公英等")
            prompt = prompt.replace("nearby_hospitals", str(hospital_info))
        except Exception as e:
            raise ValueError(f"fail to init prompt {str(e)}") from e
        self.logger.info(f"prompt -------------------------------- : {prompt}")
        full_response = ""
        async for chunk in self.enhance_llm.execute(
            text_list=[], 
            message_history=[], 
            question='根据以上提示给用户回答针对性的建议',
            prompt=prompt,
            top_k=3,
            retrieval_flag=0,
            stream_flag=1
        ):
            full_response += chunk
            
            
        
        # 根据以上response提供针对性的建议
            
        
        return full_response
    

if __name__ == '__main__':
    from whoami.llm_api.ollama_llm import OllamaLLM
    from whoami.configs.llm_config import LLMConfig
    from pathlib import Path
    
    
    llm_qwen = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))
    enhance_qwen = EnhanceRetrieval(llm=llm_qwen)
    health_advice = HealthAdvice(enhance_llm=enhance_qwen)
    async def main():
        response = await health_advice.execute(
            health_report="""
            昨天的睡眠报告显示，您的心率为80次/分钟（正常范围内但略低），而深睡时间仅为37分钟，低于建议值。因此您的睡眠质量被评为“较差”。建议关注并改善您的睡眠习惯以提高整体健康水平。
            """,
            device_sn="",
            elder_info="""[{'device_sn': '13D6F349200080712111957107', 'device_type': 'WAVVE_SLEEP_DETECTION', 'device_name': '睡眠检测', 'device_status': 1, 'dept_id': 187, 'room_id': 126, 'bed_id': 187, 'elderly_id': 197, 'elderly_name': '李桂花', 'institution_room_id': 126, 'institution_bed_id': 187, 'elderly_id_card': '142725194007120029', 'elderly_sex': '2', 'elderly_age': 85, 'elderly_birthday': None, 'elderly_address': '万荣县解店镇七庄村', 'contacts': '[{"name":"程自强","phone":"17635930777"}]'}]"""
        )
        print(response)
        return response
    
    asyncio.run(main())