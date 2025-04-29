

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
from whoami.tool.agent.tool.sleep_indices_sql_data import SleepIndicesSqlData
from whoami.tool.agent.tool import TimeExtract
from whoami.tool.agent.tool import WeatherApi
from whoami.tool.agent.tool.zhoubian import ZhouBian
from whoami.tool.agent.tool.health_advice import HealthAdvice

weather_api = WeatherApi()

class HealthReportSchema(BaseModel):
    health_report_question: str = Field(
        ...,  # 使用 ... 表示必填字段
        # description="用户咨询的睡眠报告相关的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结"
        description="用户关于睡眠健康报告的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结。系统将分析用户的睡眠监测数据并提供专业解读。问题可涉及特定日期或时间段的睡眠质量分析、睡眠趋势比较、睡眠异常解释、健康建议等。系统将根据问题自动检索相关的睡眠数据记录，并给出专业的分析和建议。"
    )
    
    
zhoubian = ZhouBian()

@tool
class HealthReport:
    """回复的内容不详细，比如对于时间跨度较大的，比如一周，回复的内容不详细
    并没有精确回复每一天的，而且对要回复的时间范围把控不仔细

    Returns:
        _type_: _description_
    """
    args_schema: BaseModel = HealthReportSchema
    # 如果希望算法更加高效，设置end_flag为1
    end_flag: int = 0
    device_sn: Optional[str] = None
    enhance_llm: Optional[EnhanceRetrieval] = None
    sql_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    time_extract: Optional[TimeExtract] = None
    time_extract: Optional[TimeExtract] = None
    health_advice: Optional[HealthAdvice] = None
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.get('enhance_llm')
        if 'device_sn' in kwargs:
            self.device_sn = kwargs.get('device_sn')
        if 'sql_data' in kwargs:
            self.sql_data = kwargs.get('sql_data')
        if 'time_extract' in kwargs:
            self.time_extract = kwargs.get('time_extract')
        if 'health_advice' in kwargs:
            self.health_advice = kwargs.get('health_advice')
            
        # 验证 enhance_llm 是否设置
        if self.enhance_llm is None:
            self.logger.error("enhance_llm 未设置，执行方法将无法正常工作")
            
        self.time_extract = TimeExtract(llm=self.enhance_llm.llm) if self.time_extract is None else self.time_extract
        self.logger.info(f"初始化 HealthReport: enhance_llm={self.enhance_llm}, device_sn={self.device_sn}")
        # 初始化 sql_data
        if self.sql_data is None:
            try:
                self.logger.info("尝试创建 SleepIndicesSqlData 实例...")
                sql_data_instance = SleepIndicesSqlData()
                
                # 使用 sql_data 属性获取结果
                self.sql_data = sql_data_instance.sql_data
                # self.logger.info(f"成功获取 sql_data: {self.sql_data}")
                
                # 后处理数据（仅适用于健康报告的问答）
                
                
                if not self.sql_data:
                    self.logger.warning("获取到的 sql_data 为空，使用空字典作为默认值")
                    self.sql_data = {}
            except Exception as e:
                self.logger.error(f"SQL数据初始化失败: {str(e)}")
                self.sql_data = {}
                
        self.health_advice = HealthAdvice(enhance_llm=self.enhance_llm) if self.health_advice is None else self.health_advice    
    
    def filter_sleep_data_by_date_range(self, data, time_range: Dict):
        if not time_range:
            return data
        start_date = datetime.datetime.strptime(time_range["start"], '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(time_range["end"], '%Y-%m-%d').date()
        filtered_data = {}
    
        for device_sn, (sleep_reports, device_info) in data.items():
            filtered_reports = []
            
            for report in sleep_reports:
                report_date = report.get('查询日期')
                
                # 检查日期是否在指定范围内
                if isinstance(report_date, datetime.date) and start_date <= report_date <= end_date:
                    filtered_reports.append(report)
            
            # 只有在过滤后的报告非空时才添加到结果中
            if filtered_reports:
                filtered_data[device_sn] = (filtered_reports, device_info)
        
        return filtered_data
            
    
    
    async def execute(
        self, 
        health_report_question: str,
        device_sn: Optional[str] = None
    ) -> str:
        self.device_sn = device_sn if device_sn is not None else self.device_sn
        device_sn_ = [self.device_sn]
        # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_string = datetime.datetime.now().strftime("%Y-%m-%d")
        try:
            sql_data_ = {key: self.sql_data[key] for key in device_sn_ if key in self.sql_data}
        except Exception as e:
            # 报错说明是自定义上传的数据，直接使用它自己
            sql_data_ = self.sql_data
        self.logger.info(f"成功获取 sql_data: {sql_data_}")
        prompt = ""
        try:
            prompt = self.system_prompt.replace("current_time", date_string)
            prompt = self.system_prompt.replace("latest_data_date", date_string)
        except Exception as e:
            raise ValueError(f"fail to init prompt {str(e)}") from e
        result = await self.time_extract.execute(question=health_report_question)
        time_range = {}
        if result.get('found', False):
            time_range = result.get('time_range', {})
        time_range = {'start': date_string, 'end': date_string} if not time_range else time_range
        sql_data_ = self.filter_sleep_data_by_date_range(sql_data_, time_range)
        self.logger.info(f"成功获取 sql_data: {sql_data_}")
        
        self.logger.info(f"time_range -------------------------------- : {time_range}")
        self.logger.info(f"prompt -------------------------------- : {prompt}")
        full_response = ""
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
            full_response += chunk
            
            
        final_result = await self.health_advice.execute(
            health_report=full_response, 
            device_sn=self.device_sn, 
            elder_info=str(sql_data_[self.device_sn][1])
        )
        final = full_response + "\n" + final_result
        return final