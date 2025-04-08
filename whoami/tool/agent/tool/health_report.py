

from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Any
)


from whoami.tool.agent.base_tool import tool
from whoami.tool.llm_application.enhance_retrieval import EnhanceRetrieval
from whoami.tool.agent.tool.sleep_indices_sql_data import SleepIndicesSqlData


class HealthReportSchema(BaseModel):
    health_report_question: str = Field(
        ...,  # 使用 ... 表示必填字段
        # description="用户咨询的睡眠报告相关的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结"
        description="用户关于睡眠健康报告的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结。系统将分析用户的睡眠监测数据并提供专业解读。问题可涉及特定日期或时间段的睡眠质量分析、睡眠趋势比较、睡眠异常解释、健康建议等。系统将根据问题自动检索相关的睡眠数据记录，并给出专业的分析和建议。"
    )
    
    

@tool
class HealthReport:
    """回复的内容不详细，比如对于时间跨度较大的，比如一周，回复的内容不详细
    并没有精确回复每一天的，而且对要回复的时间范围把控不仔细

    Returns:
        _type_: _description_
    """
    args_schema: BaseModel = HealthReportSchema
    # 如果希望算法更加高效，设置end_flag为1
    end_flag: int = 1
    device_sn: Optional[str] = None
    enhance_llm: Optional[EnhanceRetrieval] = None
    sql_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.get('enhance_llm')
        if 'device_sn' in kwargs:
            self.device_sn = kwargs.get('device_sn')
        if 'sql_data' in kwargs:
            self.sql_data = kwargs.get('sql_data')
        self.logger.info(f"初始化 HealthReport: enhance_llm={self.enhance_llm}, device_sn={self.device_sn}")

        # 初始化 sql_data
        if self.sql_data is None:
            try:
                self.logger.info("尝试创建 SleepIndicesSqlData 实例...")
                sql_data_instance = SleepIndicesSqlData()
                
                # 使用 sql_data 属性获取结果
                self.sql_data = sql_data_instance.sql_data
                self.logger.info(f"成功获取 sql_data: {self.sql_data}")
                
                if not self.sql_data:
                    self.logger.warning("获取到的 sql_data 为空，使用空字典作为默认值")
                    self.sql_data = {}
            except Exception as e:
                self.logger.error(f"SQL数据初始化失败: {str(e)}")
                self.sql_data = {}
                
                
            # 根据 device_sn 筛选数据
            if self.device_sn is not None and self.sql_data:
                try:
                    if self.device_sn in self.sql_data:
                        self.logger.info(f"正在筛选设备 {self.device_sn} 的数据")
                        device_data = self.sql_data[self.device_sn]
                        self.sql_data = device_data
                    else:
                        self.logger.warning(f"设备编号 {self.device_sn} 不存在于sql_data中，将使用空列表")
                        self.sql_data = []
                except Exception as e:
                    self.logger.error(f"筛选设备数据失败: {str(e)}")
                    self.sql_data = []
                    
            # 验证 enhance_llm 是否设置
            if self.enhance_llm is None:
                self.logger.error("enhance_llm 未设置，执行方法将无法正常工作")
                
            
    
    async def execute(
        self, 
        health_report_question: str,
        device_sn: Optional[str] = None
    ) -> str:
        sql_data_ = self.sql_data[device_sn] if device_sn is not None else self.sql_data
        full_response = ""
        async for chunk in self.enhance_llm._run(
            text_list=[], 
            message_history=[], 
            query=health_report_question,
            rewritten_query=health_report_question,
            prompt=self.system_prompt,
            database_retrieval_data=sql_data_,
            top_k=3,
            retrieval_flag=0,
            stream_flag=1
        ):
            full_response += chunk
        return full_response