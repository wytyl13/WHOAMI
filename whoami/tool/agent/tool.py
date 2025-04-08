from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Any
)

from whoami.tool.llm_application.enhance_retrieval import EnhanceRetrieval
from whoami.tool.agent.base_tool import tool
from whoami.tool.search.google_search_provider import GoogleSearch


class DirectLLMSchema(BaseModel):
    question: str = Field(
        ...,  # 使用 ... 表示必填字段
        # description="用户的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结"
        description="用户关于舜熙科技及其产品的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结。包括但不限于公司信息、产品功能、操作指南、技术规格、使用方法、售后服务等任何与舜熙科技相关的查询。系统将自动分析问题并从专有知识库中检索最相关的信息。"
    )
    
    
class GoogleSearchSchema(BaseModel):
    google_query: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="用于谷歌搜索的精确查询词。应提取用户问题中的关键实体、概念和查询意图，组织成能够获取最相关搜索结果的简洁查询字符串。查询词应聚焦于非舜熙科技相关的外部信息需求，如时事新闻、行业趋势、科学知识等。"
    )
    


class HealthReportSchema(BaseModel):
    health_report_question: str = Field(
        ...,  # 使用 ... 表示必填字段
        # description="用户咨询的睡眠报告相关的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结"
        description="用户关于睡眠健康报告的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结。系统将分析用户的睡眠监测数据并提供专业解读。问题可涉及特定日期或时间段的睡眠质量分析、睡眠趋势比较、睡眠异常解释、健康建议等。系统将根据问题自动检索相关的睡眠数据记录，并给出专业的分析和建议。"
    )

 
@tool
class DirectLLMTool:
    args_schema: Type[BaseModel] = DirectLLMSchema
    end_flag: int = 1
    # you can define the private attribution but not implement it here.
    enhance_llm: Optional[EnhanceRetrieval] = None
    
    def __init__(self, **kwargs):
        # you should implement any private attribute here. 
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.pop('enhance_llm')
        super().__init__(**kwargs)
    
    async def execute(self, question: str, message_history: List[Dict[str, Any]] = None) -> str:
        response = ""
        if self.enhance_llm:
            async for chunk in self.enhance_llm._run(
                text_list=[],
                query=question,
                message_history=message_history,
                retrieval_flag=False,
                stream_flag=0
            ):
                response += chunk
        return response
    
    
    
@tool
class DirectLLMTool:
    args_schema: Type[BaseModel] = DirectLLMSchema
    end_flag: int = 1
    # you can define the private attribution but not implement it here.
    enhance_llm: Optional[EnhanceRetrieval] = None
    
    def __init__(self, **kwargs):
        # you should implement any private attribute here. 
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.pop('enhance_llm')
        super().__init__(**kwargs)
    
    async def execute(self, question: str, message_history: List[Dict[str, Any]] = None) -> str:
        response = ""
        if self.enhance_llm:
            async for chunk in self.enhance_llm._run(
                text_list=[],
                query=question,
                message_history=message_history,
                retrieval_flag=False,
                stream_flag=0
            ):
                response += chunk
        return response
    
    

@tool
class GoogleSearchTool:
    args_schema: BaseModel = GoogleSearchSchema
    end_flag: int = 0
    google_search: Optional[GoogleSearch] = None # 自定义新的属性一定要在构造函数中初始化，否则会出现深拷贝错误
    enhance_llm: Optional[EnhanceRetrieval] = None
    
    def __init__(self, **kwargs):
        # 先初始化google_search
        if 'google_search' in kwargs:
            self.google_search = kwargs.pop('google_search')
        else:
            self.google_search = GoogleSearch(snippet_flag=0, 
                                                search_config_path='/work/ai/WHOAMI/whoami/scripts/test/search_config.yaml', 
                                                query_num=5)
        
        # you should implement any private attribute here. 
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.pop('enhance_llm')
        super().__init__(**kwargs)

    async def execute(self, google_query: str) -> float:
        param = {
            "query": google_query
        }
        status, result = self.google_search(**param)
        self.logger.info(result)
        text_list = [{item["link"]: item.get("fetch_url_content", item["html_snippet"])} for item in result]
        retrieval_nodes = self.enhance_llm.retrieve(text_list=text_list, top_k=2, query=google_query) if text_list else []
        context_texts = [node.node.text for node in retrieval_nodes]
        # self.logger.info("context_texts: -------------------------- {context_texts}")
        context = "没有检索到任何信息！" if not context_texts else "\n\n".join(context_texts)
        return context





    
if __name__ == '__main__':
    direct_llm_tool = DirectLLMTool()
    google_search_tool = GoogleSearchTool()
    print(direct_llm_tool)
    print(google_search_tool)

    