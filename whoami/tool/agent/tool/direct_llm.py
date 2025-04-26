
from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Any
)


from whoami.tool.agent.base_tool import tool
# from whoami.tool.llm_application.enhance_retrieval import EnhanceRetrieval
from whoami.tool.agent.tool import EnhanceRetrieval


class DirectLLMSchema(BaseModel):
    question: str = Field(
        ...,  # 使用 ... 表示必填字段
        # description="用户的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结"
        description="用户关于舜熙科技及其产品的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结。包括但不限于公司信息、产品功能、操作指南、技术规格、使用方法、售后服务等任何与舜熙科技相关的查询。系统将自动分析问题并从专有知识库中检索最相关的信息。"
    )

@tool
class DirectLLM:
    args_schema: Type[BaseModel] = DirectLLMSchema
    # 如果希望算法更加高效，设置end_flag为1，但是不能保证回复的质量
    end_flag: int = 1
    # you can define the private attribution but not implement it here.
    enhance_llm: Optional[EnhanceRetrieval] = None
    
    def __init__(self, **kwargs):
        
        # you should implement any private attribute here first. 
        super().__init__(**kwargs)
        
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.pop('enhance_llm')
    
    async def execute(self, question: str, message_history: List[Dict[str, Any]] = None) -> str:
        response = ""
        if self.enhance_llm:
            async for chunk in self.enhance_llm.execute(
                text_list=[],
                question=question,
                message_history=message_history,
                retrieval_flag=1,
                stream_flag=1
            ):
                response += chunk
        return response