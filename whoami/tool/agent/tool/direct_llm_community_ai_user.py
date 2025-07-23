
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


class DirectLLMCommunityAiUserSchema(BaseModel):
    question: str = Field(
        ...,  # 使用 ... 表示必填字段
        # description="用户的完整问题，需根据当前问题和历史对话上下文进行综合理解和总结"
        description="用户任何问题！"
    )

@tool
class DirectLLMCommunityAiUser:
    args_schema: Type[BaseModel] = DirectLLMCommunityAiUserSchema
    # 如果希望算法更加高效，设置end_flag为1，但是不能保证回复的质量
    end_flag: int = 1
    # you can define the private attribution but not implement it here.
    enhance_llm: Optional[EnhanceRetrieval] = None
    
    def __init__(self, **kwargs):
        
        # you should implement any private attribute here first. 
        super().__init__(**kwargs)
        
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.pop('enhance_llm')
    
    async def execute(
        self, 
        question: str, 
        message_history: List[Dict[str, Any]] = None, 
        username: Optional[str] = None, 
        location: Optional[str] = None,
        role: Optional[str] = None
    ):
        user_info = username + ", " + location
        prompt = self.system_prompt.replace("user_info", user_info)
        self.logger.info(f"prompt: ---------------------------------- {prompt}")
        if self.enhance_llm:
            async for chunk in self.enhance_llm.execute(
                text_list=[],
                question=question,
                message_history=message_history,
                prompt=prompt,
                stream_flag=1,
                retrieval_flag=True
            ):
                # response += chunk
                yield chunk
        # return response