from typing import (
    Optional,
    Dict,
    List,
    Any
)
from datetime import datetime


from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool import JsonProcessor
from whoami.llm_api.ollama_llm import OllamaLLM

@tool
class InfoExtract(JsonProcessor):
    
    llm: Optional[OllamaLLM] = None
    default_extract_result: Optional[Dict] = None
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'llm' in kwargs:
            self.llm = kwargs.get('llm')
            
        if 'default_extract_result' in kwargs:
            self.default_extract_result = kwargs.get('default_extract_result')
            
         
    async def extract_info(self, 
        query: str, 
        message_history: List[Dict[str, str]] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        current_time = datetime.now()
        current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        message = self.system_prompt + f"\n\n历史会话消息：{message_history}" + f"\n\n当前时间：{current_time}" + f"\n\n当前问题：{query}"
        namespace_message_history = [{"role": "user", "content": message}]
        try:
            # 调用LLM生成回答
            self.logger.info(namespace_message_history)
            response = await self.llm._whoami_text(namespace_message_history, timeout=30, user_stop_words=[])
            # 使用基类的JSON解析方法处理回答
            result = self.parse_json_response(response, self.default_extract_result)
            return result
            
        except Exception as e:
            self.logger.error(f"提取信息时出错: {str(e)}")
            return self.default_extract_result
        
    
    async def execute(self, question: str) -> Any:
        """
        执行工具逻辑，需要子类实现
        
        子类应该重写这个方法，实现实际的工具功能。
        框架会自动处理参数验证和类型检查。
        """
        return await self.extract_info(question, message_history=[])