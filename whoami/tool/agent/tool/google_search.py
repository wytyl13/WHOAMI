from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Dict,
    Type,
    Any,
    overload
)


from whoami.tool.agent.tool import Retrieval
from whoami.tool.agent.base_tool import tool
from whoami.tool.search.google_search_provider import GoogleSearchProvider


class GoogleSearchSchema(BaseModel):

    google_query: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="用于谷歌搜索的精确查询词。应提取用户问题中的关键实体、概念和查询意图，组织成能够获取最相关搜索结果的简洁查询字符串。查询词应聚焦于非舜熙科技相关的外部信息需求，如时事新闻、行业趋势、科学知识等。"
    )
    
    
    
@tool
class GoogleSearch:
    """已完成谷歌检索
    待优化：
    1、并行提取网页文本
    2、使用代理访问google但是要使用本地网络访问国内的页面，否则会造成很大的延迟
    3、google检索结果待优化（不是最新消息、不是最相关的消息）
    4、网页文本解析有错误

    Args:
        BaseModel (_type_): _description_
    """
    args_schema: BaseModel = GoogleSearchSchema
    # 如果希望算法更加高效，设置end_flag为1
    end_flag: int = 0
    google_search_provider: Optional[GoogleSearchProvider] = None # 自定义新的属性一定要在构造函数中初始化，否则会出现深拷贝错误
    retrieval: Optional[Retrieval] = None
    
    
    @overload
    def __init__(
        self,
        retrieval: Optional[Retrieval] = None,
        google_search_provider: Optional[GoogleSearchProvider] = None
    ):
        ...
    
    
    def __init__(self, *args, **kwargs):
        self.logger.info("初始化GoogleSearch！")
        # you should implement any private attribute here first. 
        super().__init__(*args, **kwargs)
        # 先保存关键参数
        if 'retrieval' in kwargs:
            self.retrieval = kwargs.get('retrieval')

        # 初始化google_search
        if 'google_search_provider' in kwargs:
            self.google_search_provider = kwargs.get('google_search_provider')
            self.logger.info("使用传入的 google_search")
        else:
            self.logger.info("创建新的 GoogleSearchProvider 实例")
            self.google_search_provider = GoogleSearchProvider(
                snippet_flag=0, 
                search_config_path='/work/ai/WHOAMI/whoami/scripts/test/search_config.yaml', 
                query_num=5)
        
        
        #  验证组件初始化
        if self.google_search_provider is None:
            self.logger.error("google_search_provider 初始化失败")
        
        if self.retrieval is None:
            self.logger.error("retrieval 未设置，可能会影响功能")

    async def execute(self, google_query: str, username: Optional[str] = None, location: Optional[str] = None, role: Optional[str] = None) -> float:
        """执行谷歌搜索查询"""

        # 验证组件
        if self.google_search_provider is None:
            return "搜索服务初始化失败，无法执行查询。"
        
        if self.retrieval is None:
            return "增强检索组件未设置，无法处理搜索结果。"
        
        # 执行搜索
        self.logger.info(f"执行搜索查询: {google_query}")
        param = {
            "query": google_query
        }
        
        try:
            status, result = self.google_search_provider(**param)
            self.logger.info(result)
            text_list = [{item["link"]: item.get("fetch_url_content", item["html_snippet"])} for item in result]
            retrieval_nodes = await self.retrieval.execute(
                text_list=text_list, 
                top_k=3, 
                retrieval_word=google_query,
                static_flag=0
            ) if text_list else []
            context_texts = [node.node.text for node in retrieval_nodes]
            # self.logger.info("context_texts: -------------------------- {context_texts}")
            context = "没有检索到任何信息！" if not context_texts else "\n\n".join(context_texts)
            return context
        except Exception as e:
            error_msg = f"执行搜索或检索时出错: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
        
if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    from whoami.configs.llm_config import LLMConfig
    from whoami.llm_api.ollama_llm import OllamaLLM
    llm_qwen = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))
    retrieval = Retrieval()
    google_search = GoogleSearch(retrieval=retrieval)
    async def main():
        result = await google_search.execute(google_query="小米su7自燃事件")
        print(result)
    asyncio.run(main())