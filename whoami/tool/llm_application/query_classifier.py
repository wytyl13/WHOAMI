import re
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RouterQueryEngine

class QueryClassifier:
    def __init__(self, llm=None):
        self.llm = llm

    def build_router(self, db_engine, web_engine, direct):
        """构建查询路由器，决定使用哪种查询引擎"""
        query_engine_tools = [
            QueryEngineTool.from_defaults(
                query_engine=db_engine,
                description=(
                    "用于查询已有数据库中的信息，适用于历史数据、产品信息、"
                    "用户记录等结构化或半结构化存储的信息。当问题涉及到系统"
                    "内部已存储的信息时使用。"
                ),
            ),
            QueryEngineTool.from_defaults(
                query_engine=web_engine,
                description=(
                    "用于查询最新的网络信息，适用于实时数据、新闻、天气、"
                    "股票价格等需要最新信息的查询。当问题涉及到可能变化的"
                    "外部信息或系统内部没有的信息时使用。"
                ),
            ),
            QueryEngineTool.from_defaults(
                query_engine=direct,
                description=(
                    "用于直接回答的检索增强生成引擎，适用于一般性知识、"
                    "常识问题、概念解释等不需要特定查询的问题。当问题是"
                    "通用知识且不需要最新信息时使用。"
                ),
            ),
        ]
        # 创建路由选择器
        selector = LLMSingleSelector.from_defaults()
        # 构建路由查询引擎
        router = RouterQueryEngine(
            selector=selector,
            query_engine_tools=query_engine_tools,
            select_multi=False,
        )

        return router


    def classifier_query(self, query_str):
        """直接对查询进行分类，返回分类结果"""
        prompt = f"""
        请分析以下用户查询，并判断应该使用哪种信息源来回答：

        查询："{query_str}"

        请从以下三种类型中选择一种最合适的：
        1. DATABASE: 使用数据库查询 
        - 适用于查询系统内已存储的信息和历史数据
        - 特别是涉及用户个人信息（如账户、设置、偏好）
        - 用户的历史行为记录（如睡眠时长、活动记录、使用历史、建议等）
        - 包含个人指标的查询（如"我的"、"我昨天"、"我上周"等）
        - 系统内的具体记录（如床位信息、设备状态）
        - 任何需要访问用户专属数据的查询

        2. WEB: 使用联网搜索
        - 适用于需要最新外部信息的查询
        - 新闻、天气、市场动态等实时信息
        - 系统数据库中没有存储的通用知识
        - 外部事实、事件、人物相关的查询
        - 包含"最新"、"现在"等时效性词语的查询
        
        3. DIRECT: 直接回答
        - 适用于通用常识性问题
        - 不涉及个人数据或外部最新信息的概念解释
        - 简单的计算或推理
        - 不需要查询具体数据的建议或观点

        重要：如果查询包含"我的"、"我"加时间词（如"我昨晚"、"我上周"）等表示个人数据的词语，几乎总是应该选择DATABASE。

        只返回一个结果：DATABASE, WEB, DIRECT
        """
        response = self.llm.complete(prompt)
        result = response.text.strip().upper()
        filtered_output = re.sub(r'<THINK>.*?</THINK>', '', result, flags=re.DOTALL).strip()
        if "DATABASE" in filtered_output:
            return "DATABASE"
        elif "WEB" in filtered_output:
            return "WEB"
        else:
            return "DIRECT"
