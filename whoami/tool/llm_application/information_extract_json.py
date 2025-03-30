
from typing import (
    Dict,
    List,
    Any,
    Union,
    Optional
)
from pathlib import Path
from datetime import datetime, timedelta
import copy


from whoami.tool.llm_application.base_json_processor import BaseJsonProcessor
from whoami.tool.llm_application.prompt_config import PromptTemplatesConfig
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig
from whoami.tool.llm_application.classify_query_intent_template import ClassifyQueryIntentTemplate



llm=OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml')))
llm_qwen_=OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))
class InformationExtractJson(BaseJsonProcessor):
    
    prompt_templates: Optional[Dict] = None
    default_results: Optional[Dict] = None
    classify_query_intent_template: ClassifyQueryIntentTemplate = None
    def __init__(self, config_path: str=None):
        
        super().__init__()
        self.prompt_templates = {}
        self.default_results = {}
        self._llm = None
        self._llm_qwen = None
        self.classify_query_intent_template: ClassifyQueryIntentTemplate = ClassifyQueryIntentTemplate()
        # 加载提示词模板配置
        if config_path:
            self.load_templates_from_config(config_path)
        self.customer_templates()
    @property
    def llm(self):
        self._llm = llm if self._llm is None else self._llm
        return self._llm
    
    @property
    def llm_qwen(self):
        self._llm_qwen = llm_qwen_ if self._llm_qwen is None else self._llm_qwen
        return self._llm_qwen
    
    
    def customer_templates(self):
        
        # intent_health_report start
        self.prompt_templates["intent_health_report"] = self.classify_query_intent_template._run(
            classify_topic="睡眠报告",
            classify_topic_info="""
            - 睡眠时长：总监测时长、总在床时长、睡眠时长、深度睡眠时长、浅睡时长、清醒时长、入睡时长、离床时间
            - 睡眠效率：睡眠效率、深睡效率
            - 睡眠评分：评分、评分归类、超越人数百分比
            - 睡眠状态：上床时间、入睡时间、醒来时间、离床时间、离床次数、夜醒次数
            - 生理指标：平均心率、最大心率、最小心率、平均呼吸率、最大呼吸率、最小呼吸率、心率状态、呼吸率状态
            - 体动数据：体动次数、体动指数、平均体动次数、最大体动次数、最小体动次数、体动状态
            - 呼吸异常：呼吸异常次数、呼吸异常指数、典型呼吸异常事件
            - 图表数据：睡眠阶段划分图、心率折线图、体动图、呼吸异常图、呼吸率图、心率图
            - 健康建议：健康建议内容
            """,
            classify_topic_info_standard="""
            1. 相关（is_report_related=true）：
            - 查询明确要获取睡眠报告中的具体数据或指标值（如"昨晚深睡时长是多少"）
            - 查询具体询问用户的睡眠数据表现（如"用户睡眠质量怎么样"）
            - 查询包含对睡眠报告数据的统计或分析（如"上周平均心率是多少"）
            - 查询特定设备或用户的睡眠相关指标数据
            
            2. 不相关（is_report_related=false）：
            - 查询是关于如何操作界面、如何查看界面、如何使用功能的问题（如"如何查看睡眠报告"、"在哪个页面可以看到心率"）
            - 查询是关于睡眠报告功能、界面等的问题，而非数据本身
            - 查询是关于系统或设备管理的问题，且不涉及获取任何睡眠数据（如"如何连接设备"）
            """,
            sleep_keywords = ["睡眠", "心率", "呼吸", "体动", "深睡", "浅睡", "夜醒", "入睡", "报告", 
                        "效率", "评分", "清醒", "床", "监测", "异常"]
        )

        self.default_results["intent_health_report"] = {
                "is_related": False,
                "confidence": 0.5,
                "reasoning": "默认值"
        }
        # intent_health_report end
    
    
    def load_templates_from_config(self, config_path: Union[str, Path, Dict]):
        """
        从配置文件加载提示词模板和默认结果
        
        参数:
            config_path: 配置文件路径(字符串或Path对象)或已加载的配置字典
        """
        try:
            config = PromptTemplatesConfig(config_path)
            
            # 加载所有任务的提示词和默认结果
            for task_name in config.get_task_names():
                prompt = config.get_prompt(task_name)
                default_result = config.get_default_result(task_name)
                
                if prompt:
                    self.prompt_templates[task_name] = prompt
                if default_result:
                    self.default_results[task_name] = default_result
                    
        except Exception as e:
            error_info = f"加载提示词模板配置失败: {str(e)}"
            self.logger.error(error_info)
            raise ValueError(error_info) from e
    
    
    def add_prompt_template(self, task_name: str, template: str) -> None:
        """
        添加提示词模板
        
        参数:
            task_name: 任务名称，用于后续引用此模板
            template: 提示词模板字符串
        """
        self.prompt_templates[task_name] = template
        self.logger.debug(f"已添加任务 '{task_name}' 的提示词模板")
    
    
    def add_default_result(self, task_name: str, default_result: Dict) -> None:
        """
        添加任务的默认结果
        
        参数:
            task_name: 任务名称
            default_result: 默认结果字典
        """
        self.default_results[task_name] = default_result
        
        
    def _get_default_result(self, task_name: str) -> Dict:
        """
        获取任务的默认结果
        
        参数:
            task_name: 任务名称
            
        返回:
            默认结果字典
        """
        return self.default_results.get(task_name, {})
    
    def _prepare_prompt(self, template, query, context=None):
        """准备完整的提示词
    
        参数:
            template: 提示词模板
            query: 用户查询
            context: 额外上下文
            
        返回:
            完整的提示词字符串
        """
        context_str = ""
        if context:
            for key, value in context.items():
                context_str += f"{key}: {value}\n"
        
        return f"{template}\n\n{context_str}\n查询：\"{query}\"\n返回："
    
    async def extract_info(self, 
        query: str, 
        task_name: str,
        message_history: List[Dict[str, str]] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        从查询中提取信息
        
        参数:
            query: 用户查询
            task_name: 任务名称
            context: 额外上下文
            temperature: LLM温度参数
            
        返回:
            提取的信息，JSON格式
        """
        if task_name not in self.prompt_templates:
            raise ValueError(f"未找到任务 '{task_name}' 的提示词模板")
        
        if message_history is None and task_name == "intent_database":
            # 如果没有历史会话消息，默认需要检索
            return self._get_default_result("intent_database")
        
        template = self.prompt_templates[task_name]
        # Notice extract_info not influence message_history.
        # 系统提示词因为太多，放在最前面，不要影响重要信息
        # namespace_message_history = [{"role": "system", "content": template}]
        # namespace_message_history.extend(message_history)
        # namespace_message_history.append({"role": "user", "content": query})
        message = template + f"\n\n历史会话消息：{message_history}" + f"\n\n当前问题：{query}"
        namespace_message_history = [{"role": "user", "content": message}]
        self.logger.info(f"信息提取任务：{task_name} ---------------------------------- {namespace_message_history}")
        try:
            # 调用LLM生成回答
            response = await self.llm_qwen._whoami_text(namespace_message_history, timeout=30, user_stop_words=[])
            # 使用基类的JSON解析方法处理回答
            result = self.parse_json_response(response, self._get_default_result(task_name))
            return result
            
        except Exception as e:
            self.logger.error(f"提取信息时出错: {str(e)}")
            return self._get_default_result(task_name)
    
    
    # 便捷方法
    async def extract_name_id(self, query, message_history=None):
        """提取名字或ID信息"""
        return await self.extract_info(query, "name_id", message_history)
    
    
    async def extract_time_info(self, query, message_history=None):
        """提取时间信息"""
        current_time = datetime.now()
        current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        namespace_message_history = copy.deepcopy(message_history)
        namespace_message_history.append({"role": "system", "content": f"当前时间：{current_time}"})
        
        return await self.extract_info(query, "time", namespace_message_history)
    
    
    async def query_rewrite(self, query, message_history=None):
        """重写query"""
        # 预处理message_history，仅需要角色为user的数据去分析用户的意图
        current_time = datetime.now()
        current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        user_message_history = [{"role": "system", "content": f"当前系统时间：{current_time}"}]
        self.logger.info(f"message_history: ------------------------------------- {message_history}")
        user_message_history.extend([message for message in message_history if message["role"] == "user"])
    
        return await self.extract_info(query, "query_rewrite", message_history=user_message_history)
    
    async def analyze_intent_health_report(self, query, message_history=None):
        """分析查询意图"""
        # 预处理message_history，仅需要角色为user的数据去分析用户的意图
        return await self.extract_info(query, "intent_health_report", message_history=message_history)

    async def analyze_intent_database(self, query, message_history=None):
        """根据历史会话消息分析是否需要进行数据库检索"""
        return await self.extract_info(query, "intent_database", message_history)

    async def analyze_intent_database_combine(self, query, message_history=None):
        """判断是否需要数据库检索（包含睡眠报告相关意图识别）"""
        current_time = datetime.now()
        current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        user_message_history = [{"role": "system", "content": f"当前系统时间：{current_time}"}]
        self.logger.info(f"message_history: ------------------------------------- {message_history}")
        user_message_history.extend([message for message in message_history if message["role"] == "user"])
        return await self.extract_info(query, "intent_database_combine", message_history=user_message_history)


    def _run(self, *args, **kwargs):
        pass

    