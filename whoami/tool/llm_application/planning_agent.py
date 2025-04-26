
from pathlib import Path
from pydantic import BaseModel, Field
from typing import (
    List,
    Optional,
    Dict,
    Any
)
import asyncio
import json
import datetime
from datetime import datetime, timedelta
import inspect

from whoami.tool.llm_application.information_extract_json import InformationExtractJson
from whoami.utils.log import Logger
from whoami.tool.agent.base_tool import BaseTool 
from whoami.tool.llm_application.enhance_retrieval import EnhanceRetrieval
from whoami.configs.llm_config import LLMConfig
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.tool.search.google_search_provider import GoogleSearchProvider
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sleep_indices import SleepIndices




class SqlData:
    sql_result: Optional[List[Dict[str, Any]]] = None
    sql_provider: Optional[SqlProvider] = None
    sql_config_path: str = "/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
    def __init__(self):
        try:
            self.sql_provider = SqlProvider(model=SleepIndices, sql_config_path=self.sql_config_path)
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            result = self.sql_provider.get_record_by_condition(
                condition={},
                exclude_fields=[
                    'health_advice',
                    'sleep_stage_image_x_y',
                    'body_move_image_x_y',
                    'breath_exception_image_sixty_x_y',
                    'heart_bpm_image_x_y',
                    'breath_bpm_image_x_y',
                    'breath_exception_image_x_y',
                    'deep_sleep_second',
                    'total_num_second',
                    'total_num_second_on_bed',
                    'sleep_second',
                    'deep_sleep_second',
                    'waking_second',
                    'to_sleep_second',
                    'leave_bed_total_second',
                    'save_file_path',
                    'creator',
                    'create_time',
                    'updater',
                    'update_time',
                    'deleted',
                    'tenant_id',
                    'id'
                ],
                date_range={"date_field": "query_date", "start_date": start_date, "end_date": end_date}
            )
            field_names_and_descriptions = self.sql_provider.get_field_names_and_descriptions()
            result_list = [self.convert_keys_to_chinese(field_names_and_descriptions, item) for item in result]
            self.sql_result = self.group_by_device_sn(result_list)
            
        except Exception as e:
            raise ValueError(f"Fail to get sql data! {str(e)}") from e

    @property
    def sql_data(self):
        return self.sql_result
        
    def group_by_device_sn(self, data_list):
        result = {}
    
        for item in data_list:
            device_sn = item.get('设备编号')
            if device_sn:
                if device_sn not in result:
                    result[device_sn] = []
                result[device_sn].append(item)
        
        return result

    def convert_keys_to_chinese(self, field_to_chinese_dict, data_dict):
            """
            将数据字典的键从字段名转换为中文名
            
            参数:
            field_to_chinese_dict (dict): 字段名到中文名的映射字典
            data_dict (dict): 需要转换的数据字典，键为字段名，值为字段值
            
            返回:
            dict: 转换后的字典，键为中文名，值为字段值
            """
            # 创建新字典以存储转换后的结果
            converted_dict = {}
            
            # 遍历原始数据字典
            for field, value in data_dict.items():
                # 如果字段在映射字典中存在，使用中文名作为新键
                if field in field_to_chinese_dict:
                    chinese_key = field_to_chinese_dict[field]
                    converted_dict[chinese_key] = value
                else:
                    # 如果字段在映射字典中不存在，保留原始字段名
                    converted_dict[field] = value
            
            # 清空原始字典并用新值更新它
            data_dict.clear()
            data_dict.update(converted_dict)
            
            return data_dict


print("started .................................................... ")
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


class DirectLLMTool(BaseTool):
    """专门用于回答与舜熙科技及其产品相关的所有内容，作为默认的首选工具。当问题涉及公司信息、产品功能、用户指南、操作方法、技术支持、常见问题、使用教程、产品优势、价格政策等任何与舜熙科技及其产品相关的信息时，应优先使用此工具。此工具拥有最全面、最准确的舜熙科技专有知识库。"""
    name: str = "DirectLLMTool"
    description: str = "专门用于回答与舜熙科技及其产品相关的所有内容，作为默认的首选工具。当问题涉及公司信息、产品功能、用户指南、操作方法、技术支持、常见问题、使用教程、产品优势、价格政策等任何与舜熙科技及其产品相关的信息时，应优先使用此工具。此工具拥有最全面、最准确的舜熙科技专有知识库。"
    args_schema: BaseModel = DirectLLMSchema
    end_flag: int = 1
    enhance_llm: Optional[EnhanceRetrieval] = None
    
    async def execute(self, question: str, message_history: List[Dict[str, any]] = None) -> float:
        response = ""
        async for chunk in self.enhance_llm._run(
            text_list=[],
            query=question,
            message_history=[],
            retrieval_flag=False,
            stream_flag=0
        ):
            response = chunk
        return response
    
    
class GoogleSearchTool(BaseTool):
    """谷歌检索工具，专门用于查询与舜熙科技无关的外部信息。当用户询问的问题涉及时事新闻、政策法规、行业趋势、市场数据、公众人物、科学知识、全球事件等需要最新外部信息的内容时，应使用此工具。此工具不应用于查询舜熙科技及其产品的相关信息。"""
    name: str = "GoogleSearchTool"
    description: str = "谷歌检索工具，专门用于查询与舜熙科技无关的外部信息。当用户询问的问题涉及时事新闻、政策法规、行业趋势、市场数据、公众人物、科学知识、全球事件等需要最新外部信息的内容时，应使用此工具。此工具不应用于查询舜熙科技及其产品的相关信息。"
    args_schema: BaseModel = GoogleSearchSchema
    end_flag: int = 0
    google_search: Optional[GoogleSearchProvider] = None # 自定义新的属性一定要在构造函数中初始化，否则会出现深拷贝错误
    enhance_llm: Optional[EnhanceRetrieval] = None
    def __init__(self, google_search: Optional[GoogleSearchProvider] = None, enhance_llm: Optional[EnhanceRetrieval] = None):
        # 先初始化google_search
        if google_search is None:
            google_search = GoogleSearchProvider(snippet_flag=0, 
                                                search_config_path='/work/ai/WHOAMI/whoami/scripts/test/search_config.yaml', 
                                                query_num=5)
        super().__init__(google_search=google_search, enhance_llm=enhance_llm)

    async def execute(self, google_query: str) -> float:
        param = {
            "query": google_query
        }
        status, result = self.google_search(**param)
        self.logger.info(result)
        text_list = [{item["link"]: item.get("fetch_url_content", item["html_snippet"])} for item in result]
        retrieval_nodes = self.enhance_llm.retrieve(
            text_list=text_list, 
            top_k=2, 
            query=google_query
        ) if text_list else []
        context_texts = [node.node.text for node in retrieval_nodes]
        # self.logger.info("context_texts: -------------------------- {context_texts}")
        context = "没有检索到任何信息！" if not context_texts else "\n\n".join(context_texts)
        return context


class HealthReportTool(BaseTool):
    """
    专用于睡眠健康报告分析和解读的工具。当用户询问有关其睡眠数据、睡眠质量评估、睡眠趋势分析、睡眠建议或任何与睡眠监测报告相关的问题时，应使用此工具。此工具能够访问用户的睡眠监测历史数据，提供专业的睡眠健康分析和个性化建议。工具将分析包括但不限于睡眠时长、睡眠效率、睡眠评分、睡眠状态、生理指标、体动数据、呼吸异常等睡眠健康指标。
    """
    name: str = "HealthReportTool"
    description: str = """
    专用于睡眠报告相关的回答：
    睡眠报告包含以下指标：
    - 睡眠时长（总监测时长、总在床时长、睡眠时长、深度睡眠时长等）
    - 睡眠效率（睡眠效率、深睡效率）
    - 睡眠评分（评分、评分归类、超越人数百分比）
    - 睡眠状态（上床时间、入睡时间、醒来时间等）
    - 生理指标（心率、呼吸率及其状态）
    - 体动数据（体动次数、体动指数等）
    - 呼吸异常（次数、指数、典型事件）
    """
    args_schema: BaseModel = HealthReportSchema
    end_flag: int = 1
    sql_data: SqlData = SqlData().sql_data
    system_prompt: str = """
    你是一名专业的睡眠健康分析师，擅长解读睡眠监测数据。
    【重要规则】：
    
    简明扼要地回答用户问题，控制回复总长度在150字以内。

    确认系统中实际包含的数据日期范围。
    使用系统时间或最新数据日期作为参考点处理相对时间表述。
    只使用系统中实际存在的数据字段和数值。
    优先展示最重要的指标和明显异常。
    针对问题直接给出简短的评估和建议。

    回复要简洁专业，突出重点，避免冗长描述。

    记住：你的分析必须基于系统提供的实际数据，回答必须控制在150字以内。
    """
    device_sn: Optional[str] = None
    enhance_llm: Optional[EnhanceRetrieval] = None
    
    async def execute(self, health_report_question: str) -> str:
        # sql_data_ = self.sql_data["13D6F349200080712111957107"]
        sql_data_ = self.sql_data[self.device_sn]
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
            stream_flag=0
        ):
            full_response += chunk
        return full_response


class PlanningAgent:
    """一个可以完成简单问题的react
    已实现：
    一个固定的react提示词
    非固定的可调用工具
    对工具域的数量有上限
    尝试3次调用，每次调用式while循环但是有限制（目前是同样的工具成功或者失败两次以上直接返回或者退出while循环）
    后续需要优化为调用策略
    策略1：while循环没有退出机制
    策略2：while循环的退出机制为：相同的工具连续两次失败（break）或成功（直接返回）（现采纳）
    策略3：while循环的退出机制为：相同的工具连续两次调用失败（break）或成功（直接返回），连续三次调用（最后一次成功则返回，失败则break）
    策略4：while循环的退出机制为：针对策略3放宽条件，不必要求连续的工具必须相同，不相同也使用策略3。

    后续精炼planning agnet为react架构，每一个工具都可以选择称谓react架构。。。
    """
    def __init__(self, tools: list, llm: OllamaLLM) -> None:
        self.tools = tools
        self.tool_names = ', '.join([tool.name for tool in self.tools])
        self.tool_descs =self._init_descs()
        self.logger = Logger('PlanningAgent')
        self.llm: OllamaLLM = llm

        # 提示词中，注意每个工具的准确描述对于工具调用正确率很重要，但是首先在整个工具域层面去宏观制定工具调用规则更重要
        self.prompt_tpl = """
        Today is {today} {weekday}. 位置：山西运城， Please Answer the following questions as best you can. You have access to the following tools:
        {tool_description}
        系统调用逻辑：
        1. 优先级规则：当用户问题涉及个人健康数据（包括但不限于睡眠、心率、呼吸、体动等指标）时，必须优先调用HealthReport，即使问题表述简短或模糊。
        2. 当问题明确寻求最新外部信息且与舜熙科技及个人健康数据无关时，调用GoogleSearch。
        3. 其他所有情况（包括舜熙科技信息、产品功能、系统身份等）使用DirectLLM作为兜底工具。
        
        工具使用指导：
        1. 效率原则：每个问题尽量只调用一次工具。一旦获得足够回答用户问题的信息，立即提供Final Answer，不要进行重复或冗余的工具调用，除非第一次调用工具失败。
        2. 信息完整性评估：收到工具返回的Observation后，立即评估信息是否完整、准确且直接回答了用户问题。如信息完整，直接提供Final Answer；只有在信息明显不足时才考虑再次调用工具。
        3. 避免重复调用：不要使用相同的参数重复调用同一工具。如需更多信息，应调整参数或尝试其他工具。
        
        
        These are chat history before:
        {chat_history}
        
        参数规范：
        1. DirectLLM工具：使用参数 {{"question": "用户完整问题"}}
        2. GoogleSearch工具：使用参数 {{"google_query": "搜索关键词"}}
        3. HealthReport工具：使用参数 {{"health_report_question": "健康相关完整问题"}}
        4. 必须严格按照以上参数名称和格式构造参数，不得使用其他参数名称或格式。
        5. 参数值应包含用户的完整问题，若涉及时间段，直接包含在问题中，而非作为单独参数。
        
        格式要求：
        1. 严格遵循下方格式模板进行输出，保持格式的一致性与简洁性。
        2. Action行只能包含工具名称，不要添加任何解释、注释或额外信息。
        3. 每个部分必须严格按照指定格式填写，避免不必要的冗余信息。
        4. 一旦从工具获得足够信息，立即提供Final Answer，不要进行重复调用。

        Use the following format:
        - Question: 用户的问题
        - Thought: 详细分析当前问题以及历史对话上下文，判断问题是否与舜熙科技或其产品相关。考虑对话历史中提到的实体和信息，进行完整的问题理解。
        - Action: 选择的工具名称 [{tool_names}]
        - Action Input: {{"参数名": "参数值"}} - 完整的JSON格式
        - Observation: 工具返回的结果
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        - Thought: 基于返回结果进行分析
        - Final Answer: 最终答案，应直接回答用户问题


        Question: {query}
        {agent_scratchpad}
        """
        # 重要规则：
        # 1. 所有涉及"舜熙科技"的问题必须使用DirectLLMTool，无一例外。这包括但不限于：
        # - 公司基本信息（地址、电话、成立时间等）
        # - 产品及服务信息
        # - 官网及联系方式
        # - 使用说明和操作指南
        # - 任何与舜熙科技相关的知识
        # 2. 只有完全与舜熙科技无关的问题才能使用GoogleSearchTool
        # 3. 当用户使用"它"、"它们"、"该公司"、"你"、"你们"等指代词时，应该从上下文会话中找到该指代词的实际内容
        #     错误示例: {{"question": "它们公司地址是多少"}}
        #     正确示例: {{"question": "舜熙科技的地址是多少"}}
        # 4. 即使不确定是否有答案，也必须先尝试使用DirectLLMTool而非GoogleSearchTool
      
        
    def _init_descs(self):
        """初始化工具描述信息

        Returns:
            _type_: _description_
        """
        # tool_descs = [str(t.tool_schema) for t in self.tools]
        tool_descs = [t.get_simple_tool_description() for t in self.tools]
        tool_descs = '\n\n'.join(tool_descs)
        return tool_descs


    async def agent_execute(self, query, chat_history=[]):
        global tools, tool_names, tool_descs, prompt_tpl, llm, tokenizer

        agent_scratchpad = ''  # agent执行过程
        
        # Add counter dictionaries to track tool calls
        tool_success_counter = {}  # Format: {tool_name: count}
        tool_error_counter = {}  # Format: {tool_name: count}
        last_tool = None  # Track the last used tool
        
        while True:
            # 1 格式化提示词并输入大语言模型
            history = '\n'.join(['Question:%s\nAnswer:%s' % (his[0], his[1]) for his in chat_history])
            # 兼容qwen2.5和其他模型
            model_name = 'qwen2.5'
            history = ';'.join(['Question:%s;Answer:%s' % (his[0], his[1]) for his in chat_history])
            
            today = datetime.now().strftime('%Y-%m-%d')
            weekday_num = datetime.now().weekday()

            # 中文星期名称列表，Monday对应“星期一”
            weekday_cn = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
            weekday = weekday_cn[weekday_num]
            prompt = self.prompt_tpl.format(today=today, weekday=weekday, chat_history=history, tool_description=self.tool_descs, tool_names=self.tool_names,
                                    query=query, agent_scratchpad=agent_scratchpad)
            self.logger.info(f"---等待LLM返回... ...\n{prompt}")
            user_stop_words = ['Observation:'] if model_name == 'qwen2' else ['- Observation:']
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm._whoami_text(messages=messages, timeout=30, user_stop_words=user_stop_words)
            self.logger.info(f"---LLM返回... ...\n{response}")

            # 2 解析 thought+action+action input+observation or thought+final answer
            thought_i_str = 'Thought:' if model_name == 'qwen2' else '- Thought:'
            final_answer_i_str = '\nFinal Answer:' if model_name == 'qwen2' else '\n- Final Answer:'
            action_i_str = '\nAction:' if model_name == 'qwen2' else '\n- Action:'
            action_input_i_str = '\nAction Input:' if model_name == 'qwen2' else '\n- Action Input:'
            observation_i_str = '\nObservation:' if model_name == 'qwen2' else '\nObservation:'
            
            thought_i = response.rfind(thought_i_str)
            final_answer_i = response.rfind(final_answer_i_str)
            action_i = response.rfind(action_i_str)
            action_input_i = response.rfind(action_input_i_str)
            observation_i = response.rfind(observation_i_str)
            self.logger.info(f"=============工具调用提取的参数位置信息============={thought_i, action_i, action_input_i, observation_i}")
            
            # 3 返回final answer，执行完成
            if final_answer_i != -1 and thought_i < final_answer_i:
                final_answer = response[final_answer_i + len(final_answer_i_str):].strip()
                chat_history.append((query, final_answer))
                return True, final_answer, chat_history

            # 4 解析action
            if not (thought_i < action_i < action_input_i):
                return False, 'LLM回复格式异常', chat_history
            if observation_i == -1:
                observation_i = len(response)
                response = response + '\nObservation: '
            thought = response[thought_i + len(thought_i_str):action_i].strip()
            action = response[action_i + len(action_i_str):action_input_i].strip()
            action_input = response[action_input_i + len(action_input_i_str):observation_i].strip()
            self.logger.info(f"=============工具调用提取的参数信息============={action, action_input}")
            # 5 匹配tool
            the_tool = None
            for t in self.tools:
                # if t.name == action: # 使用更加严格的工具匹配
                if t.name in action:
                    the_tool = t
                    break
            if the_tool is None:
                observation = 'the tool not exist'
                agent_scratchpad = agent_scratchpad + response + observation + '\n'
                
                # Reset counters when tool changes
                last_tool = None
                continue

            # {"url": "http://localhost:8000/user/", "filed_value": '{"realname":"李四"}'}

            # Initialize counters for this tool if not exist
            tool_name = the_tool.name
            if tool_name not in tool_success_counter:
                tool_success_counter[tool_name] = 0
            if tool_name not in tool_error_counter:
                tool_error_counter[tool_name] = 0

            # If tool changed, reset consecutive error counter
            if last_tool != tool_name:
                tool_error_counter[tool_name] = 0
                last_tool = tool_name


            # 6 执行tool
            try:
                # 注意上一步工具的输出结果最好不要有嵌套json，否则解析会出错
                # 因为大语言模型对嵌套json字符串的返回不是转义格式，这不符合python中的json工具对json字符串的解析要求
                action_input = json.loads(action_input)
                
                signature = inspect.signature(the_tool.execute)
                if "message_history" in signature.parameters or any(
                    param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL) 
                    for param in signature.parameters.values()
                ):
                    action_input["message_history"] = chat_history
                
                self.logger.info(f"---action_input结果... ...\n{action_input}")
                tool_ret = await the_tool.execute(**action_input)
                self.logger.info(f"---执行tool结果... ...\n{tool_ret}")
                
                # 如果the_tool是终止tool，直接返回并结束当前agent
                if the_tool.end_flag == 1:
                    chat_history.append((query, tool_ret))
                    return True, tool_ret, chat_history
                
                # Tool executed successfully
                tool_success_counter[tool_name] += 1
                tool_error_counter[tool_name] = 0  # Reset error counter
                observation = str(tool_ret)
                
                # If same tool called successfully 3 or more times, generate final answer
                if tool_success_counter[tool_name] >= 2:
                    # Add final observation to scratchpad
                    agent_scratchpad = agent_scratchpad + response + observation + '\n'
                    
                    # Generate final answer from LLM
                    # 这里要根据end_flag参数去决定是否直接返回最后一步的执行结果
                    final_answer = response
                    if the_tool.end_flag == 0:
                        final_prompt = self.prompt_tpl.format(
                            today=today, 
                            chat_history=history, 
                            tool_description=self.tool_descs, 
                            tool_names=self.tool_names,
                            query=query, 
                            agent_scratchpad=agent_scratchpad + "\n- Thought: I've collected sufficient information after multiple tool calls. Let me provide a final answer.\n"
                        )
                    
                        final_messages = [{"role": "user", "content": final_prompt}]
                        final_response = await self.llm._whoami_text(messages=final_messages, timeout=30, user_stop_words=[])
                        
                        # Extract final answer
                        final_answer_i = final_response.rfind(final_answer_i_str)
                        if final_answer_i != -1:
                            final_answer = final_response[final_answer_i + len(final_answer_i_str):].strip()
                        else:
                            # If no Final Answer format is found, use the whole response
                            self.logger.info(f"final_messages: ---------------------- {final_messages}")
                            self.logger.info(f"final_response: ---------------------- {final_response}")
                            final_answer = final_response.strip().replace("Final Answer:", "")
                    
                    chat_history.append((query, final_answer))
                    return True, final_answer, chat_history
                
            except Exception as e:
                observation = 'the tool has error:{}'.format(e)
                tool_error_counter[tool_name] += 1
                tool_success_counter[tool_name] = 0  # Reset success counter
                
                # If same tool failed 3 or more times in a row, break out
                if tool_error_counter[tool_name] >= 3:
                    error_message = f"工具 {tool_name} 连续调用失败超过限制次数，无法完成请求。请尝试其他方式提问或联系客服。"
                    chat_history.append((query, error_message))
                    return False, error_message, chat_history
            # except Exception as e:
            #     observation = 'the tool has error:{}'.format(e)
            # else:
            #     observation = str(tool_ret)
            agent_scratchpad = agent_scratchpad + response + observation + '\n'


    async def agent_execute_with_retry_async(self, query, chat_history=[], retry_times=3):
        for i in range(retry_times):
            status, result, chat_history = await self.agent_execute(query, chat_history=chat_history)
            if status:
                return status, result, chat_history
        return status, result, chat_history

        
    def agent_execute_with_retry(self, query, chat_history=[], retry_times=3):
        """
        Synchronous wrapper for the async agent_execute method
        """
        # 检查是否在事件循环中
        if asyncio.get_event_loop().is_running():
            # 如果已在事件循环中，直接异步调用但不等待结果
            # 这将导致警告，应该在调用处使用 await
            return self.agent_execute_with_retry_async(query, chat_history, retry_times)
        else:
            # 如果不在事件循环中，创建新的事件循环运行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.agent_execute_with_retry_async(query, chat_history, retry_times)
                )
            finally:
                loop.close()


# import asyncio

# async def test_direct_llm():
#     # 初始化 DirectLLM 实例
#     direct_llm_tool = DirectLLMTool()
#     google_search_tool = GoogleSearchTool()
#     health_report_tool = HealthReportTool(device_sn='13D6F349200080712111957107')
    
#     # 定义测试问题
#     question = "最近一周的睡眠报告？"
#     chat_history = [
#         ["舜熙科技的主要产品是什么？", "不知道"], 
#         ["它们公司官网是多少", "舜熙科技的官网是 https://shunxikj.com/ ，您可以了解产品详情、解决方案和成功案例。"],
#         ["它们公司地址是多少", "舜熙科技的总部位于山西省运城市盐湖区黄河金三角科创城C1栋。"],
#         ["你们公司的核心算法有哪些？", "舜熙科技的核心算法主要包括：\n1. 跌倒检测算法：能准确识别老人的异常姿态；\n2. 行为模式学习算法：系统会“学习”老人的日常活动规律，在偏离正常模式时及时预警；\n3. 语音语义理解算法：精准识别老人口音和方言；\n4. 健康状态异常分析算法：结合生理数据和行为数据判断是否出现疾病预警。所有算法都会随着使用时间增长而不断优化。"],
#         ["我昨晚睡的怎么样？", """根据您的睡眠监测数据显示，您昨晚（2025年4月1日）的睡眠情况如下：
#   - 上床时间：2025年3月31日晚上7点0分
#   - 入睡时间：2025年3月31日晚上7点9分57秒
#   - 醒来时间：2025年4月1日早上5点54分40秒

#   其他相关信息：
#   - 睡眠时长为8小时25分钟
#   - 深度睡眠时间为2小时7分钟，占总睡眠时间的25%
#   - 浅度睡眠时间为6小时17分钟，占总睡眠时间的77.39%
#   - 夜间醒来次数为3次
#   - 睡眠效率为0.72（即72%）
#   - 体动指数为4.63

#   总体评分：57.97，属于较差水平。

#   基于以上数据，建议您注意改善睡眠质量。如有需要可以咨询医生或专业人士。"""]
#     ]
#     planning_agent = PlanningAgent(tools=[direct_llm_tool, google_search_tool, health_report_tool])
#     print(planning_agent.tool_descs)
#     print(planning_agent.tool_names)
    
    
#     status, result, chat_history = await planning_agent.agent_execute_with_retry(question, chat_history=chat_history)
#     print(f"问题: {question}")
#     print(f"回答: {result}")
    
#     return result

# 运行测试函数
if __name__ == "__main__":
    asyncio.run(test_direct_llm())

    

