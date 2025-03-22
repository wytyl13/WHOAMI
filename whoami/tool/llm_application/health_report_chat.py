from typing import (
    Optional,
    List,
    Dict
)
from pathlib import Path
import asyncio
import re
import copy
import json
import datetime


from whoami.tool.base.base_tool import BaseTool
from whoami.tool.llm_application.classify_query_intent import ClassifyQueryIntent
from whoami.tool.llm_application.enhance_retrieval import EnhanceRetrieval
from whoami.configs.llm_config import LLMConfig
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.tool.llm_application.information_extract_json import InformationExtractJson
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sleep_indices import SleepIndices


CONFIG_PATH = "/work/ai/WHOAMI/whoami/tool/llm_application/info_extract_prompt.yaml"
SQL_CONFIG_PATH = "/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"


llm=OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml')))
llm_qwen_=OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))
class HealthReportChat(BaseTool):

    query: Optional[str] = None
    full_response: Optional[List[str]] = None
    sql_provider: Optional[SqlProvider] = None
    sql_provider_user_info: Optional[SqlProvider] = None
    
    def __init__(
        self, 
        query: Optional[str] = None,
        sql_provider: Optional[SqlProvider] = None,
        sql_provider_user_info: Optional[SqlProvider] = None
    ):
        super().__init__()
        self.query = query
        self._classify_query_intent = None
        self._enhance_retrieval = None
        self._enhance_retrieval_qwen = None
        self._llm = None
        self._llm_qwen = None
        self._information_extract_json = None
        self.sql_provider = sql_provider if sql_provider is not None else self.sql_provider
        self.sql_provider_user_info = sql_provider_user_info if sql_provider_user_info is None else self.sql_provider_user_info
        self.full_response = []
        
        if self.sql_provider is None:
            self.sql_provider = SqlProvider(model=SleepIndices, sql_config_path=SQL_CONFIG_PATH)
            
        if self.sql_provider_user_info is None:
            self.sql_provider_user_info = SqlProvider(model=SleepIndices, sql_config_path=SQL_CONFIG_PATH)
        
    
    @property
    def llm(self):
        # 只在第一次访问时创建LLM
        self._llm = llm if self._llm is None else self._llm
        return self._llm
    
    @property
    def llm_qwen(self):
        # 只在第一次访问时创建LLM
        self._llm_qwen = llm_qwen_ if self._llm_qwen is None else self._llm_qwen
        return self._llm_qwen
    
    
    @property
    def information_extract_json(self):
        self._information_extract_json = InformationExtractJson(config_path=CONFIG_PATH) if self._information_extract_json is None else self._information_extract_json
        return self._information_extract_json
    
    
    
    @property
    def classify_query_intent(self):
        self._classify_query_intent = ClassifyQueryIntent(
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
            1. 如果查询明确提及睡眠报告的任何指标或数据，应判断为相关
            2. 如果查询间接询问用户睡眠质量、睡眠情况等，应判断为相关
            3. 如果查询与睡眠报告完全无关，应判断为不相关
            """,
            sleep_keywords = ["睡眠", "心率", "呼吸", "体动", "深睡", "浅睡", "夜醒", "入睡", "报告", 
                        "效率", "评分", "清醒", "床", "监测", "异常"]
        ) if self._classify_query_intent is None else self._classify_query_intent
        return self._classify_query_intent
    
    
    @property
    def enhance_retrieval(self):
        self._enhance_retrieval = EnhanceRetrieval(llm=self.llm) if self._enhance_retrieval is None else self._enhance_retrieval
        return self._enhance_retrieval
    
    @property
    def enhance_retrieval_qwen(self):
        self._enhance_retrieval_qwen = EnhanceRetrieval(llm=self.llm_qwen) if self._enhance_retrieval_qwen is None else self._enhance_retrieval_qwen
        return self._enhance_retrieval_qwen
    

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


    async def string_to_chat_stream(self, text):
        """将普通字符串转换为模拟LLM流式响应的格式"""
        # 可以根据需要调整分块的大小和策略
        # 这里我们按句子或短语分割
        chunks = re.split(r'(?<=[.!?,:;])\s+', text)
        
        for chunk in chunks:
            if chunk.strip():  # 跳过空块
                yield chunk + " "
                # 模拟网络延迟，使流看起来更自然，实际使用可以删除或调整
                await asyncio.sleep(0.05)

    def convert_dates_to_strings(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_dates_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_dates_to_strings(item) for item in obj]
        elif str(type(obj)) == "<class 'datetime.date'>":
            return obj.isoformat()  # Convert date to ISO format string
        elif str(type(obj)) == "<class 'datetime.datetime'>":
            return obj.isoformat()  # Convert datetime to ISO format string
        else:
            return obj
        

    async def _run(
        self, 
        query: Optional[str] = None,
        message_history: List[Dict[str, str]] = None
    ):
        query = query if query is not None else self.query
        self.full_response = []
        self.logger.info(f"【开始处理】: query: {query}\n\n")
        self.logger.info(f"【正在处理】：开始阶段1 - 重写query： {query}")
        query_rewrite_result = await self.information_extract_json.query_rewrite(query=query, message_history=message_history)
        rewritten_query = query_rewrite_result["rewritten_query"]
        rewritten_query = query if rewritten_query == "" else rewritten_query
        self.logger.info(f"【正在处理】：完成阶段1 - 重写query，重写结果 {rewritten_query}\n\n")
        stage = 2
        # get health_report status
        self.logger.info(f"【正在处理】：开始阶段{stage} - 识别用户意图：是否询问睡眠报告相关内容？\n\n")
        intent_health_report_result = await self.information_extract_json.analyze_intent_health_report(query=rewritten_query, message_history=[])
        status = intent_health_report_result["is_related"]
        self.logger.info(f"【正在处理】：完成阶段{stage} - 识别用户意图结果：是否询问睡眠报告相关内容？{status}\n\n")
        chat_stream = None
        stage += 1
        if not status:
            # Not health_report, call the llm directly.
            self.logger.info(f"【正在处理】：开始阶段{stage} - 无需数据库辅助，直接回答用户！\n\n")
            chat_stream = self.enhance_retrieval._run(
                text_list=[], 
                message_history=message_history, 
                query=query,
                rewritten_query=rewritten_query
            )
                
        else:

            # Yes health_report, enhance retrieval generate the health_report.
            self.logger.info(f"【正在处理】：开始阶段{stage} - 是否需要数据库辅助？\n\n")
            # database_status

            intent_database_result = await self.information_extract_json.analyze_intent_database(query=rewritten_query, message_history=message_history)
            database_status = intent_database_result.get("need_database", True)
            if not database_status:
                self.logger.info(f"【正在处理】：完成阶段{stage} - 不需要数据库辅助，直接回答\n\n")
                
                # 根据历史消息回答重写问题
                chat_stream = self.enhance_retrieval._run(
                    text_list=[], 
                    message_history=message_history, 
                    query=query,
                    rewritten_query=rewritten_query
                )
            else:
                self.logger.info(f"【正在处理】：完成阶段{stage} - 需要数据库辅助，开始检索\n\n")
                stage += 1
                self.logger.info(f"【正在处理】：开始阶段{stage} - 并行提取用户查询信息！\n\n")
                name_id_result = await self.information_extract_json.extract_name_id(query=rewritten_query, message_history=[])
                self.logger.info(name_id_result)
                time_info_result = await self.information_extract_json.extract_time_info(query=rewritten_query, message_history=[])
                self.logger.info(time_info_result)
                name_id = None
                name_type = None
                time_info = None
                if name_id_result["found"]:
                    name_id = name_id_result["value"]
                    name_type = name_id_result["type"]
                else:
                    yield "请告诉我您要查询哪个用户！"
                    return
                    
                if time_info_result["found"]:
                    time_info = time_info_result["time_range"]
                else:
                    yield "请告诉我您要查询的时间范围！"
                    return
                self.logger.info(f"【正在处理】：完成阶段{stage} - 提取到的信息为：name_id: {name_id}, time_info: {time_info}！\n\n")
            
                
                # 检索增强生成
                stage += 1
                self.logger.info(f"【正在处理】：开始阶段{stage} 用户信息数据库检索！\n\n")
                
                # 首先需要根据用户的编号获取用户的所有信息
                # 其次也要根据用户的姓名获取用户的所有信息
                # 最后还要兼容用户输入的模糊信息，比如查询某个房间的所有老人睡眠情况
                # 查询某个床位的老人睡眠情况
                
                device_sn = name_id if name_type == "id" else None
                elderly_name = name_id if name_type == "name" else None
                dept_id = None
                room_id = None
                bed_id = None
                
                try:
                    elderly_info = self.sql_provider.get_device_info(
                        device_sn=device_sn, 
                        elderly_name=elderly_name,
                        dept_id=dept_id,
                        room_id=room_id,
                        bed_id=bed_id
                    )
                    self.logger.info(f"【正在处理】：完成阶段{stage} 用户{elderly_name}信息：{elderly_info}！\n\n")
                    stage += 1
                    if not elderly_info:
                        yield "用户信息获取失败！"
                        return  
                except Exception as e:
                    error_info = f"用户信息获取失败！{str(e)}"
                    self.logger.error(f"【正在处理】：完成阶段{stage}, {error_info}\n\n")
                    yield "用户信息获取失败！"
                    return 
                
                self.logger.info(f"【正在处理】：开始阶段{stage} 用户睡眠数据库检索！\n\n")
                field_descriptions = self.sql_provider.get_field_names_and_descriptions()
                sql_result_list = []
                try:
                    start_date = time_info["start"]
                    end_date = time_info["end"]
                except Exception as e:
                    yield "请提供具体时间！"
                    return
                for item in elderly_info:
                    # 去数据库检索 elderly_info
                    device_sn = item["device_sn"]
                    elderly_name = item["elderly_name"]
                    if device_sn is None or device_sn == "":
                        yield f"抱歉！{elderly_name}没有绑定监测设备！"
                        return
                    
                    try:
                        sql_result = self.sql_provider.get_record_by_condition(
                            condition={"device_sn": device_sn}, 
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
                    except Exception as e:
                        self.logger.error(f"Fail to exec sql check! {str(e)}")
                        yield "数据库查询失败！"
                        return
                    new_sql_result = []
                    for result in sql_result:
                        result = self.convert_keys_to_chinese(field_descriptions, result)
                        new_sql_result.append(result)
                    if not new_sql_result:
                        yield f"抱歉！{elderly_name}: 设备编号{device_sn}没有监测数据！"
                        return
                    sql_result_list.append({"elderly_info": item, "sql_result": new_sql_result})
                self.logger.info(f"【正在处理】：完成阶段{stage} 用户睡眠数据库检索结果：{sql_result_list}\n\n")
                system_prompt = """
                你是一名专业的睡眠健康分析师，擅长解读睡眠监测数据。
                【重要规则】：

                首先必须确认系统中实际包含的数据日期范围，格式如："系统中包含的数据：2025年3月14日至2025年3月20日的记录，共7天。"
                时间参考处理：

                注意历史会话消息中提供的"当前系统时间"作为时间参考点
                将相对时间表述（如"昨天"、"当天"、"最近一周"）基于提供的系统时间计算
                如果未提供系统时间，则基于数据中最新的日期作为"今天"


                时间段处理规则：

                对于"最近X天/周/月"的查询：确认系统中是否有足够的数据
                对于具体日期的查询（如"2025年3月15日"）：确认该日期的数据是否存在
                对于相对表述（如"昨天"、"前天"、"当天"）：基于系统时间或最新数据日期推算
                对于日期范围（如"3月15日至3月18日"）：验证范围内所有日期是否有数据


                数据完整性说明：

                如数据不足或缺失，明确指出："系统中[有/没有]覆盖[请求时间段]的完整数据。实际可用数据为[起始日期]至[结束日期]，共[X]天。以下分析基于这些可用数据。"
                对于部分可用的情况，清楚说明哪些日期有数据，哪些没有


                数据分析原则：

                严格使用系统中实际存在的数据字段和数值
                不创造不存在的指标名称或数值
                保持数据的原始精度
                按时间顺序呈现数据趋势
                提供客观的整体评估和具体可行的建议


                回复格式要求：

                使用简洁专业的语言
                适当使用分段和列表呈现数据
                重点突出异常值和重要变化



                记住：你的分析必须完全基于系统提供的实际数据，不添加不存在的数据，也不使用不存在的字段名称。
                """
                # database_message_history = []
                # if not any(msg.get('role') == 'system' and "你是一名专业的睡眠分析师" in msg.get('content', '') for msg in database_message_history):
                #     database_message_history.append({"role": "system", "content": system_prompt})
                
                text_list = [{f"用户询问的关于{name_id}的数据库检索信息": f"{str(elderly_info)}\n\n"}]
                for item in sql_result_list:
                    item_sql_result = item["sql_result"]
                    elderly_info = item["elderly_info"]
                    elderly_name_i = elderly_info.get("elderly_name", None)
                    device_sn_i = elderly_info.get("device_sn", None)
                    for i in item_sql_result:
                        query_date = i["查询日期"]
                        text_list.append({"报告姓名": elderly_name_i, "报告编号": device_sn_i, "报告日期": str(query_date), "睡眠报告" : i})
                
                text_list = self.convert_dates_to_strings(text_list)
                
                # 仅根据检索消息和重写query回答用户问题
                chat_stream = self.enhance_retrieval_qwen._run(
                    text_list=text_list, 
                    message_history=[], 
                    query=query,
                    rewritten_query=rewritten_query,
                    prompt=system_prompt
                )
            
        if not chat_stream or chat_stream is None:
            self.logger.error("Stream is empty or None!")
            chat_stream = self.string_to_chat_stream("抱歉！您的问题太深奥了！以至于我无法回答！")
        try:
            async for chunk in chat_stream:
                # 只处理非空内容
                if chunk:
                    # 返回当前块
                    self.full_response.append(chunk)
                    yield chunk
            # 记录完整响应
            complete_response = "".join(self.full_response)
            self.logger.info(f"Complete response length: {len(complete_response)}")
            self.logger.info(f"First 100 chars: {complete_response[:100]}")
        except Exception as e:
            self.logger.error(f"处理流时出错: {str(e)}")
            yield f"Error: {str(e)}"

            

async def main():
    health_report_chat = HealthReportChat(query="睡眠")
    async for chunk in health_report_chat._run(message_history=[]):
        print(chunk, end="", flush=True)
    print()  # 打印最后的换行

            
if __name__ == '__main__':
    asyncio.run(main())
         
    
    
    

