#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/26 17:22
@Author  : weiyutao
@File    : planning_agent.py
"""

from typing import (
    Optional,
    Type,
    List,
    Dict
)
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
import json
import inspect
import asyncio
from contextvars import ContextVar
import uuid

from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool.enhance_retrieval import EnhanceRetrieval
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig
from whoami.tool.agent.base_tool import BaseTool



execution_context: ContextVar[dict] = ContextVar('execution_context')

class PlanningAgentCommunityAiUserSchema(BaseModel):
    question: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="用户需要解决的问题"
    )
    
    tools: List = Field(
        ...,  # 使用 ... 表示必填字段
        description="Agent可以使用的工具域"
    )
    



@tool
class PlanningAgentCommunityAiUser:
    end_flag: int = 0
    args_schema: Type[BaseModel] = PlanningAgentCommunityAiUserSchema
    enhance_llm: Optional[EnhanceRetrieval] = None
    tools: Optional[List[BaseTool]] = None
    tool_descs: Optional[str] = None
    tool_names: Optional[str] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'enhance_llm' in kwargs:
            self.enhance_llm = kwargs.pop('enhance_llm')
            
        if 'tools' in kwargs:
            self.tools = kwargs.pop('tools')
    
        if self.enhance_llm is None:
            raise ValueError("EnhanceLLM must not be null!")
        
        if self.tools:
            self._init_descs_names()



    def _get_context(self):
        """获取当前执行上下文"""
        try:
            return execution_context.get()
        except LookupError:
            # 如果没有上下文，创建一个新的
            ctx = {
                "execution_id": str(uuid.uuid4()),
                "status": "ready",
                "error_message": ""
            }
            execution_context.set(ctx)
            return ctx

    
    def _set_status(self, status: str, error_message: str = ""):
        """设置执行状态"""
        ctx = self._get_context()
        ctx["status"] = status
        ctx["error_message"] = error_message
    
    


    def _init_descs_names(self):
        """初始化工具描述信息

        Returns:
            _type_: _description_
        """
        try:
            # tool_descs = [str(t.tool_schema) for t in self.tools]
            tool_descs = [t.get_simple_tool_description() for t in self.tools]
            self.tool_descs = '\n\n'.join(tool_descs)
            self.tool_names = ', '.join([tool.name for tool in self.tools])
        except Exception as e:
            raise ValueError("Fail to init the tool_descs and tool_names!")
    
    
    async def agent_execute(
        self, 
        query, 
        chat_history=[], 
        retrieval_flag=False, 
        retry_count=0, 
        max_retries=3,
        username: str = None,
        location: str = None,
        role: str = None,

    ):
        
        
        # 初始化上下文
        self._set_status("running")
        
        global tools, tool_names, tool_descs, system_prompt, llm, tokenizer

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
            
            # retrieval anytime
            context = "无可用上下文信息"
            if retrieval_flag:
                nodes = await self.enhance_llm.retrieval.execute(text_list=[], top_k=1, retrieval_word=query)
                context_texts = [node.node.text.replace('\n', '') for node in nodes]
                if context_texts:
                    context = "\n\n".join(context_texts)
            
            prompt = self.system_prompt.format(today=today, weekday=weekday, chat_history=history, context=context, tool_description=self.tool_descs, tool_names=self.tool_names,
                                    query=query, agent_scratchpad=agent_scratchpad, location=location)
            
            self.logger.info(f"---等待LLM返回... ...\n{prompt}")
            user_stop_words = ['Observation:'] if model_name == 'qwen2' else ['- Observation:']
            messages = [{"role": "user", "content": prompt}]
            response = await self.enhance_llm.llm._whoami_text(messages=messages, timeout=30, user_stop_words=user_stop_words)
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
                self._set_status("success")
                final_answer = response[final_answer_i + len(final_answer_i_str):].strip()
                chat_history.append((query, final_answer))
                # return True, final_answer, chat_history
                for char in final_answer:
                    yield char
                return

            # 4 解析action
            if not (thought_i < action_i < action_input_i):
                # 递归重试
                if retry_count < max_retries:
                    self.logger.info(f"LLM回复格式异常，进行第{retry_count + 1}次重试...")
                    async for chunk in self.agent_execute(query, chat_history, retrieval_flag, retry_count + 1, max_retries, username=username, location=location, role=role):
                        yield chunk
                    return
                else:
                    # 达到最大重试次数，返回错误
                    self._set_status("error", 'LLM回复格式异常')
                    for char in 'LLM回复格式异常':
                        yield char
                    return
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
                
                action_input["username"] = username
                action_input["location"] = location
                action_input["role"] = role
                self.logger.info(f"action_input: ----------------------------------------- {action_input}")
                # 如果the_tool是终止tool，直接返回并结束当前agent
                self.logger.info(f"---action_input结果... ...\n{action_input}")
                tool_ret = ""
                if the_tool.end_flag == 1:
                    self._set_status("success")
                    async for chunk in the_tool.execute(**action_input):
                        yield chunk
                        tool_ret += chunk
                    self.logger.info(f"---执行tool结果... ...\n{tool_ret}")
                    chat_history.append((query, tool_ret))
                    return
                    # return False, tool_ret, chat_history
                
                
                tool_ret = await the_tool.execute(**action_input)
                self.logger.info(f"---执行tool结果... ...\n{tool_ret}")
                
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
                        final_response = await self.enhance_llm.llm._whoami_text(messages=final_messages, timeout=30, user_stop_words=[])
                        
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
                    # return False, final_answer, chat_history
                    self._set_status("success")
                    for char in final_answer:
                        yield char
                    return
                
            except Exception as e:
                observation = 'the tool has error:{}'.format(e)
                tool_error_counter[tool_name] += 1
                tool_success_counter[tool_name] = 0  # Reset success counter
                
                # If same tool failed 3 or more times in a row, break out
                if tool_error_counter[tool_name] >= 3:
                    # 递归重试
                    if retry_count < max_retries:
                        error_message = f"工具 {tool_name} 连续调用失败，进行第{retry_count + 1}次重试..."
                        self.logger.info(error_message)
                        async for chunk in self.agent_execute(query, chat_history, retrieval_flag, retry_count + 1, max_retries, username=username, location=location, role=role):
                            yield chunk
                        return
                    else:
                        # 最终失败
                        error_message = f"工具 {tool_name} 连续调用失败超过限制次数，无法完成请求。"
                        self._set_status("error", error_message)
                        for char in error_message:
                            yield char
                        return
                    
                
                # 工具执行异常
                if retry_count < max_retries:
                    self.logger.info(f"工具执行异常，进行第{retry_count + 1}次重试: {str(e)}")
                    async for chunk in self.agent_execute(query, chat_history, retrieval_flag, retry_count + 1, max_retries, username=username, location=location, role=role):
                        yield chunk
                    return
                else:
                    error_msg = f"工具执行异常: {str(e)}"
                    self._set_status("error", error_msg)
                    for char in error_msg:
                        yield char
                    return
                
                
            # except Exception as e:
            #     observation = 'the tool has error:{}'.format(e)
            # else:
            #     observation = str(tool_ret)
            agent_scratchpad = agent_scratchpad + response + observation + '\n'
    
    
    async def execute(
        self, 
        tools: Optional[list[any]] = None,
        question: str = None,
        chat_history: Optional[List] = None,
        retry_times: Optional[int] = 3,
        retrieval_flag: Optional[bool] = True,
        username: str = None,
        location: str = None,
        role: str = None
    ):
        if tools:
            self.tools = tools
            self._init_descs_names()

        async for chunk in self.agent_execute(
            query=question,
            chat_history=chat_history or [],
            retrieval_flag=retrieval_flag,
            retry_count=0,
            max_retries=retry_times,
            username=username,
            location=location,
            role=role
        ):
            yield chunk


