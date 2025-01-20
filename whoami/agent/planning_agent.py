
import json
import datetime
from whoami.llm_api.ollama_llm import OllamLLM
from whoami.configs.llm_config import LLMConfig
from pathlib import Path

from whoami.utils.log import Logger


llm = OllamLLM(
    LLMConfig.from_file(Path("/home/weiyutao/work/WHOAMI/whoami/scripts/test/ollama_config.yaml")), 
    temperature=0.0
)

class PlanningAgent:
    def __init__(self, tools: list) -> None:
        self.tools = tools
        self.tool_names = ', '.join([tool.name for tool in self.tools])
        self.tool_descs =self._init_descs()
        self.logger = Logger('PlanningAgent')
        self.llm = llm
        self.prompt_tpl = """Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:
        {tool_description}

        These are chat history before:
        {chat_history}
        
        Use the following format:
        - Question: the input question you must answer
        - Thought: you should always think about what to do.
        - Action: the action to take, should be one of [{tool_names}]
        - Action Input: the input to the action
        - Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        - Thought: I now know the final answer
        - Final Answer: the final answer to the original input question

        Begin!

        Question: {query}
        {agent_scratchpad}
        """
        
    def _init_descs(self):
        """初始化工具描述信息

        Returns:
            _type_: _description_
        """
        tool_descs = []
        for t in self.tools:
            args_desc = []
            for name, info in t.args().items():
                print(name, info)
                args_desc.append(
                    {'name': name, 'description': info['description'] if 'description' in info else '', 'type': info['type']})
            args_desc = json.dumps(args_desc, ensure_ascii=False)
            tool_descs.append('%s: %s, args: %s' % (t.name, t.description, args_desc))
        tool_descs = '\n'.join(tool_descs)
        return tool_descs

    def agent_execute(self, query, chat_history=[]):
        global tools, tool_names, tool_descs, prompt_tpl, llm, tokenizer

        agent_scratchpad = ''  # agent执行过程
        while True:
            # 1 格式化提示词并输入大语言模型
            history = '\n'.join(['Question:%s\nAnswer:%s' % (his[0], his[1]) for his in chat_history])
            # 兼容qwen2.5和其他模型
            model_name = 'qwen2'
            # history = ';'.join(['Question:%s;Answer:%s' % (his[0], his[1]) for his in chat_history])
            
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            prompt = self.prompt_tpl.format(today=today, chat_history=history, tool_description=self.tool_descs, tool_names=self.tool_names,
                                    query=query, agent_scratchpad=agent_scratchpad)
            self.logger.info(f"---等待LLM返回... ...\n{prompt}")
            user_stop_words = ['Observation:'] if model_name == 'qwen2' else ['- Observation:']
            response = self.llm.whoami(prompt, user_stop_words=user_stop_words)
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
                if t.name == action:
                    the_tool = t
                    break
            if the_tool is None:
                observation = 'the tool not exist'
                agent_scratchpad = agent_scratchpad + response + observation + '\n'
                continue

            # {"url": "http://localhost:8000/user/", "filed_value": '{"realname":"李四"}'}

            # 6 执行tool
            try:
                # 注意上一步工具的输出结果最好不要有嵌套json，否则解析会出错
                # 因为大语言模型对嵌套json字符串的返回不是转义格式，这不符合python中的json工具对json字符串的解析要求
                action_input = json.loads(action_input)
                self.logger.info(f"---action_input结果... ...\n{action_input}")
                tool_ret = the_tool._run(**action_input)
                self.logger.info(f"---执行tool结果... ...\n{tool_ret}")
                
                # 如果the_tool是终止tool，直接返回并结束当前agent
                if the_tool.end_flag == 1:
                    chat_history.append((query, tool_ret))
                    return True, tool_ret, chat_history
                
            except Exception as e:
                observation = 'the tool has error:{}'.format(e)
            else:
                observation = str(tool_ret)
            agent_scratchpad = agent_scratchpad + response + observation + '\n'
    
    def agent_execute_with_retry(self, query, chat_history=[], retry_times=3):
        for i in range(retry_times):
            status, result, chat_history = self.agent_execute(query, chat_history=chat_history)
            if status:
                return status, result, chat_history
        return status, result, chat_history

