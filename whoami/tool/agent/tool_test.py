from pydantic import BaseModel, Field
from typing import List, Type, Optional
import asyncio


from whoami.tool.agent.base_tool import BaseTool
from whoami.tool.agent.workflow import WorkFlow
from whoami.tool.agent.tool_node import ToolNode
from whoami.tool.agent.execution_enum import ExecutionResult


# too1(a, b), tool2(a)
# c


# context
# 我要计算1+2
# too1： 加法计算  两个int型参数
# tool2: 乘法计算   1个参数是整型

# context
# a, b, c

# 可选：为工具创建参数schema
class CalculatorSchema(BaseModel):
    numbers: List[float] = Field(
        ...,  # 使用 ... 表示必填字段
        description="要进行计算的数字列表"
    )
    operation: str = Field(
        ...,
        description="要执行的数学运算，如 'add'、'multiply'、'subtract' 或 'divide'"
    )
    flag: Optional[str] = Field(
        default=None,
        description="flag"
    )

# 继承BaseTool创建自定义工具
class CalculatorAdd(BaseTool):
    """一个简单的计算器工具"""
    name: str = "Calculator"
    description: str = "执行基本数学运算，加法"
    args_schema: BaseModel = CalculatorSchema
    
    def execute(self, numbers: List[float]) -> float:
        return sum(numbers)
       
       
# 继承BaseTool创建自定义工具
class CalculatorAddMulti(BaseTool):
    """一个简单的计算器工具"""
    name: str = "Calculator"
    description: str = "执行基本数学运算，乘法"
    args_schema: BaseModel = CalculatorSchema
    
    def execute(self, numbers: float) -> float:
        result = 1
        result *= numbers
        return result
        # 其他操作...
        
# 继承BaseTool创建自定义工具
class CalculatorAddMultiPlus(BaseTool):
    """一个简单的计算器工具"""
    name: str = "Calculator"
    description: str = "执行基本数学运算，乘法"
    args_schema: BaseModel = CalculatorSchema
    
    def execute(self, numbers1: int, numbers2: int) -> float:
        return numbers1 * numbers2
        # 其他操作...
        
from pathlib import Path

from whoami.tool.agent.tool.direct_llm import DirectLLM
from whoami.tool.agent.tool.google_search import GoogleSearch
from whoami.tool.agent.tool.health_report import HealthReport
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig
from whoami.tool.llm_application.enhance_retrieval import EnhanceRetrieval
from whoami.tool.llm_application.planning_agent import PlanningAgent
from whoami.tool.agent.tool.sleep_indices_sql_data import SleepIndicesSqlData


# import asyncio

async def test_direct_llm():
    
    llm_finetune = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml')))
    llm_qwen = OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config_qwen.yaml')))

    enhance_finetune = EnhanceRetrieval(llm=llm_finetune)
    enhance_qwen = EnhanceRetrieval(llm=llm_qwen)

    direct_llm_tool = DirectLLM(enhance_llm=enhance_finetune)
    google_search_tool = GoogleSearch(enhance_llm=enhance_qwen)
    health_report_tool = HealthReport(enhance_llm=enhance_qwen, device_sn='13D6F349200080712111957107')
    planning_agent = PlanningAgent(tools=[direct_llm_tool, google_search_tool, health_report_tool], llm=llm_qwen)
    # sql_data = SleepIndicesSqlData()
    
    print(direct_llm_tool)
    print()
    print()
    print(google_search_tool)
    print()
    print()
    # print(health_report_tool)
    print()
    print()
    question = "卫宇涛多大年龄？"
    chat_history = [
        ["舜熙科技的主要产品是什么？", "不知道"], 
        ["它们公司官网是多少", "舜熙科技的官网是 https://shunxikj.com/ ，您可以了解产品详情、解决方案和成功案例。"],
        ["它们公司地址是多少", "舜熙科技的总部位于山西省运城市盐湖区黄河金三角科创城C1栋。"],
        ["你们公司的核心算法有哪些？", "舜熙科技的核心算法主要包括：\n1. 跌倒检测算法：能准确识别老人的异常姿态；\n2. 行为模式学习算法：系统会“学习”老人的日常活动规律，在偏离正常模式时及时预警；\n3. 语音语义理解算法：精准识别老人口音和方言；\n4. 健康状态异常分析算法：结合生理数据和行为数据判断是否出现疾病预警。所有算法都会随着使用时间增长而不断优化。"],
        ["我昨晚睡的怎么样？", """根据您的睡眠监测数据显示，您昨晚（2025年4月1日）的睡眠情况如下：
  - 上床时间：2025年3月31日晚上7点0分
  - 入睡时间：2025年3月31日晚上7点9分57秒
  - 醒来时间：2025年4月1日早上5点54分40秒

  其他相关信息：
  - 睡眠时长为8小时25分钟
  - 深度睡眠时间为2小时7分钟，占总睡眠时间的25%
  - 浅度睡眠时间为6小时17分钟，占总睡眠时间的77.39%
  - 夜间醒来次数为3次
  - 睡眠效率为0.72（即72%）
  - 体动指数为4.63

  总体评分：57.97，属于较差水平。

  基于以上数据，建议您注意改善睡眠质量。如有需要可以咨询医生或专业人士。"""]
    ]
    status, result, chat_history = await planning_agent.agent_execute_with_retry(question, chat_history=chat_history)
    print(f"问题: {question}")
    print(f"回答: {result}")
    
    return result


if __name__ == '__main__':

    asyncio.run(test_direct_llm())
    
    
    # calculator_add = CalculatorAdd()
    # calculator_multi = CalculatorAddMulti()
    # calculator_add_multi_plus = CalculatorAddMultiPlus()
    
    # calculator_add_node = ToolNode(calculator_add)
    # calculator_multi_node = ToolNode(calculator_multi).add_dependency(calculator_add_node)
    
    # workflow = WorkFlow()
    
    # calculator_add_node_plus = ToolNode(calculator_add_multi_plus)
    
    
    # 因为在创建Tool工具的时候定义了输入参数要求
    # 而在建立节点的时候如果存在多个依赖节点，需要指定对应的依赖节点和输入参数的映射关系
    # 这种映射关系使用硬编码的方式不方便
    # 思考：如果在输出参数和输入参数的字段上能对应上，就不需要建立映射关系，因为get_required_input会从上下文信息中自动找到对应的参数
    # 但是这种有个弊端，就是如果在上下文中多个节点存在重复的输出字段呢？会存在歧义
    # 所以使用在这种办法需要优化get_required_input更加智能
    # 如果在某个节点的输出参数中加入node_id，这样在上下文中肯定不会存在重复的输出字段，消除歧义
    # 但是这种如何实现自动定义映射？比如一个节点的依赖节点是  A B节点，那么我的本节点的get_required_input方法会仅从我的山下文中筛选这两个节点的输出找对应的输入参数
    # 现在的逻辑是这样的，如果我需要A B节点的输出，我有两个对应的输入参数，一个是number1  一个是number2
    # A节点的输出不可能定义或者智能定义为number1，因为A节点的定义有可能在当前节点的定义之前，而且A节点的定义是基于某个工具，该工具的定义是为了定义不同的节点，因为不同的节点可能使用
    # 相同的工具不同的初始化参数去定义，而相同的工具的返回值是固定的（是否可以在定义节点的时候改变工具的返回值），假如我在定义节点的时候已经知道了依赖该节点的节点需要什么样的返回参数？
    # 那么我可以动态修改对应工具的返回参数，那么工具会很容易找到该节点的返回参数去赋值给自己的参数
    # 工作流需要人定义。（Agent）
    # 
    # 任务：
    #   输入 （ai报告生成 + tool）   输出工作流{a, b, c}
    # 大模型完成    # 输入到工作流  （Agent）
    # 也就是说我需要先定义工作流，然后使用工作流节点之间的依赖关系去定义节点，然后节点再去动态修改tool的返回值
    
    # 这样可以实现吗？
    # 我感觉可以，直接将节点的定义设置为动态即可。
    # 由工作流自上而下定义
    # 先定义节点之间的依赖关系
    # 然后依靠依赖关系去定义节点
    # 然后再由节点定义工具
    # 这样就可以实现动态获取
    # 我真是一个人才
    
    # calculator_add_node_plus = calculator_add_node_plus.add_dependency(calculator_add_node, ["numbers1"])
    # calculator_add_node_plus = calculator_add_node_plus.add_dependency(calculator_multi_node, ["numbers2"])
    
    # print(calculator_add_node.dependencies)
    # print(calculator_multi_node.dependencies)
    # print(calculator_add_node_plus.dependencies)
    
    # # 打印节点依赖信息
    # print("\n===== 节点依赖信息 =====")
    # print("加法节点依赖:", calculator_add_node.dependencies)
    # print("乘法节点依赖:", calculator_multi_node.dependencies)
    # print("乘法plus节点依赖:", calculator_add_node_plus.dependencies)
    
    # # 提取并打印乘法节点期望的输入参数名
    # for dep_node, required_inputs in calculator_add_node_plus.dependencies:
    #     print(f"乘法plus节点依赖 {dep_node.tool.name} 需要的输入字段: {required_inputs}")

    # 20 


    # new_workflow = workflow.add_node(calculator_add_node)
    # new_workflow = new_workflow.add_node(calculator_multi_node)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)

    # print(new_workflow.visualize())
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # new_workflow = new_workflow.add_node(calculator_add_node_plus)
    # context = new_workflow.execute()
    
    # # always
    # # chuanxing bingxing
    # # 
    
    # # AGENT

    # # BaseTool
    # # Customer_Tool(BaseTool)
    # # ToolNode
    # # Workflow
    # a -> b -> -d
    #     e
    #     finally----------------------------------------------------------------------
    #     g

    # Workflow.add(a)
    # Workflow.add(b)
    # Workflow.add(d)
    
    
    # #  BaseTool
    # # Customer_Tool(BaseTool)
    # # 
    # {
    #     "a" -> "b" -> "c"
    # }
    
    
    






    
    
    # print(context)
    