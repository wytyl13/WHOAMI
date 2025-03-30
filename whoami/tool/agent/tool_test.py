from pydantic import BaseModel, Field
from typing import List, Type, Optional

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
        
        
if __name__ == '__main__':
    calculator_add = CalculatorAdd()
    calculator_multi = CalculatorAddMulti()
    calculator_add_multi_plus = CalculatorAddMultiPlus()
    
    calculator_add_node = ToolNode(calculator_add)
    calculator_multi_node = ToolNode(calculator_multi).add_dependency(calculator_add_node)
    
    workflow = WorkFlow()
    
    calculator_add_node_plus = ToolNode(calculator_add_multi_plus)
    
    
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
    
    calculator_add_node_plus = calculator_add_node_plus.add_dependency(calculator_add_node, ["numbers1"])
    calculator_add_node_plus = calculator_add_node_plus.add_dependency(calculator_multi_node, ["numbers2"])
    
    print(calculator_add_node.dependencies)
    print(calculator_multi_node.dependencies)
    print(calculator_add_node_plus.dependencies)
    
    # 打印节点依赖信息
    print("\n===== 节点依赖信息 =====")
    print("加法节点依赖:", calculator_add_node.dependencies)
    print("乘法节点依赖:", calculator_multi_node.dependencies)
    print("乘法plus节点依赖:", calculator_add_node_plus.dependencies)
    
    # 提取并打印乘法节点期望的输入参数名
    for dep_node, required_inputs in calculator_add_node_plus.dependencies:
        print(f"乘法plus节点依赖 {dep_node.tool.name} 需要的输入字段: {required_inputs}")

    20 


    new_workflow = workflow.add_node(calculator_add_node)
    new_workflow = new_workflow.add_node(calculator_multi_node)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)

    print(new_workflow.visualize())
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    new_workflow = new_workflow.add_node(calculator_add_node_plus)
    context = new_workflow.execute()
    
    # always
    # chuanxing bingxing
    # 
    
    # AGENT

    # BaseTool
    # Customer_Tool(BaseTool)
    # ToolNode
    # Workflow
    a -> b -> -d
        e
        finally----------------------------------------------------------------------
        g

    Workflow.add(a)
    Workflow.add(b)
    Workflow.add(d)
    
    
    #  BaseTool
    # Customer_Tool(BaseTool)
    # 
    {
        "a" -> "b" -> "c"
    }
    
    
    






    
    
    print(context)
    