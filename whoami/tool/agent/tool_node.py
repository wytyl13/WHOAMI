from typing import List, Dict, Any, Tuple, Optional, Type
import uuid


from whoami.tool.agent.base_tool import BaseTool
from whoami.tool.agent.execution_enum import ExecutionStatus
from whoami.tool.agent.conditional_rule import ConditionalRule
from whoami.tool.agent.execution_enum import ExecutionResult


class ToolNode(BaseTool):
    """工具节点类，表示工作流中的一个工具节点

    Args:
        BaseTool (_type_): _description_
    """
    def __init__(self, tool: BaseTool, strict_required_input_flag: Optional[int] = None, node_id: str = None):
        super().__init__()
        self.tool: BaseTool = tool
        self.node_id = node_id or f"{tool.name}_{uuid.uuid4().hex[:8]}"  # 生成唯一ID
        self.dependencies: List[Tuple[ToolNode, List[str]]] = []
        self.conditional_rules: List[ConditionalRule] = []
        self.status: ExecutionStatus = ExecutionStatus.NOT_STARTED
        self.result: str = None
        self.strict_required_input_flag: int = strict_required_input_flag if strict_required_input_flag is not None else 1


        # 创建输入参数辅助对象
        # 在初始化时创建参数访问对象
        self.param = self.tool.args_schema
    
    
    def add_dependency(self, dependency: 'ToolNode', required_inputs: List[str] = None):
        """添加依赖节点

        Args:
            dependency (ToolNode): 添加依赖工具节点
            required_inputs (List[str], optional): 从依赖结果中需要提取的输入参数名列表. Defaults to None.
        """
        if required_inputs is None:
            required_inputs = self.tool.get_input_names()
        self.dependencies.append((dependency, required_inputs))
        return self
    
    
    def add_conditional_rule(self, rule: ConditionalRule):
        self.conditional_rules.append(rule)
        return self
    
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        """判断是否应该执行该节点"""
        # 检查所有依赖是否成功完成
        for dep, _ in self.dependencies:
            dep_node_id = dep.node_id
            if dep_node_id not in context:
                return False
            
            
            dep_result: ExecutionResult = context[dep_node_id]
            if not dep_result.is_successful:
                return False
            
        
        # 如果没有条件规则，只要依赖满足就执行
        if not self.conditional_rules:
            return True
        
        
        # 检查是否满足至少一个条件规则
        return any(rule.evaluate(context) for rule in self.conditional_rules)
    
    
    def get_required_inputs(self, context: Dict[str, 'ToolNode'], strict_required_input_flag: Optional[int] = None) -> Dict[str, Any]:
        """
        从上下文中获取执行所需的输入
        参数：
            context: 执行上下文，包含之前工具的执行结果
        返回：
            执行当前工具所需的输入参数字典
        """
        
        strict_required_input_flag = self.strict_required_input_flag if strict_required_input_flag is None else strict_required_input_flag
        
        inputs = {}
        missing_inputs = set()
        # 从依赖中收集输入
        for dep, require_fields in self.dependencies:
            dep_node_id = dep.node_id
            if dep_node_id in context and context[dep_node_id].is_successful:
                dep_result = context[dep_node_id].result

                # 收集未找到的字段
                found_fields = set()
                
                # 如果依赖结果是字典，可以提取特定字段
                if isinstance(dep_result, dict):
                    for field in require_fields:
                        if field in dep_result:
                            inputs[field] = dep_result[field]
                            found_fields.add(field)
                # 如果依赖结果是单值，直接使用
                elif len(require_fields) == 1:
                    inputs[require_fields[0]] = dep_result
                    found_fields.add(require_fields[0])

                # 记录缺失的字段
                missing_fields = set(require_fields) - found_fields
                if missing_fields:
                    missing_inputs.update(missing_fields)

        if missing_inputs:
            # 选择1：抛出异常
            if strict_required_input_flag:
                raise ValueError(f"Missing required inputs from dependencies: {missing_inputs}")
            else:
                self.logger.warning(f"Missing inputs {missing_inputs} from dependencies, trying global context")
        
        # 从全局上下文中获取其他输入
        for input_name in self.tool.get_input_names():
            if input_name in context and input_name not in inputs:
                inputs[input_name] = context[input_name]
        
        return inputs

    def execute(self, **kwargs):
        raise NotImplementedError

    