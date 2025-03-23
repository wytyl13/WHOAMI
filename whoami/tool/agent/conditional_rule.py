from typing import (
    Callable,
    Dict,
    Any
)


from whoami.tool.agent.base_tool import BaseTool
from whoami.tool.agent.execution_enum import ExecutionResult

class ConditionalRule(BaseTool):

    def __init__(
        self,
        condition_func: Callable[[Dict[str, Any]], bool]
    ):
        """条件规则类，用于确定工具是否应该执行

        Args:
            condition_func (Callable[[Dict[str, Any]], bool]): _description_
        """
        self.condition_func = condition_func
    
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        评估条件是否满足
        """
        return self.condition_func(context)
    
    
    @staticmethod
    def from_dependency_result(dependency_name: str, expected_value: Any) -> 'ConditionalRule':
        """
        创建基于依赖结果的条件规则
        """
        def check_result(context):
            if dependency_name not in context:
                return False
            
            dep_result = context[dependency_name]
            if not isinstance(dep_result, ExecutionResult):
                return False
            
            if not dep_result.is_successful:
                return False
            
            return dep_result.result == expected_value 
        return ConditionalRule(check_result) 
    
    
    @staticmethod
    def always() -> 'ConditionalRule':
        """创建始终执行的条件规则"""
        return ConditionalRule(lambda _: True)
 
        
    