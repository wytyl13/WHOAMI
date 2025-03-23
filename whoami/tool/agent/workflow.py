from typing import (
    Optional,
    Dict,
    Set,
    List
)


from whoami.tool.agent.base_tool import BaseTool
from whoami.tool.agent.tool_node import ToolNode
from whoami.tool.agent.execution_enum import ExecutionResult
from whoami.tool.agent.execution_enum import ExecutionStatus

class WorkFlow(BaseTool):
    """工作流类，管理工具节点之间的依赖和执行顺序

    Args:
        BaseTool (_type_): _description_
    """
    
    def __init__(self):
        super().__init__()
        self.nodes: Dict[str, ToolNode] = {}
        self.execution_order = None


    def add_node(self, node: ToolNode) -> 'WorkFlow':
        """添加工具节点"""
        self.nodes[node.node_id] = node
        self.execution_order = None
        return self
    
    
    def get_node(self, node_id: str) -> Optional[ToolNode]:
        """根据节点id获取节点"""
        return self.nodes.get(node_id)


    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """构建依赖图，用于拓扑排序"""
        graph = {node_id: set() for node_id in self.nodes}
        
        for node_id, node in self.nodes.items():
            for dep, _ in node.dependencies:
                graph[node_id].add(dep.node_id)
        
        return graph
    
    
    def _topological_sort(self) -> List[str]:
        """
        对节点进行拓扑排序，确定执行顺序
        返回：
            排序后的节点名称列表
        """
        graph = self._build_dependency_graph()
        visited = {node_id: False for node_id in graph}
        temp = {node_id: False for node_id in graph}
        order = []
        
        def dfs(node_id):
            if temp[node_id]:
                # 循环依赖
                raise ValueError(f'Circular dependency detected involving {node_id}')
            if not visited[node_id]:
                temp[node_id] = True
                for neighbor in graph[node_id]:
                    dfs(neighbor)
                temp[node_id] = False
                visited[node_id] = True
                order.append(node_id)
            
            
        for node_id in graph:
            if not visited[node_id]:
                dfs(node_id)
        
        return list(order)
    
    
    def get_execution_order(self) -> List[str]:
        """
        获取工具的执行顺序
        """
        if self.execution_order is None:
            self.execution_order = self._topological_sort()
        return self.execution_order
    
    
    def visualize(self) -> str:
        """
        生成工作流的文本可视化表示
        """
        if not self.nodes:
            return "Empty workflow"
        
        order = self.get_execution_order()
        result = [f"WorkFlow: {self.name}", ""]
        result.append("Execution Order:")
        
        for i, node_id in enumerate(order):
            node: ToolNode = self.nodes[node_id]
            deps = [dep.node_id for dep, _ in node.dependencies]
            deps_str = ", ".join(deps) if deps else "None"
            
            result.append(f"{i+1}. {node_id} - Dependencies: {deps_str}")
            
            # 添加条件信息
            if node.conditional_rules:
                result.append(f"    Conditions: {len(node.conditional_rules)} rule(s)")
                
        
        return "\n".join(result)


    def execute(self, **kwargs):
        """
        执行工作流中的所有节点
        
        Args:
            initial_context: 初始上下文，包含输入数据
            
        Returns:
            执行后的上下文，包含所有节点的执行结果
        """
        execution_result = ExecutionResult()
        # 初始化上下文
        initial_context = {"numbers": [2, 3, 4]}
        context = initial_context or {}
        
        # 获取执行顺序
        execution_order = self.get_execution_order()
        print(f"执行顺序: {execution_order}\n")
        print(f"Execution Order: \n{self.visualize()}")
        # 按顺序执行每个节点
        for node_id in execution_order:
            node = self.get_node(node_id)
            
            print(f"\n===== 执行节点: {node_id} =====")
            print(f"当前上下文: {context}")
            
            # 检查是否应该执行
            if node.should_execute(context):
                try:
                    # 获取输入参数
                    inputs = node.get_required_inputs(context)
                    print(f"节点输入: {inputs}")
                    
                    # 执行工具
                    result = node.tool.execute(**inputs)
                    print(f"执行结果: {result}")
                    
                    
                    # 将结果存入上下文
                    context[node_id] = execution_result(status=ExecutionStatus.COMPLETED, result=result)
                except Exception as e:
                    print(f"执行错误: {e}")
                    context[node_id] = execution_result(status=ExecutionStatus.FAILED, result=result, error=e)
            else:
                print(f"跳过节点 {node_id}，条件不满足")
        
        return context

