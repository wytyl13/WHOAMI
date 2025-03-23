from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Optional
)



class ExecutionStatus(Enum):
    """工具执行状态

    Args:
        Enum (_type_): _description_
    """
    NOT_STARTED = auto() # 尚未开始执行
    RUNNING = auto() # 正在执行
    COMPLETED = auto() # 执行完成
    SKIPPED = auto() # 被跳过
    FAILED = auto() # 执行失败
    
    


class ExecutionResult:
    """工具执行结果

    Args:
        Enum (_type_): _description_
    """
    status: ExecutionStatus
    result: Any = None
    error: Exception = None
    metadata: Dict[str, Any] = None
    
    
    def __call__(
        self, 
        status: Optional[ExecutionStatus] = None, 
        result: Optional[Any] = None,
        error: Optional[Exception] = None,
        metadata: Dict[str, Any] = None
    ):
        self.status = status if status is not None else self.status
        self.result = result if result is not None else self.result
        self.error = error if error is not None else self.error
        self.metadata = metadata if metadata is not None else self.metadata
        return self
    
    
    @property
    def is_successful(self) -> bool:
        """检查执行是否成功"""
        return self.status == ExecutionStatus.COMPLETED