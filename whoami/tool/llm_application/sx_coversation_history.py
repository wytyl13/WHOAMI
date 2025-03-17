from sqlalchemy import Column, Integer, String, Enum, Text, BigInteger, Index, func, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mysql import TIMESTAMP
Base = declarative_base()

class SxConversationHistory(Base):
    """对话历史数据模型"""
    __tablename__ = 'sx_conversation_history'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(String(100), nullable=False)
    user_id = Column(String(50), nullable=False)
    role = Column(Enum('user', 'assistant', 'system', name='role_enum'), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(fsp=6), server_default=func.current_timestamp(6))
    deleted = Column(Boolean, nullable=False, default=False)
    
    # 创建索引
    __table_args__ = (
        Index('idx_conversation_id_created_at', conversation_id, created_at),
        Index('idx_user_id', user_id),
    )
    
    def __repr__(self):
        return f"<ConversationHistory(id={self.id}, conversation_id={self.conversation_id}, role={self.role})>"
    
    def to_dict(self):
        """将对象转换为字典"""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'user_id': self.user_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'deleted': self.deleted
        }