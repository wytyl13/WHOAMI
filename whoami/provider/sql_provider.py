#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/24 11:35
@Author  : weiyutao
@File    : sql_provider.py
"""
import traceback
from pydantic import BaseModel, model_validator, ValidationError
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
    Generic,
    TypeVar,
    Any,
    Type,
    List
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
from contextlib import contextmanager
import traceback
import urllib.parse
from sqlalchemy import and_

from whoami.provider.base_provider import BaseProvider
from whoami.configs.sql_config import SqlConfig

# 定义基类
class Base(DeclarativeBase):
    pass

# 定义泛型类型变量
ModelType = TypeVar("ModelType", bound=Base)

class SqlProvider(BaseProvider, Generic[ModelType]):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    sql_connection: Optional[sessionmaker] = None
    model: Type[ModelType] = None
    
    def __init__(
        self, 
        model: Type[ModelType] = None,
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None
    ) -> None:
        super().__init__()
        self._init_param(sql_config_path, sql_config, model)
    
    def _init_param(self, sql_config_path: Optional[str] = None, sql_config: Optional[SqlConfig] = None, model : Type[ModelType] = None):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.sql_config = SqlConfig.from_file(self.sql_config_path) if self.sql_config is None and self.sql_config_path is not None else self.sql_config
        # if self.sql_config is None and self.data is None:
        #     raise ValueError("config config_path and data must not be null!")
        self.sql_connection = self.get_sql_connection() if self.sql_config is not None else self.sql_connection
        self.model = model
        if self.model is None:
            raise ValueError("model must not be null!")
        
    def get_sql_connection(self):
        try:
            sql_info = self.sql_config
            username = sql_info.username
            database = sql_info.database
            password = sql_info.password
            host = sql_info.host
            port = sql_info.port
        except Exception as e:
            raise ValueError(f"fail to init the sql connect information!\n{self.sql_config}") from e
        # 因为url中的密码可能存在冲突的字符串，因此需要在进行数据库连接前对其进行编码
        # urllib.parse.quote_plus() 函数将特殊字符替换为其 URL 编码的对应项。例如，! 变为 %21，@ 变为 %40。这确保了密码被视为单个字符串，并且不会破坏 URL 语法。
        encoded_password = urllib.parse.quote_plus(password)
        database_url = f"mysql+mysqlconnector://{username}:{encoded_password}@{host}:{port}/{database}"
        
        try:
            engine = create_engine(database_url, pool_size=10, max_overflow=20)
            SessionLocal = sessionmaker(bind=engine)
        except Exception as e:
            raise ValueError("fail to create the sql connector engine!") from e
        return SessionLocal
    
    def set_model(self, model: Type[ModelType] = None):
        """reset model"""
        if model is None:
            raise ValueError('model must not be null!')
        self.model = model
    
    @contextmanager
    def get_db_session(self):
        """提供数据库会话的上下文管理器"""
        if not self.sql_connection:
            raise ValueError("Database connection not initialized")
        
        session = self.sql_connection()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_record(self, data: Dict[str, Any]) -> int:
        """添加记录"""
        with self.get_db_session() as session:
            try:
                record = self.model(**data)
                session.add(record)
                session.flush()  # 刷新以获取ID
                record_id = record.id
                return record_id
            except Exception as e:
                error_info = f"Failed to add record: {e}"
                self.logger.error(error_info)
                self.logger.error(traceback.print_exc())
                raise ValueError(error_info) from e
    
    def delete_record(self, record_id: int) -> bool:
        """软删除记录"""
        with self.get_db_session() as session:
            try:
                result = session.query(self.model).filter(
                    self.model.id == record_id,
                    self.model.deleted == False
                ).update({"deleted": True})
                return result > 0
            except Exception as e:
                error_info = f"Failed to delete record: {record_id}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
    
    def update_record(self, record_id: int, data: Dict[str, Any]) -> bool:
        """更新记录"""
        with self.get_db_session() as session:
            try:
                result = session.query(self.model).filter(
                    self.model.id == record_id,
                    self.model.deleted == False
                ).update(data)
                return result > 0
            except Exception as e:
                error_info = f"Failed to update record {record_id} with data: {data}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e


    def upsert_record_by_unique_field(
        self, 
        unique_field: Union[str, List[str]] = None, 
        data: Dict[str, Any] = None,
        db_model: Type[Base] = None
    ) -> Dict[str, Any]:

        db_model = self.model if db_model is None else db_model
        """
        根据唯一字段进行记录的更新或插入
        
        Args:
            unique_field (str): 用于判断记录唯一性的字段名
            data (Dict[str, Any]): 要插入或更新的数据字典
            db_model (Type[Base]): 数据库模型类
        
        Returns:
            Dict[str, Any]: 插入或更新后的记录
        """
        
        def convert_numpy_types(value):
            """转换numpy数据类型为Python原生类型"""
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, np.bool_):
                return bool(value)
            return value
        
        
        with self.get_db_session() as session:
            try:
                
                if not data:
                    raise ValueError("Empty data dictionary provided")
                
                # 转换数据类型
                converted_data = {
                    key: convert_numpy_types(value)
                    for key, value in data.items()
                }

                # 将单个字段转换为列表，统一处理
                unique_fields = [unique_field] if isinstance(unique_field, str) else unique_field
                
                # 检查唯一字段是否存在于模型中
                for field in unique_fields:
                    if not hasattr(db_model, field):
                        raise ValueError(f"Unique field {field} not found in model")
                
                # 构建唯一键的查询条件
                filter_conditions = []
                for field in unique_fields:
                    field_value = converted_data.get(field)
                    if field_value is None:
                        raise ValueError(f"Unique field {field} value is None")
                    filter_conditions.append(getattr(db_model, field) == field_value)
                
                # 添加未删除条件
                filter_conditions.append(db_model.deleted == False)
                
                # 查询是否存在记录
                existing_record = session.query(db_model).filter(
                    and_(*filter_conditions)
                ).first()
                
                # 构建要更新的数据字典
                valid_data = {
                    key: value 
                    for key, value in converted_data.items() 
                    if hasattr(db_model, key) and key != 'id'  # 排除id和不存在的字段
                }
                if not valid_data:
                    raise ValueError("No valid fields to update")
                
                # 如果记录已存在，更新记录
                if existing_record:
                    for key, value in valid_data.items():
                        setattr(existing_record, key, value)
                    record = existing_record
                
                # 如果记录不存在，创建新记录
                else:
                    # 移除可能的id字段，防止主键冲突
                    record = db_model(**valid_data)
                    session.add(record)
                
                # 提交事务
                session.commit()
                session.refresh(record)
                
                # 转换为字典返回
                result = {}
                for key in valid_data.keys():
                    value = getattr(record, key)
                    # 处理SQLAlchemy对象关系
                    if hasattr(value, '__table__'):
                        continue  # 跳过关联对象
                    result[key] = value
                
                return result
            except Exception as e:
                session.rollback()
                if isinstance(unique_field, str):
                    error_info = f"Failed to upsert record with {unique_field}={data.get(unique_field)}"
                else:
                    unique_values = {field: data.get(field) for field in unique_field}
                    error_info = f"Failed to upsert record with unique fields: {unique_values}"
                self.logger.error(f"{error_info}. Error: {str(e)}")
                raise ValueError(error_info) from e
    
    

    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """根据ID查询记录"""
        with self.get_db_session() as session:
            try:
                record = session.query(self.model).filter(
                    self.model.id == record_id,
                    self.model.deleted == False
                ).first()
                return record.__dict__ if record else None
            except Exception as e:
                error_info = f"Failed to get record by id: {record_id}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
    
    def get_record_by_condition(
        self, 
        condition: Optional[Dict[str, Any]],
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        with self.get_db_session() as session:
            try:
                
                # 获取模型的所有字段
                all_fields = [column.key for column in self.model.__table__.columns]
                
                if fields:
                    # 如果指定了字段，只查询指定字段
                    query_fields = fields
                else:
                    query_fields = all_fields
                
                # 排除不需要的字段
                if exclude_fields:
                    query_fields = [f for f in query_fields if f not in exclude_fields]
                    
                # 构建查询条件
                query = session.query(*[getattr(self.model, field) for field in query_fields])
                
                # 添加未删除条件
                query = query.filter(self.model.deleted == False)

                # Apply filters based on the provided condition
                if condition:
                    for key, value in condition.items():
                        # Assuming that keys in condition match the model's attributes
                        query = query.filter(getattr(self.model, key) == value)

                # 执行查询
                records = query.all()

                # 处理查询结果
                if not records:
                    return []
                
                # 返回查询结果
                return [dict(zip(query_fields, record)) for record in records]
                # if fields:
                #     # 如果指定了字段，返回包含指定字段的字典列表
                #     return [dict(zip(fields, record)) for record in records]
                # else:
                #     return [{
                #         key: value 
                #         for key, value in record.__dict__.items() 
                #         if key != '_sa_instance_state'
                #         } for record in records]
            except Exception as e:
                error_info = f"Failed to get records by condition: {condition}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
    
    def get_field_names_and_descriptions(self) -> Dict[str, str]:
        field_info = {}
        # 获取模型的所有字段
        for column in self.model.__table__.columns:
            # 假设中文描述存储在列的 doc 属性中
            # 如果没有中文描述，可以使用其他方法来获取
            field_info[column.name] = column.comment  if column.comment else "无描述"
        return field_info
    
    def update_rank_by_id(self, record_id: int, new_rank: int) -> Optional[Dict[str, Any]]:
        with self.get_db_session() as session:
            try:
                # 查询要更新的记录
                record = session.query(self.model).filter(self.model.id == record_id, self.model.deleted == False).one_or_none()
                
                if record is None:
                    error_info = f"Record with ID {record_id} not found."
                    self.logger.error(error_info)
                    raise ValueError(error_info)

                # 更新 rank 字段
                record.score_rank = new_rank
                
                # 提交更改
                session.commit()
                
                # 返回更新后的记录（可选）
                return {key: value for key, value in record.__dict__.items() if key != '_sa_instance_state'}
            
            except Exception as e:
                error_info = f"Failed to update rank for record ID {record_id}: {str(e)}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
    
    def update_health_advice_by_id(self, record_id: int, new_health_advice) -> Optional[Dict[str, Any]]:
        with self.get_db_session() as session:
            try:
                # 查询要更新的记录
                record = session.query(self.model).filter(self.model.id == record_id, self.model.deleted == False).one_or_none()
                
                if record is None:
                    error_info = f"Record with ID {record_id} not found."
                    self.logger.error(error_info)
                    raise ValueError(error_info)

                # 更新 rank 字段
                record.health_advice = new_health_advice
                
                # 提交更改
                session.commit()
                
                # 返回更新后的记录（可选）
                return {key: value for key, value in record.__dict__.items() if key != '_sa_instance_state'}
            
            except Exception as e:
                error_info = f"Failed to update health advice for record ID {record_id}: {str(e)}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
        
    
    def exec_sql(self, query: Optional[str] = None):
        """query words check data"""
        with self.sql_connection() as db:
            try:
                result = db.execute(text(query)).fetchall()
                db.commit()
            except Exception as e:
                db.rollback()        
                error_info = f"Failed to execute SQL query: {query}!"
                self.logger.error(f"{traceback.print_exc()}\n{error_info}")
                raise ValueError(error_info) from e
            if result is not None:
                # 将 RowProxy 转换为列表，然后再转换为 NumPy 数组
                numpy_array = np.array(result)
                return numpy_array
            return result