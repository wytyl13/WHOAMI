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
from datetime import datetime, date
from sqlalchemy.exc import IntegrityError


from whoami.provider.base_provider import BaseProvider
from whoami.configs.sql_config import SqlConfig
from whoami.tool.health_report.device_info import DeviceInfo
from whoami.tool.health_report.institution_elderly_bed import InstitutionElderlyBed
from whoami.tool.health_report.elderly_info import ElderlyInfo

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
        self.sql_connection = self.get_sql_connection() if self.sql_connection is None else self.sql_connection
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
    
    
    def bulk_insert_with_update(self, data_list: List[Dict[str, Any]]) -> int:
        """批量插入，遇到重复数据时覆盖旧数据"""
        if not data_list:
            return 0
        
        try:
            from sqlalchemy import text
            
            # 获取表名
            table_name = self.model.__tablename__
            
            # 构建字段列表（排除自增主键id）
            sample_data = data_list[0]
            columns = [col for col in sample_data.keys() if col != 'id']
            columns_str = ', '.join(columns)
            
            # 构建VALUES占位符
            values_placeholder = ', '.join([f':{col}' for col in columns])
            
            # 构建UPDATE部分（覆盖所有字段）
            update_assignments = []
            for col in columns:
                update_assignments.append(f'{col} = VALUES({col})')
            update_str = ', '.join(update_assignments)
            
            # 构建完整SQL
            sql = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({values_placeholder})
            ON DUPLICATE KEY UPDATE {update_str}
            """
            
            success_count = 0
            with self.get_db_session() as session:
                for data in data_list:
                    try:
                        # 移除id字段（如果存在）
                        clean_data = {k: v for k, v in data.items() if k != 'id'}
                        session.execute(text(sql), clean_data)
                        success_count += 1
                    except Exception as e:
                        self.logger.error(f"插入失败: {e}")
                        continue
                
                session.commit()
            
            self.logger.info(f"批量插入/更新完成: {success_count}/{len(data_list)} 条成功")
            return success_count
            
        except Exception as e:
            self.logger.error(f"批量插入/更新失败: {e}")
            return 0
    
    
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


    def update_record_enhanced(self, record_id: int, data: Dict[str, Any], return_updated: bool = True) -> Optional[Dict[str, Any]]:
        """
        增强版更新记录函数
        
        Args:
            record_id (int): 要更新的记录ID
            data (Dict[str, Any]): 包含要更新字段的字典
            return_updated (bool): 是否返回更新后的记录，默认为True
            
        Returns:
            Optional[Dict[str, Any]]: 如果return_updated为True，返回更新后的记录字典；否则返回None
            
        Raises:
            ValueError: 当记录不存在、数据为空或更新失败时抛出
        """
        with self.get_db_session() as session:
            try:
                if not data:
                    raise ValueError("更新数据不能为空")
                
                # 查询要更新的记录
                record = session.query(self.model).filter(
                    self.model.id == record_id,
                    self.model.deleted == False
                ).first()
                
                if not record:
                    raise ValueError(f"ID为 {record_id} 的记录不存在或已被删除")
                
                # 过滤掉不存在的字段
                valid_data = {}
                invalid_fields = []
                
                for key, value in data.items():
                    if hasattr(self.model, key):
                        # 跳过主键字段
                        if key != 'id':
                            valid_data[key] = value
                    else:
                        invalid_fields.append(key)
                
                if invalid_fields:
                    self.logger.warning(f"以下字段在模型中不存在，将被忽略: {invalid_fields}")
                
                if not valid_data:
                    raise ValueError("没有有效的字段需要更新")
                
                # 执行更新操作
                result = session.query(self.model).filter(
                    self.model.id == record_id,
                    self.model.deleted == False
                ).update(valid_data)
                
                if result == 0:
                    raise ValueError(f"更新失败，记录ID {record_id} 不存在")
                
                # 如果需要返回更新后的记录
                if return_updated:
                    session.commit()
                    updated_record = session.query(self.model).filter(
                        self.model.id == record_id
                    ).first()
                    
                    if updated_record:
                        return {
                            key: value for key, value in updated_record.__dict__.items() 
                            if key != '_sa_instance_state'
                        }
                
                return None
                
            except Exception as e:
                session.rollback()
                error_info = f"更新记录失败 ID: {record_id}, 数据: {data}, 错误: {str(e)}"
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
    
    
    def get_record_by_condition_bake(
        self, 
        condition: Optional[Dict[str, Any]],
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None
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

                # Apply date range filter
                if date_range:
                    date_field = date_range.get('date_field')
                    start_date_str = date_range.get('start_date')
                    end_date_str = date_range.get('end_date')

                    if not date_field:
                        raise ValueError("date_field must be specified for date range filtering")

                    # Convert string dates to datetime objects
                    try:
                        if start_date_str:
                            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                            query = query.filter(getattr(self.model, date_field) >= start_date)
                        
                        if end_date_str:
                            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                            query = query.filter(getattr(self.model, date_field) <= end_date)
                    except ValueError as e:
                        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. {str(e)}")


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
    
    
    def get_record_by_condition(
        self, 
        condition: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        增强版条件查询函数 - 支持精确到秒的时间查询
        
        Args:
            condition: 查询条件字典 {'device_id': 'DEV001', 'state': '呼吸暂停'}
            fields: 指定返回字段列表 ['timestamp', 'state', 'heart_bpm']
            exclude_fields: 排除字段列表 ['id', 'create_time']
            date_range: 日期范围查询
                {
                    'date_field': 'timestamp',  # 日期字段名
                    'start_date': '2025-06-27 15:30:45',  # 开始时间
                    'end_date': '2025-06-27 16:30:45'     # 结束时间
                }
        
        支持的时间格式：
            - '2025-06-27 15:30:45' (精确到秒)
            - '2025-06-27 15:30' (精确到分钟)
            - '2025-06-27' (整天范围)
            - '1751011266.382772' (时间戳)
        
        Returns:
            List[Dict]: 查询结果列表
        """
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

                # 应用基础查询条件
                if condition:
                    for key, value in condition.items():
                        # 支持范围查询
                        if isinstance(value, dict) and 'min' in value or 'max' in value:
                            field_attr = getattr(self.model, key)
                            if 'min' in value:
                                query = query.filter(field_attr >= value['min'])
                            if 'max' in value:
                                query = query.filter(field_attr <= value['max'])
                        # 支持列表查询 (IN 操作)
                        elif isinstance(value, (list, tuple)):
                            query = query.filter(getattr(self.model, key).in_(value))
                        # 普通等值查询
                        else:
                            query = query.filter(getattr(self.model, key) == value)

                # 🔧 增强版日期范围过滤 - 支持精确到秒
                if date_range:
                    date_field = date_range.get('date_field')
                    start_date_str = date_range.get('start_date')
                    end_date_str = date_range.get('end_date')

                    if not date_field:
                        raise ValueError("date_field must be specified for date range filtering")

                    try:
                        if start_date_str:
                            start_date = self._parse_datetime_unified(start_date_str, is_end_date=False)
                            query = query.filter(getattr(self.model, date_field) >= start_date)
                        
                        if end_date_str:
                            end_date = self._parse_datetime_unified(end_date_str, is_end_date=True)
                            query = query.filter(getattr(self.model, date_field) <= end_date)
                    except ValueError as e:
                        raise ValueError(f"Invalid date format. {str(e)}")

                # 执行查询
                records = query.all()

                # 处理查询结果
                if not records:
                    return []
                
                # 返回查询结果
                return [dict(zip(query_fields, record)) for record in records]
                
            except Exception as e:
                error_info = f"Failed to get records by condition: {condition}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e


    def _parse_datetime_unified(self, datetime_str: str, is_end_date: bool = False) -> datetime:
        """
        统一的日期时间解析方法，支持多种格式
        
        支持的格式：
        - '2025-06-27' → 2025-06-27 00:00:00 (开始) 或 2025-06-27 23:59:59 (结束)
        - '2025-06-27 15:30:45' → 2025-06-27 15:30:45
        - '2025-06-27 15:30' → 2025-06-27 15:30:00
        - '1751011266.382772' → 时间戳转换
        - '1751011266' → 整数时间戳转换
        
        Args:
            datetime_str: 时间字符串
            is_end_date: 是否为结束时间（影响只有日期时的处理）
            
        Returns:
            datetime: 解析后的datetime对象
        """
        # 尝试解析时间戳（浮点数）
        try:
            timestamp = float(datetime_str)
            return datetime.fromtimestamp(timestamp)
        except ValueError:
            pass
        
        # 尝试解析整数时间戳
        try:
            timestamp = int(datetime_str)
            return datetime.fromtimestamp(timestamp)
        except ValueError:
            pass
        
        # 定义支持的日期格式（按精确度排序）
        formats = [
            '%Y-%m-%d %H:%M:%S.%f',  # 2025-06-27 15:30:45.123456
            '%Y-%m-%d %H:%M:%S',     # 2025-06-27 15:30:45
            '%Y-%m-%d %H:%M',        # 2025-06-27 15:30
            '%Y-%m-%d',              # 2025-06-27
            '%Y/%m/%d %H:%M:%S',     # 2025/06/27 15:30:45
            '%Y/%m/%d %H:%M',        # 2025/06/27 15:30
            '%Y/%m/%d',              # 2025/06/27
        ]
        
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(datetime_str, fmt)
                
                # 如果只有日期，需要特殊处理
                if fmt in ['%Y-%m-%d', '%Y/%m/%d']:
                    if is_end_date:
                        # 结束日期：设置为当天的23:59:59.999999
                        parsed_date = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                    # 开始日期：保持00:00:00（默认）
                
                return parsed_date
                
            except ValueError:
                continue
        
        # 所有格式都失败
        raise ValueError(
            f"Unsupported datetime format: '{datetime_str}'. "
            f"Supported formats: 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD HH:MM:SS', "
            f"'YYYY-MM-DD HH:MM:SS.fff', 'YYYY/MM/DD...', or timestamp"
        )

    
    
    
    def get_device_info(
        self, 
        device_sn: str = None,
        elderly_name: str = None,
        dept_id: str = None,
        room_id: str = None,
        bed_id: str = None
    ):
        """
        根据条件查询设备和老人的关联信息
        
        查询优先级: device_sn > elderly_name > 其他条件
        
        Args:
            device_sn (str, optional): 设备SN码
            elderly_name (str, optional): 老人姓名
            dept_id (str, optional): 部门ID过滤条件
            room_id (str, optional): 房间ID过滤条件
            bed_id (str, optional): 床位ID过滤条件
        
        Returns:
            list[dict]: 包含设备和老人信息的字典列表
        """
        result = []
        with self.get_db_session() as session:
            try:
                # 1. 优先按device_sn查询
                if device_sn:
                    result = self._query_by_device_sn(session, device_sn)
                    if not result:
                        return [{
                            "device_sn": device_sn,
                            "elderly_name": "",
                            "dept_id": "",
                            "dept_name": "",
                            "room_id": "",
                            "room_name": "",
                            "bed_id": "",
                            "bed_name": ""
                        }]
                # 2. 其次按elderly_name查询
                if elderly_name:
                    result = self._query_by_elderly_name(session, elderly_name, dept_id, room_id, bed_id)
                    if result:
                        return result
                
                # 3. 最后按其他条件组合查询
                if dept_id:
                    # 如果有dept_id和room_id和bed_id，可以精确查询到床位信息
                    if room_id and bed_id:
                        result = self._query_by_bed_info(session, dept_id, room_id, bed_id)
                    # 如果只有dept_id和room_id，查询房间信息
                    elif room_id:
                        result = self._query_by_room_info(session, dept_id, room_id)
                    # 如果只有dept_id，查询部门信息
                    else:
                        result = self._query_by_dept_info(session, dept_id)
                
                return result
            
            except Exception as e:
                self.logger.error(f"查询设备和老人关联信息失败: 错误: {str(e)}")
                return []
    
    
    def _query_by_device_sn(self, session, device_sn: str) -> list[dict]:
        """根据设备SN查询关联信息"""
        result = []
        
        # 查询设备信息
        device_info = session.query(DeviceInfo).filter(
            DeviceInfo.device_sn == device_sn,
        ).first()
        self.logger.info(f"device_info:  {device_info}")
        if not device_info:
            return []
        
        # 获取床位关联的老人信息
        if device_info.bed_id:
            elderly_bed = session.query(InstitutionElderlyBed).filter(
                InstitutionElderlyBed.institution_bed_id == device_info.bed_id,
                InstitutionElderlyBed.deleted == b'0'
            ).first()
            
            if elderly_bed:
                # 查询老人详细信息
                elderly_info = session.query(ElderlyInfo).filter(
                    ElderlyInfo.elderly_name == elderly_bed.elderly_name,
                    DeviceInfo.device_name == "睡眠检测",
                    ElderlyInfo.deleted == b'0'
                ).first()
                
                # 组合结果
                info_dict = self._combine_info(device_info, elderly_bed, elderly_info)
                if info_dict:
                    result.append(info_dict)
        
        return result
        
        
    def _query_by_elderly_name(self, session, elderly_name: str, dept_id: str = None, room_id: str = None, bed_id: str = None) -> list[dict]:
        """根据老人姓名查询关联信息"""
        result = []
        # 构建查询条件
        query = session.query(InstitutionElderlyBed).filter(
            InstitutionElderlyBed.elderly_name == elderly_name,
        )
        
        # 添加额外过滤条件
        if dept_id:
            query = query.filter(InstitutionElderlyBed.dept_id == dept_id)
        if room_id:
            query = query.filter(InstitutionElderlyBed.institution_room_id == room_id)
        if bed_id:
            query = query.filter(InstitutionElderlyBed.institution_bed_id == bed_id)
        
        elderly_bed = query.first()
        if not elderly_bed:
            return []
        
        # 查询老人详细信息
        elderly_info = session.query(ElderlyInfo).filter(
            ElderlyInfo.elderly_name == elderly_bed.elderly_name,
            ElderlyInfo.deleted == b'0'
        ).first()
        
        # 查询设备信息
        device_info = session.query(DeviceInfo).filter(
            DeviceInfo.bed_id == elderly_bed.institution_bed_id,
            DeviceInfo.device_name == "睡眠检测",
            DeviceInfo.deleted == b'0'
        ).first()
        
        # 如果没有找到设备信息，尝试通过room_id查询
        if not device_info and elderly_bed.institution_room_id:
            device_info = session.query(DeviceInfo).filter(
                DeviceInfo.room_id == elderly_bed.institution_room_id,
                DeviceInfo.deleted == b'0'
            ).first()
        
        self.logger.warning(device_info)
        self.logger.warning(elderly_bed)
        self.logger.warning(elderly_info)
        # 组合结果
        info_dict = self._combine_info(device_info, elderly_bed, elderly_info)
        if info_dict:
            result.append(info_dict)
        self.logger.warning(result)
        return result
    
    
    def _query_by_bed_info(self, session, dept_id: str, room_id: str, bed_id: str) -> list[dict]:
        """根据床位信息查询关联信息"""
        result = []
        
        # 查询床位关联的老人信息
        elderly_bed = session.query(InstitutionElderlyBed).filter(
            InstitutionElderlyBed.dept_id == dept_id,
            InstitutionElderlyBed.institution_room_id == room_id,
            InstitutionElderlyBed.institution_bed_id == bed_id,
        ).first()

        if not elderly_bed:
            return []
        
        # 查询老人详细信息
        elderly_info = session.query(ElderlyInfo).filter(
            ElderlyInfo.elderly_name == elderly_bed.elderly_name,
            ElderlyInfo.deleted == b'0'
        ).first()
        
        # 查询设备信息
        device_info = session.query(DeviceInfo).filter(
            DeviceInfo.dept_id == dept_id,
            DeviceInfo.room_id == room_id,
            DeviceInfo.bed_id == bed_id,
            DeviceInfo.device_name == "睡眠检测",
        ).first()
        
        # 组合结果
        info_dict = self._combine_info(device_info, elderly_bed, elderly_info)
        if info_dict:
            result.append(info_dict)
        
        return result
    
    
    def _query_by_room_info(self, session, dept_id: str, room_id: str) -> list[dict]:
        """根据房间信息查询该房间内所有床位和老人信息"""
        result = []
        
        # 查询房间关联的所有床位和老人信息
        elderly_beds = session.query(InstitutionElderlyBed).filter(
            InstitutionElderlyBed.dept_id == dept_id,
            InstitutionElderlyBed.institution_room_id == room_id,
            InstitutionElderlyBed.deleted == b'0'
        ).all()
        
        if not elderly_beds:
            return []
        
        # 遍历所有床位信息，查询详细信息并组合
        for elderly_bed in elderly_beds:
            # 查询老人详细信息
            elderly_info = session.query(ElderlyInfo).filter(
                ElderlyInfo.elderly_name == elderly_bed.elderly_name,
                ElderlyInfo.deleted == b'0'
            ).first()
            
            # 查询设备信息
            device_info = session.query(DeviceInfo).filter(
                DeviceInfo.dept_id == dept_id,
                DeviceInfo.room_id == room_id,
                DeviceInfo.bed_id == elderly_bed.institution_bed_id,
                DeviceInfo.device_name == "睡眠检测",
                DeviceInfo.deleted == b'0'
            ).first()
            
            # 组合结果
            info_dict = self._combine_info(device_info, elderly_bed, elderly_info)
            if info_dict:
                result.append(info_dict)
        
        return result
    
    
    def _query_by_dept_info(self, session, dept_id: str) -> list[dict]:
        """根据部门信息查询该部门下所有人员信息"""
        result = []
        
        # 查询部门下所有老人床位信息
        elderly_beds = session.query(InstitutionElderlyBed).filter(
            InstitutionElderlyBed.dept_id == dept_id,
            InstitutionElderlyBed.deleted == b'0'
        ).all()
        
        if not elderly_beds:
            return []
        
        # 遍历所有床位信息，查询详细信息并组合
        for elderly_bed in elderly_beds:
            # 查询老人详细信息
            elderly_info = session.query(ElderlyInfo).filter(
                ElderlyInfo.elderly_name == elderly_bed.elderly_name,
                ElderlyInfo.deleted == b'0'
            ).first()
            
            # 查询设备信息
            device_info = None
            if elderly_bed.institution_bed_id:
                device_info = session.query(DeviceInfo).filter(
                    DeviceInfo.dept_id == dept_id,
                    DeviceInfo.bed_id == elderly_bed.institution_bed_id,
                    DeviceInfo.device_name == "睡眠检测",
                    DeviceInfo.deleted == b'0'
                ).first()
            
            # 如果没有找到设备信息，尝试通过room_id查询
            if not device_info and elderly_bed.institution_room_id:
                device_info = session.query(DeviceInfo).filter(
                    DeviceInfo.dept_id == dept_id,
                    DeviceInfo.room_id == elderly_bed.institution_room_id,
                    DeviceInfo.device_name == "睡眠检测",
                    DeviceInfo.deleted == b'0'
                ).first()
            
            # 组合结果
            info_dict = self._combine_info(device_info, elderly_bed, elderly_info)
            if info_dict:
                result.append(info_dict)
        
        return result
    
    
    def _combine_info(self, device_info, elderly_bed, elderly_info) -> dict:
        """组合设备、床位和老人信息"""
        if not (device_info or elderly_bed):
            return None
        
        result = {}
        
        # 添加设备信息
        if device_info:
            result.update({
                "device_sn": device_info.device_sn,
                "device_type": device_info.device_type,
                "device_name": device_info.device_name,
                "device_status": device_info.device_status,
                "dept_id": device_info.dept_id,
                "room_id": device_info.room_id,
                "bed_id": device_info.bed_id
            })
        
        # 添加老人床位信息
        if elderly_bed:
            result.update({
                "elderly_id": elderly_bed.elderly_id,
                "elderly_name": elderly_bed.elderly_name,
                "institution_room_id": elderly_bed.institution_room_id,
                "institution_bed_id": elderly_bed.institution_bed_id
            })
        
        # 添加老人详细信息
        if elderly_info:
            result.update({
                "elderly_id_card": elderly_info.elderly_id_card,
                "elderly_sex": elderly_info.elderly_sex,
                "elderly_age": elderly_info.elderly_age,
                "elderly_birthday": elderly_info.elderly_birthday,
                "elderly_address": elderly_info.elderly_address,
                "contacts": elderly_info.contacts
            })
        
        return result
    
    
    def _get_device_based_name(
        self, 
        person_name: str,
        dept_id: str = None,
        room_id: str = None,
        bed_id: str = None
    ):
        # 根据老人姓名查询设备SN的函数
        """
        根据老人姓名查询关联的所有设备SN
        
        Args:
        person_name (str): 老人姓名
        dept_id (str, optional): 部门ID过滤条件
        room_id (str, optional): 房间ID过滤条件
        bed_id (str, optional): 床位ID过滤条件
        
        Returns:
            str: 单个设备SN；如果没有找到则返回空字符串
        """
        with self.get_db_session() as session:
            try:
                # 1. 首先在老人表中查找老人信息
                person_query = session.query(InstitutionElderlyBed)
                person_query = person_query.filter(
                    InstitutionElderlyBed.elderly_name == person_name,
                    InstitutionElderlyBed.deleted == b'0'
                )
                
                # 如果传入了额外的过滤条件，则添加到查询中
                if dept_id is not None:
                    person_query = person_query.filter(InstitutionElderlyBed.dept_id == dept_id)
                if room_id is not None:
                    person_query = person_query.filter(InstitutionElderlyBed.institution_room_id == room_id)
                if bed_id is not None:
                    person_query = person_query.filter(InstitutionElderlyBed.institution_bed_id == bed_id)
                
                person_records = person_query.all()
                
                if not person_records:
                    return ""
                
                for person in person_records:
                    # 策略1: 通过elderly_id直接匹配
                    device_query = session.query(DeviceInfo.device_sn).filter(
                        DeviceInfo.elderly_id == person.elderly_id,
                        DeviceInfo.deleted == b'0'
                    )
                    
                    # 如果传入了dept_id，则添加dept_id过滤条件
                    if dept_id is not None:
                        device_query = device_query.filter(DeviceInfo.dept_id == dept_id)
                    
                    device = device_query.first()
                    if device:
                        return device[0]
                    
                    # 策略2: 通过room_id匹配
                    if person.institution_room_id:
                        device_query = session.query(DeviceInfo.device_sn).filter(
                            DeviceInfo.room_id == person.institution_room_id,
                            DeviceInfo.deleted == b'0'
                        )
                        
                        # 如果传入了dept_id，则添加dept_id过滤条件
                        if dept_id is not None:
                            device_query = device_query.filter(DeviceInfo.dept_id == dept_id)
                        
                        device = device_query.first()
                        if device:
                            return device[0]
                    
                    # 策略3: 通过bed_id匹配
                    if person.institution_bed_id:
                        device_query = session.query(DeviceInfo.device_sn).filter(
                            DeviceInfo.bed_id == person.institution_bed_id,
                            DeviceInfo.deleted == b'0'
                        )
                        
                        # 如果传入了dept_id，则添加dept_id过滤条件
                        if dept_id is not None:
                            device_query = device_query.filter(DeviceInfo.dept_id == dept_id)
                        
                        device = device_query.first()
                        if device:
                            return device[0]
                
                # 如果所有策略都未找到设备，返回空字符串
                return ""
            
            except Exception as e:
                self.logger.error(f"根据老人姓名查询设备失败: {person_name}, 错误: {str(e)}")
                return ""
    
    
    def _get_elderly_name_by_device_sn(
        self, 
        device_sn: str,
        dept_id: str = None,
        room_id: str = None,
        bed_id: str = None
    ):
        """
        根据设备SN查询关联的老人姓名
        
        Args:
            device_sn (str): 设备序列号
            dept_id (str, optional): 部门ID过滤条件
            room_id (str, optional): 房间ID过滤条件
            bed_id (str, optional): 床位ID过滤条件
        
        Returns:
            str: 老人姓名；如果没有找到则返回空字符串
        """
        with self.get_db_session() as session:
            try:
                # 1. 首先在设备信息表中查找设备
                device_query = session.query(DeviceInfo).filter(
                    DeviceInfo.device_sn == device_sn,
                    DeviceInfo.deleted == b'0'
                )
                
                # 添加额外的过滤条件
                if dept_id is not None:
                    device_query = device_query.filter(DeviceInfo.dept_id == dept_id)
                if room_id is not None:
                    device_query = device_query.filter(DeviceInfo.room_id == room_id)
                if bed_id is not None:
                    device_query = device_query.filter(DeviceInfo.bed_id == bed_id)
                
                device = device_query.first()
                
                if device:
                    # 优先使用elderly_name字段
                    if device.elderly_name:
                        return device.elderly_name
                    
                    # 如果elderly_name为空，尝试通过elderly_id查询
                    if device.elderly_id:
                        elderly = session.query(InstitutionElderlyBed).filter(
                            InstitutionElderlyBed.elderly_id == device.elderly_id,
                            InstitutionElderlyBed.deleted == b'0'
                        ).first()
                        
                        if elderly:
                            return elderly.elderly_name
                    
                    # 尝试通过room_id查询
                    if device.room_id:
                        elderly = session.query(InstitutionElderlyBed).filter(
                            InstitutionElderlyBed.institution_room_id == device.room_id,
                            InstitutionElderlyBed.deleted == b'0'
                        ).first()
                        
                        if elderly:
                            return elderly.elderly_name
                    
                    # 尝试通过bed_id查询
                    if device.bed_id:
                        elderly = session.query(InstitutionElderlyBed).filter(
                            InstitutionElderlyBed.institution_bed_id == device.bed_id,
                            InstitutionElderlyBed.deleted == b'0'
                        ).first()
                        
                        if elderly:
                            return elderly.elderly_name
                
                # 如果所有策略都未找到老人，返回空字符串
                return ""
            
            except Exception as e:
                self.logger.error(f"根据设备SN查询老人姓名失败: {device_sn}, 错误: {str(e)}")
                return ""
    
    
    def get_elderly_info_by_device_sn(
        self, 
        device_sn: str,
        dept_id: str = None,
        room_id: str = None,
        bed_id: str = None
    ):
        """
        根据设备SN查询关联的老人完整信息
        
        Args:
            device_sn (str): 设备序列号
            dept_id (str, optional): 部门ID过滤条件
            room_id (str, optional): 房间ID过滤条件
            bed_id (str, optional): 床位ID过滤条件
        
        Returns:
            dict: 老人信息字典，包含姓名、性别、年龄、地址、电话、部门编号、房间编号、床位编号等信息；
                  如果没有找到则返回空字典
        """
        with self.get_db_session() as session:
            try:
                # 1. 首先在设备信息表中查找设备
                device_query = session.query(DeviceInfo).filter(
                    DeviceInfo.device_sn == device_sn,
                    DeviceInfo.deleted == b'0'
                )
                
                # 添加额外的过滤条件
                if dept_id is not None:
                    device_query = device_query.filter(DeviceInfo.dept_id == dept_id)
                if room_id is not None:
                    device_query = device_query.filter(DeviceInfo.room_id == room_id)
                if bed_id is not None:
                    device_query = device_query.filter(DeviceInfo.bed_id == bed_id)
                
                device = device_query.first()
                
                if not device:
                    return {}
                
                # 初始化结果字典，添加设备相关信息
                result = {
                    "device_sn": device_sn,
                    "dept_id": device.dept_id,
                    "room_id": device.room_id,
                    "bed_id": device.bed_id
                }
                
                # 2. 获取老人ID
                elderly_id = device.elderly_id
                
                # 3. 如果设备上有老人ID，直接查询老人信息
                if elderly_id:
                    elderly_info = session.query(ElderlyInfo).filter(
                        ElderlyInfo.id == elderly_id,
                        ElderlyInfo.deleted == b'0'
                    ).first()
                    
                    if elderly_info:
                        result.update({
                            "elderly_id": elderly_id,
                            "elderly_name": elderly_info.elderly_name,
                            "elderly_sex": elderly_info.elderly_sex,
                            "elderly_age": elderly_info.elderly_age,
                            "elderly_address": elderly_info.elderly_address,
                            "contacts": elderly_info.contacts,
                            "elderly_id_card": elderly_info.elderly_id_card
                        })
                        return result
                
                # 4. 如果设备没有直接关联老人ID，尝试通过床位信息查询
                elderly_bed = None
                
                # 尝试通过bed_id查询
                if device.bed_id:
                    elderly_bed = session.query(InstitutionElderlyBed).filter(
                        InstitutionElderlyBed.institution_bed_id == device.bed_id,
                        InstitutionElderlyBed.deleted == b'0'
                    ).first()
                
                # 如果床位信息没有找到，尝试通过room_id查询
                if not elderly_bed and device.room_id:
                    elderly_bed = session.query(InstitutionElderlyBed).filter(
                        InstitutionElderlyBed.institution_room_id == device.room_id,
                        InstitutionElderlyBed.deleted == b'0'
                    ).first()
                
                # 5. 如果找到床位信息，再查询老人详细信息
                if elderly_bed:
                    result.update({
                        "elderly_id": elderly_bed.elderly_id,
                        "elderly_name": elderly_bed.elderly_name
                    })
                    
                    # 查询老人详细信息
                    elderly_info = session.query(ElderlyInfo).filter(
                        ElderlyInfo.id == elderly_bed.elderly_id,
                        ElderlyInfo.deleted == b'0'
                    ).first()
                    
                    if elderly_info:
                        result.update({
                            "elderly_sex": elderly_info.elderly_sex,
                            "elderly_age": elderly_info.elderly_age,
                            "elderly_address": elderly_info.elderly_address,
                            "contacts": elderly_info.contacts,
                            "elderly_id_card": elderly_info.elderly_id_card
                        })
                
                return result
            
            except Exception as e:
                self.logger.error(f"根据设备SN查询老人完整信息失败: {device_sn}, 错误: {str(e)}")
                return {}
    
    
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
    
    
    def update_deep_health_advice_by_id(self, record_id: int, new_health_advice) -> Optional[Dict[str, Any]]:
        with self.get_db_session() as session:
            try:
                # 查询要更新的记录
                record = session.query(self.model).filter(self.model.id == record_id, self.model.deleted == False).one_or_none()
                
                if record is None:
                    error_info = f"Record with ID {record_id} not found."
                    self.logger.error(error_info)
                    raise ValueError(error_info)

                # 更新 rank 字段
                record.deep_health_advice = new_health_advice
                
                # 提交更改
                session.commit()
                
                # 返回更新后的记录（可选）
                return {key: value for key, value in record.__dict__.items() if key != '_sa_instance_state'}
            
            except Exception as e:
                error_info = f"Failed to update health advice for record ID {record_id}: {str(e)}"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
    
    
    
    
    def delete_records_by_condition(self, condition: Dict[str, Any]) -> int:
        """
        按照指定条件硬删除多条记录（永久从数据库中删除）
        
        Args:
            condition (Dict[str, Any]): 删除条件，格式为 {字段名: 值}
        
        Returns:
            int: 成功删除的记录数量
        """
        with self.get_db_session() as session:
            try:
                query = session.query(self.model)
                
                # 添加条件过滤，忽略不存在的字段
                valid_conditions = {}
                for key, value in condition.items():
                    if hasattr(self.model, key):
                        query = query.filter(getattr(self.model, key) == value)
                        valid_conditions[key] = value
                    else:
                        self.logger.warning(f"Field '{key}' not found in model, ignoring this condition")
                
                if not valid_conditions:
                    self.logger.warning("No valid conditions found, no records will be deleted")
                    return 0
                    
                # 获取要删除的记录数量
                count_to_delete = query.count()
                
                # 执行硬删除操作
                query.delete(synchronize_session=False)
                
                return count_to_delete
            except Exception as e:
                error_info = f"Failed to delete records by condition: {condition}"
                self.logger.error(error_info)
                self.logger.error(traceback.format_exc())
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