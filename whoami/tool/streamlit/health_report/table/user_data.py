#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/07/02 16:00
@Author  : weiyutao
@File    : user_data.py
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, BigInteger, ForeignKey, BINARY, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class UserData(Base):
    """
    用户数据表
    对应 SX_USER_DATA 表
    """
    __tablename__ = 'sx_user_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    username = Column(String(64), nullable=False, unique=True, comment='用户名')
    password = Column(String(128), nullable=False, comment='密码')
    full_name = Column(String(64), nullable=False, comment='姓名')
    gender = Column(String(8), nullable=True, comment='性别')
    age = Column(Integer, nullable=True, comment='年龄')
    address = Column(String(256), nullable=True, comment='地址')
    phone = Column(String(20), nullable=True, comment='手机号')
    email = Column(String(128), nullable=True, comment='邮箱')
    status = Column(String(16), default='active', nullable=False, comment='用户状态')
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')
    deleted = Column(BINARY(1), default=b'0', nullable=True, comment='是否删除')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')
    role = Column(String(32), nullable=True, comment='角色')
    community = Column(String(128), nullable=True, comment='机构社区')