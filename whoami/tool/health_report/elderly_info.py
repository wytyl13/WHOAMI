#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/20 10:21
@Author  : weiyutao
@File    : elderly_info.py
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, BigInteger, ForeignKey, BINARY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional, List, Dict, Any

Base = declarative_base()

class ElderlyInfo(Base):
    """
    老人基本信息表
    对应 SX_ELDERLY_INFO 表
    """
    __tablename__ = 'sx_elderly_info'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    dept_id = Column(BigInteger, nullable=True, comment='部门ID')
    family_id = Column(BigInteger, nullable=True, comment='家庭ID')
    dept_type = Column(String(32), nullable=True, comment='部门类型')
    elderly_name = Column(String(32), nullable=False, comment='老人姓名')
    elderly_id_card = Column(String(32), nullable=False, comment='老人身份证号')
    elderly_nation = Column(String(32), nullable=True, comment='老人民族')
    elderly_sex = Column(String(2), nullable=True, comment='老人性别')
    elderly_age = Column(Integer, nullable=True, comment='老人年龄')
    elderly_birthday = Column(String(32), nullable=True, comment='老人生日')
    elderly_address = Column(String(32), nullable=True, comment='老人地址')
    contacts = Column(String(255), nullable=True, comment='联系人信息')
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')
    deleted = Column(BINARY(1), default=b'0', nullable=True, comment='是否删除')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')