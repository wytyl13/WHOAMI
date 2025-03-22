#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/18 17:56
@Author  : weiyutao
@File    : institution_elderly_bed.py
"""



from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, BigInteger, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional, List, Dict, Any

Base = declarative_base()

class InstitutionElderlyBed(Base):
    """
    老人床位信息表
    对应 SX_INSTITUTION_ELDERLY_BED 表
    """
    __tablename__ = 'sx_institution_elderly_bed'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    dept_id = Column(BigInteger, nullable=True, comment='部门ID')
    institution_room_id = Column(BigInteger, nullable=True, comment='房间ID')
    institution_bed_id = Column(BigInteger, nullable=False, comment='床位ID')
    elderly_id = Column(BigInteger, nullable=False, comment='老人ID')
    elderly_name = Column(String(64), nullable=False, comment='老人姓名')
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')
    deleted = Column(Boolean, default=False, comment='是否删除')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')