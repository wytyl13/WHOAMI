#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/07/017 16:30
@Author  : weiyutao
@File    : community_real_time_data.py
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class CommunityRealTimeData(Base):
    """
    社区实时数据表
    对应 community_real_time_data 表
    """
    __tablename__ = 'community_real_time_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键ID')
    type = Column(String(32), nullable=True, comment='类型')
    content = Column(Text, nullable=True, comment='内容')
    url = Column(String(255), nullable=True, comment='链接地址')
    creator = Column(String(64), nullable=True, comment='创建人')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新人')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')