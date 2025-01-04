#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/04 18:09
@Author  : weiyutao
@File    : standard_breath_heart.py
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Boolean, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from sqlalchemy import Text

Base = declarative_base()

class StandardBreathHeart(Base):
    __tablename__ = 'sx_device_wavve_vital_sign_config'
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键id')
    device_sn = Column(String(255), comment='设备SN码')
    breath_bpm_low = Column(Float, comment='呼吸频率阈值低')
    breath_bpm_high = Column(Float, comment='呼吸频率阈值高')
    heart_bpm_low = Column(Float, comment='心率阈值低')
    heart_bpm_high = Column(Float, comment='心率阈值高')
    alarm_time_interval = Column(Integer, comment='告警时间间隔 单位：分钟')
    creator = Column(String(64), comment='创建者')
    create_time = Column(DateTime, default=datetime.now, comment='创建时间')
    updater = Column(String(64), comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')
    deleted = Column(Boolean, default=False, comment='是否删除')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')