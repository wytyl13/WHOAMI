#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/27 11:53
@Author  : weiyutao
@File    : sleep_data_state.py
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, BigInteger, ForeignKey, BINARY, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class SleepDataState(Base):
    """
    睡眠数据状态表
    对应 SX_SLEEP_DATA_STATE 表
    """
    __tablename__ = 'sx_sleep_data_state_post'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    device_id = Column(String(64), nullable=False, comment='设备编号')
    timestamp = Column(Float, nullable=False, comment='业务时间戳')
    breath_bpm = Column(Float, nullable=True, comment='呼吸率')
    breath_line = Column(Float, nullable=True, comment='呼吸线')
    heart_bpm = Column(Float, nullable=True, comment='心率')
    heart_line = Column(Float, nullable=True, comment='心线')
    reconstruction_error = Column(Float, nullable=True, comment='重建误差值')
    state = Column(String(32), nullable=False, comment='状态')
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')
    deleted = Column(BINARY(1), default=b'0', nullable=True, comment='是否删除')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')