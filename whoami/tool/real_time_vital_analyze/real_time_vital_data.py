
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/07/05
@Author  : weiyutao  
@File    : sleep_sensor_data.py
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, BigInteger, ForeignKey, BINARY, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class RealTimeVitalData(Base):
    """
    睡眠传感器数据表
    对应传感器实时数据
    """
    __tablename__ = 'sx_real_time_vital_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    device_sn = Column(String(64), nullable=False, comment='设备编号')
    timestamp = Column(Float, nullable=False, comment='业务时间戳')
    breath_bpm = Column(Float, nullable=True, comment='呼吸率')
    breath_line = Column(Float, nullable=True, comment='呼吸线')
    heart_bpm = Column(Float, nullable=True, comment='心率')
    heart_line = Column(Float, nullable=True, comment='心线')
    target_distance = Column(Float, nullable=True, comment='目标距离')
    signal_strength = Column(Float, nullable=True, comment='信号强度')
    valid_bit_id = Column(Integer, nullable=True, comment='有效位标识')
    body_move_energy = Column(Float, nullable=True, comment='体动能量')
    body_move_range = Column(Float, nullable=True, comment='体动范围')
    in_bed = Column(Integer, nullable=False, default=0, comment='在床状态(0:不在床, 1:在床)')
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')
    deleted = Column(BINARY(1), default=b'0', nullable=True, comment='是否删除')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')