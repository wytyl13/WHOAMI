#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/28 
@Author  : weiyutao
@File    : sleep_statistics_model.py
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, BigInteger, ForeignKey, BINARY, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class SleepStatistics(Base):
    """
    睡眠统计数据表
    对应 SX_SLEEP_STATISTICS 表
    """
    __tablename__ = 'sx_sleep_statistics'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    device_sn = Column(String(64), nullable=False, comment='设备序列号')
    sleep_start_time = Column(DateTime, nullable=False, comment='睡眠区间起始时间')
    sleep_end_time = Column(DateTime, nullable=False, comment='睡眠区间终止时间')
    
    # 生理指标
    avg_breath_rate = Column(Float, nullable=True, comment='平均呼吸率')
    avg_heart_rate = Column(Float, nullable=True, comment='平均心率')
    heart_rate_variability = Column(Float, nullable=True, comment='心率变异性系数')
    
    # 行为统计
    body_movement_count = Column(Integer, nullable=True, comment='体动次数')
    apnea_count = Column(Integer, nullable=True, comment='呼吸暂停次数')
    rapid_breathing_count = Column(Integer, nullable=True, comment='呼吸急促次数')
    leave_bed_count = Column(Integer, nullable=True, comment='离床次数')
    
    # 时长统计 (以小时字符串格式存储，如"6小时49分22秒")
    total_duration = Column(String(32), nullable=True, comment='统计总时长(小时字符串格式)')
    in_bed_duration = Column(String(32), nullable=True, comment='在床时长(小时字符串格式)')
    out_bed_duration = Column(String(32), nullable=True, comment='离床时长(小时字符串格式)')
    deep_sleep_duration = Column(String(32), nullable=True, comment='深睡眠时长(小时字符串格式)')
    light_sleep_duration = Column(String(32), nullable=True, comment='浅睡眠时长(小时字符串格式)')
    awake_duration = Column(String(32), nullable=True, comment='清醒时长(小时字符串格式)')
    
    # 系统字段
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')
    deleted = Column(BINARY(1), default=b'0', nullable=True, comment='是否删除')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')