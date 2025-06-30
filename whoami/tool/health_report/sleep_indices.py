#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/26 16:16
@Author  : weiyutao
@File    : sleep_indices.py
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Boolean, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from sqlalchemy import Text

Base = declarative_base()

class SleepIndices(Base):
    __tablename__ = 'sx_device_wavve_sleep_indices'
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键id')
    average_heart_bpm = Column(Integer, comment='平均心率')
    average_breath_bpm = Column(Integer, comment='平均呼吸率')
    total_num_hour = Column(String(255), comment='总监测时长（小时）')
    total_num_hour_on_bed = Column(String(255), comment='总在床时长（小时）')
    sleep_hour = Column(String(255), comment='睡眠时长（小时）')
    deep_sleep_hour = Column(String(255), comment='深度睡眠时长（小时）')
    waking_hour = Column(String(255), comment='清醒时长（小时）')
    to_sleep_hour = Column(String(255), comment='入睡时长（小时）')
    leave_bed_total_hour = Column(String(255), comment='离床总时间（小时）')
    light_sleep_hour = Column(String(255), comment='浅睡时长（小时）')
    total_num_second = Column(Integer, comment='总监测时长（秒）')
    total_num_second_on_bed = Column(Integer, comment='总在床时长（秒）')
    sleep_second = Column(Integer, comment='睡眠时长（秒）')
    deep_sleep_second = Column(Integer, comment='深度睡眠时长（秒）')
    waking_second = Column(Integer, comment='清醒时长（秒）')
    to_sleep_second = Column(Integer, comment='入睡时长（秒）')
    leave_bed_total_second = Column(Integer, comment='离床总时间（秒）')
    light_sleep_second = Column(Integer, comment='浅睡时长（秒）')
    leave_count = Column(Integer, comment='离床次数')
    sleep_efficiency = Column(Float, comment='睡眠效率')
    deep_sleep_efficiency = Column(Float, comment='深睡效率')
    score = Column(Float, comment='评分')
    score_name = Column(String(255), comment='评分归类')
    consist_count_waking = Column(Integer, comment='连续?晚夜醒时长超过31分钟')
    consist_count_sleep_efficiency = Column(Integer, comment='连续?晚睡眠效率小于80%')
    query_date = Column(Date, comment='查询日期')
    save_file_path = Column(String(255), comment='睡眠阶段划分-心率折线图')
    creator = Column(String(64), comment='创建者')
    create_time = Column(DateTime, default=datetime.now, comment='创建时间')
    updater = Column(String(64), comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')
    deleted = Column(Boolean, default=False, comment='是否删除')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')
    waking_count = Column(Integer, comment='夜醒次数（次）')
    on_bed_time = Column(DateTime, comment='上床时间（节点）')
    sleep_time = Column(DateTime, comment='入睡时间（节点）')
    waking_time = Column(DateTime, comment='醒来时间（节点）')
    sleep_stage_image_x_y = Column(Text, comment='睡眠分区绘图')
    max_breath_bpm = Column(Integer, comment='最大呼吸率')
    min_breath_bpm = Column(Integer, comment='最小呼吸率')
    max_heart_bpm = Column(Integer, comment='最大心率')
    min_heart_bpm = Column(Integer, comment='最小心率')
    body_move_count = Column(Integer, comment='体动次数')
    body_move_exponent = Column(Float, comment='体动指数')
    average_body_move_count = Column(Integer, comment='平均体动次数')
    max_body_move_count = Column(Integer, comment='最大体动次数')
    min_body_move_count = Column(Integer, comment='最小体动次数')
    breath_bpm_status = Column(String(64), comment='呼吸率状态')
    heart_bpm_status = Column(String(64), comment='心率状态')
    body_move_status = Column(String(64), comment='体动状态')
    body_move_image_x_y = Column(Text, comment='体动绘图')
    breath_exception_image_sixty_x_y = Column(Text, comment='典型呼吸异常事件')
    breath_exception_count = Column(Integer, comment='呼吸异常次数')
    breath_exception_exponent = Column(Float, comment='呼吸异常指数')
    breath_exception_image_x_y = Column(Text, comment='呼吸异常绘图')
    breath_bpm_image_x_y = Column(Text, comment='呼吸率绘图数据')
    heart_bpm_image_x_y = Column(Text, comment='心率绘图数据')
    leave_bed_time=Column(DateTime, comment='离床时间')
    device_sn=Column(String(255), comment='设备编号')
    score_rank=Column(Float, comment='超越人数百分比')
    health_advice=Column(Text, comment='睡眠建议')
    deep_health_advice=Column(Text, comment='详细睡眠建议')
    