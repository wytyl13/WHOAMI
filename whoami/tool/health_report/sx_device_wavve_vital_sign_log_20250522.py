#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/23 16:12
@Author  : weiyutao
@File    : sx_device_wavve_vital_sign_log_20250522.py
"""


from sqlalchemy import Column, BigInteger, Float, String, Date, DateTime, Boolean, Integer, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class SxDeviceWavveVitalSignLog(Base):
    __tablename__ = 'sx_device_wavve_vital_sign_log_20250716'

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    device_sn = Column(String(64), nullable=False, index=True, comment='设备SN码')
    breath_bpm = Column(Float, nullable=True, comment='呼吸频率')
    breath_line = Column(Float, nullable=True, comment='呼吸线')
    heart_bpm = Column(Float, nullable=True, comment='心率')
    heart_line = Column(Float, nullable=True, comment='心率线')
    distance = Column(Float, nullable=True, comment='距离')
    signal_intensity = Column(Float, nullable=True, comment='信号强度')
    state = Column(Float, nullable=True, comment='置信度')
    in_out_bed = Column(Integer, nullable=True, comment='在床离床 【0:离床，1:在床】')
    body_move_data = Column(Float, nullable=True, comment='体动能量值')
    create_date = Column(Date, nullable=True, comment='创建日期')
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, nullable=True, server_default=func.now(), comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, nullable=True, server_default=func.now(), onupdate=func.now(), comment='更新时间')
    deleted = Column(LargeBinary(1), nullable=True, server_default='0', comment='是否删除')  # bit(1) 类型对应 LargeBinary(1)
    tenant_id = Column(BigInteger, nullable=False, default=0, comment='租户编号')

    __table_args__ = {
        'comment': '设备生命体征日志表'
    }