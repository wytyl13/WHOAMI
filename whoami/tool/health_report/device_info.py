
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/18 17:55
@Author  : weiyutao
@File    : device_info.py
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Boolean, BigInteger, BINARY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from sqlalchemy import Text

Base = declarative_base()
class DeviceInfo(Base):
    """
    设备信息表
    对应 SX_DEVICE_INFO 表
    """
    __tablename__ = 'sx_device_info'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    device_sn = Column(String(128), nullable=False, index=True, comment='设备SN码')
    device_type = Column(String(64), nullable=False, comment='设备类型')
    device_name = Column(String(64), nullable=True, comment='设备名称')
    dept_type = Column(String(255), nullable=True, comment='部门类型')
    dept_id = Column(BigInteger, nullable=False, comment='部门ID')
    family_id = Column(BigInteger, nullable=True, comment='家庭ID')
    room_id = Column(BigInteger, nullable=True, comment='房间ID')
    bed_id = Column(BigInteger, nullable=True, comment='床位ID')
    elderly_id = Column(BigInteger, nullable=True, comment='老人ID')
    device_kind = Column(String(2), nullable=True, comment='设备种类')
    device_status = Column(Integer, nullable=False, comment='设备状态')
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, default=datetime.now, nullable=True, comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间')
    deleted = Column(Boolean, default=False, comment='是否删除')
    tenant_id = Column(BigInteger, default=0, nullable=False, comment='租户编号')