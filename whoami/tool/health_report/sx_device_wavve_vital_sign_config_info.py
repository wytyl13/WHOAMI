from sqlalchemy import Column, BigInteger, String, Float, Integer, DateTime, Boolean, Date, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from sqlalchemy import Text

Base = declarative_base()

class DeviceWavveVitalSignConfigInfo(Base):
    __tablename__ = 'sx_device_wavve_vital_sign_config_info'

    # Primary Key
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    
    # Device Information
    device_sn = Column(String(64), nullable=False, comment='设备SN码')
    
    # Configuration Thresholds
    breath_bpm_low = Column(Float, nullable=True, comment='呼吸率下限')
    breath_bpm_high = Column(Float, nullable=True, comment='呼吸率上限')
    heart_bpm_low = Column(Float, nullable=True, comment='心率下限')
    heart_bpm_high = Column(Float, nullable=True, comment='心率上限')
    
    # Statistics Fields
    min_breath_bpm = Column(Float, nullable=True, comment='最小呼吸率')
    max_breath_bpm = Column(Float, nullable=True, comment='最大呼吸率')
    min_heart_bpm = Column(Float, nullable=True, comment='最小心率')
    max_heart_bpm = Column(Float, nullable=True, comment='最大心率')
    
    # Query Date
    query_date = Column(String(10), nullable=False, index=True, comment='查询日期')
    error_info = Column(Text, nullable=False, index=True, comment='错误信息')
    
    # Audit Fields
    creator = Column(String(64), nullable=True, comment='创建者')
    create_time = Column(DateTime, nullable=True, server_default=func.now(), comment='创建时间')
    updater = Column(String(64), nullable=True, comment='更新者')
    update_time = Column(DateTime, nullable=True, server_default=func.now(), onupdate=func.now(), comment='更新时间')
    
    # System Fields
    deleted = Column(Boolean, nullable=True, server_default='0', comment='是否删除')
    tenant_id = Column(BigInteger, nullable=False, default=0, comment='租户编号')