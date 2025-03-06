from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class SxDiseasePredict(Base):
    __tablename__ = 'sx_disease_predict'
    
    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键id')
    device_sn = Column(String(255), comment='设备SN码')
    cardiovascular_disease = Column(Float, comment='心血管疾病')
    respiratory_disease = Column(Float, comment='呼吸系统疾病')
    neurological_disease = Column(Float, comment='神经系统疾病')
    sleep_disorder = Column(Float, comment='睡眠障碍')
    metabolic_disease = Column(Float, comment='代谢性疾病')
    mental_health_disease = Column(Float, comment='心理健康问题')
    infectious_disease = Column(Float, comment='感染性疾病')
    creator = Column(String(64), comment='创建者')
    create_time = Column(DateTime, default=datetime.now, comment='创建时间')
    deleted = Column(Boolean, default=False, comment='是否删除')