#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/30 15:37
@Author  : weiyutao
@File    : realtime_health_report.py
"""
from typing import (
    Optional,
    Type    
)

from whoami.provider.base_provider import BaseProvider
from whoami.tool.



class RealtimeHealthReport(BaseProvider):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    sql_provider: Optional[SqlProvider] = None
    data_provider: Optional[SxDataProvider] = None
    query_date: Optional[str] = None
    device_sn: Optional[str] = None
    model: Type[ModelType] = SleepIndices
    standard_breath_heart: Optional[SqlProvider] = None
    breath_bpm_low: Optional[int] = None
    breath_bpm_high: Optional[int] = None
    heart_bpm_low: Optional[int] = None
    heart_bpm_high: Optional[int] = None
    threshold_value: Optional[ThresholdValue] = None
    def __init__(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        sql_provider: Optional[SqlProvider] = None,
        data_provider: Optional[SxDataProvider] = None,
        query_date: Optional[str] = None,
        device_sn: Optional[str] = None,
        model: Type[ModelType] = None
    ) -> None:
        super().__init__()
        self.threshold_value = ThresholdValue(sql_config_path=sql_config_path, sql_config=sql_config, sql_provider=sql_provider, model=model, device_sn=device_sn)
        self._init_param(sql_config_path, sql_config, sql_provider, data_provider, query_date, device_sn, model=model)
        # 查询是否已经存在，否则直接返回
        first_check_condition = {"device_sn": self.device_sn, "query_date": self.query_date}
        check_fields = ['id']
        record_ = self.data_provider.sql_provider.get_record_by_condition(condition=first_check_condition, fields=check_fields)
        if record_:
            raise ValueError(f"exists! {first_check_condition}")
        
        # init the breath heart param
        try:
            self.standard_breath_heart = SqlProvider(StandardBreathHeart, sql_config_path=self.sql_config_path)
            records = self.standard_breath_heart.get_record_by_condition(condition={"device_sn": "default_config"})
            if not records:
                raise ValueError("找不到默认呼吸心率配置")
            standard_breath_heart_default_data = records[0]
            self.breath_bpm_low = standard_breath_heart_default_data["breath_bpm_low"] # read from sql where table is sx_device_wavve_vital_sign_config
            self.breath_bpm_high = standard_breath_heart_default_data["breath_bpm_high"] # read from sql where table is sx_device_wavve_vital_sign_config
            self.heart_bpm_low = standard_breath_heart_default_data["heart_bpm_low"] # read from sql where table is sx_device_wavve_vital_sign_config
            self.heart_bpm_high = standard_breath_heart_default_data["heart_bpm_high"] # read from sql where table is sx_device_wavve_vital_sign_config
        except Exception as e:
            raise ValueError("fail to init breath heart bpm low and high!") from e
        
    
    
    def _init_param(self, sql_config_path, sql_config, sql_provider, data_provider, query_date, device_sn, model):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.sql_provider = sql_provider
        self.data_provider = data_provider if data_provider is not None else self.data_provider
        self.query_date = query_date
        self.device_sn = device_sn
        self.model=model if model is not None else self.model
        
        if self.sql_config_path is None and self.sql_config is None and self.data_provider is None:
            raise ValueError('sql_config_path, sql_config, data_provider must not be none!')
        if self.data_provider is None:
            if self.model is None:
                raise ValueError('model must not be null!')
            if self.query_date is not None and self.device_sn is not None:
                current_date = datetime.strptime(self.query_date, '%Y-%m-%d')
                current_date_str = current_date.strftime('%Y-%m-%d')
                pre_date_str = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
                start = pre_date_str + ' 19:00:00'
                end = current_date_str + ' 07:00:00'
                self.logger.info(start)
                self.logger.info(end)
                sql_query = f"SELECT in_out_bed, signal_intensity, breath_line, heart_line, breath_bpm, heart_bpm, state, body_move_data, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log WHERE device_sn='{self.device_sn}' AND create_time >= '{start}' AND create_time < '{end}'"
                # sql_query = f"SELECT in_out_bed, distance, breath_line, heart_line, breath_bpm, heart_bpm, state, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log WHERE device_sn='{self.device_sn}' AND create_time >= '{start}' AND create_time < '{end}'"
                self.data_provider = SxDataProvider(sql_config_path=self.sql_config_path, sql_config=self.sql_config, sql_provider=self.sql_provider, sql_query=sql_query, model=self.model)
            else:
                self.data_provider = SxDataProvider(sql_config_path=self.sql_config_path, sql_config=self.sql_config, sql_provider=self.sql_provider, model=self.model)

    def init_data(self, batch_size: Optional[int] = 60*60*14):
        dataloader = DataLoader(self.data_provider, batch_size=batch_size, shuffle=False)
        data_list = []
        for batch in dataloader:
            float_array = batch.numpy()
            # float_array = float_array[float_array[:, 1] != 0]
            data_list.append(float_array)
        return data_list