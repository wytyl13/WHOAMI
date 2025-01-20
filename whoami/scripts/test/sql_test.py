#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/04 18:33
@Author  : weiyutao
@File    : sql_test.py
"""


from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.standard_breath_heart import StandardBreathHeart
from whoami.tool.health_report.sleep_indices import SleepIndices

if __name__ == '__main__':
    # sql_provider = SqlProvider(model=SleepIndices, sql_config_path='/home/weiyutao/work/WHOAMI/whoami/scripts/health_report/sql_config.yaml')
    # result = sql_provider.get_record_by_condition(
    #     condition={"device_sn": "13CFF349200080712111155D07", "query_date": "2025-1-17"}, 
    #     fields=['id', 'score'],
    #     exclude_fields=['breath_bpm_image_x_y', 'heart_bpm_image_x_y', 'sleep_stage_image_x_y']
    # )
    # print(result)

    standard_breath_heart = SqlProvider(StandardBreathHeart, sql_config_path='/home/weiyutao/work/WHOAMI/whoami/scripts/health_report/sql_config.yaml')
    standard_breath_heart_default_data = standard_breath_heart.get_record_by_condition(condition={"device_sn": "default_config"})[0]
    print(standard_breath_heart_default_data)

    
    