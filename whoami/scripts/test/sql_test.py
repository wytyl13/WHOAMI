#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/01/04 18:33
@Author  : weiyutao
@File    : sql_test.py
"""


from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.standard_breath_heart import StandardBreathHeart

if __name__ == '__main__':
    sql_provider = SqlProvider(model=StandardBreathHeart, sql_config_path='/home/weiyutao/work/WHOAMI/whoami/scripts/health_report/sql_config.yaml')
    result = sql_provider.get_record_by_condition({"device_sn": "default_config"})
    print(result)

    
    