#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/25 12:30
@Author  : weiyutao
@File    : scripts.py
"""
import os
import numpy as np
from datetime import datetime

from whoami.tool.health_report.health_report import HealthReport
ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SQL_CONFIG_PATH = os.path.join(ROOT_DIRECTORY, 'sql_config.yaml')



if __name__ == '__main__':
    health_report = HealthReport(sql_config_path=SQL_CONFIG_PATH, query_date='2024-12-25', device_sn='13D7F349200080712111150807')
    result = health_report.process()
    # print(result)
    # for t in time_list:
    #     print(datetime.utcfromtimestamp((t).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S'))
    # # print(data_list[0][199, 0])
    