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
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
import uvicorn
from dataclasses import dataclass, field
from pydantic import BaseModel, model_validator, ValidationError
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
    Type,
    Any
)


from whoami.tool.health_report.health_report import HealthReport
from whoami.tool.health_report.sleep_indices import SleepIndices
from whoami.utils.log import Logger
from whoami.utils.R import R
logger = Logger('health_report_fastapi')
app = FastAPI()

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SQL_CONFIG_PATH = os.path.join(ROOT_DIRECTORY, 'sql_config.yaml')

@dataclass
class RequestData:
    device_sn: Optional[Union[list, str]] = None
    query_date: Optional[str] = None

@app.post('/sleep_indices')
async def sleep_indices(request_data: RequestData, background_tasks: BackgroundTasks):
    logger.info(request_data)
    try:
        device_sn = request_data.device_sn
        query_date = request_data.query_date
    except Exception as e:
        return R.fail(f"传参错误！{request_data}")

    if not isinstance(device_sn, (list, str)) or not device_sn:
        return R.fail(f"device_sn must be list or string! device_sn: {device_sn}")
    
    if query_date is None:
        return R.fail(f"query date must not be null! query_date: {query_date}")
    
    # 定义一个后台任务函数
    async def process_health_report(device_sn_i, last_flag: int):
        logger.info(f"To start process report for device {device_sn_i}")
        health_report = None
        try:
            health_report = HealthReport(
                sql_config_path=SQL_CONFIG_PATH,
                query_date=query_date,
                device_sn=device_sn_i,
                model=SleepIndices
            )
            result = health_report.process()
        except Exception as e:
            logger.error(e)
        finally:
            logger.info(f"Processed report for device {device_sn_i}")
            if last_flag > 0 and health_report is not None:
                health_report.rank()
                health_report.health_advice()
    
    device_sn_size = 0
    last_flag = 0
    if isinstance(device_sn, str):
        device_sn_size += 1
        background_tasks.add_task(process_health_report, device_sn, 1)
    else:
        for device_sn_i in device_sn:
            device_sn_size += 1
            if device_sn_size == len(device_sn):
                last_flag = 1
            background_tasks.add_task(process_health_report, device_sn_i, last_flag)
    waste_time = device_sn_size * 1.5
    
    return R.success(f"To start process background. It will take approximately {waste_time} minutes.")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)