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
import time
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from whoami.tool.disease_predict.sx_device_info import SxDeviceInfo
from whoami.tool.disease_predict.disease_predict import DiseasePredict
from whoami.provider.sql_provider import SqlProvider
from whoami.utils.log import Logger
from whoami.utils.R import R
logger = Logger('disease_predict_fastapi')
app = FastAPI()

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SQL_CONFIG_PATH = os.path.join(ROOT_DIRECTORY, 'sql_config.yaml')

@dataclass
class RequestData:
    dept_id: Optional[str] = None
    sick_bed_id: Optional[str] = None
    
sql_provider = SqlProvider(model=SxDeviceInfo, sql_config_path=SQL_CONFIG_PATH)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    # 记录原始请求体
    body = await request.body()
    print(f"Invalid request body: {body.decode()}")
    
    # 返回自定义错误响应
    return JSONResponse(
        status_code=422,
        content={
            "detail": f"Invalid request body: {body.decode()}",
            "errors": exc.errors(),
            "body": exc.body,
        },
    )

@app.post('/disease_predict')
async def disease_predict(request_data: RequestData):
    logger.info(request_data)
    logger.info(sql_provider)
    
    try:
        sick_bed_id = request_data.sick_bed_id
        dept_id = request_data.dept_id
    except Exception as e:
        return R.fail(f"传参错误！{request_data}")
    
    if sick_bed_id is None or dept_id is None:
        return R.fail("dept_id and sick_bed_id must not be null!")
    disease_predict = DiseasePredict(sql_config_path=SQL_CONFIG_PATH, sql_provider=sql_provider, model=SxDeviceInfo)
    result = disease_predict._run(dept_id, sick_bed_id)
    return R.success(result)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=48080)
