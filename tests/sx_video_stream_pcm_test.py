#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/04 14:10
@Author  : weiyutao
@File    : sx_video_stream_pcm_test.py
"""
import os
import pytest
from fastapi import FastAPI
from dataclasses import dataclass, field
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
import os
import ctypes
import threading
import uvicorn
import argparse
import time
import copy
import signal
import sys
from typing import Dict
import torch
import gc
from queue import Queue

from whoami.tool.detect.sx_video_stream_pcm import SxVideoStreamPCM
from whoami.configs.detector_config import DetectorConfig
from whoami.tool.base.consumer_tool_pool import ConsumerToolPool
from whoami.tool.base.model_info import ModelInfo
from whoami.tool.detect.ultralitics_detector import UltraliticsDetector
from whoami.utils.R import R

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = '/work/ai/WHOAMI/whoami/scripts/detect/detect_config.yaml'
app = FastAPI()
CONFIG = DetectorConfig.from_file(CONFIG_PATH).__dict__
TOPIC_DICT = CONFIG['topics']

topic_list_flag = False


@dataclass
class RequestData:
    device_sn: str = None
    video_stream_url: str = ""
    sampling_interval: float = 0.3
    topic_list: list = field(default_factory=lambda: '/fire/smoke/warning')
    base64_flag: int = 0
    mqtt_flag: int = 0

@pytest.mark.parametrize(
    "device_sn, topic_list",
    [
        pytest.param(
            'BD3202818', ["/fire/smoke/warning"]
        ),
    ]
)
def test_sx_video_stream_pcm(
    device_sn,
    topic_list
):
    conf_dict = CONFIG["conf"]
    model_path_dict = CONFIG["model_path"]
    class_list_dict = CONFIG["class_list"]
    topic_list = TOPIC_DICT
    
        
    # model_paths = {
    #     "/fire/smoke/warning": ModelInfo("/work/ai/WHOAMI/whoami/models/detect/fire_smoke_yolov10m_v2_epochs_250.pt", UltraliticsDetector, [0, 1], 0.5),
    #     "/fallen/falling/warning": ModelInfo("/work/ai/WHOAMI/whoami/models/detect/fall_yolov10_7000_218.pt", UltraliticsDetector, [0], 0.91),
    #     "/mouse/warning": ModelInfo("/work/ai/WHOAMI/whoami/models/detect/mouse_yolov10l_epochs_250.pt", UltraliticsDetector, [0], 0.95),
    #     "/violence/warning": ModelInfo("/work/ai/WHOAMI/whoami/models/detect/fight_yolov10m_199_epoch.pt", UltraliticsDetector, [0], 0.97),
    # }
    
    model_paths = {}
    for conf_key, conf_value in conf_dict.items():
        for topic_name in topic_list:
            topic_key = conf_key + topic_name
            model_paths[topic_key] = ModelInfo(
                model_path="/work/ai/WHOAMI/whoami/"+model_path_dict[topic_name],
                model_type_class=UltraliticsDetector,
                classes=class_list_dict[topic_name],
                conf=conf_value[topic_name]
            )
    print(f"model_paths: --------------------------------------\n {model_paths}")
    consumer_tool_pool = ConsumerToolPool(model_paths=model_paths)
    sx_video_stream_pcm = SxVideoStreamPCM(consumer_tool_pool=consumer_tool_pool)

    
    @app.get('/list_all_topic')
    async def list_all_topic():
        return R.success(TOPIC_DICT)

    @app.post('/fire_smoke_warning')
    async def warning_fastapi(request_data: RequestData, background_tasks: BackgroundTasks):
        # logger.info(request_data)
        try:
            video_stream_url = request_data.video_stream_url
            device_sn = request_data.device_sn
            sampling_interval = request_data.sampling_interval
            topic_list = request_data.topic_list
        except Exception as e:
            return R.fail(f"传参错误！{request_data}")

        current_memory = sx_video_stream_pcm.memory_monitor.check_memory_usage()
        print(f"current_memory: ----------------------------------------- {current_memory}")
        def run_process():
            sx_video_stream_pcm._run(
                topic_dict=TOPIC_DICT, 
                device_sn=device_sn, 
                topic_list=topic_list,
                topic_list_flag=topic_list_flag
            )
        background_tasks.add_task(run_process)
        return R.success("start process")
    
    uvicorn.run(app, host='0.0.0.0', port=9999)


    
    
    
