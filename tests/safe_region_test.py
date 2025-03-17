#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/07 14:13
@Author  : weiyutao
@File    : safe_region_test.py
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
from whoami.tool.detect.coordinate_transform import CoordinateTramsform

def test_sx_video_stream_pcm(
):
    safe_zone = [
        (
          1694.0289855072463,
          1284.9565217391305
        ),
        (
          1956.3478260869565,
          882.0579710144928
        ),
        (
          2141.8550724637685,
          590.7536231884058
        ),
        (
          2296.927536231884,
          590.7536231884058
        ),
        (
          2301.2753623188405,
          1289.304347826087
        )
      ]
    
    coordinate_transform = CoordinateTramsform(
        original_image="/work/ai/WHOAMI/tests/c92bf1d61933cf40d3e2d333b4571dca2c8af8adc4c5980d6472df5d6abaa47a.png",
        input_size=(640, 640),
        polygon_points=safe_zone
    )

    # result = coordinate_transform._calculate_transform_params()

    # detector = UltraliticsDetector(model_path="/work/ai/WHOAMI/whoami/models/detect/fall_yolov10_7000_218.pt", class_list=[0], conf=0.5)
    # results = detector.predict(image="/work/ai/WHOAMI/tests/c92bf1d61933cf40d3e2d333b4571dca2c8af8adc4c5980d6472df5d6abaa47a.png")
    # results[0].save("test.png")
    # for result in results:
    #     boxes = result.boxes.xyxy
    #     for box in boxes:
    #         print(coordinate_transform.calculate_overlap_ratio(point1=(box[0].item(), box[1].item()), point2=(box[2].item(), box[3].item())))
    result = coordinate_transform.calculate_overlap_ratio(
        point1=(149.39955139160156, 464.7524719238281), 
        point2=(742.2192993164062, 909.7803344726562),
        polygon_points=[
            (
                195.47826086956536,
                337.1304347826088
            ),
            (
                480.9855072463769,
                224.08695652173918
            ),
            (
                841.8550724637682,
                1292.2028985507247
            ),
            (
                302.72463768115955,
                1289.304347826087
            ),
            (
                247.65217391304364,
                1225.536231884058
            ),
            (
                180.98550724637695,
                719.7391304347826
            )
      ]
    )

    print(result)