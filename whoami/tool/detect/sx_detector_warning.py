#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/19 18:00
@Author  : weiyutao
@File    : sx_detector_warning.py
"""
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
)
import cv2
import requests
import time
from datetime import datetime

from whoami.tool.detect.detector_warning import DetectorWarning

class SxDetectorWarning(DetectorWarning):
    url_str_flag: Optional[str] = None,
    device_sn: Optional[str] = None,
    stream_url: Optional[str] = None, # drop this variable.
    topic_name: Optional[str] = None,
    def __init__(
        self, 
        config_path: Optional[str] = None,
        url_str_flag: Optional[str] = None, 
        device_sn: Optional[str] = None,
        stream_url: Optional[str] = None, # drop this variable.
        topic_name: Optional[str] = None,
    ):
        super().__init__(
            config_path=config_path,
        )
        """notice, all valid function will not influence the init value in construct function"""
        self._valid_variable(url_str_flag, device_sn, stream_url, topic_name)
        if hasattr(self, 'logger'):  # 检查是否已经初始化
            return
    
    def tostring(self):
        return {
            "name": self.name,
            "config_path": self.config_path, 
            "pre_warning_time": self.pre_warning_time, 
            "warning_gap": self.warning_gap, 
            "warning_infomation": self.warning_infomation, 
            "url_str_flag": self.url_str_flag,
            "device_sn": self.device_sn,
            "stream_url": self.stream_url,
            "topic_name": self.topic_name
        }
       
    def _valid_variable(self, url_str_flag, device_sn, stream_url, topic_name):
        """valid and init all variable in SxDetectorWarning"""
        self.url_str_flag = url_str_flag
        self.device_sn = device_sn
        self.stream_url = stream_url # drop this variable.
        self.topic_name = topic_name 
        if self.url_str_flag is None or self.device_sn is None or self.topic_name is None:
            raise ValueError('url_str_flag, device_sn, topic_name must not be null!')
        return True
    
    def customer_send_warning(self, warning_information):
        """send warning function implemented by inherited class."""
        
        # upload the image
        upload_url = self.config['upload_url'][self.url_str_flag]
        _, encoded_image = cv2.imencode('.png', warning_information)
        image_bytes = encoded_image.tobytes()
        files = {'file': ('image.png', image_bytes, 'image/png')}
        response = requests.post(upload_url, files=files)
        self.logger.info(response.json())
        if response.status_code != 200:
            raise ValueError("fail to upload warning file!")
        try:
            url_res = response.json()["data"]
            response_data = {
                "deviceSn": self.device_sn,
                "videoStreamUrl": self.stream_url,
                "imageUrl": url_res,
                "alarmType": self.config["topics"][self.topic_name],
                "alarmTime": datetime.fromtimestamp(int(time.time())).isoformat()
            }
        except Exception as e:
            raise ValueError("fail to imageUrl") from e
        if url_res:
            response_data["imageUrl"] = url_res
        else:
            raise ValueError("valid imageUrl!")
                                                                                   
        self.logger.info(f"response_data: {response_data}")
        
        # send the warning information
        warning_url = self.config['warning_url'][self.url_str_flag]
        response_ = requests.post(warning_url, json=response_data)
        if response_.status_code != 200:
            raise ValueError("fail to send warning information!")
        try:
            json_response = response_.json()
        except Exception as e:
            raise ValueError("faile to parse the json response for requesting send warning!") from e
        self.logger.info(json_response)
        return True
        
        