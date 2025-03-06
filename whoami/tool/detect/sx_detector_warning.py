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
import pytz

from whoami.tool.detect.detector_warning import DetectorWarning
tz = pytz.timezone('Asia/Shanghai')

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
        warning_gap: Optional[int] = None
    ):
        super().__init__(
            config_path=config_path,
            warning_gap=warning_gap
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
    
    def customer_send_warning(self, warning_information, device_sn: str = None, stream_url: str = None, topic_name: str = None):
        """send warning function implemented by inherited class."""
        device_sn = self.device_sn if device_sn is None else device_sn
        stream_url = self.stream_url if stream_url is None else stream_url
        topic_name = self.topic_name if topic_name is None else topic_name
        # upload the image
        upload_url = self.config['upload_url'][self.url_str_flag]
        _, encoded_image = cv2.imencode('.png', warning_information)
        image_bytes = encoded_image.tobytes()
        files = {'file': ('image.png', image_bytes, 'image/png')}
        try:
            response = requests.post(upload_url, files=files, timeout=(5, 10))
        except Exception as e:
            self.logger.error(f"fail to upload file, upload_url: {upload_url}, error_info: {str(e)}")
            raise ValueError(f"fail to upload file, upload_url: {upload_url}") from e
        if response.status_code != 200:
            self.logger.error(f"fail to upload warning file, upload_url: {upload_url}, response: {response}")
            raise ValueError(f"fail to upload warning file, upload_url: {upload_url}, response: {response}") from e
        try:
            url_res = response.json()["data"]
            response_data = {
                "deviceSn": device_sn,
                "videoStreamUrl": stream_url,
                "imageUrl": url_res,
                "alarmType": self.config["topics"][topic_name],
                "alarmTime": datetime.fromtimestamp(int(time.time()), tz=tz).isoformat()
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
        try:
            response_ = requests.post(warning_url, json=response_data)
        except Exception as e:
            raise ValueError(f"fail to send warning information! warning_url: {warning_url}") from e
        if response_.status_code != 200:
            raise ValueError(f"fail to send warning information! response_: {response_}")
        try:
            json_response = response_.json()
        except Exception as e:
            raise ValueError("faile to parse the json response for requesting send warning!") from e
        self.logger.info(json_response)
        return True
        
        
