#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/18 17:44
@Author  : weiyutao
@File    : sx_video_stream_detector.py
"""
from pydantic import BaseModel, model_validator, ValidationError, field_validator, validator
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
from abc import ABC, abstractmethod
import os
import requests
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import ast
# import ray

from whoami.tool.detect.video_stream_detector import VideoStreamDetector
from whoami.tool.detect.detector_warning import DetectorWarning
from whoami.tool.detect.detector import Detector
from whoami.utils.log import Logger
from whoami.utils.utils import Utils
from whoami.tool.detect.sx_detector_warning import SxDetectorWarning
from whoami.tool.detect.ultralitics_detector import UltraliticsDetector
from whoami.configs.detector_config import DetectorConfig 

utils = Utils()

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# ray.init()

# @ray.remote # faild to use ray to implemented the multi threads. because the pydantic version.
class SxVideoStreamDetector(VideoStreamDetector):
    # 注意：如果子类的特有属性在父类的构造函数中被使用了，应用无效，因为父类的构造函数在子类的构造函数之前执行
    # 解决办法是在子类的构造函数中将该属性初始化在调用父类构造函数之前
    
    def __init__(
        self, 
        name: Optional[str] = None,
        detector_warning: Optional[DetectorWarning] = None,
        device_sn: Optional[str] = None,
        detector: Optional[Detector] = None,
        stream_url: Optional[str] = None,
        url_str_flag: Optional[str] = None,
        config_path: Optional[str] = None,
        topic_name: Optional[str] = None
    ):
        """notice, all valid function will not influence the init value in construct function"""
        """notice, the special attribution url_str_flag for child class SxVideoStreamDetector must init before the super function."""
        """because we have overwrite the method get_sql_connection and call it in the construction function of father class VideoStreamDetector"""
        """if not do like this, will fail to use the url_str_flag value in the construction of child class SxVideoStreamDetector."""
        """but i have failed to do that, so we should reset the correspond attribution in the construction of child class."""
        super().__init__(
            name=name, 
            detector_warning=detector_warning, 
            device_sn=device_sn, 
            stream_url=stream_url, 
            url_str_flag=url_str_flag,
            detector=detector,
            config_path=config_path,
            topic_name=topic_name
        )
        # self.url_str_flag = url_str_flag if url_str_flag is not None else self.url_str_flag
        # if you init the special attribution for child class. you should consider whether overwrite some method.
        # self.sql_connection = self.get_sql_connection() if url_str_flag is not None else self.sql_connection
        # self.detector_warning = self._valid_detector_warning()
        if hasattr(self, 'logger'):  # 检查是否已经初始化
            return
    

    
    def tostring(self):
        return {
            "name": self.name,
            "detector_warning": self.detector_warning.tostring(), 
            "device_sn": self.device_sn, 
            "detector": self.detector.tostring(), 
            "stream_url": self.stream_url, 
            "config_path": self.config_path, 
            "sql_connection": self.sql_connection, 
            "topic_name": self.topic_name, 
            "url_str_flag": self.url_str_flag
        }
    
    def _init_detector(self, model_path):
        """init detector implemented by inherited class."""
        return UltraliticsDetector(model_path=model_path)
    
    @model_validator(mode="before")
    @classmethod
    def _valid_before(cls, values):
        return values
    
    @model_validator(mode="after")
    @classmethod
    def _valid_after(cls, data: any):
        """model_validator after will exec after the construction function __init__"""
        """valid model and model path and init them implemented by inherited class"""
        return data
    
    @model_validator(mode="after")
    @classmethod
    def _valid_url_str_flag_after(cls, data: any):
        """model_validator after will exec after the construction function __init__"""
        """valid model and model path and init them implemented by inherited class"""
        """but i have found this valid will exec before __init__ function."""
        """but the init value in model_validator after will not overwrite the init value in construct function."""
        """use the default config file url_str_flag, if you want customer, pass the variable in construct function."""
        try:
            config = DetectorConfig.from_file(data.config_path).__dict__
            if "url_str_flag" not in data or not data.url_str_flag:
                data.url_str_flag = config["url_str_flag"][0]
        except Exception as e:
            raise ValueError("fail to init the url_str_flag variable! you can pass it in your construct function!")
        return data
    
    def _valid_detector_warning(self, detector_warning: DetectorWarning = None):
        """init detector warning implemented by inherited class."""
        self.detector_warning = detector_warning if detector_warning is not None else self.detector_warning
        if self.detector_warning is None or self.device_sn is not None or self.topic_name is not None or self.stream_url is not None or self.url_str_flag is not None:
            self.detector_warning = SxDetectorWarning(
                self.config_path, 
                self.url_str_flag,
                self.device_sn, 
                self.stream_url,
                self.topic_name
            )
        return True
    
    def get_sql_connection(self):
        try:
            sql_info = self.config["sql"][self.url_str_flag]
            username = sql_info["username"]
            database = sql_info["database"]
            password = sql_info["password"]
            host = sql_info["host"]
            port = sql_info["port"]
        except Exception as e:
            raise ValueError(f"fail to init the sql connect information!\n{self.config}") from e
        database_url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
        try:
            engine = create_engine(database_url, pool_size=10, max_overflow=20)
            SessionLocal = sessionmaker(bind=engine)
        except Exception as e:
            raise ValueError("fail to create the sql connector engine!") from e
        return SessionLocal
    
    def check_sql_video_stream_status(self):
        with self.sql_connection() as db:
            try:
                query = "SELECT * FROM webcam_ai_config WHERE device_sn = :device_sn;"
                result = db.execute(text(query), {"device_sn": self.device_sn}).fetchone()
                db.commit()
            except Exception as e:
                db.rollback()        
                error_info = "fail to update sql!"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
        try:
            real_topic_list = ast.literal_eval(result[-1]) if result else []
        except Exception as e:
            error_info = f"topic list is not a valid list string!{result[-1]}"
            self.logger.error(error_info)
            raise ValueError(error_info) from e
        return real_topic_list
    
    def update_sql_video_stream_status(self, topic_list):
        with self.sql_connection() as db:
            try:
                if not topic_list:
                    delete_sql = "DELETE FROM webcam_ai_config WHERE device_sn = :device_sn;"
                    db.execute(text(delete_sql), {"device_sn": self.device_sn})
                    db.commit()
                    return True, f"success to delete device_sn: {self.device_sn}"
                
                sql = """
                    INSERT INTO webcam_ai_config (device_sn, video_url, topic_list)
                    VALUES (:device_sn, :video_url, :topic_list)
                    ON DUPLICATE KEY UPDATE 
                        video_url = VALUES(video_url), 
                        topic_list = VALUES(topic_list);
                """
                db.execute(text(sql), {
                    "device_sn": self.device_sn,
                    "video_url": self.stream_url,
                    "topic_list": str(topic_list)
                })
                db.commit()
            except Exception as e:
                db.rollback()        
                error_info = "fail to update sql!"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
        return True
    
    def truncate_sql_table(self):
        with self.sql_connection() as db:
            try:
                truncate_sql = "TRUNCATE TABLE webcam_ai_config;"
                db.execute(text(truncate_sql))
                db.commit()
            except Exception as e:
                db.rollback()        
                error_info = "fail to truncate the webcam_ai_config table!"
                self.logger.error(error_info)
                raise ValueError(error_info) from e
        return True, f"successfully truncate the webcam_ai_config table"
    
    def get_real_topic_list(self):
        """get real topic list implemented by inherited class"""
        # check the real time topic list
        real_topic_list = self.check_sql_video_stream_status()
        return real_topic_list

    def get_topic_based_topic_name(self):
        """get real topic list implemented by inherited class"""
        """overwrite the get_topic_based_topic_name function, code return self.topic_name if topic = self.topic_name"""
        try:
            topic = self.topic_name + "?" + self.config['topics'][self.topic_name]
        except Exception as e:
            error_info = "fail to get topic based topic name!"
            raise ValueError(error_info) from e
        return topic

    def get_warning_information(self, results):
        """set the warning information based on the predict result implemented by inherited class."""
        warning_flag = False
        try:
            for index, result in enumerate(results):
                predict_result = result.to_json()
                if predict_result and predict_result != '[]':
                    warning_flag = True
                image = result.plot()
        except Exception as e:
            raise RuntimeError("fail to get warning information based on the detector predict results!") from e
        return warning_flag, image
        
    def get_video_stream_url(self, device_sn: Optional[str] = None):
        """overwrite the get video stream url method if you need. and notice, if you
        have not provided one stream url in the StreamDetector instance, you must overwrite this method.
        """
        get_video_stream_url = self.config["get_video_stream_url"]
        url = get_video_stream_url["url"][self.url_str_flag]
        request_json = get_video_stream_url["request_json"][self.url_str_flag]
        request_json[next(iter(request_json))] = self.device_sn
        
        print(type(self.logger))
        self.logger.info(f"request_json: {request_json}")
        self.logger.info(f"request_url: {url}")
        result = requests.post(url, json=request_json)
        self.logger.info(f"get_video_stream_url result: {result.json()}")
        url = ""
        if result.status_code != 200:
            raise ConnectionError(f"fail to get video url stream! the reason is api error or invalid device_sn: {self.device_sn}")
        try:
            json_result = result.json()
        except Exception as e:
            raise json.JSONDecodeError("fail to parse result json in get_video_url_stream function!") from e
        
        if "data" in json_result:
            return json_result["data"]["url"]
        else:
            raise ValueError(json_result["msg"])
        