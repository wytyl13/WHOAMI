#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/23 17:48
@Author  : weiyutao
@File    : scripts.py
"""

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

from whoami.tool.detect.sx_video_stream_detector import SxVideoStreamDetector
from whoami.utils.log import Logger
from whoami.utils.R import R
from whoami.utils.utils import Utils
from whoami.configs.detector_config import DetectorConfig
from whoami.tool.detect.ultralitics_detector import UltraliticsDetector

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIRECTORY, "detect_config.yaml"))

CONFIG = DetectorConfig.from_file(CONFIG_PATH).__dict__
TOPIC_DICT = CONFIG['topics']
default_topic_list = ['/fallen/falling/warning']
app = FastAPI()
logger = Logger('warning_fastapi')
url_str_flag = 'new'

threads = {}
sx_video_stream_detector = None


class DetectorPool:
    def __init__(self, model_paths: Dict[str, str], max_pool_size=20):
        """
        初始化检测器对象池
        
        :param model_paths: 模型路径字典 
        :param max_pool_size: 每个模型最大实例数
        """
        self.pools = {}
        self.locks = {}
        
        # 为每个模型创建线程安全的对象池
        for topic, model_path in model_paths.items():
            self.pools[topic] = Queue(maxsize=max_pool_size)
            self.locks[topic] = threading.Lock()
            
            # 预先创建实例
            for _ in range(max_pool_size):
                detector = UltraliticsDetector(model_path=model_path)
                self.pools[topic].put(detector)
    
    def get_detector(self, topic):
        """
        获取指定主题的检测器实例
        
        :param topic: 检测器主题
        :return: 检测器实例
        """
        if topic not in self.pools:
            raise ValueError(f"No detector pool for topic: {topic}")
        
        # 从池中获取实例
        detector = self.pools[topic].get()
        return detector
    
    def release_detector(self, topic, detector):
        """
        将检测器实例返回到池中
        
        :param topic: 检测器主题
        :param detector: 检测器实例
        """
        if topic not in self.pools:
            raise ValueError(f"No detector pool for topic: {topic}")
        
        # 将实例放回池中
        self.pools[topic].put(detector)
model_paths = {
    "/fallen/falling/warning": "/work/ai/WHOAMI/whoami/models/detect/falldetect-11x.pt",
    "/fire/smoke/warning": "/work/ai/WHOAMI/whoami/models/detect/fire_smoke_yolov10m_v2_epochs_250.pt",
    "/violence/warning": "/work/ai/WHOAMI/whoami/models/detect/fight_yolov10m_199_epoch.pt"
}

detector_pool = DetectorPool(model_paths)

@dataclass
class RequestData:
    device_sn: str
    video_stream_url: str = ""
    sampling_interval: float = 0.3
    topic_list: list = field(default_factory=lambda: default_topic_list)
    base64_flag: int = 0
    mqtt_flag: int = 0

def _async_raise(tid, exctype):
    """Raises the exception in the thread with id tid."""
    tid = ctypes.c_long(tid)
    if not isinstance(exctype, type):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    try:
        if res == 0:
            raise ValueError("Invalid thread id")
        elif res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
    except Exception as e:
        logger.error(str(e))
    return True

def stop_thread(thread):
    logger.info(f"thread: {thread}")
    """Stop a thread by raising an exception in it."""
    _async_raise(thread.ident, SystemExit)
    thread.join(timeout=15)
    if thread.is_alive():
        logger.warning(f"Thread {thread} did not stop gracefully")
        return False
    return True

def stop_all_thread():
    values = [value for value in threads.values()]
    for value in values:
        print(value)
        _async_raise(value.ident, SystemExit)

def background_run(sx_video_stream_detector: SxVideoStreamDetector):
    try:
        return sx_video_stream_detector.process()
    except Exception as e:
        logger.error(f"Error in thread: {e}")
    finally:
        # 释放GPU资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def check_running_threads():
    running_threads = {}
    for thread_id, thread in threads.items():
        if thread.is_alive():
            running_threads[thread_id] = thread
    return running_threads

@app.post('/fire_smoke_warning')
async def warning_fastapi(request_data: RequestData):
    # logger.info(request_data)
    try:
        video_stream_url = request_data.video_stream_url
        device_sn = request_data.device_sn
        sampling_interval = request_data.sampling_interval
        topic_list = request_data.topic_list
    except Exception as e:
        return R.fail(f"传参错误！{request_data}")
    
    TOPIC_LIST = list(TOPIC_DICT.keys())
    for topic in topic_list:
        if topic not in TOPIC_DICT:
            return R.fail(f"topic: {topic}错误！应该属于：{TOPIC_LIST}")
        
    sx_video_stream_detector.set_device_sn(device_sn)

    logger.info(sx_video_stream_detector.tostring())
    topic_list = [topic + "?" + TOPIC_DICT[topic] for topic in topic_list]
    
    # check the status for this video stream url
    try:
        real_topic_list = sx_video_stream_detector.check_sql_video_stream_status()
    except Exception as e:
        return R.fail(e)
    
    logger.info(f"topic_list: {topic_list}")
    logger.info(f"real_topic_list: {real_topic_list}")

    intersection_topic = list(set(topic_list) & set(real_topic_list))
    difference = list(set(real_topic_list) - set(topic_list))
    logger.info(f"del_topic: {difference}")
    
    for topic_del in difference:
        # delete the deleted topic for the current video stream url in mysql table.
        task_id_to_stop = device_sn + topic_del
        logger.info(f"delete thread: {task_id_to_stop}")
        if task_id_to_stop in threads:
            thread = threads[task_id_to_stop]
            if stop_thread(thread):
                del threads[task_id_to_stop]
            else:
                logger.error(f"Failed to stop thread for {task_id_to_stop}")
        
    try:
        sx_video_stream_detector.update_sql_video_stream_status(topic_list)
    except Exception as e:
        return R.fail(e)

    # what topic need to open.
    need_open_topic_list = list(set(topic_list) - set(intersection_topic))
    logger.info(f"need to open topic list: {need_open_topic_list}")
    for topic in need_open_topic_list:
        topic_name = topic.split('?')[0]
        thread_id = device_sn + topic
        if topic_name not in TOPIC_DICT:
            return R.fail(f"topic: {topic_name}错误！应该属于：{TOPIC_LIST}")
        # sx_video_stream_detector_thread = SxVideoStreamDetector(device_sn=device_sn, url_str_flag=url_str_flag, topic_name=topic_name, config_path=CONFIG_PATH)
        detector = detector_pool.get_detector(topic_name)
        try:
            sx_video_stream_detector_thread = SxVideoStreamDetector(device_sn=device_sn, url_str_flag=url_str_flag, topic_name=topic_name, config_path=CONFIG_PATH, detector=detector)
            logger.info(sx_video_stream_detector_thread.tostring())
            print(f"------------------开启任务：{sx_video_stream_detector_thread.topic_name}")
            def wrapped_background_run(detector_wrapper):
                try:
                    result = background_run(detector_wrapper)
                    return result
                finally:
                    # 完成后将检测器实例返回对象池
                    detector_pool.release_detector(topic_name, detector)
            
            thread = threading.Thread(target=wrapped_background_run,
                            args=(sx_video_stream_detector_thread,))
            thread.start()
            threads[thread_id] = thread
        except Exception as e:
            # 如果创建线程失败，也要确保将检测器实例返回对象池
            detector_pool.release_detector(topic_name, detector)
            logger.error(f"创建线程失败: {e}")
    return R.success(f"视频流解析成功{video_stream_url}！开始后台执行！")

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down application...")
    stop_all_thread()
    print("All threads stopped.")

@app.get('/list_all_topic')
async def list_all_topic():
    return R.success(TOPIC_DICT)

@app.get('/check_running_thread')
async def check_running_thread():
    return R.success(check_running_threads())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='new', help='the default url_str_flag!')
    args = parser.parse_args()
    url_str_flag = args.url
    sx_video_stream_detector = SxVideoStreamDetector(device_sn='', url_str_flag=url_str_flag, topic_name=default_topic_list[0], config_path=CONFIG_PATH)
    sx_video_stream_detector.truncate_sql_table()
    uvicorn.run(app, host='0.0.0.0', port=CONFIG["port_dict"][url_str_flag])
    


