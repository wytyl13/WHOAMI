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

from whoami.tool.detect.sx_video_stream_detector import SxVideoStreamDetector
from whoami.utils.log import Logger
from whoami.utils.R import R
from whoami.utils.utils import Utils
from whoami.configs.detector_config import DetectorConfig

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIRECTORY, "../../tool/detect/default_config.yaml"))

CONFIG = DetectorConfig.from_file(CONFIG_PATH).__dict__
TOPIC_DICT = CONFIG['topics']
default_topic_list = ['/fallen/falling/warning']
app = FastAPI()
logger = Logger('warning_fastapi')
url_str_flag = 'new'

threads = {}
sx_video_stream_detector = None

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
    if res == 0:
        raise ValueError("Invalid thread id")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    logger.info(f"thread: {thread}")
    """Stop a thread by raising an exception in it."""
    _async_raise(thread.ident, SystemExit)

def stop_all_thread():
    values = [value for value in threads.values()]
    for value in values:
        print(value)
        _async_raise(value.ident, SystemExit)

def background_run(sx_video_stream_detector: SxVideoStreamDetector):
    return sx_video_stream_detector.process()

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
    # sx_video_stream_detector.set_url_str_flag(url_str_flag)

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
        stop_thread(threads[task_id_to_stop])
        
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
        sx_video_stream_detector_thread = SxVideoStreamDetector(device_sn=device_sn, url_str_flag=url_str_flag, topic_name=topic_name)
        print(f"------------------开启任务：{sx_video_stream_detector_thread.topic_name}")
        # logger.info(sx_video_stream_detector)
        thread = threading.Thread(target=background_run,
                        args=(sx_video_stream_detector_thread,))
        thread.start()
        threads[thread_id] = thread
    return R.success(f"视频流解析成功{video_stream_url}！开始后台执行！")


@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down application...")
    stop_all_thread()
    print("All threads stopped.")

@app.get('/list_all_topic')
async def list_all_topic():
    return R.success(TOPIC_DICT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='new', help='the default url_str_flag!')
    args = parser.parse_args()
    url_str_flag = args.url
    sx_video_stream_detector = SxVideoStreamDetector(device_sn='', url_str_flag=url_str_flag, topic_name=default_topic_list[0])
    sx_video_stream_detector.truncate_sql_table()
    uvicorn.run(app, host='0.0.0.0', port=CONFIG["port_dict"][url_str_flag])
    


