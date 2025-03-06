import pytest
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
    Any,
    List
)
import requests
import threading
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from dataclasses import dataclass, field
from queue import Queue

from whoami.tool.disease_predict.sx_device_wavve_vital_sign_log import SxDeviceWavveVitalSignLog
from whoami.tool.disease_predict.threshold_value import ThresholdValue
from whoami.configs.sql_config import SqlConfig
from whoami.provider.sql_provider import SqlProvider
from whoami.provider.sql_provider import ModelType
from whoami.tool.health_report.sx_device_wavve_vital_sign_realtime import SxDeviceWavveVitalSignLogRealTime
from whoami.tool.health_report.standard_breath_heart import StandardBreathHeart
from whoami.tool.health_report.sx_device_wavve_vital_sign_config_info import DeviceWavveVitalSignConfigInfo
from whoami.tool.detect.video_stream_manager import VideoStreamManager
from whoami.utils.log import Logger
from whoami.tool.detect.ultralitics_detector import UltraliticsDetector
from whoami.tool.detect.video_stream_manager import warning_fastapi_improved
from whoami.configs.detector_config import DetectorConfig


logger = Logger("VideoStreamManager")
CONFIG_PATH = "/work/ai/WHOAMI/whoami/scripts/detect/detect_config.yaml"
default_topic_list = ['/fallen/falling/warning']
CONFIG = DetectorConfig.from_file(CONFIG_PATH).__dict__
TOPIC_DICT = CONFIG['topics']
@dataclass
class RequestData:
    device_sn: str
    video_stream_url: str = ""
    sampling_interval: float = 0.3
    topic_list: list = field(default_factory=lambda: default_topic_list)
    base64_flag: int = 0
    mqtt_flag: int = 0

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

app = FastAPI()
@pytest.mark.parametrize(
    "sql_config_path, sql_config, sql_provider, model, device_sn",
    [
        pytest.param(
            '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml', 
            None, 
            None, 
            SxDeviceWavveVitalSignLog,
            '13CFF349200080712111157C07'
        ),
    ]
)
def test_video_stream_manager(
    sql_config_path, 
    sql_config, 
    sql_provider,
    model,
    device_sn
):
    video_stream_manager = VideoStreamManager(max_readers=3, max_processors=5)
    video_stream_manager.set_logger(logger)
    video_stream_manager.set_detector_pool(detector_pool)

    @app.post('/fire_smoke_warning')
    async def warning_fastapi(request_data: RequestData):
        return await warning_fastapi_improved(request_data, video_stream_manager, detector_pool, logger, TOPIC_DICT)