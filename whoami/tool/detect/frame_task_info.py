import numpy as np
import time
from typing import Union, List

class FrameTaskInfo:
    def __init__(self, device_sn: str, topic: Union[str, List[str]], frame: np.ndarray, topic_model_key: str):
        self.device_sn = device_sn
        self.topic = topic
        self.frame = frame
        self.topic_model_key = topic_model_key
        self.timestamp = time.perf_counter()