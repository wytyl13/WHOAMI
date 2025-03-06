import numpy as np
import time

class FrameTaskInfo:
    def __init__(self, device_sn: str, topic: str, frame: np.ndarray):
        self.device_sn = device_sn
        self.topic = topic
        self.frame = frame
        self.timestamp = time.perf_counter()