import time

class ProductionLineInfo:
    def __init__(self, device_sn, topic, stream_url, detector):
        self.device_sn = device_sn
        self.topic = topic
        self.stream_url = stream_url
        self.detector = detector
        self.last_update_time = time.perf_counter()
        self.frame_count = 0
        self.real_topic_list = []
        self.pre_warning_time = None

    def set_real_topic_list(self, real_topic_list):
        self.real_topic_list = real_topic_list

    def set_pre_warning_time(self, pre_warning_time):
        self.pre_warning_time = pre_warning_time

    def set_stream_url(self, stream_url):
        self.stream_url = stream_url