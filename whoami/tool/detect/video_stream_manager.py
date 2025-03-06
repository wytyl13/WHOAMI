import cv2
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set
import requests
import json
from whoami.tool.detect.sx_detector_warning import SxDetectorWarning

@dataclass
class FrameTask:
    device_sn: str
    topic: str
    frame: Any
    timestamp: float


class VideoStreamManager:
    def __init__(self, max_reader=3, max_processors=5, queue_size=100):
        self.reader_pool = ThreadPoolExecutor(max_workers=max_reader)
        self.processor_pool = ThreadPoolExecutor(max_workers=max_processors)
        self.frame_queue = queue.Queue(maxsize=queue_size)

        self.active_streams = {}
        self.stream_locks = {}
        self.stop_flags = {}
        
        self.logger = None
        self.consumer_running = True
        self.consumer_thread = threading.Thread(target=self._process_frames_worker)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()
        
        self.detector_pool = None
    
    def set_logger(self, logger):
        self.logger = logger
    
    def set_detector_pool(self, detector_pool):
        self.detector_pool = detector_pool
    
    def start_stream(self, device_sn, topic_name, stream_url, detector):
        task_id = device_sn + topic_name
        if task_id not in self.stream_locks:
            self.stream_locks[task_id] = threading.Lock()
        
        with self.stream_locks[task_id]:
            if task_id in self.active_streams:
                self.logger.info(f"Stream {task_id} is already running, not starting again")
                return
            
            self.stop_flags[task_id] = False
            self.active_streams[task_id] = {
                "device_sn": device_sn,
                "topic_name": topic_name,
                "stream_url": stream_url,
                "detector": detector,
                "last_update_time": time.perf_counter(),
                "frame_count": 0, 
                "real_topic_list": []
            }

            self.reader_pool.submit(self._read_stream_worker, task_id)
            self.logger.info(f"Started video stream {task_id}")
        
    def stop_stream(self, task_id):
        """Stop a running a video stream"""
        if task_id not in self.active_streams:
            self.logger.warn(f"Cannot stop sream {task_id} - not found in active streams")
            return False

        with self.stream_locks.get(task_id, threading.Lock()):
            self.stop_flags[task_id] = True
            self.logger.info(f"Setting stop flag for stream {task_id}")
        
        # Release the detector back to the pool
        if self.detector_pool and "topic_name" in self.active_streams[task_id]:
            topic_name = self.active_streams[task_id]["topic_name"]
            detector = self.active_streams[task_id].get("detector")
            if detector:
                self.detector_pool.release_detector(topic_name.split('?')[0], detector)
                self.logger.info(f"Released detector for {topic_name} back to pool")
        
        stream_info = self.active_streams.pop(task_id, None)
        return True
    
    def get_active_stream_topics(self):
        """Get a list of all active stream topics"""
        active_topics = []
        for task_id, stream_info in self.active_streams.items():
            topic = stream_info["topic_name"]
            active_topics.append(topic)
        return active_topics
    
    def update_db_status(self, task_id, real_topic_list):
        """Update database with stream status"""
        if task_id in self.active_streams:
            with self.stream_locks.get(task_id, threading.Lock()):
                self.active_streams[task_id]["real_topic_list"] = real_topic_list.copy()
        self.reader_pool.submit(self._update_db_async, real_topic_list.copy())
    
    def _update_db_async(self, real_topic_list):
        try:
            pass
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error updating database: {str(e)}")


    def _read_stream_worker(self, task_id):
        """Worker function that reads frames from a video stream"""
        if task_id not in self.active_streams:
            self.logger.error(f"Stream {task_id} not found in active streams")
            return
        stream_info = self.active_streams[task_id]
        device_sn = stream_info["device_sn"]
        topic = stream_info["topic_name"]
        stream_url = stream_info["stream_url"]
        self.logger.info(f"Starting reader for stream {task_id}, URL: {stream_url}")

        while not self.stop_flags.get(task_id, True):
            try:
                real_topic_list = []
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    self.logger.error(f"Failed to open stream_url: {stream_url}")
                    if topic in real_topic_list:
                        real_topic_list.remove(topic)
                    self.update_db_status(task_id, real_topic_list)
                    time.sleep(1)
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.logger.info(f"Stream {task_id} - FPS: {fps}, Resolution: {width}x{height}")

                if topic not in real_topic_list:
                    real_topic_list.append(topic)
                self.update_db_status(task_id, real_topic_list)

                # Read frame loop
                frame_count = 0
                last_db_update_time = time.perf_counter()
                while cap.isOpened() and not self.stop_flags.get(task_id, True):
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        self.logger.error(f"Failed to read frame from: {stream_url}")
                        if topic in real_topic_list:
                            real_topic_list.remove(topic)
                        self.update_db_status(task_id, real_topic_list)
                        break
                    current_time = time.perf_counter()
                    self.logger.info(current_time)
                    frame_task = FrameTask(
                        device_sn=device_sn,
                        topic=topic,
                        frame=frame.copy(),
                        timestamp=current_time
                    )

                    try:
                        self.frame_queue.put(frame_task, timeout=0.1)
                        frame_count += 1
                    except queue.Full:
                        self.logger.warning(f"Frame queue full, dropping frame for {task_id}")

                    # Periodically update database status (every 3 seconds)
                    if current_time - last_db_update_time >= 3.0:
                        elapsed_time = current_time - last_db_update_time
                        actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        self.logger.info(f"Stream {task_id} - Actual FPS: {actual_fps:.2f}")

                        if topic not in real_topic_list:
                            real_topic_list.append(topic)
                        self.update_db_status(task_id, real_topic_list)
                        frame_count = 0
                        last_db_update_time = current_time
                cap.release()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in stream reader for {task_id}: {str(e)}")
                time.sleep(1.5)
        self.logger.info(f"Reader for stream {task_id} stopped")

    
    def _process_frames_worker(self):
        """Worker function that processes frames from the queue"""
        while self.consumer_running:
            try:
                # Get a frame task from the queue
                frame_task = self.frame_queue.get(timeout=0.5)
                
                # Process the frame in the processor thread pool
                self.processor_pool.submit(self._process_single_frame, frame_task)
                
            except queue.Empty:
                # Queue is empty, just continue
                continue
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in frame processor: {str(e)}")
                time.sleep(0.1)  # Short delay on error


    def _get_warning_information(self, results):
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

    def _process_single_frame(self, frame_task):
        """Process a single frame with the appropriate detector"""
        task_id = frame_task.device_sn + frame_task.topic
        
        if task_id not in self.active_streams:
            # Stream was stopped while frame was in queue
            return
        
        stream_info = self.active_streams[task_id]
        detector = stream_info.get('detector')
        
        if not detector:
            self.logger.error(f"No detector found for {task_id}")
            return
        
        try:
            # Run inference on the frame
            self.logger.info(f"{task_id} processing frame")
            results = detector.predict(frame_task.frame)
            
            # Get and process any warnings
            warning_flag, warning_information = self._get_warning_information(results)
            if warning_flag:
                # We would call detector_warning.warning() here
                detector_warning = SxDetectorWarning(
                    config_path="/work/ai/WHOAMI/whoami/scripts/detect/detect_config.yaml", 
                    url_str_flag="new",
                    device_sn=stream_info["device_sn"], 
                    stream_url=stream_info["stream_url"],
                    topic_name=stream_info["topic_name"]
                )
                detector_warning.warning(warning_information)
                self.logger.warning(f"Warning detected: {warning_information}")
        
        except Exception as e:
            self.logger.error(f"Error processing frame for {task_id}: {str(e)}")

    def shutdown(self):
        """Shutdown the manager and all thread pools"""
        # Stop all active streams
        for task_id in list(self.active_streams.keys()):
            self.stop_stream(task_id)
        
        # Signal consumer thread to stop
        self.consumer_running = False
        if self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=5.0)
        
        # Shutdown thread pools
        self.reader_pool.shutdown(wait=True)
        self.processor_pool.shutdown(wait=True)
        
        self.logger.info("Video stream manager shutdown complete")

def get_video_stream_url(device_sn: Optional[str] = None):
        """overwrite the get video stream url method if you need. and notice, if you
        have not provided one stream url in the StreamDetector instance, you must overwrite this method.
        """
        request_json = {
            "deviceSn": device_sn,
            "channelNo": 1,
            "protocol": 2,
            "quality": 1, 
            "bitRateType": 0, # 变码率，默认为固定码率
            "videoFrameRate": 0, # 0为全帧率一般为15
            "resolution": "VGA", # 分辨率，使用较低的640*480。对应较低的码率，可以减少带宽占用
            "videoBitRate": "21", # 码率决定视频流的带宽占用，较低的视频流分辨率适配较低的码率，码率=宽*高*位深*帧率=2304*1296*8*15=358318080bps=358318kbps=358.318Mbps=358.318/8(Mb/s)=44.8MB/s
            "encodeType": "H264",
            "expireTime": "86400"
        }

        url = "http://1.71.15.102:48080/admin-api/device/yingShiWebcam/getWebcamLiveAddress"
        
        print(f"request_json: {request_json}")
        print(f"request_url: {url}")
        result = requests.post(url, json=request_json)
        print(f"get_video_stream_url result: {result.json()}")
        if result.status_code != 200:
            raise ConnectionError(f"fail to get video url stream! the reason is api error or invalid device_sn: {device_sn}")
        try:
            json_result = result.json()
        except Exception as e:
            raise json.JSONDecodeError("fail to parse result json in get_video_url_stream function!") from e
        
        if "data" in json_result:
            return json_result["data"]["url"]
        else:
            raise ValueError(json_result["msg"])


# Example FastAPI endpoint that uses the manager
async def warning_fastapi_improved(request_data, video_stream_manager, detector_pool, logger, TOPIC_DICT):
    try:
        video_stream_url = request_data.video_stream_url
        device_sn = request_data.device_sn
        sampling_interval = request_data.sampling_interval
        topic_list = request_data.topic_list
    except Exception as e:
        logger.error(f"Parameter error: {str(e)}")
        return {"code": 500, "message": f"Parameter error: {request_data}"}
    print(f"device_sn: --------------------------- {device_sn}")
    video_stream_url = get_video_stream_url(device_sn) if video_stream_url == "" else video_stream_url
    # Validate topics
    TOPIC_LIST = list(TOPIC_DICT.keys())
    for topic in topic_list:
        if topic not in TOPIC_DICT:
            return {"code": 500, "message": f"Invalid topic: {topic}. Should be one of: {TOPIC_LIST}"}
    
    # Format topics with their dictionary values
    formatted_topic_list = [topic + "?" + TOPIC_DICT[topic] for topic in topic_list]
    
    # Get currently active topics
    active_topics = video_stream_manager.get_active_stream_topics()
    logger.info(f"Requested topics: {formatted_topic_list}")
    logger.info(f"Currently active topics: {active_topics}")
    
    # Find topics to stop (active but not in request)
    topics_to_stop = list(set(active_topics) - set(formatted_topic_list))
    logger.info(f"Topics to stop: {topics_to_stop}")
    
    # Stop streams that should be stopped
    for topic_to_stop in topics_to_stop:
        task_id = device_sn + topic_to_stop
        logger.info(f"Stopping stream: {task_id}")
        video_stream_manager.stop_stream(task_id)
    
    # Find topics to start (in request but not active)
    topics_to_start = list(set(formatted_topic_list) - set(active_topics))
    logger.info(f"Topics to start: {topics_to_start}")
    
    # Start new streams
    for topic in topics_to_start:
        topic_name = topic.split('?')[0]
        
        # Get detector from pool
        try:
            detector = detector_pool.get_detector(topic_name)
            
            # Start the stream with the video manager
            video_stream_manager.start_stream(
                device_sn=device_sn,
                topic_name=topic,  # Use the full topic string
                stream_url=video_stream_url,
                detector=detector
            )
        except Exception as e:
            # If stream start fails, return detector to pool
            detector_pool.release_detector(topic_name, detector)
            logger.error(f"Failed to start stream for {topic_name}: {str(e)}")
    
    return {"code": 200, "message": f"Video stream processing started: {video_stream_url}"}
