#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/03/04 10:29
@Author  : weiyutao
@File    : sx_video_stream_pcm.py
"""
import threading
import time
import cv2
import queue
import signal
import gc

from whoami.tool.base.producer_consumer_manager import ProducerConsumerManager
from whoami.tool.detect.production_line_info import ProductionLineInfo
from whoami.tool.detect.sx_video_stream_detector import SxVideoStreamDetector
from whoami.tool.detect.frame_task_info import FrameTaskInfo
from whoami.tool.detect.sx_detector_warning import SxDetectorWarning
from whoami.tool.base.consumer_tool_pool import ConsumerToolPool
from whoami.tool.detect.detector import Detector

from whoami.tool.detect.ultralitics_detector import UltraliticsDetector
from whoami.tool.detect.coordinate_transform import CoordinateTramsform

default_device_sn = 'BD3202818'
default_topic_name = '/fire/smoke/warning'
config_path = '/work/ai/WHOAMI/whoami/scripts/detect/detect_config.yaml'


production_id_safe_region = {
    "BC8796159/fallen/falling/warning?跌倒预警": [
        [
            (
                2141.8550724637685,
                590.7536231884058
            ),
            (
                2294.0289855072465,
                583.5072463768116
            ),
            (
                2115.768115942029,
                1290.7536231884058
            ),
            (
                1666.4927536231883,
                1280.608695652174
            )
        ],
        [
            (
                127.36231884057986,
                842.927536231884
            ),
            (
                237.50724637681176,
                813.9420289855072  
            ),
            (
                505.6231884057972,
                1247.2753623188407
            ),
            (
                280.9855072463769,
                1286.4057971014493
            )
        ]
    ],
    "BD3202818/fallen/falling/warning?跌倒预警": [[
        (
            195.47826086956536,
            337.1304347826088
        ),
        (
            480.9855072463769,
            224.08695652173918
        ),
        (
            841.8550724637682,
            1292.2028985507247
        ),
        (
            302.72463768115955,
            1289.304347826087
        ),
        (
            247.65217391304364,
            1225.536231884058
        ),
        (
            180.98550724637695,
            719.7391304347826
        )
    ]]
}

class SxVideoStreamPCM(ProducerConsumerManager):
    detector_warning: SxDetectorWarning = None
    sx_video_stream_detector: SxVideoStreamDetector = None
    _on_run_complete_callback = None
    coordinate_transform: CoordinateTramsform = CoordinateTramsform()

    def __init__(self, max_producers=15, max_consumers=20, production_queue_size=1000, consumer_tool_pool: ConsumerToolPool = None):
        super().__init__(max_producers, max_consumers, production_queue_size, consumer_tool_pool)
        self.detector_warning = SxDetectorWarning(
            config_path=config_path, 
            url_str_flag="new",
            device_sn=default_device_sn,
            stream_url="",
            topic_name=default_topic_name,
            warning_gap=20
        )
        self.sx_video_stream_detector = SxVideoStreamDetector(
            device_sn=default_device_sn, 
            detector_warning=self.detector_warning,
            url_str_flag='new', 
            config_path=config_path, 
            topic_name=default_topic_name
        )
        # 保持主线程运行
        # self.keep_main_thread_alive()

    def _run(self, topic_dict: dict = None, device_sn: str = None, topic_list: list = None, video_stream_url: str = None, topic_list_flag: bool = False):
        """run function"""
        video_stream_url = self.sx_video_stream_detector.get_video_stream_url(device_sn) if video_stream_url is None else video_stream_url
        TOPIC_LIST = list(topic_dict.keys())
        for topic in topic_list:
            if topic not in topic_dict:
                raise ValueError(f"Invalid topic: {topic}. Should be one of: {TOPIC_LIST}")
        formatted_topic_list = [topic + "?" + topic_dict[topic] for topic in topic_list]
        active_topics_current_device_sn_formatted = self.get_active_topics_base_device_sn(device_sn)
        self.logger.info(f"Requested topics: {formatted_topic_list}")
        self.logger.info(f"Currently active topics: {active_topics_current_device_sn_formatted}")

        # Find topics to stop (active but not in request)
        topics_to_stop = list(set(active_topics_current_device_sn_formatted) - set(formatted_topic_list))
        self.logger.info(f"Topics to stop: {topics_to_stop}")

        # Stop streams that should be stopped
        for topic_to_stop in topics_to_stop:
            production_id = device_sn + topic_to_stop
            self.logger.info(f"Stopping production line: {production_id}")
            self.stop_produce_worker(production_id, device_sn)
        
        # Find topics to start (in request but not active)
        topics_to_start = list(set(formatted_topic_list) - set(active_topics_current_device_sn_formatted))
        self.logger.info(f"Topics to start: {topics_to_start}")

        # Start new streams
        for topic in topics_to_start:
            topic_name = topic.split('?')[0]
            topic_model_key = device_sn + topic_name
            consumer_tool = self.consumer_tool_pool.get_consumer_tool(topic_model_key) # 修改
            try:
                # Get detector from pool
                if consumer_tool is None:
                    raise ValueError(f'Invalid consumer tool! consumer_tool: {consumer_tool}, topic: {topic}, topic_name: {topic_name}, consumer_tool_pool: {self.consumer_tool_pool}')
                # Start the stream with the video manager
                # 注意这里不能传递topic_model_key，而是需要经过处理传递实际使用的topic_model_key，因为在实际从线程池中获取模型实例的时候
                # 如果topic_model_key不存在则使用默认的实例m，如果在while循环中再获取真实的topic_model_key将会有大量的时间消耗
                self.start_produce_worker(
                    device_sn=device_sn,
                    topic=topic,  # Use the full topic string
                    stream_url=video_stream_url,
                    detector=consumer_tool,
                    topic_list_flag=topic_list_flag,
                    topic_model_key=self.consumer_tool_pool.get_consumer_tool_name(topic_model_key)
                )
                f"Production line started: topic: {topic}, video_stream_url: {video_stream_url}"
            except Exception as e:
                # If stream start fails, return detector to pool
                self.consumer_tool_pool.release_consumer_tool(topic_model_key, consumer_tool) # 修改
                raise ValueError(f"Failed to start stream for {topic}: Error in start new streams: {str(e)}") from e
        
        if not topic_list and not topics_to_start:
            self.sx_video_stream_detector.update_sql_video_stream_status(topics_to_start, device_sn=device_sn, stream_url=video_stream_url)
            
        try:
            while self._is_running:
                # 每隔一段时间检查一次运行状态
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Run method interrupted, shutting down...")
        finally:
            # 确保优雅地关闭所有资源
            self._is_running = False
            self.shutdown()
            self.sx_video_stream_detector.truncate_sql_table()
        return "Production lines stopped" 

    def start_produce_worker(
            self, 
            device_sn: str = None, 
            topic: str = None,
            stream_url: str = None,
            detector: Detector = None,
            topic_list_flag: bool = False,
            topic_model_key: str = None
        ):
        production_id = device_sn + topic

        no_submit_flag = self.get_active_device_status(device_sn) and topic_list_flag

        if production_id not in self.production_line_locks:
            # lock it before starting it.
            self.production_line_locks[production_id] = threading.Lock()

        with self.production_line_locks[production_id]:
            # lock status during the producer.
            if production_id in self.active_production_lines:
                self.logger.info(f"Production line {production_id} is already running, not starting again!")
                return
            
            self.production_line_stop_flags[production_id] = False
            self.active_production_lines[production_id] = ProductionLineInfo(device_sn=device_sn, topic=topic, stream_url=stream_url, detector=detector)
            
            # start one single video stream line for multi topic task. so if have started the same video stream line, not need to start again.
            if no_submit_flag:
                self.logger.warning(f"The same video stream line have started {production_id}!")
                return

            self.producer_pool.submit(
                self._read_stream_worker, 
                production_id, 
                topic_list_flag,
                topic_model_key
            )
            self.logger.info(f"Started video stream {production_id}!")


    def get_active_device_status(self, device_sn):
        """Check whether the same device video stream have started"""
        for production_id, production_line_info in self.active_production_lines.items():
            if production_line_info.device_sn == device_sn:
                return True
        return False

    
    def get_active_topics_base_device_sn(self, device_sn: str = None):
        """Get a list of all active stream topics"""
        active_topics = []
        for production_id, production_line_info in self.active_production_lines.items():
            if production_line_info.device_sn == device_sn:
                active_topics.append(production_line_info.topic)
        return active_topics


    def _read_stream_worker_multi(self, production_id):
        """Worker function that reads frames from a video stream"""
        try:
            if production_id not in self.active_production_lines:
                self.logger.error(f"Stream {production_id} not found in active streams")
                return
            production_info: ProductionLineInfo = self.active_production_lines[production_id]
            device_sn = production_info.device_sn
            topic = production_info.topic
            stream_url = production_info.stream_url
            self.logger.info(f"Starting reader for stream {production_id}, URL: {stream_url}")
        except Exception as e:
            raise ValueError(f"fail to get stream url!")
        
        # 如果连续报错直接终止任务
        stop_outer_loop = False
        error_count = 0
        last_error_time = time.time()
        start_current_error_time = 0
        while not self.production_line_stop_flags.get(production_id, True):
            try:
                real_topic_list = self.get_active_topics_base_device_sn(device_sn=device_sn)
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    self.logger.error(f"Failed to open stream_url: {stream_url}")
                    
                    
                    if topic in real_topic_list:
                        real_topic_list.remove(topic)
                    self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn, stream_url=stream_url)

                    # 错误次数过多直接终止while循环
                    error_count += 1
                    # 检查错误次数和时间间隔
                    start_current_error_time = time.time() if start_current_error_time == 0 else start_current_error_time
                    if error_count >= 5 and start_current_error_time - last_error_time < 10:
                        self.logger.error(f"Error count exceeded 5 within 10 seconds. Stopping stream {production_id}.")
                        stop_outer_loop = True
                        self.active_production_lines.__delitem__(production_id)
                        break  
                    last_error_time = start_current_error_time

                    stream_url = self.sx_video_stream_detector.get_video_stream_url(device_sn=device_sn)
                    self.active_production_lines[production_id].set_stream_url(stream_url)
                    cap.release()
                    time.sleep(1)
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.logger.info(f"Stream {production_id} - FPS: {fps}, Resolution: {width}x{height}")

                if topic not in real_topic_list:
                    real_topic_list.append(topic)
                self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn, stream_url=stream_url)

                # Read frame loop
                frame_count = 0
                last_db_update_time = time.perf_counter()
                while cap.isOpened() and not self.production_line_stop_flags.get(production_id, True):
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        self.logger.error(f"Failed to read frame from: {stream_url}")
                        
                        
                        
                        real_topic_list = self.get_active_topics_base_device_sn(device_sn=device_sn)
                        if topic in real_topic_list:
                            real_topic_list.remove(topic)
                        self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn, stream_url=stream_url)

                        # 错误次数过多直接终止while循环
                        start_current_error_time = time.time() if start_current_error_time == 0 else start_current_error_time
                        error_count += 1
                        if error_count >= 5 and start_current_error_time - last_error_time < 10:
                            self.logger.error(f"Error count exceeded 5 within 10 seconds. Stopping stream {production_id}.")
                            stop_outer_loop = True
                            self.active_production_lines.__delitem__(production_id)
                            break  # 终止循环
                        last_error_time = start_current_error_time

                        stream_url = self.sx_video_stream_detector.get_video_stream_url(device_sn=device_sn)
                        self.active_production_lines[production_id].set_stream_url(stream_url)
                        cap.release()
                        break
                    
                    current_time = time.perf_counter()
                    frame_task = FrameTaskInfo(device_sn=device_sn, topic=real_topic_list, frame=frame.copy())
                    try:
                        self.production_queue.put(frame_task, timeout=0.1)
                        frame_count += 1
                    except queue.Full:
                        self.logger.warning(f"Frame queue full, dropping frame for {production_id}")

                    # Periodically update database status (every 3 seconds)
                    if current_time - last_db_update_time >= 5.0:
                        elapsed_time = current_time - last_db_update_time
                        production_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        self.logger.info(f"Stream {production_id} - Production FPS: {production_fps:.2f}, Current queue size is: {self.production_queue.qsize()}!")
                        real_topic_list = self.get_active_topics_base_device_sn(device_sn=device_sn)
                        
                        # 测试内存泄漏
                        # self.logger.info(f"memory_usage: ------------------------------------ {self.memory_monitor.check_memory_usage()}")

                        if topic not in real_topic_list:
                            real_topic_list.append(topic)
                        if self.active_production_lines.get(production_id, False):
                            # 有可能造成延迟，另一个线程关闭了当前视频流水线，但是这块还在读取
                            self.logger.info(f"device_sn: {device_sn}, real_topic_list--------------------- {real_topic_list}")
                            self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn, stream_url=stream_url)
                        frame_count = 0
                        last_db_update_time = current_time
                cap.release()
                time.sleep(1)
                
                # 直接终止while循环
                if stop_outer_loop:
                    break  # 终止最外层循环
            except Exception as e:
                self.logger.error(f"Error in stream reader for {production_id}: {str(e)}")
                time.sleep(1.5)
        # 已经关闭当前流水线
        self.logger.info(f"Reader for stream {production_id} stopped")
    

    def _read_stream_worker(
        self, 
        production_id, 
        topic_list_flag: bool = False, 
        topic_model_key: str = None
    ):
        """Worker function that reads frames from a video stream"""
        self.logger.info(f"topic_model_key---------------------------: {topic_model_key}")
        try:
            if production_id not in self.active_production_lines:
                self.logger.error(f"Stream {production_id} not found in active streams")
                return
            production_info: ProductionLineInfo = self.active_production_lines[production_id]
            self.logger.info(f"Starting reader for stream {production_id}")
            device_sn = production_info.device_sn
            topic = production_info.topic
            stream_url = production_info.stream_url
            self.logger.info(f"Starting reader for stream {production_id}, URL: {stream_url}")
        except Exception as e:
            raise ValueError(f"fail to get stream url!")
        
        # 如果连续报错直接终止任务
        stop_outer_loop = False
        error_count = 0
        last_error_time = time.time()
        start_current_error_time = 0
        while not self.production_line_stop_flags.get(production_id, True):
            try:
                real_topic_list = self.get_active_topics_base_device_sn(device_sn=device_sn)
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    self.logger.error(f"Failed to open stream_url: {stream_url}")
                    
                    
                    if topic in real_topic_list and not topic_list_flag:
                        real_topic_list.remove(topic)
                        
                    if topic_list_flag:
                        real_topic_list = []
                    self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn, stream_url=stream_url)

                    # 错误次数过多直接终止while循环
                    error_count += 1
                    # 检查错误次数和时间间隔
                    start_current_error_time = time.time() if start_current_error_time == 0 else start_current_error_time
                    if error_count >= 5 and start_current_error_time - last_error_time < 10:
                        self.logger.error(f"Error count exceeded 5 within 10 seconds. Stopping stream {production_id}.")
                        stop_outer_loop = True
                        self.active_production_lines.__delitem__(production_id)
                        break  
                    last_error_time = start_current_error_time

                    stream_url = self.sx_video_stream_detector.get_video_stream_url(device_sn=device_sn)
                    self.active_production_lines[production_id].set_stream_url(stream_url)
                    cap.release()
                    time.sleep(1)
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.logger.info(f"Stream {production_id} - FPS: {fps}, Resolution: {width}x{height}")

                if topic not in real_topic_list:
                    real_topic_list.append(topic)
                self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn, stream_url=stream_url)

                # Read frame loop
                frame_count = 0
                last_db_update_time = time.perf_counter()
                while cap.isOpened() and not self.production_line_stop_flags.get(production_id, True):
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        self.logger.error(f"Failed to read frame from: {stream_url}")
                        
                        
                        
                        real_topic_list = self.get_active_topics_base_device_sn(device_sn=device_sn)
                        if topic in real_topic_list and not topic_list_flag:
                            real_topic_list.remove(topic)
                            
                        if topic_list_flag:
                            real_topic_list = []
                        self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn, stream_url=stream_url)

                        # 错误次数过多直接终止while循环
                        start_current_error_time = time.time() if start_current_error_time == 0 else start_current_error_time
                        error_count += 1
                        if error_count >= 5 and start_current_error_time - last_error_time < 10:
                            self.logger.error(f"Error count exceeded 5 within 10 seconds. Stopping stream {production_id}.")
                            stop_outer_loop = True
                            self.active_production_lines.__delitem__(production_id)
                            break  # 终止循环
                        last_error_time = start_current_error_time

                        stream_url = self.sx_video_stream_detector.get_video_stream_url(device_sn=device_sn)
                        self.active_production_lines[production_id].set_stream_url(stream_url)
                        cap.release()
                        break
                    
                    current_time = time.perf_counter()
                    topic_to_frame_instance = topic if not topic_list_flag else real_topic_list
                    frame_task = FrameTaskInfo(
                        device_sn=device_sn, 
                        topic=topic_to_frame_instance, 
                        frame=frame.copy(), 
                        topic_model_key=topic_model_key
                    )
                    try:
                        self.production_queue.put(frame_task, timeout=0.1)
                        frame_count += 1
                    except queue.Full:
                        self.logger.warning(f"Frame queue full, dropping frame for {production_id}")

                    # Periodically update database status (every 3 seconds)
                    if current_time - last_db_update_time >= 5.0:
                        elapsed_time = current_time - last_db_update_time
                        production_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        self.logger.info(f"Stream {production_id} - Production FPS: {production_fps:.2f}, Current queue size is: {self.production_queue.qsize()}!")
                        real_topic_list = self.get_active_topics_base_device_sn(device_sn=device_sn)
                        
                        # 测试内存泄漏
                        # self.logger.info(f"memory_usage: ------------------------------------ {self.memory_monitor.check_memory_usage()}")

                        if topic not in real_topic_list:
                            real_topic_list.append(topic)
                        if self.active_production_lines.get(production_id, False):
                            # 有可能造成延迟，另一个线程关闭了当前视频流水线，但是这块还在读取
                            self.logger.info(f"device_sn: {device_sn}, real_topic_list--------------------- {real_topic_list}")
                            self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn, stream_url=stream_url)
                        frame_count = 0
                        last_db_update_time = current_time
                cap.release()
                time.sleep(1)
                
                # 直接终止while循环
                if stop_outer_loop:
                    break  # 终止最外层循环
            except Exception as e:
                self.logger.error(f"Error in stream reader for {production_id}: {str(e)}")
                time.sleep(1.5)
        # 已经关闭当前流水线
        self.logger.info(f"Reader for stream {production_id} stopped")

    def stop_produce_worker(self, production_id, device_sn):
        if production_id not in self.active_production_lines:
            self.logger.warning(f"Cannot stop stream {production_id} - not found in active production line!")
            return False
        
        try:
            with self.production_line_locks.get(production_id, threading.Lock()):
                self.production_line_stop_flags[production_id] = True
                self.logger.info(f"Setting stop flag for production line {production_id}")
        except Exception as e:
            raise ValueError(f"Fail to set stop flag for production line! production_id: {production_id}, {str(e)}")
        production_info: ProductionLineInfo = self.active_production_lines[production_id]
        # remove the current production id from the active_production_lines
        del self.active_production_lines[production_id]

        # Release the consumer tool back to the pool if need.
        try:
            if self.consumer_tool_pool and hasattr(production_info, 'topic') and hasattr(production_info, 'detector'):
                # self.logger.info(f"release_consumer_tool: -----------------------------------------------------------------------")
                # self.logger.info(f"release_consumer_tool: {production_id}, {device_sn}")
                # self.logger.info(f"release_consumer_tool: -----------------------------------------------------------------------")
                topic = production_info.topic
                tool_ = production_info.detector
                topic_name = topic.split('?')[0]
                topic_model_key = device_sn + topic_name
                if tool_:
                    self.consumer_tool_pool.release_consumer_tool(topic_model_key, tool_) # 修改
        except Exception as e:
            raise ValueError(f"Fail to release the consumer tool back to the pool! topic: {topic}, {str(e)}")

        # update sql
        try:
            real_topic_list = self.get_active_topics_base_device_sn(device_sn=device_sn)
            self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn)
        except Exception as e:
            raise ValueError(f"Fail to update sql status, device_sn: {device_sn}, topic_list: {real_topic_list}, {str(e)}")
        return True

    def _start_consumer_worker(self):
        """Worker function that processes frames from the queue in batches grouped by topic"""
        frame_count = 0
        last_queue_empty_time = time.perf_counter()
        last_actual_cal_time = time.perf_counter()

        # Initialize batch tracking dictionaries.
        topic_batches = {} # Format: {topic: [FrameTaskInfo, ...]}
        topic_batch_times = {} # Format: {topic: start_time}

        max_batch_size = 16 # Maximum frames per topic batch
        max_batch_wait_time = 1 # Maximum wait time in seconds

        while self.consumer_worker_running:
            try:
                current_time = time.perf_counter()

                # Get frame from queue
                try:
                    frame_task_info: FrameTaskInfo = self.production_queue.get(timeout=0.1)
                    topic_model_keys = frame_task_info.topic_model_key
                    topic_model_keys = [topic_model_keys] if isinstance(topic_model_keys, str) else topic_model_keys

                    # Initialize new topic batch if need.
                    for topic_model_key in topic_model_keys:
                        if topic_model_key not in topic_batches:
                            topic_batches[topic_model_key] = []
                            topic_batch_times[topic_model_key] = current_time
                        
                        # Add frame to its topic batch
                        topic_batches[topic_model_key].append(frame_task_info)
                except queue.Empty:
                    if current_time - last_queue_empty_time > 5 * max_batch_wait_time:
                        self.logger.warning(f"Production queue is empty!")
                        last_queue_empty_time = current_time
                    continue
                
                
                topics_to_process = []
                try:
                    for topic_model_key, batch in topic_batches.items():
                        if len(batch) >= max_batch_size or (current_time - topic_batch_times[topic_model_key] > max_batch_wait_time and batch):
                            topics_to_process.append(topic_model_key)
                except Exception as e:
                    raise ValueError("FAIL TO GET topics_to_process") from e
                
                # Process the selected topic batches.
                for topic_model_key in topics_to_process:
                    batch = topic_batches[topic_model_key]
                    # 检查线程池是否已经关闭
                    if self.consumer_pool._broken or self.consumer_pool._shutdown:
                        self.logger.error("Thread pool has been shut down")
                        return
                    
                    try:
                        future = self.consumer_pool.submit(self._process_batch_frame, batch)
                        frame_count += len(batch)
                    except RuntimeError as runtime_err:
                        # 捕获关闭时的异常
                        # self.sx_video_stream_detector.truncate_sql_table()
                        if "cannot schedule new futures after interpreter shutdown" in str(runtime_err):
                            self.logger.warning("解释器正在关闭，停止提交新任务")
                        error_info = f"Executor shutdown error: {str(runtime_err)}"
                        self.logger.error(error_info)
                        raise RuntimeError(error_info) from runtime_err
                    
                    # Clear the processed batch
                    topic_batches[topic_model_key] = []
                    topic_batch_times[topic_model_key] = current_time

                # 计算消费线程fps
                elapsed_time  = current_time - last_actual_cal_time
                if elapsed_time >= 3 * max_batch_wait_time:
                    actual_fps = frame_count / elapsed_time
                    self.logger.info(f"processing frame, actual consumer fps: {actual_fps:.2f}, Current queue size is: {self.production_queue.qsize()}!")
                    self.logger.info(f"memory_usage: ------------------------------------ {self.memory_monitor.check_memory_usage()}")
                    frame_count = 0
                    last_actual_cal_time = current_time
            except Exception as e:
                self.logger.error(f"Error in frame processor: {str(e)}")
                time.sleep(0.1)  # Short delay on error

    def _start_consumer_worker_bake(self):
        """Worker function that processes frames from the queue"""
        frame_count = 0
        last_db_update_time = time.perf_counter()
        while self.consumer_worker_running:
            try:
                current_time = time.perf_counter()
                frame_tasks_info = []
                # Get a frame task from the queue
                try:
                    for _ in range(16):
                        frame_task_info = self.production_queue.get(timeout=0.5)
                        frame_tasks_info.append(frame_task_info)
                except queue.Empty:
                    # Queue is empty, just continue
                    if current_time - last_db_update_time > 5:
                        self.logger.warning(f"Production_queue is empty!")
                        last_db_update_time = current_time
                    continue
                
                try:
                    # 检查线程池是否已经关闭
                    if self.consumer_pool._broken or self.consumer_pool._shutdown:
                        self.logger.error("Thread pool has been shut down")
                        break

                    # Process the frame in the processor thread pool
                    try:
                        future = self.consumer_pool.submit(self._process_single_frame, frame_tasks_info)
                    except Exception as e:
                        self.logger.error(f"Task submission error: {e}")
                    frame_count += 1
                except RuntimeError as runtime_err:
                    # 捕获关闭时的异常
                    if "cannot schedule new futures after interpreter shutdown" in str(runtime_err):
                        self.logger.warning("解释器正在关闭，停止提交新任务")
                        continue
                    else:
                        raise
                    self.logger.error(f"Executor shutdown error: {runtime_err}")
                    continue

                # 计算消费线程fps
                elapsed_time  = current_time - last_db_update_time
                if elapsed_time >= 3.0:
                    actual_fps = frame_count / elapsed_time
                    self.logger.info(f"processing frame, actual consumer fps: {actual_fps:.2f}, Current queue size is: {self.production_queue.qsize()}!")
                    self.logger.info(f"memory_usage: ------------------------------------ {self.memory_monitor.check_memory_usage()}")
                    frame_count = 0
                    last_db_update_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in frame processor: {str(e)}")
                time.sleep(0.1)  # Short delay on error

    def _filter_safe_region(self, result, polygon_regions):
        for item in result:
            boxes = item.boxes.xyxy
            for box in boxes:
                try:
                    box_point1 = (box[0].item(), box[1].item())
                    box_point2 = (box[2].item(), box[3].item())
                    # Check against each polygon in the list
                    for polygon_points in polygon_regions:
                        if self.coordinate_transform.calculate_overlap_ratio(
                            point1=box_point1, 
                            point2=box_point2, 
                            polygon_points=polygon_points
                        ):
                            return True
                except Exception as e:
                    error_info = f"Fail to cal coordinate transform overlap ratio {str(e)}"
                    self.logger.info(error_info)
        return False

    def _process_batch_frame(
            self, 
            frame_tasks_info: list[FrameTaskInfo] = None, 
        ):
        """Process a batch of frames grouped by topic with the appropriate detector"""
        if not frame_tasks_info or len(frame_tasks_info) == 0:
            return
        
        # All frames in the batch have the same topic
        # 有可能存在topic不同但是topic_model相同的情况
        # 但是frame_tasks_info中的所有元素topic_model肯定是相同的，因此使用不同的production_line_info肯定可以获取到相同的detector
        sample_task = frame_tasks_info[0]
        topic = sample_task.topic
        production_id = sample_task.device_sn + topic
        # Skip if production line is no longer active
        # 这里使用第一个元素的production_id不是科学的，因为frame_tasks_info中的所有元素可能存在不相同的production_id
        # 但是这个细微差别可以忽略不计
        if production_id not in self.active_production_lines:
            self.logger.warning(f"Stream was stopped while frame was in queue for {production_id}")
            return
        
        production_line_info = self.active_production_lines[production_id]
        detector = production_line_info.detector
        if not detector:
            self.logger.error(f"No detector found for {production_id}")
            return
        # log_info = {"topic": topic, "model_path": detector.model_path, "conf": detector.conf}
        # self.logger.info(f"log_info: ---------------------------------------  {log_info}")
        try:
            # Extract frames from task info
            frames = [task.frame for task in frame_tasks_info]
            
            # Run batch inference using the detector
            try:
                results = detector.predict(frames)
            except Exception as e:
                self.logger.info(str(e))
            # Process each result
            for idx, result in enumerate(results):
                task_info = frame_tasks_info[idx]
                current_production_id = task_info.device_sn + topic
                if current_production_id in production_id_safe_region:
                    # 过滤安全区域
                    if self._filter_safe_region(result, polygon_regions=production_id_safe_region[task_info.device_sn + topic]):
                        continue
                # Get and process any warnings
                try:
                    warning_flag, warning_information = self.sx_video_stream_detector.get_warning_information(result)
                except Exception as e:
                    self.logger.error(str(e))
                if warning_flag:
                    warning_status = self.detector_warning.warning(
                        warning_information, 
                        device_sn=task_info.device_sn, 
                        stream_url="",  # Empty string as you suggested
                        topic_name=topic.split('?')[0],
                        warning_gap=20,
                        topic=topic
                    )
                    if warning_status:
                        self.logger.warning(f"Warning detected: {topic}")
                        
        except Exception as e:
            error_info = f"Error processing batch for {production_id}: {str(e)}"
            self.logger.error(error_info, exc_info=True)
        
    def _process_single_frame(
            self, 
            frame_task_info: FrameTaskInfo = None, 
        ):
        """Process a single frame with the appropriate detector"""
        production_id = frame_task_info.device_sn + frame_task_info.topic
        if production_id not in self.active_production_lines:
            # Stream was stopped while frame was in queue
            error_info = "Stream was stopped while frame was in queue!"
            raise ValueError(error_info)
        
        production_line_info: ProductionLineInfo = self.active_production_lines[production_id]
        detector = production_line_info.detector
        if not detector:
            error_info = f"No detector found for {production_id}"
            raise ValueError(error_info)
        try:
            # Run inference on the frame
            frame = frame_task_info.frame
            results = detector.predict(frame)

            # Get and process any warnings
            warning_flag, warning_information = self.sx_video_stream_detector.get_warning_information(results)

            # 显式删除大型对象引用，但是依然无法解决内存泄漏的问题
            """
            del frame
            del results
            del detector
            self.logger.info(f"memory_usage: ------------------------------------ {self.memory_monitor.check_memory_usage()}")
            """

            if warning_flag:
                # We would call detector_warning.warning() here
                warning_status = self.detector_warning.warning(
                    warning_information, 
                    device_sn=frame_task_info.device_sn, 
                    stream_url=production_line_info.stream_url, 
                    topic_name=frame_task_info.topic.split('?')[0],
                    warning_gap=20,
                    topic=frame_task_info.topic
                )
                if warning_status:
                    self.logger.warning(f"Warning detected: {frame_task_info.topic}")
        except Exception as e:
            error_info = f"Error processing frame for {production_id}: {str(e)}"
            raise ValueError(error_info) from e

    def keep_main_thread_alive(self):
        """保持主线程运行"""
        try:
            while self.consumer_worker_running:
                time.sleep(1)  # 定期检查线程状态
        except KeyboardInterrupt:
            self.logger.info("收到中断信号，正在关闭...")
        finally:
            self.consumer_worker_running = False

    def _setup_signal_handling(self):
        """
        设置信号处理机制
        支持的信号包括:
        - SIGINT (Ctrl+C)
        - SIGTERM (kill 命令)
        - SIGQUIT (终端退出信号)
        """
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGQUIT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """
        统一的信号处理函数
        
        :param signum: 信号数字 
        :param frame: 当前栈帧
        """
        signal_names = {
            signal.SIGINT: "SIGINT (Ctrl+C)",
            signal.SIGTERM: "SIGTERM",
            signal.SIGQUIT: "SIGQUIT"
        }
        
        signal_name = signal_names.get(signum, f"Signal {signum}")
        self.logger.warning(f"接收到 {signal_name} 信号，准备优雅退出...")

        # 设置停止运行标志
        self._is_running = False
        
        # 执行关闭操作
        self.shutdown()

        # 可选：如果是测试环境可以打印更多调试信息
        if hasattr(self, 'sx_video_stream_detector'):
            self.logger.info("Received exception signal, truncate the sql table!")
            self.sx_video_stream_detector.truncate_sql_table()

    def shutdown(self):
        try:
            # Stop all active production lines
            for production_id in list(self.active_production_lines.keys()):
                device_sn = self.active_production_lines[production_id].device_sn
                self.stop_produce_worker(production_id, device_sn)
            
            # Signal consumer thread to stop
            self.consumer_worker_running = False
            if hasattr(self, 'consumer_worker_thread') and self.consumer_worker_thread.is_alive():
                self.consumer_worker_thread.join(timeout=5.0)
            
            # Shutdown thread pools
            if hasattr(self, 'producer_pool'):
                self.producer_pool.shutdown(wait=True)
            if hasattr(self, 'consumer_pool'):
                self.consumer_pool.shutdown(wait=True)

            # 清空数据库
            self.logger.info("Truncate the sql table!")
            self.sx_video_stream_detector.truncate_sql_table()
            
            self.logger.info("视频流管理器已完全关闭!")
            self.logger.info("Sx video stream producers consumers manager shutdown complete!")
            time.sleep(5)
        except Exception as e:
            self.logger.error(f"shutdown 方法异常! {str(e)}")


