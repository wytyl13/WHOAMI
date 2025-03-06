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

from whoami.tool.base.producer_consumer_manager import ProducerConsumerManager
from whoami.tool.detect.production_line_info import ProductionLineInfo
from whoami.tool.detect.sx_video_stream_detector import SxVideoStreamDetector
from whoami.tool.detect.frame_task_info import FrameTaskInfo
from whoami.tool.detect.sx_detector_warning import SxDetectorWarning
from whoami.tool.base.consumer_tool_pool import ConsumerToolPool
from whoami.tool.detect.detector import Detector

default_device_sn = 'BD3202818'
default_topic_name = '/fire/smoke/warning'
config_path = '/work/ai/WHOAMI/whoami/scripts/detect/detect_config.yaml'
class SxVideoStreamPCM(ProducerConsumerManager):
    detector_warning: SxDetectorWarning = None
    sx_video_stream_detector: SxVideoStreamDetector = None
    _on_run_complete_callback = None
    def __init__(self, max_producers=3, max_consumers=5, production_queue_size=1000, consumer_tool_pool: ConsumerToolPool = None):
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

    def _run(self, topic_dict: dict = None, device_sn: str = None, topic_list: list = None, video_stream_url: str = None):
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
            try:
                topic_name = topic.split('?')[0]
                # Get detector from pool
                consumer_tool = self.consumer_tool_pool.get_consumer_tool(topic_name)
                if consumer_tool is None:
                    raise ValueError(f'Invalid consumer tool! consumer_tool: {consumer_tool}, topic_name: {topic_name}, consumer_tool_pool: {self.consumer_tool_pool}')
                # Start the stream with the video manager
                self.start_produce_worker(
                    device_sn=device_sn,
                    topic=topic,  # Use the full topic string
                    stream_url=video_stream_url,
                    detector=consumer_tool,
                )
                f"Production line started: topic: {topic}, video_stream_url: {video_stream_url}"
            except Exception as e:
                # If stream start fails, return detector to pool
                self.consumer_tool_pool.release_consumer_tool(topic_name, consumer_tool)
                raise ValueError(f"Failed to start stream for {topic}: Error in start new streams: {str(e)}") from e
            
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
        ):
        production_id = device_sn + topic
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
            self.producer_pool.submit(self._read_stream_worker, production_id)
            self.logger.info(f"Started video stream {production_id}!")

    def get_active_topics_base_device_sn(self, device_sn: str = None):
        """Get a list of all active stream topics"""
        active_topics = []
        for production_id, production_line_info in self.active_production_lines.items():
            if production_line_info.device_sn == device_sn:
                active_topics.append(production_line_info.topic)
        return active_topics

    def _read_stream_worker(self, production_id):
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
        while not self.production_line_stop_flags.get(production_id, True):
            try:
                real_topic_list = self.get_active_topics_base_device_sn(device_sn=device_sn)
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    self.logger.error(f"Failed to open stream_url: {stream_url}")
                    if topic in real_topic_list:
                        real_topic_list.remove(topic)
                    self.sx_video_stream_detector.update_sql_video_stream_status(topic_list=real_topic_list, device_sn=device_sn, stream_url=stream_url)
                    stream_url = self.sx_video_stream_detector.get_video_stream_url(device_sn=device_sn)
                    self.active_production_lines[production_id].set_stream_url(stream_url)
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
                        stream_url = self.sx_video_stream_detector.get_video_stream_url(device_sn=device_sn)
                        self.active_production_lines[production_id].set_stream_url(stream_url)
                        break
                    current_time = time.perf_counter()
                    frame_task = FrameTaskInfo(device_sn=device_sn, topic=topic, frame=frame.copy())

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
            if self.consumer_tool_pool and hasattr(ProductionLineInfo, 'topic') and hasattr(ProductionLineInfo, 'detector'):
                topic = production_info.topic
                tool_ = production_info.detector
                if tool_:
                    self.consumer_tool_pool.release_consumer_tool(topic.split('?')[0], tool_)
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
        """Worker function that processes frames from the queue"""
        frame_count = 0
        last_db_update_time = time.perf_counter()
        while self.consumer_worker_running:
            try:
                current_time = time.perf_counter()
                # Get a frame task from the queue
                try:
                    frame_task_info = self.production_queue.get(timeout=0.5)
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
                    self.consumer_pool.submit(self._process_single_frame, frame_task_info)
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
                    frame_count = 0
                    last_db_update_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in frame processor: {str(e)}")
                time.sleep(0.1)  # Short delay on error

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
            results = detector.predict(frame_task_info.frame)
            
            # Get and process any warnings
            warning_flag, warning_information = self.sx_video_stream_detector.get_warning_information(results)
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


