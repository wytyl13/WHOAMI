#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/25 10:08
@Author  : weiyutao
@File    : health_report.py
"""
from pydantic import BaseModel, model_validator, ValidationError
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
    Any
)
import os
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from scipy import stats
from scipy.signal import find_peaks
import json

from whoami.configs.sql_config import SqlConfig
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sx_data_provider import SxDataProvider
from whoami.provider.base_provider import BaseProvider
from whoami.provider.base_ import ModelType
from whoami.tool.health_report.sleep_indices import SleepIndices

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class HealthReport(BaseProvider):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    data_provider: Optional[SxDataProvider] = None
    query_date: Optional[str] = None
    device_sn: Optional[str] = None
    model: Type[ModelType] = SleepIndices
    
    def __init__(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        data_provider: Optional[SxDataProvider] = None,
        query_date: Optional[str] = None,
        device_sn: Optional[str] = None,
        model: Type[ModelType] = None
    ) -> None:
        super().__init__()
        self._init_param(sql_config_path, sql_config, data_provider, query_date, device_sn, model=model)
        # 查询是否已经存在，否则直接返回
        first_check_condition = {"device_sn": self.device_sn, "query_date": self.query_date}
        record_ = self.data_provider.sql_provider.get_record_by_condition(first_check_condition)
        if record_:
            raise ValueError(f"exists! {first_check_condition}")
    
    def _init_param(self, sql_config_path, sql_config, data_provider, query_date, device_sn, model):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.data_provider = data_provider if data_provider is not None else self.data_provider
        self.query_date = query_date
        self.device_sn = device_sn
        self.model=model if model is not None else self.model
        if self.sql_config_path is None and self.sql_config is None and self.data_provider is None:
            raise ValueError('sql_config_path, sql_config, data_provider must not be none!')
        if self.data_provider is None:
            if self.model is None:
                raise ValueError('model must not be null!')
            if self.query_date is not None and self.device_sn is not None:
                current_date = datetime.strptime(self.query_date, '%Y-%m-%d')
                current_date_str = current_date.strftime('%Y-%m-%d')
                pre_date_str = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
                start = pre_date_str + ' 20:00:00'
                end = current_date_str + ' 09:00:00'
                sql_query = f"SELECT in_out_bed, distance, breath_line, heart_line, breath_bpm, heart_bpm, state, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log WHERE device_sn='{self.device_sn}' AND create_time >= '{start}' AND create_time < '{end}'"
                self.data_provider = SxDataProvider(sql_config_path=self.sql_config_path, sql_config=self.sql_config, sql_query=sql_query, model=self.model)
            else:
                self.data_provider = SxDataProvider(sql_config_path=self.sql_config_path, sql_config=self.sql_config, model=self.model)

    def init_data(self, batch_size: Optional[int] = 60*60*14):
        dataloader = DataLoader(self.data_provider, batch_size=batch_size, shuffle=False)
        data_list = []
        # 去掉distance=0的情况
        for batch in dataloader:
            float_array = batch.numpy()
            float_array = float_array[float_array[:, 1] != 0]
            data_list.append(float_array)
        return data_list
    
    def count_consecutive_zeros(self, data: Optional[Union[list, np.ndarray]] = None, max_consecutive_count: int = 300):
        try:
            data = np.asarray(data) if isinstance(data, list) else data
            all_result = []
            all_index_list = []
            count = 0
            start_index = None
            for index, item in enumerate(data):
                if item == 0:
                    if count == 0:  # 记录起始索引
                        start_index = index
                    count += 1
                elif count > 0:  # 当遇到非0值且之前有统计到0时
                    all_result.append(count)
                    all_index_list.append([start_index, index - 1])
                    count = 0
                    start_index = None
            # 处理最后一次情况
            if count > 0:
                all_result.append(count)
                all_index_list.append([start_index, len(data) - 1])
            filtered_results = [(item_result, item_index) for item_result, item_index in zip(all_result, all_index_list) if item_result > max_consecutive_count]
            real_result, real_index_list = zip(*filtered_results) if filtered_results else ([], [])
        except Exception as e:
            raise ValueError('fail to exec the count_consecutive_zeros function!') from e
        return real_result, real_index_list
    
    def _in_out_bed(self, data: Optional[np.ndarray] = None):
        other_column = data[:, 2:-1]
        condition = np.all(other_column == 0, axis=1)
        result = np.where(condition, 0, 1)
        return result
    
    def _calculate_adaptive_thresholds(self, signal, window_size=300):
        """
        计算信号的自适应阈值
        
        参数:
        signal: np.array - 输入信号
        window_size: int - 窗口大小
        
        返回:
        tuple - (wake_threshold, deep_threshold)
        """
        # 计算整个信号的统计特征
        all_stds = []
        
        # 使用滑动窗口计算局部标准差
        for i in range(0, len(signal) - window_size, window_size//2):
            window_data = signal[i:i+window_size]
            all_stds.append(np.std(window_data))
        
        all_stds = np.array(all_stds)
        # 使用分位数来确定阈值
        wake_threshold = np.percentile(all_stds, 75)  # 75分位数作为清醒阈值
        deep_threshold = np.percentile(all_stds, 25)  # 25分位数作为深睡阈值
        
        return wake_threshold, deep_threshold
    
    def _score(self, data: Optional[Union[list, np.ndarray]] = None):
        reference = {
            "waking_second": [0, 1860],
            "sleep_efficiency": [0.8, 1],
            "sleep_second": [21600, 36000],
            "deep_sleep_efficiency": [0.2, 0.6],
            "leave_count": [0, 2],
            "to_sleep_second": [0, 1800],
            "body_move_exponent": [1.25, 15],
            "breath_bpm": [8, 22],
            "heart_bpm": [40, 100]
        }

        weights = {
            "sleep_efficiency": 20,        # 睡眠效率最重要
            "deep_sleep_efficiency": 15,   # 深睡眠效率次之
            "sleep_second": 15,            # 睡眠时长
            "heart_bpm": 10,              # 心率
            "breath_bpm": 10,             # 呼吸率
            "to_sleep_second": 10,         # 入睡时间
            "body_move_exponent": 10,      # 体动指数
            "waking_second": 5,            # 清醒时间
            "leave_count": 5               # 离床次数
        }

        scores = {}
        total_score = 0
        for metric, weight in weights.items():
            if metric not in data or metric not in reference:
                continue
            actual = data[metric]
            ref_min, ref_max = reference[metric]
            # 计算单项得分
            if actual < ref_min:
                # 低于最小值，按比例得分
                score = (actual / ref_min) * weight
            elif actual > ref_max:
                # 高于最大值，按比例得分
                score = (ref_max / actual) * weight
            else:
                # 在区间内，满分
                score = weight
            # 特殊处理：对于一些指标，越低越好
            if metric in ["waking_second", "to_sleep_second", "body_move_exponent", "leave_count"]:
                if actual <= ref_min:
                    score = weight
                elif actual >= ref_max:
                    score = 0
                else:
                    score = ((ref_max - actual) / (ref_max - ref_min)) * weight
            scores[metric] = round(score, 2)
            total_score += score
        return round(total_score, 2), scores
    
    def convert_seconds_to_hhmm(self, seconds):
        # 计算小时
        hours = int(seconds // 3600)
        # 计算剩余的秒数
        remaining_seconds = seconds % 3600
        # 计算分钟
        minutes = int(remaining_seconds // 60)
        # 计算剩余的秒数
        remaining_seconds = int(remaining_seconds % 60)
        return f"{hours}小时{minutes}分钟"
    
    def find_continuous_sequences(self, data: Optional[Union[list, np.ndarray]] = None):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if not data:
            return [], []
        # 初始化结果列表
        index_ranges = []
        value_sequences = []
        start = 0
        # 遍历列表找连续序列
        for i in range(1, len(data)):
            if data[i] != data[i-1]:
                # 当发现不连续时，记录当前序列
                index_ranges.append([start, i-1])
                value_sequences.append(int(data[start]))
                start = i
        # 添加最后一个序列
        index_ranges.append([start, len(data) - 1])
        value_sequences.append(int(data[start]))
        return [index_ranges, value_sequences]
    
    def _sleep_stage(self, breath_line, heart_line, window_size: int = 300):
        # cal the adptive threshold
        hr_wake_thresh, hr_deep_thresh = self._calculate_adaptive_thresholds(heart_line, window_size)
        br_wake_thresh, br_deep_thresh = self._calculate_adaptive_thresholds(breath_line, window_size)
        
        # init the stages.
        stages = np.zeros(len(heart_line))
        def extract_features(signal, start_idx, end_idx):
            window_data = signal[start_idx:end_idx]
            # 计算统计特征
            std = np.std(window_data)
            range_val = np.ptp(window_data)
            peaks, _ = find_peaks(window_data)
            n_peaks = len(peaks)
            hist, _ = np.histogram(window_data, bins=20)
            entropy = stats.entropy(hist + 1e-8)
            
            return np.array([std, range_val, n_peaks, entropy])
        
        # Calculate number of windows
        n_windows = (len(heart_line) - window_size) // (window_size // 2) + 1
        
        # Process each window
        for i in range(n_windows):
            # Calculate window boundaries with 50% overlap
            start_idx = i * (window_size // 2)
            end_idx = min(start_idx + window_size, len(heart_line))
            
            # Extract features for the entire window
            hr_features = extract_features(heart_line, start_idx, end_idx)
            br_features = extract_features(breath_line, start_idx, end_idx)
            
            # Determine sleep stage for the window
            if (hr_features[0] > hr_wake_thresh or br_features[0] > br_wake_thresh):
                stage = 3  # Wake
            elif (hr_features[0] < hr_deep_thresh and br_features[0] < br_deep_thresh):
                stage = 1  # Deep sleep
            else:
                stage = 2  # Light sleep
            # Assign the stage to all points in the window
            stages[start_idx:end_idx] = stage
        # Handle any remaining data points at the end
        if end_idx < len(heart_line):
            last_stage = stages[end_idx - 1]
            stages[end_idx:] = last_stage
        # Smooth transitions between windows
        def smooth_stages(stages, window=900):
            smoothed = np.copy(stages)
            half_window = window // 2
            for i in range(len(stages)):
                start = max(0, i - half_window)
                end = min(len(stages), i + half_window + 1)
                smoothed[i] = np.median(stages[start:end])
            return smoothed
        return smooth_stages(stages)
        
    def _sleep_stage_details(self, breath_line, heart_line, window_size: int = 300):
        # cal the adptive threshold
        hr_wake_thresh, hr_deep_thresh = self._calculate_adaptive_thresholds(heart_line, window_size)
        br_wake_thresh, br_deep_thresh = self._calculate_adaptive_thresholds(breath_line, window_size)
        # init the stages.
        stages = np.zeros(len(heart_line))
        def extract_features(signal, start_idx):
            # 处理窗口末尾
            end_idx = min(start_idx + window_size, len(signal))
            window_data = signal[start_idx:end_idx]
            # 计算统计特征
            std = np.std(window_data)
            range_val = np.ptp(window_data)
            peaks, _ = find_peaks(window_data)
            n_peaks = len(peaks)
            hist, _ = np.histogram(window_data, bins=20)
            entropy = stats.entropy(hist + 1e-8)
            return np.array([std, range_val, n_peaks, entropy])

        for i in range(0, len(heart_line)):
            # 计算窗口的起始位置，确保不会超出数组边界
            window_start = max(0, i - window_size // 2)
            # 提取特征
            hr_features = extract_features(heart_line, window_start)
            br_features = extract_features(breath_line, window_start)
            # 基于特征阈值判断睡眠阶段
            # 这些阈值需要根据实际数据调整
            if (hr_features[0] > hr_wake_thresh or br_features[0] > br_wake_thresh):  # 高波动性
                stages[i] = 3  # 清醒
            elif (hr_features[0] < hr_deep_thresh and br_features[0] < br_deep_thresh):  # 低波动性
                stages[i] = 1  # 深睡
            else:
                stages[i] = 2  # 浅睡
        # 平滑处理：避免频繁的状态切换
        def smooth_stages(stages, window=900):
            smoothed = np.copy(stages)
            half_window = window // 2
            
            for i in range(len(stages)):
                # 计算当前点的窗口范围
                start = max(0, i - half_window)
                end = min(len(stages), i + half_window + 1)
                # 使用中值滤波
                smoothed[i] = np.median(stages[start:end])
            return smoothed
        return smooth_stages(stages)
    
    def _sleep_state(self, breath_line, heart_line, create_time, window_size: int = 300, details_flag: int = 0):
        """
        Args:
            breath_line (_type_): breath line or breath bpm
            heart_line (_type_): heart line or heart bpm
            window_size (_type_): the size of window, stage each window size seconds.

        Returns:
            _type_: np.array the sleep stage.
        """
        stage_result = self._sleep_stage(breath_line, heart_line, window_size) if details_flag == 0 else self._sleep_stage_details(breath_line, heart_line, window_size)
        
        sleep_stage_image_x_y = self.find_continuous_sequences(stage_result)
        deep_sleep_second = int(np.sum(stage_result == 1))
        waking_second = int(np.sum(stage_result == 3))
        sleep_second = int(np.sum(stage_result != 3))
        
        waking_stage = np.where(stage_result == 3, 0, 1)
        real_waking_result, real_waking_index = self.count_consecutive_zeros(waking_stage, 30)
        waking_count = len(real_waking_result)
        
        # 夜醒时长，去除第一次入睡时长
        first_waking_sleep_time = real_waking_result[0] if waking_stage[0] == 0 else 0
        night_waking_second = waking_second - first_waking_sleep_time

        # 上床时间
        on_bed_time = create_time[0]

        # 入睡时间
        sleep_time = create_time[real_waking_index[0][-1] + 1] if waking_stage[0] == 0 else create_time[0]

        # 入睡时长，所有清醒时长平均值
        to_sleep_second = waking_second / waking_count
        
        # 醒来时间，最后一次清醒的时间
        waking_time = create_time[real_waking_index[-1][-1]]
        
        result_data = {
            "total_num_second_on_bed": len(breath_line), # 总在床时长（秒）
            "sleep_second": int(sleep_second), # 睡眠时长（秒）
            "deep_sleep_second": int(deep_sleep_second), # 深睡时长（秒）
            "waking_second": int(night_waking_second), # 夜醒时长（秒）
            "to_sleep_second": int(to_sleep_second), # 入睡时长（秒）
            "total_num_hour_on_bed": self.convert_seconds_to_hhmm(len(breath_line)), # 总在床时长（小时）
            "sleep_hour": self.convert_seconds_to_hhmm(sleep_second), # 睡眠时长（小时）
            "deep_sleep_hour": self.convert_seconds_to_hhmm(deep_sleep_second), # 深睡时长（小时）
            "waking_hour": self.convert_seconds_to_hhmm(night_waking_second), # 夜醒时长（小时）
            "to_sleep_hour": self.convert_seconds_to_hhmm(to_sleep_second), # 入睡时长（小时）
            "waking_count": waking_count, # 夜醒次数（次）
            "on_bed_time": datetime.fromtimestamp((on_bed_time).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S'), # 上床时间（节点）
            "sleep_time": datetime.fromtimestamp((sleep_time).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S'), # 入睡时间（节点）
            "waking_time": datetime.fromtimestamp((waking_time).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S'), # 醒来时间（节点）
            "sleep_stage_image_x_y": sleep_stage_image_x_y, 
            "sleep_efficiency": round(sleep_second / len(breath_line), 2), # 睡眠效率
            "deep_sleep_efficiency": round(deep_sleep_second / sleep_second, 2) # 深睡效率
        }
        self.logger.info("-----------------------------------------------------------------------------------")
        self.logger.info(result_data)
        self.logger.info("-----------------------------------------------------------------------------------")
        
        return result_data
    
    def _mean_max_min(self, data: Optional[np.ndarray] = None):
        try:
            mean_ = np.mean(data)
            max_ = np.max(data)
            min_ = np.min(data)
        except Exception as e:
            raise ValueError('fail to exec mean max min function!') from e
        return round(float(mean_), 2), round(float(max_), 2), round(float(min_), 2)
    
    def _cal_batch_body_move(self):
        try:
            all_batch_list = self.init_data(batch_size=60*30)
            body_move_count_list = []
            create_time_list = []
            for index, batch in enumerate(all_batch_list):
                batch_in_out_bed = batch[:, 0]
                batch_create_time = batch[:, -1]
                batch_state = batch[:, 6][batch_in_out_bed == 1]
                batch_state_0_1 = np.where(batch_state == 0, 0, 1)
                if index == 0:
                    body_move_count_list.append(0)
                    create_time_list.append(int(batch_create_time[0]))
                body_move_result, body_move_index = self.count_consecutive_zeros(batch_state_0_1, 0)
                body_move_count = len(body_move_result)
                body_move_count_list.append(body_move_count)
                create_time_list.append(int(batch_create_time[-1]))
                # 只考虑第一次，最后一次肯定有结尾
        except Exception as e:
            raise ValueError('fail to exec cal batch body move!') from e
        return body_move_count_list, create_time_list
    
    def classify_total_score(self, score: float) -> str:
        # 对0-100分的总分进行等级划分
        
        # Parameters:
        # score (float): 0-100之间的分数
        
        # Returns:
        # str: 等级评价（优秀/良好/一般/较差）
        
        if not 0 <= score <= 100:
            return "分数超出范围"
            
        if score >= 90:
            return "优秀"
        elif score >= 75:
            return "良好"
        elif score >= 60:
            return "一般"
        else:
            return "较差"
    
    def _insert_sleep_indices_data(self, condition: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None):
        record_ = self.data_provider.sql_provider.get_record_by_condition(condition)
        if not record_:
            recode_result = self.data_provider.sql_provider.add_record(data)
            return recode_result
        else:
            raise ValueError(f'fail to add record: {condition}, exists!')

    def _check_consist_indices(self, data: Optional[Dict[str, Any]] = None):
        """返回连续指标，连续？晚夜醒时长超标，连续？晚睡眠效率低于最低标准"""
        condition = {"device_sn": self.device_sn}
        record_ = self.data_provider.sql_provider.get_record_by_condition(condition)
        consist_count_waking_second = 0
        consist_count_sleep_efficiency = 0
        try:
            for record in record_:
                if record["query_date"].strftime('%Y-%m-%d') == self.query_date:
                    raise ValueError("fail to check consist indices, because the current data is exists!")
                if record["waking_second"] > 1860:
                    consist_count_waking_second += 1
                if record["sleep_efficiency"] < 0.8:
                    consist_count_sleep_efficiency += 1
            data["consist_count_waking"] = consist_count_waking_second + 1 if data["waking_second"] > 1860 else consist_count_waking_second
            data["consist_count_sleep_efficiency"] = consist_count_sleep_efficiency + 1 if data["sleep_efficiency"] < 0.8 else consist_count_sleep_efficiency
        except Exception as e:
            raise ValueError("fail to exec check_consist_indices!") from e
        return data
    
    def process(self):
        # in_out_bed, distance, breath_line, heart_line, breath_bpm, heart_bpm, state, create_time 
        all_data_list = self.init_data(batch_size=60*60*14)
        if not all_data_list:
            raise ValueError(f"empty data, device_sn: {self.device_sn}, query_date: {self.query_date}")
        if all_data_list[0].size == 0:
            raise ValueError(f'check data is empty! device_sn: {self.device_sn}')
        # 统计离床次数 -- 歧义：离床状态连续时间（暂定连续离床状态大于300秒定义为离床，小于300秒但是为离床状态的暂时划分到体动）
        # 根据在离床状态统计离床次数
        # 根据在离床状态统计起床时间
        # 起床时间为最晚离床状态
        in_out_bed = all_data_list[0][:, 0]
        all_create_time = all_data_list[0][:, -1]
        real_leave_count_result, real_leave_index = self.count_consecutive_zeros(in_out_bed, 300)
        leave_count = len(real_leave_count_result) - 1 if real_leave_count_result else 0
        leave_bed_time = all_create_time[real_leave_index[-1][-1]] if real_leave_index else all_create_time[-1]
        
        # 基于在床数据统计睡眠分区，并进一步计算得到上床时间、入睡时间、醒来时间、夜醒时长、睡眠时长、深睡时长、入睡时长
        in_bed_data = all_data_list[0][in_out_bed != 0]
        if in_bed_data.size == 0:
            raise ValueError(f"in bed data is empty! device_sn: {self.device_sn}, query_date: {self.query_date}")
        breath_line = in_bed_data[:, 2]
        heart_line = in_bed_data[:, 3]
        create_time = in_bed_data[:, -1]
        sleep_result = self._sleep_state(breath_line, heart_line, create_time, 300, 0)
        # 并入离床次数和离床时间 
        sleep_result["leave_count"] = leave_count # 离床次数
        sleep_result["leave_bed_time"] = datetime.fromtimestamp((leave_bed_time).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S') # 离床时间
        
        # 添加总监测时长
        sleep_result["total_num_second"] = len(in_out_bed)
        sleep_result["total_num_hour"] = self.convert_seconds_to_hhmm(len(in_out_bed))
        sleep_result["query_date"] = self.query_date
        sleep_result["save_file_path"] = "none"
        sleep_result["device_sn"] = self.device_sn # 设备编号
        
        # 根据在床数据分析呼吸率和心率数据
        # 心率、心线--红线正常(state=2)，黑线为不正常(state=0, 1)
        # 呼吸率、呼吸线--蓝线正常(state=1, 2)，黑线为不正常(state=0)
        # bug：在睡眠分区的时候没有去除state异常情况，有差异
        # 如果考虑state的情况，那就需要考虑state不稳定的时候分区，是清醒还是体动，目前是将不稳定的情况归类到体动中，所以体动数据不影响睡眠分区
        # 如果state不稳定的情况是清醒，那么会影响睡眠分区
        # 如果可以根据state的状态直接进行粗分睡眠和清醒，然后再根据心率呼吸率的情况去区分浅睡眠还是深睡眠。这样较科学。
        state = in_bed_data[:, 6]
        breath_bpm = in_bed_data[:, 4][state != 0]
        heart_bpm = in_bed_data[:, 5][state == 2]
        average_breath_bpm, max_breath_bpm, min_breath_bpm = self._mean_max_min(breath_bpm)
        average_heart_bpm, max_heart_bpm, min_heart_bpm = self._mean_max_min(heart_bpm)
        sleep_result["average_breath_bpm"] = int(average_breath_bpm) # 平均呼吸率
        sleep_result["max_breath_bpm"] = int(max_breath_bpm) # 最大呼吸率
        sleep_result["min_breath_bpm"] = int(min_breath_bpm) # 最小呼吸率
        sleep_result["average_heart_bpm"] = int(average_heart_bpm) # 平均心率
        sleep_result["max_heart_bpm"] = int(max_heart_bpm) # 最大心率
        sleep_result["min_heart_bpm"] = int(min_heart_bpm) # 最小心率
        
        # 根据在床数据分析体动数据state=0，需要分批次显示
        body_move_count_list, create_time_list = self._cal_batch_body_move()
        body_move_count = sum(body_move_count_list)
        # create_time_list.insert(0, int(create_time[0]))
        # body_move_count_list.append(0)
        sleep_result["body_move_count"] = body_move_count # 体动次数
        sleep_result["body_move_exponent"] = round(body_move_count / 30, 2) # 体动指数
        sleep_result["body_move_image_x_y"] = json.dumps([create_time_list, body_move_count_list]) # 体动绘图
        
        # 呼吸异常事件，有心率没有呼吸率，并且设置首尾数据
        breath_exception = np.where((in_bed_data[:, 4] == 0) & (in_bed_data[:, 5] != 0), 0, 1)
        real_breath_exception_result, real_breath_exception_index = self.count_consecutive_zeros(breath_exception, 0)
        real_breath_exception_index = [item[-1] for item in real_breath_exception_index]
        sleep_result["breath_exception_count"] = len(real_breath_exception_result) # 呼吸异常次数
        # 设置首尾数据
        real_breath_exception_result.insert(0, 0)
        real_breath_exception_index.insert(0, int(all_create_time[0]))
        real_breath_exception_result.append(0)
        real_breath_exception_index.append(int(all_create_time[-1]))
        sleep_result["breath_exception_image_x_y"] = json.dumps([real_breath_exception_index, real_breath_exception_result]) # 呼吸异常绘图
        
        # 在睡眠分区绘图数据的基础上考虑离床数据
        real_result, real_index_list = self.count_consecutive_zeros(in_out_bed, 300)
        real_index_list = [[int(all_create_time[item[0]]), int(all_create_time[item[1]])] for item in list(real_index_list)]
        sleep_stage_image_x_y = sleep_result["sleep_stage_image_x_y"] # 睡眠分区绘图
        sleep_stage_image_x_y[0] = [[int(create_time[index[0]]), int(create_time[index[1]])]for index in sleep_stage_image_x_y[0]]
        sleep_stage_image_x_y[0].extend(list(real_index_list))
        sleep_stage_image_x_y[1].extend([4] * len(real_index_list))
        sleep_result["sleep_stage_image_x_y"] = json.dumps(sleep_stage_image_x_y)
        total_score, detailed_scores = self._score(sleep_result)
        sleep_result["score"] = total_score
        sleep_result["score_name"] = self.classify_total_score(total_score)
        
        self.logger.info("-----------------------------------------------------------------------------------")
        self.logger.info(sleep_result)
        self.logger.info("-----------------------------------------------------------------------------------")
        sleep_result = self._check_consist_indices(sleep_result)
        
        condition = {"device_sn": self.device_sn, "query_date": self.query_date}
        recode_ = self._insert_sleep_indices_data(condition, sleep_result)
        return sleep_result


        

        
        
