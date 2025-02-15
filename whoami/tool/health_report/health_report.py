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
    Any,
    List
)
import os
import numpy as np
from datetime import datetime, timedelta, timezone
from torch.utils.data import DataLoader
from scipy import stats
from scipy.signal import find_peaks
import json
import traceback
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from collections import Counter
import matplotlib
import pandas as pd

from whoami.configs.sql_config import SqlConfig
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sx_data_provider import SxDataProvider
from whoami.provider.base_provider import BaseProvider
from whoami.provider.base_ import ModelType
from whoami.tool.health_report.sleep_indices import SleepIndices
from whoami.llm_api.ollama_llm import OllamLLM
from whoami.configs.llm_config import LLMConfig
from whoami.tool.health_report.standard_breath_heart import StandardBreathHeart
from whoami.utils.utils import Utils
from whoami.tool.health_report.pie_legend import PieLegendHandler

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROGRAM_ROOT_DIRECTORY = os.path.abspath(os.path.join(ROOT_DIRECTORY, "../../"))
font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14)
utils = Utils()

llm = OllamLLM(
    LLMConfig.from_file(Path("/home/weiyutao/work/WHOAMI/whoami/scripts/test/ollama_config.yaml")),
    temperature=0.8
)

class HealthReport(BaseProvider):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    sql_provider: Optional[SqlProvider] = None # if pass this variable, you can use the single sql_provider to handle all health_report instance.
    data_provider: Optional[SxDataProvider] = None
    query_date: Optional[str] = None
    device_sn: Optional[str] = None
    model: Type[ModelType] = SleepIndices
    standard_breath_heart: Optional[SqlProvider] = None
    breath_bpm_low: Optional[int] = None
    breath_bpm_high: Optional[int] = None
    heart_bpm_low: Optional[int] = None
    heart_bpm_high: Optional[int] = None
    
    def __init__(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        sql_provider: Optional[SqlProvider] = None,
        data_provider: Optional[SxDataProvider] = None,
        query_date: Optional[str] = None,
        device_sn: Optional[str] = None,
        model: Type[ModelType] = None
    ) -> None:
        super().__init__()

        self._init_param(sql_config_path, sql_config, sql_provider, data_provider, query_date, device_sn, model=model)
        # 查询是否已经存在，否则直接返回
        first_check_condition = {"device_sn": self.device_sn, "query_date": self.query_date}
        check_fields = ['id']
        record_ = self.data_provider.sql_provider.get_record_by_condition(condition=first_check_condition, fields=check_fields)
        if record_:
            raise ValueError(f"exists! {first_check_condition}")
        
        # init the breath heart param
        try:
            self.standard_breath_heart = SqlProvider(StandardBreathHeart, sql_config_path=self.sql_config_path)
            standard_breath_heart_default_data = self.standard_breath_heart.get_record_by_condition(condition={"device_sn": "default_config"})[0]
            self.breath_bpm_low = standard_breath_heart_default_data["breath_bpm_low"] # read from sql where table is sx_device_wavve_vital_sign_config
            self.breath_bpm_high = standard_breath_heart_default_data["breath_bpm_high"] # read from sql where table is sx_device_wavve_vital_sign_config
            self.heart_bpm_low = standard_breath_heart_default_data["heart_bpm_low"] # read from sql where table is sx_device_wavve_vital_sign_config
            self.heart_bpm_high = standard_breath_heart_default_data["heart_bpm_high"] # read from sql where table is sx_device_wavve_vital_sign_config
        except Exception as e:
            raise ValueError("fail to init breath heart bpm low and high!") from e
    
    
    def _init_param(self, sql_config_path, sql_config, sql_provider, data_provider, query_date, device_sn, model):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.sql_provider = sql_provider
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
                sql_query = f"SELECT in_out_bed, signal_intensity, breath_line, heart_line, breath_bpm, heart_bpm, state, body_move_data, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log WHERE device_sn='{self.device_sn}' AND create_time >= '{start}' AND create_time < '{end}'"
                # sql_query = f"SELECT in_out_bed, distance, breath_line, heart_line, breath_bpm, heart_bpm, state, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log WHERE device_sn='{self.device_sn}' AND create_time >= '{start}' AND create_time < '{end}'"
                self.data_provider = SxDataProvider(sql_config_path=self.sql_config_path, sql_config=self.sql_config, sql_provider=self.sql_provider, sql_query=sql_query, model=self.model)
            else:
                self.data_provider = SxDataProvider(sql_config_path=self.sql_config_path, sql_config=self.sql_config, sql_provider=self.sql_provider, model=self.model)

    def init_data(self, batch_size: Optional[int] = 60*60*14):
        dataloader = DataLoader(self.data_provider, batch_size=batch_size, shuffle=False)
        data_list = []
        for batch in dataloader:
            float_array = batch.numpy()
            # float_array = float_array[float_array[:, 1] != 0]
            data_list.append(float_array)
        return data_list
    
    def count_consecutive_zeros(self, data: Optional[Union[list, np.ndarray]] = None, max_consecutive_count: int = 300):
        try:
            data = np.asarray(data) if isinstance(data, list) else data
            if data.size == 0:
                return [], []
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
        
        # signal_std = np.std(signal)
        # wake_thresh = 1.5 * signal_std
        # deep_thresh = 0.5 * signal_std
        # return wake_thresh, deep_thresh
        # 计算整个信号的统计特征
        all_stds = []
        # 使用滑动窗口计算局部标准差
        for i in range(0, len(signal) - window_size, window_size//2):
            window_data = signal[i:i+window_size]
            all_stds.append(np.std(window_data))
        
        all_stds = np.array(all_stds)
        # if all_stds.size == 0:
        #     raise ValueError("all_stds array is empty. Cannot calculate adaptive thresholds.")
        
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
            "breath_bpm": [self.breath_bpm_low, self.breath_bpm_high],
            "heart_bpm": [self.heart_bpm_low, self.heart_bpm_high],
            "breath_exception_exponent": [0, 5]
        }

        weights = {
            "sleep_efficiency": 20,        # 睡眠效率最重要
            "deep_sleep_efficiency": 15,   # 深睡眠效率次之
            "sleep_second": 15,            # 睡眠时长
            "heart_bpm": 10,              # 心率
            "breath_bpm": 10,             # 呼吸率
            "to_sleep_second": 10,         # 入睡时间
            "body_move_exponent": 5,      # 体动指数
            "breath_exception_exponent": 5,      # 呼吸异常指数
            "waking_second": 5,            # 清醒时间
            "leave_count": 5               # 离床次数
        }

        scores = {}
        total_score = 0
        try:
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

            # breath_bpm_status, heart_bpm_status, body_move_status
            breath_bpm_status = "正常" if all([item >= min(reference["breath_bpm"]) and item <= max(reference["breath_bpm"]) for item in [data["max_breath_bpm"], data["min_breath_bpm"]]]) else "异常"
            heart_bpm_status = "正常" if all([item >= min(reference["heart_bpm"]) and item <= max(reference["heart_bpm"]) for item in [data["max_heart_bpm"], data["min_heart_bpm"]]]) else "异常"
            body_move_status = "正常" if data["body_move_exponent"] >= min(reference["body_move_exponent"]) and data["body_move_exponent"] <= max(reference["body_move_exponent"]) else "异常"
        except Exception as e:
            raise ValueError('fail to exec the function _score!') from e
        return round(total_score, 2), scores, breath_bpm_status, heart_bpm_status, body_move_status
    
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
        """
        统计方法睡眠分区
        """
        # 添加Z-score归一化函数
        def normalize_signal(signal):
            mean = np.mean(signal)
            std = np.std(signal)
            return (signal - mean) / std
        
        # 对输入信号进行归一化
        # normalized_breath = normalize_signal(breath_line)
        # normalized_heart = normalize_signal(heart_line)
        
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
            peaks, _ = find_peaks(window_data, distance=window_size//10)
            n_peaks = len(peaks)
            hist, _ = np.histogram(window_data, bins=20, range=(-3, 3))
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
        def smooth_stages(stages, window=window_size*10):
            smoothed = np.copy(stages)
            half_window = window // 2
            for i in range(len(stages)):
                start = max(0, i - half_window)
                end = min(len(stages), i + half_window + 1)
                
                # 中位数
                smoothed[i] = np.median(stages[start:end])
                
                # 众数
                # mode_result = stats.mode(stages[start:end])
                # smoothed[start:end] = int(mode_result.mode)
            return smoothed
        # return stages
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

        # stage_result = []
        # for index, item in enumerate(breath_line):
        #     try:
        #         stage_result.extend(self._sleep_stage(item, heart_line[index], window_size) if details_flag == 0 else self._sleep_stage_details(breath_line, heart_line, window_size))
        #     except Exception as e:
        #         continue

        sleep_stage_image_x_y = self.find_continuous_sequences(stage_result)
        deep_sleep_second = int(np.sum(stage_result == 1))
        waking_second = int(np.sum(stage_result == 3))
        sleep_second = int(np.sum(stage_result != 3))
        light_sleep_second = int(np.sum(stage_result == 2))
        
        waking_stage = np.where(stage_result == 3, 0, 1)
        assert (len(waking_stage) == len(create_time)), f'fail to assert the dimension of waking_stage: {len(waking_stage)} and create_time: {len(create_time)}'
        real_waking_result, real_waking_index = self.count_consecutive_zeros(waking_stage, 30)
        
        # 注意每个分区的睡眠时长都可能为0
        waking_count = len(real_waking_result)
        
        # 夜醒时长，去除第一次入睡时长
        first_waking_sleep_time = real_waking_result[0] if (waking_stage[0] == 0 and waking_count != 0) else 0
        night_waking_second = waking_second - first_waking_sleep_time

        # 上床时间
        on_bed_time = create_time[0]

        # 入睡时间
        sleep_time = create_time[real_waking_index[0][-1] + 1] if (waking_stage[0] == 0 and waking_count != 0) else create_time[0]

        # 入睡时长，所有清醒时长平均值
        to_sleep_second = waking_second / waking_count if waking_count != 0 else waking_second
        
        # 醒来时间，最后一次清醒的时间
        waking_time = create_time[real_waking_index[-1][0]] if waking_count != 0 else create_time[-1]

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
            "deep_sleep_efficiency": round(deep_sleep_second / sleep_second, 2), # 深睡效率
            "light_sleep_second": light_sleep_second, # 浅睡时长（秒）
            "light_sleep_hour": self.convert_seconds_to_hhmm(light_sleep_second) # 浅睡时长（小时）
        }
        self.logger.info("-----------------------------------------------------------------------------------")
        self.logger.info(result_data)
        self.logger.info("-----------------------------------------------------------------------------------")
        
        return result_data, waking_time
    
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
                # 统计体动基于在离床
                in_out_bed = batch[:, 0]
                batch_create_time = batch[:, -1][in_out_bed != 0]
                batch_body_move_0_1 = np.where(batch[:, 7][in_out_bed != 0] != 0, 0, 1)
                # the batch_body_move_01 size might be zero.
                if batch_create_time.size == 0:
                    continue
                assert (len(batch_create_time) == len(batch_body_move_0_1) != 0), 'fail to assert the dimension of data (batch_create_time, batch_body_move_0_1)'
                if index == 0:
                    body_move_count_list.append(0)
                    create_time_list.append(int(batch_create_time[0]))
                body_move_result, body_move_index = self.count_consecutive_zeros(batch_body_move_0_1, 0)
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
        record_ = self.data_provider.sql_provider.get_record_by_condition(condition=condition, fields=['id'])
        if not record_:
            recode_result = self.data_provider.sql_provider.add_record(data)
            return recode_result
        else:
            raise ValueError(f'fail to add record: {condition}, exists!')

    def _check_consist_indices(self, data: Optional[Dict[str, Any]] = None):
        """返回连续指标，连续？晚夜醒时长超标，连续？晚睡眠效率低于最低标准"""
        condition = {"device_sn": self.device_sn}
        record_ = self.data_provider.sql_provider.get_record_by_condition(condition, fields=['query_date', 'waking_second', 'sleep_efficiency'])
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
    
    def draw_line_hear_breath(
        self, 
        breath_bpm, 
        heart_bpm, 
        breath_line,
        heart_line,
        state,
        create_time, 
        all_breath_exception, 
        all_body_move_01,
        in_out_bed,
        sleep_stage_timestamp_range = None, 
        sleep_stage_label = None, 
        query_date_device_sn: str = None
    ):
        
        """assert the same input dimension"""
        assert (len(breath_bpm) == len(heart_bpm) == len(state) == len(create_time) == len(all_breath_exception) == len(all_body_move_01)), \
        "All input arrays must have the same length."
        
        if sleep_stage_timestamp_range is not None and sleep_stage_label is not None:
            assert len(sleep_stage_timestamp_range) == len(sleep_stage_label), \
            "sleep_stage_timestamp_range and sleep_stage_label must have the same length."
        
        # 将睡眠分区和心率呼吸率数据写入csv文件，进行分区结果核对
        intervals = np.array(sleep_stage_timestamp_range)
        sleep_state = np.array(sleep_stage_label)
        timestamps = np.array(create_time)
        
        formatted_times = np.array([
            int(ts)
            for ts in timestamps
        ])
        
        starts = intervals[:, 0]
        ends = intervals[:, 1]
        in_interval = (timestamps[:, np.newaxis] >= starts) & (timestamps[:, np.newaxis] < ends)
        # 匹配不上就是离床4
        matched_values = np.where(in_interval.any(axis=1), sleep_state[in_interval.argmax(axis=1)], 4)
        merged_data = np.column_stack((formatted_times, breath_bpm, heart_bpm, breath_line, heart_line, matched_values))
        df = pd.DataFrame(merged_data, columns=['time', 'breath_bpm', 'heart_bpm', 'breath_line', 'heart_line', 'sleep_state'])
        
        try:
            breath_bpm = breath_bpm
            heart_bpm = heart_bpm
            state = state
            create_time = create_time
            
            # Modified color selection logic for breath rate
            def get_breath_color(state_val, breath_val, exception_val, body_move_01, in_out_bed_val):
                # 考虑因为呼吸异常造成的状态不稳定情况作为状态稳定情况
                if in_out_bed_val == 0:
                    return '#FF1493'
                if body_move_01 == 0:
                    # 体动
                    return 'yellow'
                if state_val != 2:
                    if exception_val == 0:
                        # 呼吸异常造成的状态不稳定，判定为呼吸异常
                        return 'green'
                    return 'gray'  # unstable state
                if exception_val != 0:
                    return '#0F52BA' # normal range
                if breath_val < self.breath_bpm_low:
                    return 'firebrick'    # below normal range
                elif breath_val > self.breath_bpm_high:
                    return 'red'     # above normal range
                return '#0F52BA'       
            
            def get_heart_color(state_val, heart_val, body_move_01, in_out_bed_val):
                # 不考虑因为心率异常造成的状态不稳定为状态稳定情况
                if in_out_bed_val == 0:
                    return '#FF1493'
                if body_move_01 == 0:
                    # 体动
                    return 'yellow'
                if state_val != 2:
                    return 'gray'  # unstable state
                if heart_val < self.heart_bpm_low:
                    return 'firebrick' # below normal range
                elif heart_val > self.heart_bpm_high:
                    return 'red' # above normal range
                return 'lightcoral' # normal range
            
            colors_breath_1 = [get_breath_color(s, b, e, v, d) 
                          for s, b, e, v, d in zip(state, breath_bpm, all_breath_exception, all_body_move_01, in_out_bed)]
            colors_heart_1 = [get_heart_color(s, b, v, d) 
                          for s, b, v, d in zip(state, heart_bpm, all_body_move_01, in_out_bed)]
            
            
            
            # breath_exception_in_bed_low_01 = np.where((in_out_bed != 0) & (all_body_move_01 != 0) & (state == 2) & (all_breath_exception == 0) & (breath_bpm < self.breath_bpm_low), 0, 1)
            # breath_exception_in_bed_high_01 = np.where((in_out_bed != 0) & (all_body_move_01 != 0) & (state == 2) & (all_breath_exception == 0) & (breath_bpm > self.breath_bpm_high), 0, 1)
            breath_exception_in_bed_low_01 = np.where((all_breath_exception == 0) & (breath_bpm < self.breath_bpm_low), 0, 1)
            breath_exception_in_bed_high_01 = np.where((all_breath_exception == 0) & (breath_bpm > self.breath_bpm_high), 0, 1)
            heart_exception_in_bed_low_01 = np.where((in_out_bed != 0) & (all_body_move_01 != 0) & (state == 2) & (heart_bpm < self.heart_bpm_low), 0, 1)
            heart_exception_in_bed_high_01 = np.where((in_out_bed != 0) & (all_body_move_01 != 0) & (state == 2) & (heart_bpm > self.heart_bpm_high), 0, 1)
            green_colors_01 = np.where(np.array(colors_breath_1) == 'green', 0, 1)
            real_result_breath_exception_high_01, _ = self.count_consecutive_zeros(breath_exception_in_bed_high_01, 0)
            real_breath_exception_low_01, _ = self.count_consecutive_zeros(breath_exception_in_bed_low_01, 0)
            real_result_heart_exception_high_01, _ = self.count_consecutive_zeros(heart_exception_in_bed_high_01, 0)
            real_result_heart_exception_low_01, _ = self.count_consecutive_zeros(heart_exception_in_bed_low_01, 0)
            real_result_body_move_list, _ = self.count_consecutive_zeros(all_body_move_01, 0)
            real_result_leave_bed_list, _ = self.count_consecutive_zeros(in_out_bed, 0)
            real_result_green_colors_01_list, _ = self.count_consecutive_zeros(green_colors_01, 0)
            count_breath_exception_high = len(real_result_breath_exception_high_01)
            count_breath_exception_low = len(real_breath_exception_low_01)
            count_heart_exception_high = len(real_result_heart_exception_high_01)
            count_heart_exception_low = len(real_result_heart_exception_low_01)
            green_colors_count = len(real_result_green_colors_01_list)
            body_move_count = len(real_result_body_move_list)
            leave_bed_count = len(real_result_leave_bed_list)
            leave_bed_count = leave_bed_count if in_out_bed[0] != 0 else (leave_bed_count - 1)
            
            x_1 = [datetime.fromtimestamp(item).strftime('%Y-%m-%d %H:%M:%S') for item in create_time]
            
            # fig_1: 呼吸率和心率
            fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))
            
            # ax1.plot(createTime, breathBpm, color=colors, label='breathBpm')
            for i in range(len(breath_bpm) - 1):
                ax1.plot(x_1[i:i+2], breath_bpm[i:i+2], color=colors_breath_1[i], marker='.', markersize=0.1) 
            ax1.set_ylabel('呼吸率', fontproperties=font)
            
            ax2 = ax1.twinx()
            for i in range(len(heart_bpm) - 1):
                ax2.plot(x_1[i:i+2], heart_bpm[i:i+2], color=colors_heart_1[i], marker='.', markersize=0.1) 
            ax2.set_ylabel('心率', fontproperties=font)
            
            # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=10))
            ax1.tick_params(axis='x', labelsize=10)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax1.set_title(f'【{self.query_date} & {self.device_sn}】呼吸率和心率折线图', fontproperties=font, fontsize=16, fontweight='bold')
            
            legend_elements = [
                Line2D([0], [0], color='#FF1493', label=f'【离床({leave_bed_count}次)】', linewidth=2),
                Line2D([0], [0], color='yellow', label=f'【体动({body_move_count}次)】', linewidth=2),
                Line2D([0], [0], color='gray', label='【状态不稳定】', linewidth=2),
                Line2D([0], [0], color='green', label=f'【呼吸异常造成状态不稳定({green_colors_count}次)】', linewidth=2),
                Line2D([0], [0], color='firebrick', label=f'【呼吸异常({count_breath_exception_low}次)】 < {int(self.breath_bpm_low)}', linewidth=2),
                Line2D([0], [0], color='red', label=f'【呼吸异常({count_breath_exception_high}次)】 > {int(self.breath_bpm_high)}', linewidth=2),
                Line2D([0], [0], color='#0F52BA', label='【正常呼吸率】'),
                Line2D([0], [0], color='firebrick', label=f'【心率异常({count_heart_exception_low}次)】 < {int(self.heart_bpm_low)}', linewidth=2),
                Line2D([0], [0], color='red', label=f'【心率异常({count_heart_exception_high}次)】 > {int(self.heart_bpm_high)}', linewidth=2),
                Line2D([0], [0], color='lightcoral', label='【正常心率】')
            ]
            # Add legend with proper styling
            legend = ax1.legend(
                handles=legend_elements, 
                prop={'family': font.get_name(), 'size': 12},
                loc='upper left', 
                bbox_to_anchor=(1.05, 1), 
                borderaxespad=0,
                handlelength=3,  
                # 增加图例线条长度
                handleheight=2,  # 增加图例高度
                handletextpad=2, # 调整文本和图例之间的距离
                borderpad=1,     # 增加图例边框内边距
                labelspacing=0.5,  # 增加图例项之间的间距
                frameon=True 
            )
            # 增加图例的可见性
            legend.get_frame().set_alpha(1)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_linewidth(1)
            
            # fig_2: 睡眠分区
            sleep_stage_count = Counter(sleep_stage_label)
            leave_bed_count = sleep_stage_count[4]
            waking_count = sleep_stage_count[3]

            sleep_stage_label_np = np.array(sleep_stage_label)
            total_seconds = sum((end - start) for start, end in sleep_stage_timestamp_range)
            stage_durations = {
                1: sum((sleep_stage_timestamp_range[item][1] - sleep_stage_timestamp_range[item][0]) 
                    for item in np.where(sleep_stage_label_np == 1)[0]),
                2: sum((sleep_stage_timestamp_range[item][1] - sleep_stage_timestamp_range[item][0]) 
                    for item in np.where(sleep_stage_label_np == 2)[0]),
                3: sum((sleep_stage_timestamp_range[item][1] - sleep_stage_timestamp_range[item][0]) 
                    for item in np.where(sleep_stage_label_np == 3)[0]),
                4: sum((sleep_stage_timestamp_range[item][1] - sleep_stage_timestamp_range[item][0]) 
                    for item in np.where(sleep_stage_label_np == 4)[0])
            }
            
            waking_hour = self.convert_seconds_to_hhmm(stage_durations[3])
            leave_bed_hour = self.convert_seconds_to_hhmm(stage_durations[4])
            deep_sleep_hour = self.convert_seconds_to_hhmm(stage_durations[1])
            light_sleep_hour = self.convert_seconds_to_hhmm(stage_durations[2])
            
            if sleep_stage_timestamp_range is not None and sleep_stage_label is not None:
                sleep_stages_start = [datetime.fromtimestamp(item[0]) for item in sleep_stage_timestamp_range]
                sleep_stages_end = [datetime.fromtimestamp(item[1]) for item in sleep_stage_timestamp_range]
                for i in range(len(sleep_stage_label)):
                    ax3.fill_betweenx(y=[sleep_stage_label[i]-0.5, sleep_stage_label[i]+0.5],
                                  x1=sleep_stages_start[i],
                                  x2=sleep_stages_end[i],
                                  alpha=0.5,
                                  label=f'阶段 {sleep_stage_label[i]}')
                # 添加图例
                handles_sleep_stages = []
                color_map = {
                    1: '#000080',   # 深睡 - 深海军蓝
                    2: '#4169E1',   # 浅睡 - 皇家蓝
                    3: '#8B4513',   # 清醒 - 马鞍棕色
                    4: '#FF1493'    # 离床 - 深粉红色
                }
                stage_labels = {
                    1: f'【深睡({deep_sleep_hour})】',
                    2: f'【浅睡({light_sleep_hour})】',
                    3: f'【清醒({waking_count}次, {waking_hour})】',
                    4: f'【离床({leave_bed_count}次, {leave_bed_hour})】'
                }
                
                ax3.set_ylim(0.5, 4.5)  # 将y轴范围设置为0-5，确保所有值都在可见范围内
                
                for i in range(len(sleep_stage_label)):
                    stage = sleep_stage_label[i]
                    ax3.fill_betweenx(y=[stage-0.4, stage+0.4],
                                    x1=sleep_stages_start[i],
                                    x2=sleep_stages_end[i],
                                    color=color_map[stage],
                                    alpha=0.7)  # 增加不透明度
                
                ax3.tick_params(axis='x', labelsize=10)
                ax3.set_yticks([1, 2, 3, 4])
                ax3.set_yticklabels(['1', '2', '3', '4'])
                ax3.set_ylabel('睡眠分区', fontproperties=font)
                ax3.set_title(f'【{self.query_date} & {self.device_sn}】睡眠分区图', fontproperties=font, fontsize=16, fontweight='bold')

                # 添加网格线以便于查看
                ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
                
                
                """
                普通图例
                # 确保创建所有阶段的图例，即使数据中没有出现
                handles_sleep_stages = [plt.Rectangle((0,0), 1, 1, 
                                                    color=color_map[i], 
                                                    alpha=0.7, 
                                                    label=stage_labels[i],
                                                    linewidth=2
                                                ) 
                                      for i in range(1, 5)]  # 强制创建1-4的所有图例
                """
                """饼状图例"""
                handler_map = {}
                for stage in range(1, 5):
                    percentage = (stage_durations[stage] / total_seconds) * 100
                    patch = matplotlib.patches.Patch(facecolor=color_map[stage], label=stage_labels[stage])
                    handles_sleep_stages.append(patch)
                    handler_map[patch] = PieLegendHandler(percentage)

                # 添加图例，调整位置和大小
                legend = ax3.legend(
                    handles=handles_sleep_stages, 
                    prop={'family': font.get_name(), 'size': 12},
                    loc='upper left', 
                    bbox_to_anchor=(1.05, 1), 
                    borderaxespad=0,
                    handlelength=3,  
                    # 增加图例线条长度
                    handleheight=3,  # 增加图例高度
                    handletextpad=2, # 调整文本和图例之间的距离
                    borderpad=1,     # 增加图例边框内边距
                    labelspacing=0.5,  # 增加图例项之间的间距
                    frameon=True 
                )
                
                # 增加图例的可见性
                legend.get_frame().set_alpha(1)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_linewidth(1)
            # 调整子图间距，确保有足够空间显示图例
            plt.subplots_adjust(right=0.85, hspace=0.3)

            # 保存图像
            save_dir = os.path.join(PROGRAM_ROOT_DIRECTORY, f"out/{self.query_date}")
            status, result = utils.init_directory(save_dir)
            if not status:
                raise ValueError(result)
            save_file_name = f'{query_date_device_sn}_{time.strftime("%Y%m%d%H%M%S")}.png'
            save_csv_name = f'{query_date_device_sn}_{time.strftime("%Y%m%d%H%M%S")}.csv'
            save_file_path = os.path.join(save_dir, save_file_name)
            save_csv_file_path = os.path.join(save_dir, save_csv_name)
            plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
            df.to_csv(save_csv_file_path, index=False)
        except Exception as e:
            raise ValueError('fail to exec draw_line_hear_breath function!') from e
    
    def rank(self):
        try:
            condition = {"query_date": self.query_date}
            record_ = self.data_provider.sql_provider.get_record_by_condition(condition=condition, fields=['id', 'score'])
            scores = [record['score'] for record in record_]
            ids = [record['id'] for record in record_]
            def rank_elements(input_list):
                # 对输入列表进行排序，并保留原始索引
                sorted_indices = sorted(range(len(input_list)), key=lambda i: input_list[i], reverse=True)
                # 初始化排名列表
                rankings = [0] * len(input_list)
                # 根据排序后的索引填充排名
                for rank, index in enumerate(sorted_indices, start=1):
                    rankings[index] = rank
                return rankings
            rank_list = rank_elements(scores)
            total_poeple = len(rank_list)
            self.logger.info(rank_list)
            rank_ = [round((total_poeple - item) / total_poeple, 2) if total_poeple != 0 else 1 for item in rank_list]
            for index, new_rank in enumerate(rank_):
                self.data_provider.sql_provider.update_rank_by_id(ids[index], new_rank)
            return rank_
        except Exception as e:
            raise ValueError('fail to exec _rank function!') from e
    
    def health_advice(self, ):
        
        reference = {
            "waking_second": [0, 1860],
            "sleep_efficiency": [0.8, 1],
            "sleep_second": [21600, 36000],
            "deep_sleep_efficiency": [0.2, 0.6],
            "leave_count": [0, 2],
            "to_sleep_second": [0, 1800],
            "body_move_exponent": [1.25, 15],
            "breath_bpm": [8, 22],
            "heart_bpm": [40, 100],
            "breath_exception_exponent": [0, 5]
        }
        
        try:
            filed_description = self.data_provider.sql_provider.get_field_names_and_descriptions()
            condition = {"query_date": self.query_date}
            record_ = self.data_provider.sql_provider.get_record_by_condition(condition=condition, exclude_fields=['breath_bpm_image_x_y', 'heart_bpm_image_x_y', 'sleep_stage_image_x_y'])
            for health_data in record_:
                if health_data:
                    if 'breath_bpm_image_x_y' in health_data:
                        del health_data['breath_bpm_image_x_y']
                    if 'heart_bpm_image_x_y' in health_data:
                        del health_data['heart_bpm_image_x_y']
            
                health_prompt = f"""
                请您作为一位专业的睡眠健康医生，基于以下睡眠监测数据生成一段专业的健康分析描述。
                
                字段描述：
                {filed_description}
                
                健康标准区间：
                {reference}
                
                实际睡眠数据：
                {health_data}
                
                分析要求：
                1. 严格对照实际数据和标准区间进行分析，确保数值完全准确
                2. 以一段流畅的文字呈现，不使用特殊符号或分段
                3. 优先分析异常指标，重点说明其偏离标准范围的程度
                4. 结合所有指标给出专业的整体诊断判断
                5. 使用专业医学视角，但确保描述通俗易懂
                6. 在描述最后给出针对性的改善建议
                7. 控制总体描述在150字以内
                """
                content = llm.whoami(health_prompt, stream=False)
                self.logger.info(content)
                self.data_provider.sql_provider.update_health_advice_by_id(health_data['id'], content)
        except Exception as e:
            raise ValueError('fail to exec the function health advice!') from e
    
    def split_continuous_data(self, data: np.ndarray, condition_array: np.ndarray) -> List[np.ndarray]:
        """
        Split data array based on continuous non-zero values in condition array
        
        Args:
            data: The data array to be split
            condition_array: Array with boolean/integer values for splitting condition
        
        Returns:
            List of numpy arrays containing split data
        """
        # Find the indices where non-zero values start and end
        try:
                if len(data) == 0:
                        return []
                    
                # Find indices where the condition changes
                change_points = np.where(np.diff(condition_array))[0] + 1
                
                if len(change_points) == 0:
                    # If no changes, return the whole array if condition is True, empty list otherwise
                    return [data] if condition_array[0] else []
                    
                # Add start and end points
                if condition_array[0]:
                    change_points = np.insert(change_points, 0, 0)
                if condition_array[-1]:
                    change_points = np.append(change_points, len(condition_array))
                    
                # Create pairs of start and end indices
                splits = [(change_points[i], change_points[i+1]) 
                        for i in range(0, len(change_points)-1, 2)]
                        
                # Split the data and filter out empty arrays
                result = [data[start:end] for start, end in splits if end > start]
                return [arr for arr in result if arr.size > 0]
        except Exception as e:
            raise ValueError('fail to exec split_continuous_data function!') from e
    
    def get_all_breath_exception(self, all_state, all_breath_bpm, all_heart_bpm, all_body_move_01):
        """考虑因为呼吸异常造成的状态不稳定为状态稳定情况"""
        """呼吸率异常也可能早层体动情况，因此也要兼容处理因为呼吸异常造成体动而导致对呼吸异常的检测"""
        try:
            all_breath_exception = np.where(
                ((all_breath_bpm < self.breath_bpm_low) | (all_breath_bpm > self.breath_bpm_high)) & 
                (all_heart_bpm != 0) & 
                (all_state == 2) & 
                (all_body_move_01 != 0), 0, 1)

            # 重写all_state
            # 如果连续呼吸异常以后出现了连续state != 2的情况，那么将该state的状态置为2，直到下一次出现连续呼吸异常为止
            # 创建一个新的state数组
            overwrite_all_state = all_state.copy()
            
            # 找出呼吸异常的变化点（从正常变为异常或从异常变为正常）
            exception_changes = np.diff(all_breath_exception)
            
            # 找出所有变化点的索引
            change_indices = np.where(exception_changes == 1)[0] + 1
            
            # 如果数组为空，直接返回原始state
            if len(change_indices) == 0:
                return overwrite_all_state
                
            # 为处理首尾，添加起始和结束点
            change_indices = np.concatenate(([0], change_indices, [len(all_breath_exception)]))
            
            # 处理每一段区间
            for i in range(len(change_indices) - 1):
                start, end = change_indices[i], change_indices[i + 1]
                
                # 如果这段区间是非呼吸异常（all_breath_exception == 1）
                if all_breath_exception[start] == 1:
                    # 将该区间内所有非正常状态(state != 2)的点重写为2
                    overwrite_all_state[start:end] = np.where(
                        all_state[start:end] != 2,
                        2,
                        all_state[start:end]
                    )
                    
            all_breath_exception = np.where(
                ((all_breath_bpm < self.breath_bpm_low) | (all_breath_bpm > self.breath_bpm_high)) & 
                (all_heart_bpm != 0) & 
                (overwrite_all_state == 2) & 
                (all_body_move_01 != 0), 0, 1)
        except Exception as e:
            raise ValueError('fail to exec the function ')
        return all_breath_exception
    
    def process(self):
        # in_out_bed, signal_intensity, breath_line, heart_line, breath_bpm, heart_bpm, state, body_move_data, create_time
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
        all_state = all_data_list[0][:, 6]
        all_breath_bpm = all_data_list[0][:, 4]
        all_heart_bpm = all_data_list[0][:, 5]
        all_breath_line = all_data_list[0][:, 2]
        all_heart_line = all_data_list[0][:, 3]
        all_create_time = all_data_list[0][:, -1]
        all_body_move_01 = np.where(all_data_list[0][:, 7] != 0, 0, 1)
        
        # 但是需要考虑一个情况，呼吸异常造成的数据不稳定，不应该影响呼吸异常情况，因为不稳定状态在这里均不作为呼吸异常，所以可能存在漏检
        # 因此需要对该种特殊情况做兼容处理
        # change state status because we should handle the special screen above
        all_breath_exception = self.get_all_breath_exception(all_state=all_state, all_breath_bpm=all_breath_bpm, all_heart_bpm=all_heart_bpm, all_body_move_01=all_body_move_01)
        
        """
        all_breath_exception = np.where(
            ((all_breath_bpm < self.breath_bpm_low) | (all_breath_bpm > self.breath_bpm_high)) & 
            (all_heart_bpm != 0) & 
            (all_state == 2) & 
            (all_body_move_01 != 0), 0, 1)
        """

        assert (len(in_out_bed) == len(all_state) == len(all_breath_bpm) == len(all_heart_bpm) == len(all_create_time) == len(all_body_move_01) == len(all_breath_exception) != 0), \
            f'fail to assert the data (in_out_bed: {len(in_out_bed)}, all_state: {len(all_state)}, all_breath_bpm: {len(all_breath_bpm)}, all_heart_bpm: {len(all_heart_bpm)}, all_create_time: {len(all_create_time)}, all_body_move_01: {len(all_body_move_01)}, all_breath_exception: {len(all_breath_exception)}) dimension!'
        real_leave_count_result, real_leave_index = self.count_consecutive_zeros(in_out_bed, 0)
        leave_count = len(real_leave_count_result) - 1 if real_leave_count_result else 0
        leave_bed_total_second = sum(real_leave_count_result[1:])
        
        timess = datetime.fromtimestamp((all_create_time[real_leave_index[-1][0]]).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info(f"起床时间---------------{timess}")
        leave_bed_time = all_create_time[real_leave_index[-1][0]] if real_leave_index else all_create_time[-1]
        
        # 基于在床数据统计睡眠分区，并进一步计算得到上床时间、入睡时间、醒来时间、夜醒时长、睡眠时长、深睡时长、入睡时长
        # 注意睡眠分区并没有去除掉体动和不稳定状态，因为体动和不稳定也属于在床数据，需要进行分区（不影响根据心率呼吸率进行睡眠分区的结果）
        in_bed_data = all_data_list[0][in_out_bed != 0]
        
        if in_bed_data.size == 0:
            raise ValueError(f"in bed data is empty! device_sn: {self.device_sn}, query_date: {self.query_date}")
        
        try:
            # 分割呼吸线心线
            # in_bed_data_list = self.split_continuous_data(all_data_list[0], in_out_bed != 0)
            # breath_line = [item[:, 2] for item in in_bed_data_list]
            # heart_line = [item[:, 3] for item in in_bed_data_list]

            # 呼吸线心线
            breath_line = in_bed_data[:, 2]
            heart_line = in_bed_data[:, 3]

            # 呼吸率心率
            # breath_line = in_bed_data[:, 4]
            # heart_line = in_bed_data[:, 5]

            create_time = in_bed_data[:, -1]
            assert (len(breath_line) == len(heart_line) == len(create_time) != 0)
            sleep_result, waking_time = self._sleep_state(breath_line, heart_line, create_time, 300, 0)

            # 并入离床次数和离床时间 
            sleep_result["leave_count"] = leave_count # 离床次数
            sleep_result["leave_bed_total_second"] = leave_bed_total_second # 离床总时间（秒）
            sleep_result["leave_bed_total_hour"] = self.convert_seconds_to_hhmm(leave_bed_total_second) # 离床总时间（小时）
            
            # 重置离床时间，因为可能存在监测时间范围内最后一次离床后又回来睡觉，导致离床时间早于醒来时间，因此要根据醒来时间重置离床时间
            # 如果离床时间早于醒来时间，重置离床时间为最后监测时间
            leave_bed_time = leave_bed_time if leave_bed_time > waking_time else all_create_time[-1]
            sleep_result["leave_bed_time"] = datetime.fromtimestamp((leave_bed_time).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S') # 离床时间
            
            # 添加总监测时长
            sleep_result["total_num_second"] = len(in_out_bed)
            sleep_result["total_num_hour"] = self.convert_seconds_to_hhmm(len(in_out_bed))
            sleep_result["query_date"] = self.query_date
            sleep_result["save_file_path"] = "none"
            sleep_result["device_sn"] = self.device_sn # 设备编号
        except Exception as e:
            self.logger.error(traceback.format_exc())
            raise ValueError('fail to cal the basic indices!') from e
        
        # 根据在床数据分析呼吸率和心率数据
        # state != 2、体动均为过滤数据，不纳入计算范围，而在睡眠分区的时候不需要过滤这部分数据，原因请看睡眠分区部分解释
        try:
            state_in_bed = in_bed_data[:, 6]
            body_move_in_bed = np.where(in_bed_data[:, 7] != 0, 0, 1)
            assert (len(state_in_bed) == len(body_move_in_bed) != 0), 'fail to assert the dimension of data (state_in_bed, body_move_in_bed)!'

            valid_indices = (state_in_bed == 2) & (body_move_in_bed != 0)
            breath_bpm_in_bed = in_bed_data[:, 4][valid_indices]
            heart_bpm_in_bed = in_bed_data[:, 5][valid_indices]
            create_time_in_bed = in_bed_data[:, -1][valid_indices]
            assert (len(breath_bpm_in_bed) == len(heart_bpm_in_bed) == len(create_time_in_bed) != 0), 'fail to assert the dimension of data (create_time_in_bed, breath_bpm_in_bed, heart_bpm_in_bed)!'
            average_breath_bpm, max_breath_bpm, min_breath_bpm = self._mean_max_min(breath_bpm_in_bed)
            average_heart_bpm, max_heart_bpm, min_heart_bpm = self._mean_max_min(heart_bpm_in_bed)
            sleep_result["average_breath_bpm"] = int(average_breath_bpm) # 平均呼吸率
            sleep_result["max_breath_bpm"] = int(max_breath_bpm) # 最大呼吸率
            sleep_result["min_breath_bpm"] = int(min_breath_bpm) # 最小呼吸率
            sleep_result["average_heart_bpm"] = int(average_heart_bpm) # 平均心率
            sleep_result["max_heart_bpm"] = int(max_heart_bpm) # 最大心率
            sleep_result["min_heart_bpm"] = int(min_heart_bpm) # 最小心率
        except Exception as e:
            self.logger.error(traceback.format_exc())
            raise ValueError('fail to cal the indices of breath_bpm and heart_bpm!') from e
        
        
        # 根据在床数据分析体动数据，根据体动动量值判断是否属于体动状态
        try:
            body_move_count_list, create_time_list = self._cal_batch_body_move()
            body_move_count = sum(body_move_count_list)
            sleep_result["body_move_count"] = body_move_count # 体动次数
            sleep_result["average_body_move_count"] = body_move_count / len(body_move_count_list) # 平均体动次数
            sleep_result["max_body_move_count"] = max(body_move_count_list) # 最大体动次数
            sleep_result["min_body_move_count"] = min(body_move_count_list) # 最小体动次数
            sleep_result["body_move_exponent"] = round(body_move_count / 30, 2) # 体动指数
            sleep_result["body_move_image_x_y"] = json.dumps([create_time_list, body_move_count_list]) # 体动绘图
        except Exception as e:
            self.logger.error(traceback.format_exc())
            raise ValueError('fail to cal the indices of body move!') from e
        
        # 呼吸异常事件，有心率没有呼吸率，或者呼吸率不在正常范围内。并且设置首尾数据
        try:
            # 添加呼吸心率秒级绘图数据
            # 适配前端数据格式
            breath_bpm_image_x_y = [[datetime.fromtimestamp(int(all_create_time[i]), tz=timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%H:%M:%S"), int(all_breath_bpm[i])] for i in range(0, len(all_create_time.tolist()), 60)]
            heart_bpm_image_x_y = [[datetime.fromtimestamp(int(all_create_time[i]), tz=timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%H:%M:%S"), int(all_heart_bpm[i])] for i in range(0, len(all_create_time.tolist()), 60)]
            
            breath_exception_in_bed = np.where((in_out_bed != 0) & (all_breath_exception == 0), 0, 1)
            real_breath_exception_result, real_breath_exception_index = self.count_consecutive_zeros(breath_exception_in_bed, 0)
            real_breath_exception_result = [item for item in real_breath_exception_result]
            
            def get_start_end_index(breath_bpm, cur_index, data: list):
                start_index = max(0, cur_index - 30)
                end_index = min(len(breath_bpm), cur_index + 30)
                current_length = end_index - start_index
                if current_length < 60:
                    # 需要补全的数量
                    needed_length = 60 - current_length

                    # 向前补充
                    if start_index > 0:
                        additional_start = max(0, start_index - needed_length)
                        start_index = additional_start
                    else:
                        # 向后补充
                        additional_end = min(len(breath_bpm), end_index + needed_length)
                        end_index = additional_end
                    # 确保最终范围不超过 60
                    if end_index - start_index > 60:
                        end_index = start_index + 60
                return data[start_index:end_index]
            
            breath_exception_60S_x = [[datetime.fromtimestamp(int(value), tz=timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%H:%M:%S") for value in get_start_end_index(breath_bpm_in_bed, item[-1], create_time_in_bed)] for item in real_breath_exception_index]
            breath_exception_60S_y = [[int(value) for value in get_start_end_index(breath_bpm_in_bed, item[-1], breath_bpm_in_bed)] for item in real_breath_exception_index]
            breath_exception_60S_x_y = [breath_exception_60S_x, breath_exception_60S_y]
            sleep_result["breath_exception_count"] = len(real_breath_exception_result) # 呼吸异常次数
            sleep_result["breath_exception_exponent"] = round(len(real_breath_exception_result) / 14, 2) # 呼吸异常指数=呼吸异常次数/监测小时数
            
            # 设置首尾数据
            real_breath_exception_index = [int(all_create_time[item[-1]]) for item in real_breath_exception_index]
            real_breath_exception_result.insert(0, 0)
            real_breath_exception_index.insert(0, int(all_create_time[0]))
            real_breath_exception_result.append(0)
            real_breath_exception_result = [item if item == 0 else 1 for item in real_breath_exception_result]
            real_breath_exception_index.append(int(all_create_time[-1]))
            sleep_result["breath_exception_image_x_y"] = json.dumps([real_breath_exception_index, real_breath_exception_result]) # 呼吸异常绘图
            sleep_result["breath_exception_image_sixty_x_y"] = json.dumps(breath_exception_60S_x_y) # 典型呼吸异常事件
        except Exception as e:
            self.logger.error(traceback.format_exc())
            raise ValueError('fail to cal the indices of breath exception!') from e
        
        # 在睡眠分区绘图数据的基础上考虑离床数据
        try:
            real_index_list = [[int(all_create_time[item[0]]), int(all_create_time[item[1]])] for item in list(real_leave_index)]
            sleep_stage_image_x_y = sleep_result["sleep_stage_image_x_y"] # 睡眠分区绘图
            sleep_stage_image_x_y[0] = [[int(create_time[index[0]]), int(create_time[index[1]])]for index in sleep_stage_image_x_y[0]]
            leave_bed_stage_list = [list(real_index_list), [4] * len(real_index_list)]
            sleep_stage_image_x_y = utils.sort_two_list(sleep_stage_image_x_y, leave_bed_stage_list)
            
            # 适配前端，无意义工作
            new_time_stamps = []
            new_values = []
            for (start_time, end_time), value in zip(sleep_stage_image_x_y[0], sleep_stage_image_x_y[1]):
                new_time_stamps.extend(range(start_time, end_time + 1, 60))
                new_values.extend([value] * len(range(start_time, end_time + 1, 60)))
            new_time_stamps = [datetime.fromtimestamp(int(time_stamps), tz=timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%H:%M:%S") for time_stamps in new_time_stamps]
            sleep_stage_image_x_y_customer = [new_time_stamps, new_values]
            sleep_result["sleep_stage_image_x_y"] = json.dumps(sleep_stage_image_x_y_customer)
            
            # 绘图
            """
            self.draw_line_hear_breath(
                breath_bpm=all_breath_bpm, 
                heart_bpm=all_heart_bpm, 
                breath_line=all_breath_line,
                heart_line=all_heart_line,
                state=all_state, 
                create_time=all_create_time, 
                all_breath_exception=all_breath_exception,
                all_body_move_01=all_body_move_01,
                in_out_bed=in_out_bed,
                sleep_stage_timestamp_range = sleep_stage_image_x_y[0],
                sleep_stage_label = sleep_stage_image_x_y[1],
                query_date_device_sn=f'{self.query_date}_{self.device_sn}'
            )
            """

            total_score, detailed_scores, breath_bpm_status, heart_bpm_status, body_move_status = self._score(sleep_result)
            sleep_result["score"] = total_score
            sleep_result["score_name"] = self.classify_total_score(total_score)
            sleep_result["breath_bpm_status"] = breath_bpm_status # 呼吸率状态
            sleep_result["heart_bpm_status"] = heart_bpm_status # 心率状态
            sleep_result["body_move_status"] = body_move_status # 体动状态
            
            # 添加呼吸率心率秒级绘图数据
            sleep_result["breath_bpm_image_x_y"] = json.dumps(breath_bpm_image_x_y) # 呼吸率绘图数据
            sleep_result["heart_bpm_image_x_y"] = json.dumps(heart_bpm_image_x_y) # 心率绘图数据 
            
        except Exception as e:
            self.logger.error(traceback.format_exc())
            raise ValueError('fail to cal the other indices!') from e
        
        # self.logger.info("-----------------------------------------------------------------------------------")
        # self.logger.info(sleep_result)
        # self.logger.info("-----------------------------------------------------------------------------------")
        sleep_result = self._check_consist_indices(sleep_result)
        
        condition = {"device_sn": self.device_sn, "query_date": self.query_date}
        recode_ = self._insert_sleep_indices_data(condition, sleep_result)
        # return sleep_result


        

        
        
