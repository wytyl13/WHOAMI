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
    Type
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
            for index, item in enumerate(data):
                if item == 0:
                    count += 1
                elif count > 0:  # 当遇到非0值且之前有统计到0时
                    all_result.append(count)
                    all_index_list.append(index)
                    count = 0
            # 处理最后一次情况
            if count > 0:
                all_result.append(count)
                all_index_list.append(index)
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
    
    def _sleep_state(self, breath_line, heart_line, create_time, window_size: int = 300):
        """
        Args:
            breath_line (_type_): breath line or breath bpm
            heart_line (_type_): heart line or heart bpm
            window_size (_type_): the size of window, stage each window size seconds.

        Returns:
            _type_: np.array the sleep stage.
        """
        
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
        stage_result = smooth_stages(stages)
        
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
        sleep_time = create_time[real_waking_index[0] + 1] if waking_stage[0] == 0 else create_time[0]

        # 入睡时长，所有清醒时长平均值
        to_sleep_second = waking_second / waking_count
        
        # 醒来时间，最后一次清醒的时间
        waking_time = create_time[real_waking_index[-1]]
        
        result_data = {
            "total_num_second_on_bed": len(breath_line), # 总在床时长（秒）
            "sleep_second": sleep_second, # 睡眠时长（秒）
            "deep_sleep_second": deep_sleep_second, # 深睡时长（秒）
            "waking_second": night_waking_second, # 夜醒时长（秒）
            "to_sleep_second": to_sleep_second, # 入睡时长（秒）
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
            "sleep_efficiency": round(sleep_second / len(breath_line), 2) # 睡眠效率
        }
        self.logger.info("-----------------------------------------------------------------------------------")
        self.logger.info(result_data)
        self.logger.info("-----------------------------------------------------------------------------------")
        
        # self.logger.info("-----------------------------------------------------------------------------------")
        # self.logger.info(f"总监测时长（秒）：{len(breath_line)}")
        # self.logger.info(f"睡眠时长（秒）：{sleep_second}")
        # self.logger.info(f"深睡时长（秒）：{deep_sleep_second}")
        # self.logger.info(f"夜醒时长（秒）：{night_waking_second}")
        # self.logger.info(f"入睡时长（秒）：{to_sleep_second}")
        # self.logger.info(f"上床时间（节点）：{on_bed_time}")
        # self.logger.info(f"入睡时间（节点）：{sleep_time}")
        # self.logger.info(f"醒来时间（节点）：{waking_time}")
        # self.logger.info("-----------------------------------------------------------------------------------")
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
                body_move_result, body_move_index = self.count_consecutive_zeros(batch_state_0_1, 0)
                body_move_count = len(body_move_result)
                body_move_count_list.append(body_move_count)
                create_time_list.append(int(batch_create_time[-1]))
                if index == 0:
                    body_move_count_list.append(0)
                    create_time_list.append(int(batch_create_time[0]))
                # 只考虑第一次，最后一次肯定有结尾
        except Exception as e:
            raise ValueError('fail to exec cal batch body move!') from e
        return body_move_count_list, create_time_list
    
    def process(self):
        # in_out_bed, distance, breath_line, heart_line, breath_bpm, heart_bpm, state, create_time 
        all_data_list = self.init_data(batch_size=60*60*14)
        
        # 统计离床次数 -- 歧义：离床状态连续时间（暂定连续离床状态大于300秒定义为离床，小于300秒但是为离床状态的暂时划分到体动）
        # 根据在离床状态统计离床次数
        # 根据在离床状态统计起床时间、
        # 起床时间为最晚离床状态
        in_out_bed = all_data_list[0][:, 0]
        all_create_time = all_data_list[0][:, -1]
        real_leave_count_result, real_leave_index = self.count_consecutive_zeros(in_out_bed, 300)
        leave_count = len(real_leave_count_result) - 1
        leave_bed_time = all_create_time[real_leave_index[-1]]
        
        # 基于在床数据统计睡眠分区，并进一步计算得到上床时间、入睡时间、醒来时间、夜醒时长、睡眠时长、深睡时长、入睡时长
        in_bed_data = all_data_list[0][in_out_bed != 0]
        breath_line = in_bed_data[:, 2]
        heart_line = in_bed_data[:, 3]
        create_time = in_bed_data[:, -1]
        sleep_result = self._sleep_state(breath_line, heart_line, create_time, 900)
        
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
        breath_bpm = in_bed_data[:, 4]
        heart_bpm = in_bed_data[:, 5]
        average_breath_bpm, max_breath_bpm, min_breath_bpm = self._mean_max_min(breath_bpm)
        average_heart_bpm, max_heart_bpm, min_heart_bpm = self._mean_max_min(heart_bpm)
        sleep_result["average_breath_bpm"] = average_breath_bpm # 平均呼吸率
        sleep_result["max_breath_bpm"] = max_breath_bpm # 最大呼吸率
        sleep_result["min_breath_bpm"] = min_breath_bpm # 最小呼吸率
        sleep_result["average_heart_bpm"] = average_heart_bpm # 平均心率
        sleep_result["max_heart_bpm"] = max_heart_bpm # 最大心率
        sleep_result["min_heart_bpm"] = min_heart_bpm # 最小心率
        
        # 根据在床数据分析体动数据state=0，需要分批次显示
        body_move_count_list, create_time_list = self._cal_batch_body_move()
        body_move_count = sum(body_move_count_list)
        create_time_list.insert(0, int(all_create_time[0]))
        body_move_count_list.append(0)
        create_time_list.append(int(all_create_time[-1]))
        sleep_result["body_move_count"] = body_move_count # 体动次数
        sleep_result["body_move_image_x_y"] = json.dumps([create_time_list, body_move_count_list]) # 体动绘图
        
        # 呼吸异常事件，有心率没有呼吸率，并且设置首尾数据
        breath_exception = np.where((in_bed_data[:, 4] == 0) & (in_bed_data[:, 5] != 0), 0, 1)
        real_breath_exception_result, real_breath_exception_index = self.count_consecutive_zeros(breath_exception, 0)
        sleep_result["breath_exception_count"] = len(real_breath_exception_result) # 呼吸异常次数
        # 设置首尾数据
        real_breath_exception_result.insert(0, 0)
        real_breath_exception_index.insert(0, int(all_create_time[0]))
        real_breath_exception_result.append(0)
        real_breath_exception_index.append(int(all_create_time[-1]))
        sleep_result["breath_exception_image_x_y"] = json.dumps([real_breath_exception_index, real_breath_exception_result]) # 呼吸异常绘图
        
        # 在睡眠分区绘图数据的基础上考虑离床数据
        leave_bed_list = self.find_continuous_sequences(in_out_bed)
        leave_bed_list = [[int(all_create_time[index[0]]), int(all_create_time[index[1]])] for index in leave_bed_list[0] if index[1] - index[0] >= 300]
        sleep_stage_image_x_y = sleep_result["sleep_stage_image_x_y"] # 睡眠分区绘图
        sleep_stage_image_x_y[0] = [[int(create_time[index[0]]), int(create_time[index[1]])]for index in sleep_stage_image_x_y[0]]
        sleep_stage_image_x_y[0].extend(leave_bed_list)
        sleep_stage_image_x_y[1].extend([4] * len(leave_bed_list))
        sleep_result["sleep_stage_image_x_y"] = json.dumps(sleep_stage_image_x_y)
        self.logger.info("-----------------------------------------------------------------------------------")
        self.logger.info(sleep_result)
        self.logger.info("-----------------------------------------------------------------------------------")
        
        recode_ = self.data_provider.sql_provider.add_record(sleep_result)
        
        return recode_


        

        
        
