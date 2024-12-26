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
)
import os
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from scipy import stats
from scipy.signal import find_peaks

from whoami.configs.sql_config import SqlConfig
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sx_data_provider import SxDataProvider
from whoami.provider.base_provider import BaseProvider


ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class HealthReport(BaseProvider):
    sql_config_path: Optional[str] = None
    sql_config: Optional[SqlConfig] = None
    data_provider: Optional[SxDataProvider] = None
    query_date: Optional[str] = None
    device_sn: Optional[str] = None
    
    def __init__(
        self, 
        sql_config_path: Optional[str] = None, 
        sql_config: Optional[SqlConfig] = None, 
        data_provider: Optional[SxDataProvider] = None,
        query_date: Optional[str] = None,
        device_sn: Optional[str] = None
    ) -> None:
        super().__init__()
        self._init_param(sql_config_path, sql_config, data_provider, query_date, device_sn)
    
    def _init_param(self, sql_config_path, sql_config, data_provider, query_date, device_sn):
        self.sql_config_path = sql_config_path
        self.sql_config = sql_config
        self.data_provider = data_provider if data_provider is not None else self.data_provider
        self.query_date = query_date
        self.device_sn = device_sn
        
        if self.sql_config_path is None and self.sql_config is None and self.data_provider is None:
            raise ValueError('sql_config_path, sql_config, data_provider must not be none!')
        if self.data_provider is None:
            if self.query_date is not None and self.device_sn is not None:
                current_date = datetime.strptime(self.query_date, '%Y-%m-%d')
                current_date_str = current_date.strftime('%Y-%m-%d')
                pre_date_str = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
                start = pre_date_str + ' 20:00:00'
                end = current_date_str + ' 09:00:00'
                sql_query = f"SELECT in_out_bed, distance, breath_line, heart_line, breath_bpm, heart_bpm, state, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log WHERE device_sn='{self.device_sn}' AND create_time >= '{start}' AND create_time < '{end}'"
                self.data_provider = SxDataProvider(sql_config_path=sql_config_path, sql_config=sql_config, sql_query=sql_query)
            else:
                self.data_provider = SxDataProvider(sql_config_path=sql_config_path, sql_config=sql_config)

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
            print(all_result)
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
        hours = seconds // 3600
        # 计算剩余的秒数
        remaining_seconds = seconds % 3600
        # 计算分钟
        minutes = remaining_seconds // 60
        # 计算剩余的秒数
        remaining_seconds = remaining_seconds % 60
        return f"{hours}小时{minutes}分钟"
    
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
        
        deep_sleep_second = np.sum(stage_result == 1)
        waking_second = np.sum(stage_result == 3)
        sleep_second = np.sum(stage_result != 3)
        
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
            "总监测时长（秒）": len(breath_line),
            "睡眠时长（秒）": sleep_second,
            "深睡时长（秒）": deep_sleep_second,
            "夜醒时长（秒）": night_waking_second,
            "入睡时长（秒）": to_sleep_second,
            "总监测时长（小时）": len(breath_line),
            "睡眠时长（小时）": self.convert_seconds_to_hhmm(sleep_second),
            "深睡时长（小时）": self.convert_seconds_to_hhmm(deep_sleep_second),
            "夜醒时长（小时）": self.convert_seconds_to_hhmm(night_waking_second),
            "入睡时长（秒）": self.convert_seconds_to_hhmm(to_sleep_second),
            "夜醒次数（次）": waking_count,
            "上床时间（节点）": datetime.fromtimestamp((on_bed_time).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S'),
            "入睡时间（节点）": datetime.fromtimestamp((sleep_time).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S'),
            "醒来时间（节点）": datetime.fromtimestamp((waking_time).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S'),
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
        return mean_, max_, min_
    
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
                create_time_list.append(batch_create_time[-1])
                if index == 0:
                    body_move_count_list.append(0)
                    create_time_list.append(batch_create_time[0])
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
        sleep_result["leave_count"] = leave_count
        sleep_result["leave_bed_time"] = datetime.fromtimestamp((leave_bed_time).astype(np.int32)).strftime('%Y-%m-%d %H:%M:%S')
        
        # 根据在床数据分析呼吸率和心率数据
        breath_bpm = in_bed_data[:, 4]
        heart_bpm = in_bed_data[:, 5]
        mean_breath_bpm, max_breath_bpm, min_breath_bpm = self._mean_max_min(breath_bpm)
        mean_heart_bpm, max_heart_bpm, min_heart_bpm = self._mean_max_min(heart_bpm)
        sleep_result["mean_breath_bpm"] = mean_breath_bpm
        sleep_result["max_breath_bpm"] = max_breath_bpm
        sleep_result["min_breath_bpm"] = min_breath_bpm
        sleep_result["mean_heart_bpm"] = mean_heart_bpm
        sleep_result["max_heart_bpm"] = max_heart_bpm
        sleep_result["min_heart_bpm"] = min_heart_bpm
        
        # 根据在床数据分析体动数据state=0，需要分批次显示
        body_move_count_list, create_time_list = self._cal_batch_body_move()
        body_move_count = sum(body_move_count_list)
        create_time_list.insert(0, all_create_time[0])
        body_move_count_list.append(0)
        create_time_list.append(all_create_time[-1])
        sleep_result["body_move_count"] = body_move_count
        sleep_result["body_move_image_x_y"] = [create_time_list, body_move_count_list]
        
        # 呼吸异常事件，有心率没有呼吸率，并且设置首尾数据
        breath_exception = np.where((in_bed_data[:, 4] == 0) & (in_bed_data[:, 5] != 0), 0, 1)
        real_breath_exception_result, real_breath_exception_index = self.count_consecutive_zeros(breath_exception, 0)
        sleep_result["breath_exception_count"] = len(real_breath_exception_result)
        # 设置首尾数据
        real_breath_exception_result.insert(0, 0)
        real_breath_exception_index.insert(0, all_create_time[0])
        real_breath_exception_result.append(0)
        real_breath_exception_index.append(all_create_time[-1])
        sleep_result["breath_exception_image_x_y"] = [real_breath_exception_index, real_breath_exception_result]
        
        return sleep_result


        

        
        
