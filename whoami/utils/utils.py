#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/23 17:47
@Author  : weiyutao
@File    : utils.py
"""
import traceback
import os
import shutil
import re
import yaml
from typing import (
    Optional
)
from enum import Enum
import jieba
import numpy as np

from whoami.utils.log import Logger
logger = Logger('Utils')

class StrEnum(str, Enum):
    def __str__(self) -> str:
        # overwrite the __str__ method to implement enum_instance.attribution == enum_instance.attribution.value
        return self.value
    
    def __repr__(self) -> str:
        return f"'{str(self)}'"



class Utils:
    """Utils class what aims to code some generation tools what can be used in all tool, agent or other function.
    """
    def __init__(self) -> None:
        pass
        
    def get_error_info(self, error_info: str, e: Exception):
        """get the error information that involved the error code line and reason.

        Args:
            error_info (str): the error information that you want to raise.
            e (Exception): the error reason.

        Returns:
            _type_: error infomation.
        """
        error_info = traceback.format_exc()
        error = f"{error_info}{str(e)}！\n{error_info}"
        return error

    def init_directory(self, directory: str, delete_flag: int = 0):
        """_summary_

        Args:
            directory (str): the directory path.
            delete_flag (int, optional): whether delete all the files in the exist directory. Defaults to 0.

        Returns:
            _type_: (bool, error_info/success_info)
        """
        try:
            if os.path.exists(directory) and delete_flag == 1:
                shutil.rmtree(directory)
            if not os.path.exists(directory):
                os.makedirs(directory) 
                os.chmod(directory, 0o2755) # 设置setgid位
            return True, f"success to init the directory: {directory}！"
        except Exception as e:
            error_info = f"fail to init the directory: {directory}\n{str(e)}！\n{traceback.format_exc()}"
            logger.error(error_info)
            return False, error_info
    
    def get_files_based_extension(self, directory, file_extension: str):
        """list all the file with the file_extension, no recursive

        Args:
            directory (_type_): _description_
            file_extension (str): file extension just like '.txt'

        Returns:
            _type_: (bool, error_info/list)
        """
        try:
            txt_files = []
            for file in os.listdir(directory):
                if file.endswith(file_extension):
                    txt_files.append(os.path.join(directory, file))
        except Exception as e:
            error_info = self.get_error_info(f"fail to get the extention: {file_extension} file！", e)
            logger.error(error_info)
            return False, error_info
        return True, txt_files

    def count_chinese_characters(self, text):
        try:
            chinese_char_pattern = r'[\u4e00-\u9fff]'
            chinese_chars = re.findall(chinese_char_pattern, text)
        except Exception as e:
            error_info = self.get_error_info("fail to count chinese characters!", e)
            logger.error(error_info)
            return False, error_info
        return True, len(chinese_chars)

    def count_english_words(self, text):
        try:
            words = re.findall(r'\b\w+\b', text)
        except Exception as e:
            error_info = self.get_error_info("fail to count english characters!", e)
            logger.error(error_info)
            return False, error_info
        return True, len(words)

    def read_yaml(self, yaml_file: str):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            raise ValueError('fail to load yaml file!') from e
        return config
    
    def sort_two_list(self, list_one: Optional[list[list[int, int], list[int]]] = None, list_two: Optional[list[list[int, int], list[int]]] = None):
        """
        combined two list and rerank them. each list involved one timestamp range list and correspond label list.
        rerank the timestamp range list and rerank the correspond label list.
        """
        try:
            timestamp_range = list_one[0]
            timestamp_range.extend(list_two[0])
            label_value = list_one[1]
            label_value.extend(list_two[1])
            combined_data = list(zip(timestamp_range, label_value))
            combined_data.sort(key=lambda x: x[0][0])
            timestamps = set()
            for (start, end), _ in combined_data:
                timestamps.add(start)
                timestamps.add(end)
            timestamps = sorted(list(timestamps))
            result = []
            for i in range(len(timestamps) - 1):
                current_time = timestamps[i]
                next_time = timestamps[i + 1]
                active_intervals = []
                for (start, end), value in combined_data:
                    if start <= current_time and end >= next_time:
                        active_intervals.append((value, start))
                if active_intervals:
                    # Sort by start time in descending order
                    active_intervals.sort(key=lambda x: x[1], reverse=True)
                    value = active_intervals[0][0]
                    result.append(([current_time, next_time], value))        
                
            merged_result = []
            for interval in result:
                if (merged_result and 
                    merged_result[-1][1] == interval[1] and 
                    merged_result[-1][0][1] == interval[0][0]):
                    merged_result[-1] = ([merged_result[-1][0][0], interval[0][1]], interval[1])
                else:
                    merged_result.append(interval)
                    
            sorted_timestamps, sorted_labels = zip(*merged_result)
        except Exception as e:
            logger.error(traceback.print_exc())
            raise ValueError('fail to exec sort two list function!') from e
        return [sorted_timestamps, sorted_labels]
        
    
    def remove_stopwords(self, text, stop_words):
        words = jieba.cut(text)
        filtered_words = [word for word in words if word not in stop_words]
        return ''.join(filtered_words).replace(' ', '')
        
    
    def clean_text(self, text):
        try:
            cleaned_text = re.sub(r'https?://[^\s]+|www\.[^\s]+', '', text)
            cleaned_text = re.sub(r'<[^>]*>', '', cleaned_text)
            cleaned_text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5\s,.!?，。！？；：""''()《》【】（）<>{}]+', '', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = re.sub(r'([,.!?，。！？；：""''()《》【】（）<>{}])\1+', r'\1', cleaned_text)
            cleaned_text = re.sub(r'[A-Za-z0-9]{9,}', '', cleaned_text)
            cleaned_text = cleaned_text.strip()
        except Exception as e:
            raise ValueError("fail to exec clean_text function!") from e
        return cleaned_text
    
    
    def create_sliding_windows(
            self, 
            data, 
            window_size=20, 
            step_size=1, 
            field_index=None
        ):
        """
        创建滑动窗口数据
        
        Args:
            data: numpy数组，可以是1D或2D
                - 如果是1D: 直接对该数组做滑动窗口
                - 如果是2D: 需要指定field_index来选择列
            window_size: 窗口大小，默认20
            step_size: 滑动步长，默认1
            field_index: 当data是2D时，指定要处理的列索引
        
        Returns:
            windows: shape为(n_windows, window_size)的numpy数组
        """
        
        # 处理输入数据
        if data.ndim == 1:
            # 1D数据，直接使用
            time_series = data
        elif data.ndim == 2:
            # 2D数据，需要选择列
            if field_index is None:
                raise ValueError("对于2D数据，必须指定field_index")
            time_series = data[:, field_index]
        else:
            raise ValueError("数据维度不支持，只支持1D或2D数组")
        
        # 计算窗口数量
        n_samples = len(time_series)
        n_windows = (n_samples - window_size) // step_size + 1
        
        if n_windows <= 0:
            raise ValueError(f"数据长度({n_samples})小于窗口大小({window_size})")
        
        # 创建滑动窗口
        windows = np.zeros((n_windows, window_size))
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            windows[i] = time_series[start_idx:end_idx]
        return windows



    
    