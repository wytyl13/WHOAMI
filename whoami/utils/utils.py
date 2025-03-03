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

from whoami.utils.log import Logger
logger = Logger('Utils')

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
        
        
        
        


    
    