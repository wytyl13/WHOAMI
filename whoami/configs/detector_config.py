#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/21 14:14
@Author  : weiyutao
@File    : detector_config.py
"""
import yaml
import os
from typing import Dict, Optional, Union
from enum import Enum
from typing import Optional, Dict, Any
import argparse

from whoami.utils.yaml_model import YamlModel
from whoami.utils.log import Logger
ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIRECTORY, "yaml/detect_config_case.yaml"))

logger = Logger('DetectorConfig')

class DetectorConfig(YamlModel):
    
    topics: Optional[Dict] = None
    model_path: Optional[Dict] = None
    class_list: Optional[Dict] = None
    conf: Optional[Dict] = None
    warning_gap: Optional[float] = None
    url_str_flag: Optional[list] = None
    call_url: Optional[Dict] = None
    get_video_stream_url: Optional[Dict] = None
    upload_url: Optional[Dict] = None
    warning_url: Optional[Dict] = None
    port_dict: Optional[Dict] = None
    sql: Optional[Dict] = None

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default=CONFIG_PATH, help='the detect config case path!')
    args = parser.parse_args()
    detector_config = DetectorConfig.from_file().__dict__
    try:
        with open(args.file_path, "w") as yaml_file:
            yaml.dump(detector_config, yaml_file)
        logger.info(f"success to init the default config yaml file path!{args.file_path}")
    except Exception as e:
        raise ValueError(f"invalid file path!{args.file_path}") from e
    
    