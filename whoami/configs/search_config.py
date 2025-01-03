#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/30 17:08
@Author  : weiyutao
@File    : search_config.py
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
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIRECTORY, "yaml/search_config_case.yaml"))

logger = Logger('SqlConfig')

class SearchConfig(YamlModel):
    key: Optional[str] = None
    cx: Optional[str] = None
    snippet_flag: Optional[int] = None
    blocked_domains: Optional[list] = None
    query_num: Optional[int] = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default=CONFIG_PATH, help='the default search config path!')
    args = parser.parse_args()
    detector_config = SearchConfig.from_file().__dict__
    try:
        with open(args.file_path, "w") as yaml_file:
            yaml.dump(detector_config, yaml_file)
        logger.info(f"success to init the default config yaml file path!{args.file_path}")
    except Exception as e:
        raise ValueError(f"invalid file path!{args.file_path}") from e