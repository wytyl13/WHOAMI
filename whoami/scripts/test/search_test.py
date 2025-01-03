#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/30 17:05
@Author  : weiyutao
@File    : google_search_test.py
"""

from whoami.tool.search.google_search import GoogleSearch


if __name__ == '__main__':
    
    google_search = GoogleSearch(snippet_flag=1, search_config_path='/home/weiyutao/work/WHOAMI/whoami/scripts/test/search_config.yaml')
    param = {
        "query": "我是谁"
    }
    status, result = google_search(**param)
    print(len(result))