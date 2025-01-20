#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/30 17:05
@Author  : weiyutao
@File    : google_search_test.py
"""

from whoami.tool.search.google_search import GoogleSearch
from whoami.agent.planning_agent import PlanningAgent

if __name__ == '__main__':
    
    # google_search = GoogleSearch(snippet_flag=1, search_config_path='/home/weiyutao/work/WHOAMI/whoami/scripts/test/search_config.yaml')
    # param = {
    #     "query": "我是谁"
    # }
    # status, result = google_search(**param)
    # print(len(result))
    
    google_search = GoogleSearch(snippet_flag=1, search_config_path='/home/weiyutao/work/WHOAMI/whoami/scripts/test/search_config.yaml')

    tools = [google_search, google_search]
    agent = PlanningAgent(tools)
    _, result, _ = agent.agent_execute_with_retry("特朗普是谁？")
    print(result)
    