#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/30 17:05
@Author  : weiyutao
@File    : google_search_test.py
"""
import asyncio

from whoami.tool.search.google_search import GoogleSearch
from whoami.agent.planning_agent import PlanningAgent

if __name__ == '__main__':
    
    # google_search = GoogleSearch(snippet_flag=1, search_config_path='/work/ai/WHOAMI/whoami/scripts/test/search_config.yaml')
    # param = {
    #     "query": "舜熙科技"
    # }
    # status, result = google_search(**param)
    # print(len(result))
    # async def main():
    #     google_search = GoogleSearch(snippet_flag=1, search_config_path='/work/ai/WHOAMI/whoami/scripts/test/search_config.yaml')
    #     tools = [google_search, google_search]
    #     agent = PlanningAgent(tools)
    #     status, result, chat_history = await agent.agent_execute_with_retry_async("舜熙科技", retry_times=2)
    #     print(result)
    
    # asyncio.run(main())
    google_search = GoogleSearch(snippet_flag=1, search_config_path='/work/ai/WHOAMI/whoami/scripts/test/search_config.yaml')

    tools = [google_search]
    agent = PlanningAgent(tools)
    _, result, _ = agent.agent_execute_with_retry("舜熙科技")
    print(result)
    