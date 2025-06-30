
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/04 17:58
@Author  : weiyutao
@File    : age_club_content.py
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime, timedelta, date
import asyncio
from typing import (
    overload
)

from whoami.tool.agent.base_tool import tool

@tool
class AgeClubContent:
    end_flag: int = 0
    
    @overload
    def __init__(self, *args, **kwargs):
        ...
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_webpage_content(self):
        """获取资迅所在网页的内容"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
            'Connection': 'keep-alive',
        }
        try:
            url = "https://www.ageclub.net/newsflashes"
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'

            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            formatted_html = soup.prettify()
            return {
                "raw_html": html_content,
                "formatted_html": formatted_html,
                "soup": soup,
                "url": url,
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
        except requests.RequestException as e:
            raise ValueError(f"请求失败！{e}")
        
        
    def extract_date_array(
        self, 
        html_content: str = None, 
        target_date: str = None
    ):
        if html_content is None or html_content == "":
            raise ValueError("html_content must not be null!")
        # 找到目标日期的开始位置
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        # target_date = datetime.now().strftime("%Y-%m-%d")
        # target_date = datetime.now().strftime("%Y-%m-%d")
        # target_date = datetime.now().strftime("%Y-%m-%d")
        date_start = html_content.find(f'"{target_date}":[')
        if date_start == -1:
            return None
        
        # 找到数组开始的位置（[之后）
        array_start = html_content.find('[', date_start) + 1
        
        # 从数组开始位置计算括号匹配，找到对应的结束位置
        bracket_count = 1
        pos = array_start
        
        while pos < len(html_content) and bracket_count > 0:
            if html_content[pos] == '[':
                bracket_count += 1
            elif html_content[pos] == ']':
                bracket_count -= 1
            pos += 1
        
        if bracket_count == 0:
            # 找到了匹配的结束位置，提取数组内容（不包括最外层的[]）
            array_content = html_content[array_start:pos-1]
            return array_content
        else:
            return None

    
    def extract_title_content_from_string(self, data_string):
        """
        使用正则表达式解析title和content
        """
        
        if data_string is None or data_string == "":
            raise ValueError("data string must not be null!")
        result = []
        
        # len(匹配title字段的正则表达式)
        title_pattern = r'title:"([^"]*)"'
        
        # 匹配content字段的正则表达式，考虑可能包含中文引号
        content_pattern = r'content:"((?:[^"\\]|\\.)*)(?<!\\)"'
        
        title_matches = re.findall(title_pattern, data_string)
        content_matches = re.findall(content_pattern, data_string)
        
        # 确保title和content数量匹配
        min_length = min(len(title_matches), len(content_matches))
        
        for i in range(min_length):
            result.append({
                "title": title_matches[i],
                "content": content_matches[i]
            })
        
        return result
    
    
    async def execute(
        self, 
        **kwargs,
    ) -> str:
        result = self.get_webpage_content()
        if result is None:
            return []
        array_content = self.extract_date_array(result["formatted_html"])
        result = self.extract_title_content_from_string(array_content)
        return result
    
    
    
if __name__ == '__main__':
    age_club_content = AgeClubContent()
    
    async def main():
        result = await age_club_content.execute()
        return result
    
    
    result = asyncio.run(main())
    print(result)
