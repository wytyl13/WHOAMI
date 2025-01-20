#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/23 17:47
@Author  : weiyutao
@File    : google_search.py
"""
import requests
import logging
import json
import time
from gne import GeneralNewsExtractor
from readability import Document
from bs4 import BeautifulSoup
import trafilatura
from newspaper import Article
import re
import threading
import os
from simhash import Simhash, SimhashIndex
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
from pydantic import BaseModel, Field

from whoami.utils.utils import Utils
from whoami.utils.log import Logger
from whoami.tool.base.base_tool import BaseTool
from whoami.configs.search_config import SearchConfig
utils = Utils()

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(ROOT_DIRECTORY, "search_config_case.yaml"))

class GoogleSearchSchema(BaseModel):
    """TableInf tool schema."""
    query: str = Field(..., description="the search string for google search")



class GoogleSearch(BaseTool):
    name: Optional[str] = 'GoogleSearch'
    description: Optional[str] = 'the tool that you can search human, event and so on.'
    args_schema: Optional[BaseModel] = GoogleSearchSchema
    search_config: Optional[SearchConfig] = None
    search_config_path: Optional[str] = CONFIG_PATH
    key: Optional[str] = None
    cx: Optional[str] = None
    snippet_flag: Optional[int] = None
    blocked_domains: Optional[list] = None
    end_flag: Optional[int] = 0
    query_num: Optional[int] = None

    def __init__(
                self, 
                key: Optional[str] = None, 
                cx: Optional[str] = None,
                search_config: Optional[SearchConfig] = None,
                search_config_path: Optional[str] = CONFIG_PATH,
                snippet_flag: Optional[int] = 0,
                blocked_domains: Optional[list] = None,
                end_flag: Optional[int] = 0,
                query_num: Optional[int] = None
            ) -> None:
        super().__init__()
        self.end_flag = end_flag if end_flag is not None else self.end_flag
        self.search_config_path = search_config_path if search_config_path is not None else self.search_config_path
        self.search_config = search_config if search_config is not None else self.search_config
        self._init_param(key, cx, blocked_domains, snippet_flag, query_num)
       
    def remove_duplicate_simhash(self, contents: list[str]) -> list[str]:
        """remove duplicate paragraph used simhash

        Args:
            contents (list[str]): the paragraph list.
        """
        def get_features(s):
            width = 3
            s = s.lower()
            s = re.sub(r'[^\w]+', '', s)
            return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]
        
        try:
            index = SimhashIndex([], k=3)
            unique_contents = []
            for content in contents:
                simhash_value = Simhash(get_features(content))
                if not index.get_near_dups(simhash_value):
                    unique_contents.append(content)
                    index.add(content, simhash_value)
        except Exception as e:
            return False, self.get_error_info("fail to remove duplicate!", e)
        return True, unique_contents
        
    def _init_param(self, key, cx, blocked_domains, snippet_flag, query_num):
        self.key = key if key is not None else self.key
        self.cx = cx if cx is not None else self.cx
        self.blocked_domains = blocked_domains if blocked_domains is not None else self.blocked_domains
        self.snippet_flag = snippet_flag if snippet_flag is not None else self.snippet_flag
        self.query_num = query_num if query_num  is not None else self.query_num
        if self.search_config is None and self.search_config_path is not None:
            self.search_config = SearchConfig.from_file(self.search_config_path)
        
        if self.search_config is not None:
            self.key = self.search_config.key if self.key is None else self.key
            self.cx = self.search_config.cx if self.cx is None else self.cx
            self.blocked_domains = self.search_config.blocked_domains if self.blocked_domains is None else self.blocked_domains
            self.snippet_flag = self.search_config.snippet_flag if self.snippet_flag is None else self.snippet_flag
            self.query_num = self.search_config.query_num if self.query_num is None else self.query_num
        if None in {self.key, self.cx, self.snippet_flag} or self.blocked_domains is None:
            raise ValueError('invalid parameters in GoogleSearch!')

    def fetch_url_content(self, url: str, max_request_num: int = 3):
        max_request_num -= 1
        try:
            response = requests.get(
                url, timeout=3, headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            return True, response.text
        except Exception as e:
            if max_request_num > 0:
                self.logger.warning(f"fail to request, retry: {url}，the num of rest request: {max_request_num}")
                time.sleep(2)
                return False, self.fetch_url_content(url, max_request_num)
            error_info = utils.get_error_info("fail to request: {url}!", e)
            self.logger.error(error_info)
            return False, error_info
    
    def preprocess_web_content(self, url, original_content, min_paraph_length: int = 50):
        process_result = []
        
        # init the query result
        content = ""
        try:
            original_content_list = []
            if isinstance(original_content, str):
                if os.path.isfile(original_content):
                    with open(original_content, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # and drop the conetent
                else:
                    content = original_content
                original_content_list = content.split('\n')

            else:
                original_content_list = original_content
            
            # drop all the black string.
            original_content_list = [x for x in original_content_list if x and x.strip()]
            self.logger.info(original_content_list)
            total_index_sum = len(original_content_list)
            total_length = sum(len(s) for s in original_content_list)
            average_length = total_length / total_index_sum if original_content_list else 0
            self.logger.info(f"=================Go ahead the link: {url}=================") 
            self.logger.info(f"the result before processing：\n{original_content_list}") 
            self.logger.info(f"=================Number of paragraphs before processing is {total_index_sum}, the total number of words before processing is {total_length}，the average number of words for each paragraphs is {average_length:.0f}=================") 
            for p in original_content_list:
                # first, clear all the url.
                cleaned_text = re.sub(r'https?://[^\s]+|www\.[^\s]+', '', p)
                
                # second, clear all the special character except single english and 标点符号和数字
                # cleaned_text = re.sub(r'[^A-Za-z0-9\s,.!?，。！？；：“”‘’()《》【】]+', '', cleaned_text)
                cleaned_text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5\s,.!?，。！？；：“”‘’()《》【】（）<>{}]+', '', cleaned_text)
                
                # third, replace all the consistant and single space string to one space string for each paragraph.
                # 替换连续的标点为一个，过滤掉连续9个及以上的字符和数字组合
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                cleaned_text = re.sub(r'([,.!?，。！？；：“”‘’()《》【】（）<>{}])\1+', r'\1', cleaned_text)
                cleaned_text = re.sub(r'[A-Za-z0-9]{9,}', '', cleaned_text)
                # third, strip.
                cleaned_text = cleaned_text.strip()
                
                # fourth, drop the content that length less than 50
                # 分别计算清洗结果中的汉字长度和英文长度，选择最大的作为比较长度
                max_length_cleaned_text = max(utils.count_chinese_characters(cleaned_text)[1], utils.count_english_words(cleaned_text)[1])
                if max_length_cleaned_text >= min_paraph_length and not cleaned_text.startswith('Sorry'):
                    process_result.append(cleaned_text)
            self.logger.info(f"=================开始去重！=================") 
            status, process_result = self.remove_duplicate_simhash(process_result)
            if not status:
                self.logger.error(f"=================去重失败！=================") 
                return False, process_result
            self.logger.info(f"=================去重完成！=================") 
            total_index_sum = len(process_result)
            total_length = sum(len(s) for s in process_result)
            average_length = total_length / total_index_sum if process_result else 0
            self.logger.info(f"=================处理后的段落数是{total_index_sum}，总字数是{total_length}，平均段落字数是{average_length:.0f}=================") 
            self.logger.info(f"处理结果如下：\n{process_result}")
        except Exception as e:
            return False, utils.get_error_info("预处理检索结果错误！", e)
        return True, process_result
    
    def _parse_html(self, url: str, html_content: str):
        """多线程执行多个解析html的函数

        Args:
            url (_type_): _description_
            html (_type_): _description_
        """
        def general_new_extract(html):
            # 可以很好的提取百度百科，但是需要和beutiful_soup结合解析的内容才很全
            extractor = GeneralNewsExtractor()
            return extractor.extract(html)
            
        def beautiful_soup(html):
            doc = Document(html)
            content = doc.summary()
            soup = BeautifulSoup(content, "html.parser")
            paragraphs = soup.find_all("p")
            content = "\n".join(
                p.get_text().strip() for p in paragraphs if p.get_text().strip()
            )
            return content

        def traifila_extract(html):
            content = trafilatura.extract(
                html, include_links=False, include_images=False, include_tables=False
            )
            if content:
                content = re.sub(r"\s+", "\n", content).strip()
            return content
   
        def article_extract(html):
            article = Article(url)
            article.set_html(html)
            article.parse()
            content = re.sub(r"\s+", "\n", article.text).strip()
            return content
        
        def beautiful_extract_direct(html):
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = soup.find_all("p")
            content = "\n".join(
                p.get_text().strip() for p in paragraphs if p.get_text().strip()
            )
            return content
        
        functions = {
            "general_new_extract": general_new_extract,
            "beautiful_soup": beautiful_soup,
            "traifila_extract": traifila_extract,
            "article_extract": article_extract,
            "beautiful_extract_direct": beautiful_extract_direct
        }
        
        results = [None] * len(functions)
        def thread_function(index, name, func, html):
            # 因为做了多次尝试，因此这里必须使用索引的方式
            results[index] = {name: func(html)}
        
        def extract_values(d):
            values = []
            for value in d.values():
                if isinstance(value, dict):
                    values.extend(extract_values(value))  # 递归提取嵌套字典的值
                elif isinstance(value, list):
                    values.append(' '.join(map(str, value)))  # 拼接列表中的值
                else:
                    values.append(value)  # 添加非字典和非列表的值
            return ''.join(values)
        
        def get_longest_value(results):
            """result是一个json，其每个字典中对应的值可能是字符串也可能是list
            传入results，输出字典值长度最长的（如果是list，将所有元素进行拼接比较）
            另外还需做特殊处理，如果是url是百度百科，则返回beautiful_soup函数和general_new_extract函数的共同结果
            Args:
                results (_type_): _description_

            Returns:
                _type_: _description_
            """
            longest_value = ""
            baike_result = ""
            for item in results:
                for key, value in item.items():
                    if isinstance(value, str):
                        current_length = len(value)
                    elif isinstance(value, list):
                        value = ''.join(map(str, value))
                        current_length = len(value)
                    elif isinstance(value, dict):
                        value = extract_values(value)
                        current_length = len(value)
                    else:
                        continue
                    longest_value = value if current_length > len(longest_value) else longest_value
                    if 'https://baike.baidu.com' in url and (key == 'beautiful_soup' or key == 'general_new_extract'):
                        baike_result += value
                longest_value = baike_result if 'https://baike.baidu.com' in url else longest_value
            return longest_value
        
        threads = []
        for index, (name, func) in enumerate(functions.items()):
            thread = threading.Thread(target=thread_function, args=(index, name, func, html_content))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
        self.logger.info(f"不同方法从html到text的解析结果：{url}为\n{results}")
        return get_longest_value(results)    
    
    def get_input_schema(self):
        if self.args_schema is not None:
            return self.args_schema
    
    def _run(self, *args, **kwds) -> str:
        return self.__call__(*args, **kwds)
    
    def __call__(self, *args, **kwds) -> str:
        self.logger.info(f"{kwds}")
        
        query = kwds['query'] if 'query' in kwds else ''
        
        if query == '':
            return False, "query must not be null!"
        
        search_url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.key,
            "cx": self.cx,
            "q": query,
            "num": self.query_num
        }
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            search_results = response.json()
        except Exception as e:
            error_info = utils.get_error_info("fail to request google api！", e)
            self.logger.error(error_info)
            return error_info
        
        # extract the interested content from the search results.
        try:
            result = [
                {"title": item['title'], "link": item['link'], "html_snippet": item['htmlSnippet']} 
                for item in search_results['items']
                if not any(domain in item['link'] for domain in self.blocked_domains)
                and not item['link'].endswith(('pdf', 'mp4', 'mp3', 'ashx', 'avi'))
            ]
            self.logger.info(f"google search result: \n{result}")
        except Exception as e:
            error_info = utils.get_error_info("fail to extract interested content.", e)
            self.logger.error(error_info)
            return error_info
        
        if self.snippet_flag:
            return result
        
        # fetch url content
        fetch_url_content_result = []
        for index, item in enumerate(result):
            status, content = self.fetch_url_content(item['link'])
            if status:
                if content:
                    # parse the html content
                    item["fetch_url_content"] = self.preprocess_web_content(url=item['link'], original_content=self._parse_html(item['link'], content))
                    fetch_url_content_result.append(item)
        return fetch_url_content_result