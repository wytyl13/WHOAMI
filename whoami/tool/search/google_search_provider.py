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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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


class GoogleSearchProvider(BaseTool):
    name: Optional[str] = 'GoogleSearchProvider'
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
       
    def remove_duplicate_simhash(self, contents: list[str]) -> tuple[bool, Union[list[str], str]]:
        """remove duplicate paragraph used simhash

        Args:
            contents (list[str]): the paragraph list.
            
        Returns:
            tuple[bool, Union[list[str], str]]: Status and either unique contents or error message
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
            return False, utils.get_error_info("fail to remove duplicate!", e)
        return True, unique_contents
        
    def _init_param(self, key, cx, blocked_domains, snippet_flag, query_num):
        """Initialize parameters from arguments or config file"""
        self.key = key if key is not None else self.key
        self.cx = cx if cx is not None else self.cx
        self.blocked_domains = blocked_domains if blocked_domains is not None else self.blocked_domains
        self.snippet_flag = snippet_flag if snippet_flag is not None else self.snippet_flag
        self.query_num = query_num if query_num is not None else self.query_num
        
        # Load from config file if needed
        if self.search_config is None and self.search_config_path is not None:
            try:
                self.search_config = SearchConfig.from_file(self.search_config_path)
            except Exception as e:
                self.logger.error(f"Failed to load search config from {self.search_config_path}: {str(e)}")
        
        # Apply config values if they exist
        if self.search_config is not None:
            self.key = self.search_config.key if self.key is None else self.key
            self.cx = self.search_config.cx if self.cx is None else self.cx
            self.blocked_domains = self.search_config.blocked_domains if self.blocked_domains is None else self.blocked_domains
            self.snippet_flag = self.search_config.snippet_flag if self.snippet_flag is None else self.snippet_flag
            self.query_num = self.search_config.query_num if self.query_num is None else self.query_num
            
        # Set defaults if still None
        self.blocked_domains = self.blocked_domains or []
        self.snippet_flag = self.snippet_flag if self.snippet_flag is not None else 0
        self.query_num = self.query_num or 10
        
        # Validate critical parameters
        if None in {self.key, self.cx}:
            raise ValueError('Invalid parameters in GoogleSearch: key and cx must be provided')

    def fetch_url_content(self, url: str, max_request_num: int = 3) -> tuple[bool, str]:
        """Fetch content from a URL with retries

        Args:
            url (str): URL to fetch
            max_request_num (int, optional): Maximum number of retry attempts. Defaults to 3.

        Returns:
            tuple[bool, str]: Success status and content or error message
        """
        for attempt in range(max_request_num):
            try:
                response = requests.get(
                    url, timeout=5, headers={"User-Agent": "Mozilla/5.0"}
                )
                response.raise_for_status()
                return True, response.text
            except Exception as e:
                if attempt < max_request_num - 1:
                    self.logger.warning(f"Failed to request, retry {attempt+1}/{max_request_num}: {url}")
                    time.sleep(2)
                else:
                    error_info = utils.get_error_info(f"Failed to request: {url}!", e)
                    self.logger.error(error_info)
                    return False, error_info
        
        # This should never be reached due to the error in the last iteration
        return False, f"Failed to fetch {url} after {max_request_num} attempts"
    
    def preprocess_web_content(self, url, original_content, min_paraph_length: int = 50) -> tuple[bool, Union[list[str], str]]:
        """Preprocess web content to extract clean paragraphs

        Args:
            url (str): Source URL
            original_content: Content to process (str or list)
            min_paraph_length (int, optional): Minimum paragraph length. Defaults to 50.

        Returns:
            tuple[bool, Union[list[str], str]]: Success status and processed content or error message
        """
        try:
            # Convert input to list of paragraphs
            original_content_list = []
            if isinstance(original_content, str):
                if os.path.isfile(original_content):
                    with open(original_content, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    content = original_content
                original_content_list = content.split('\n')
            else:
                original_content_list = original_content
            
            # Filter empty strings
            original_content_list = [x for x in original_content_list if x and x.strip()]
            
            # Log initial statistics
            total_count = len(original_content_list)
            total_length = sum(len(s) for s in original_content_list)
            average_length = total_length / total_count if total_count else 0
            self.logger.info(f"=== Processing content from: {url} ===") 
            self.logger.info(f"Initial statistics: {total_count} paragraphs, {total_length} chars, avg {average_length:.0f} chars/paragraph") 
            
            # Process paragraphs (cleaning and filtering)
            def process_paragraph(p):
                # Clean text
                cleaned_text = re.sub(r'https?://[^\s]+|www\.[^\s]+', '', p)
                cleaned_text = re.sub(r'<[^>]*>', '', cleaned_text)
                cleaned_text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5\s,.!?，。！？；：""''()《》【】（）<>{}]+', '', cleaned_text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                cleaned_text = re.sub(r'([,.!?，。！？；：""''()《》【】（）<>{}])\1+', r'\1', cleaned_text)
                cleaned_text = re.sub(r'[A-Za-z0-9]{9,}', '', cleaned_text)
                cleaned_text = cleaned_text.strip()
                
                # Check if paragraph meets minimum length requirement
                max_length = max(
                    utils.count_chinese_characters(cleaned_text)[1], 
                    utils.count_english_words(cleaned_text)[1]
                )
                if max_length >= min_paraph_length and not cleaned_text.startswith('Sorry'):
                    return cleaned_text
                return None
            
            # Use parallel processing for larger content
            if len(original_content_list) > 10:
                with ThreadPoolExecutor(max_workers=min(20, len(original_content_list))) as executor:
                    results = list(executor.map(process_paragraph, original_content_list))
                    process_result = [r for r in results if r is not None]
            else:
                # Serial processing for smaller content
                process_result = []
                for p in original_content_list:
                    cleaned_text = process_paragraph(p)
                    if cleaned_text:
                        process_result.append(cleaned_text)
            
            # Deduplicate content
            self.logger.info("=== Removing duplicates ===")
            status, process_result = self.remove_duplicate_simhash(process_result)
            if not status:
                self.logger.error("=== Deduplication failed ===")
                return False, process_result
            
            # Log final statistics
            total_count = len(process_result)
            total_length = sum(len(s) for s in process_result)
            average_length = total_length / total_count if total_count else 0
            self.logger.info(f"=== Processing complete: {total_count} paragraphs, {total_length} chars, avg {average_length:.0f} chars/paragraph ===")
            
            return True, process_result
            
        except Exception as e:
            error_info = utils.get_error_info("Error preprocessing web content", e)
            self.logger.error(error_info)
            return False, error_info
    
    def _parse_html(self, url: str, html_content: str) -> str:
        """Parse HTML content using multiple methods and return the best result

        Args:
            url (str): Source URL
            html_content (str): HTML content to parse

        Returns:
            str: Extracted text content
        """
        def general_new_extract(html):
            # Good for Baidu Baike but needs to be combined with beautiful_soup for complete content
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
        
        # Dictionary of parsing functions
        functions = {
            "general_new_extract": general_new_extract,
            "beautiful_soup": beautiful_soup,
            "traifila_extract": traifila_extract,
            "article_extract": article_extract,
            "beautiful_extract_direct": beautiful_extract_direct
        }
        
        # Process all methods in parallel
        results = []
        with ThreadPoolExecutor(max_workers=len(functions)) as executor:
            future_to_name = {executor.submit(func, html_content): name for name, func in functions.items()}
            
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    if result:  # Only add non-empty results
                        results.append({name: result})
                except Exception as e:
                    self.logger.error(f"Error processing {name}: {e}")
        
        def extract_values(d):
            """Recursively extract all values from nested dictionaries and lists"""
            values = []
            for value in d.values():
                if isinstance(value, dict):
                    values.extend(extract_values(value))
                elif isinstance(value, list):
                    values.append(' '.join(map(str, value)))
                else:
                    values.append(str(value))
            return ''.join(values)
        
        def get_longest_value(results):
            """Get the longest result, with special handling for Baidu Baike"""
            longest_value = ""
            baike_result = ""
            
            for item in results:
                for key, value in item.items():
                    # Convert value to string and get its length
                    if isinstance(value, str):
                        current_value = value
                    elif isinstance(value, list):
                        current_value = ''.join(map(str, value))
                    elif isinstance(value, dict):
                        current_value = extract_values(value)
                    else:
                        current_value = str(value)
                    
                    current_length = len(current_value)
                    
                    # Keep track of longest value
                    if current_length > len(longest_value):
                        longest_value = current_value
                    
                    # Special handling for Baidu Baike
                    if 'https://baike.baidu.com' in url and (key == 'beautiful_soup' or key == 'general_new_extract'):
                        baike_result += current_value
            
            # For Baidu Baike, combine results from beautiful_soup and general_new_extract
            if 'https://baike.baidu.com' in url and baike_result:
                return baike_result
                
            return longest_value
        
        # Get and return the best result
        self.logger.info(f"HTML parsing results for {url}: {len(results)} methods successful")
        return get_longest_value(results)
    
    def get_input_schema(self):
        """Return the input schema for the tool"""
        return self.args_schema
    
    def _run(self, *args, **kwds) -> str:
        """Run method required by BaseTool"""
        return self.__call__(*args, **kwds)
    
    def fetch_all_url_contents(self, result_items):
        """Fetch and process all URL contents in parallel

        Args:
            result_items (list): List of search result items with links

        Returns:
            list: Processed items with content
        """
        processed_items = []
        
        def process_item(item):
            """Process a single search result item"""
            try:
                # Fetch URL content
                status, content = self.fetch_url_content(item['link'])
                if not status or not content:
                    self.logger.warning(f"Failed to fetch content from {item['link']}")
                    return None
                    
                # Parse HTML content
                parsed_html = self._parse_html(item['link'], content)
                if not parsed_html:
                    self.logger.warning(f"Failed to parse HTML from {item['link']}")
                    return None
                    
                # Preprocess web content
                status, result = self.preprocess_web_content(url=item['link'], original_content=parsed_html)
                if not status or not result:
                    self.logger.warning(f"Failed to preprocess content from {item['link']}")
                    return None
                    
                # Create a copy of the item with content
                item_copy = item.copy()
                item_copy["fetch_url_content"] = '\n\n'.join(result)
                return item_copy
                
            except Exception as e:
                self.logger.error(f"Error processing URL {item.get('link', 'unknown')}: {str(e)}")
                return None
        
        # Process all items in parallel
        with ThreadPoolExecutor(max_workers=min(10, len(result_items))) as executor:
            futures = [executor.submit(process_item, item) for item in result_items]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        processed_items.append(result)
                except Exception as e:
                    self.logger.error(f"Error in fetch_all_url_contents: {str(e)}")
        
        return processed_items
    
    def __call__(self, *args, **kwds) -> tuple[bool, Union[list, str]]:
        """Main method to handle search requests

        Returns:
            tuple[bool, Union[list, str]]: Success status and search results or error message
        """
        self.logger.info(f"GoogleSearchProvider called with: {kwds}")
        
        # Get and validate query
        query = kwds.get('query', '')
        if not query:
            return False, "Query must not be empty!"
        
        # Set up Google search API parameters
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.key,
            "cx": self.cx,
            "q": query,
            "num": self.query_num or 10,
            "dateRestrict": "d3"
        }
        
        # Make the API request
        try:
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            
            # Check if the response contains results
            if 'error' in search_results:
                error_msg = search_results.get('error', {}).get('message', 'Unknown API error')
                self.logger.error(f"Google API error: {error_msg}")
                return False, f"Google API error: {error_msg}"
                
            # Check if 'items' exists in the response (handle empty results)
            if 'items' not in search_results:
                self.logger.warning(f"No search results found for query: {query}")
                return True, []  # Return empty list instead of error
                
        except requests.exceptions.RequestException as e:
            error_info = utils.get_error_info("Failed to request Google API", e)
            self.logger.error(error_info)
            return False, error_info
        except json.JSONDecodeError as e:
            error_info = utils.get_error_info("Failed to parse Google API response", e)
            self.logger.error(error_info)
            return False, error_info
        except Exception as e:
            error_info = utils.get_error_info("Unexpected error during Google API request", e)
            self.logger.error(error_info)
            return False, error_info
        
        # Extract search results and filter out unwanted content
        try:
            # Extract and filter results
            result = []
            for item in search_results['items']:
                # Skip items from blocked domains or with unwanted extensions
                if any(domain in item['link'] for domain in self.blocked_domains):
                    continue
                if item['link'].endswith(('pdf', 'mp4', 'mp3', 'ashx', 'avi', 'xlsx', 'docx')):
                    continue
                    
                result.append({
                    "title": item.get('title', ''),
                    "link": item.get('link', ''),
                    "html_snippet": item.get('htmlSnippet', '')
                })
                
            self.logger.info(f"Google search returned {len(result)} results after filtering")
            
            # Return snippet results if snippet_flag is set
            if self.snippet_flag:
                return True, result
                
            # Fetch and process URL contents
            fetch_url_content_result = self.fetch_all_url_contents(result)
            return True, fetch_url_content_result
            
        except KeyError as e:
            error_info = utils.get_error_info("Failed to extract search results - missing field", e)
            self.logger.error(error_info)
            return False, error_info
        except Exception as e:
            error_info = utils.get_error_info("Failed to process search results", e)
            self.logger.error(error_info)
            return False, error_info



class GoogleSearchProvider_bake(BaseTool):
    name: Optional[str] = 'GoogleSearchProvider'
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
            "q": query + " recent news",
            "num": self.query_num,
        }
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            search_results = response.json()
        except Exception as e:
            error_info = utils.get_error_info("fail to request google api！", e)
            self.logger.error(error_info)
            return False, error_info
        
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
            return False, error_info
        
        if self.snippet_flag:
            return True, result
        
        # fetch url content
        fetch_url_content_result = []
        for index, item in enumerate(result):
            status, content = self.fetch_url_content(item['link'])
            if status:
                if content:
                    # parse the html content
                    status, result = self.preprocess_web_content(url=item['link'], original_content=self._parse_html(item['link'], content))
                    content = '\n\n'.join(result)
                    item["fetch_url_content"] = content
                    fetch_url_content_result.append(item)
        return True, fetch_url_content_result