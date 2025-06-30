#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/28 16:29
@Author  : weiyutao
@File    : zhoubian.py
"""
import requests 
from typing import (
    Optional,
    Type
)
from pydantic import BaseModel, Field
import asyncio
import json

from whoami.tool.agent.tool import ApiTool
from whoami.tool.agent.base_tool import tool



class ZhouBianSchema(BaseModel):
    query_district: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="需要查询周边的地址"
    )
    query_type: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="需要查询到目的地的周边类型"
    )
    

@tool
class ZhouBian(ApiTool):
    ak: Optional[str]= None
    url: Optional[str] = None
    args_schema: Type[BaseModel] = ZhouBianSchema
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.url = "https://api.map.baidu.com/geocoding/v3"
        self.ak = "lEnj3LGZkkyUYhdkIF0yYcfw603jq284"
        if 'ak' in kwargs:
            self.ak = kwargs.pop('ak')
        if 'url' in kwargs:
            self.url = kwargs.pop('url')

    
    def parse_result(self, input_json):
        """
        Transform input JSON to required output format.
        
        Args:
            input_json (dict): Input JSON with POI information
            
        Returns:
            list: List of transformed POI dictionaries
        """
        result = []
        try:
            if input_json.get('status') == 0 and 'results' in input_json:
                for poi in input_json['results']:
                    transformed_poi = {
                        "名称": poi.get('name', ''),
                        "地址": poi.get('address', ''),
                        "电话": poi.get('telephone', ''),
                        "距离": f"{poi.get('detail_info', {}).get('distance', '')}米"
                    }
                    result.append(transformed_poi)
        except Exception as e:
            raise ValueError(f"Fail to exec parse_result function!{str(e)}") from e
        return result
    

    async def request_url(self,  **kwargs) -> str:
        url = None
        ak = None
        query_district = None
        query_type = None
        
        if 'ak' in kwargs:
            ak = kwargs.pop('ak')
        if 'url' in kwargs:
            url = kwargs.pop('url')
            
        if 'query_district' in kwargs:
            query_district = kwargs.pop('query_district')
        else:
            if query_district == "" or query_district is None:
                raise ValueError("query_district must not be null!")
            
        if 'query_type' in kwargs:
            query_type = kwargs.pop('query_type')
        else:
            if query_type == "" or query_type is None:
                raise ValueError("query_type must not be null!")
        params = {
            "address": query_district,
            "output": "json",
            "ak": ak,
        }
        try:
            response = requests.get(url=url, params=params, proxies=None, timeout=3)
            # 解析经纬度
            result = response.json()
            if result.get('status') != 0 or 'result' not in result:
                return f"地理编码错误: {result.get('message', '未知错误')}"
            
            # 返回响应的文本内容
            location = result['result']['location']
            lng = location['lng']
            lat = location['lat']
            params_ = {
                "query": query_type,
                "location": f"{lat:.5f},{lng:.5f}",
                "radius": "5000",
                "output": "json",
                "ak": ak,
                "coordtype": "bd09ll",  # 明确指定坐标系
                "scope": 2
            }
            print(f"周边搜索参数: {params_}")
            search_url = 'https://api.map.baidu.com/place/v2/search'
            response = requests.get(url=search_url, params=params_, proxies=None)
            response.raise_for_status()
            if response:
                return self.parse_result(response.json())
        except requests.exceptions.RequestException as e:
            print(f"请求发生错误: {e}")
            return None
    
    
if __name__ == '__main__':
    async def main():
        zhoubian = ZhouBian()
        result = await zhoubian.execute(query_district="山西省运城市盐湖区安邑西路科创城C4三楼301室", query_type="公园$大学")
        print(result)
    
    
    asyncio.run(main())    