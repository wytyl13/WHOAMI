#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/24 17:56
@Author  : weiyutao
@File    : weather_api.py
"""
from typing import (
    Type,
    Optional,
    Union
)
import requests
import asyncio
import json
from pydantic import BaseModel, Field


from whoami.tool.agent.tool import ApiTool
from whoami.tool.agent.base_tool import tool



class WeatherSchema(BaseModel):
    query_key: str = Field(
        ...,  # 使用 ... 表示必填字段
        description="需要查询天气的地区（可以是直辖市、区县）"
    )



@tool
class WeatherApi(ApiTool):
    ak: Optional[str]= None
    url: Optional[str] = None
    args_schema: Type[BaseModel] = WeatherSchema
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.url = "https://api.map.baidu.com/weather/v1/"
        self.ak = "lEnj3LGZkkyUYhdkIF0yYcfw603jq284"
        if 'ak' in kwargs:
            self.ak = kwargs.pop('ak')
        if 'url' in kwargs:
            self.url = kwargs.pop('url')

        
    def get_district_id_query(self, query):
        return 140802
    
    
    
    
    def parse_weather_json(self, json_str):
        # 解析JSON字符串
        data = json.loads(json_str)
        
        # 提取位置信息
        location = data['result']['location']
        country = location['country']
        province = location['province']
        city = location['city']
        district = location['name']
        
        # 提取当前天气
        now = data['result']['now']
        weather_text = now['text']
        temp = now['temp']
        feels_like = now['feels_like']
        humidity = now['rh']
        wind_class = now['wind_class']
        wind_dir = now['wind_dir']
        update_time = now['uptime']
        formatted_time = f"{update_time[:4]}-{update_time[4:6]}-{update_time[6:8]} {update_time[8:10]}:{update_time[10:12]}"
        
        # 提取预报信息
        forecasts = data['result']['forecasts']
        today = forecasts[0]
        tomorrow = forecasts[1]
        
        # 格式化为一段连贯的话
        result = (
            f"{formatted_time}更新：{country}{province}{city}{district}当前天气{weather_text}，气温{temp}°C，"
            f"体感温度{feels_like}°C，相对湿度{humidity}%，{wind_dir}{wind_class}。"
            f"今天{today['week']}白天{today['text_day']}，{today['wd_day']}{today['wc_day']}，"
            f"夜间{today['text_night']}，气温{today['low']}°C至{today['high']}°C；"
            f"明天{tomorrow['week']}白天{tomorrow['text_day']}，{tomorrow['wd_day']}{tomorrow['wc_day']}，"
            f"夜间{tomorrow['text_night']}，气温{tomorrow['low']}°C至{tomorrow['high']}°C。"
            f"未来几天气温总体在{min([f['low'] for f in forecasts])}°C至{max([f['high'] for f in forecasts])}°C之间，"
            f"以{'晴好' if all(f['text_day'] == '晴' or f['text_night'] == '晴' for f in forecasts) else '多云'}"
            f"天气为主。"
        )
        
        return result
    
        
    async def request_url(self, **kwargs) -> str:
        url = None
        ak = None
        query_key = None
        if 'ak' in kwargs:
            ak = kwargs.pop('ak')
        if 'url' in kwargs:
            url = kwargs.pop('url')
        if 'query_key' in kwargs:
            query_key = kwargs.pop('query_key')
        else:
            if query_key == "" or query_key is None:
                raise ValueError("query_key must not be null!")
            
        district_id = self.get_district_id_query(query_key)
        request_url = f"{url}?district_id={district_id}&data_type=all&ak={ak}"
        try:
            response = requests.get(request_url, proxies=None)
            # 检查请求是否成功
            response.raise_for_status()
            # 返回响应的文本内容
            result = self.parse_weather_json(response.text)
            return result
        except requests.exceptions.RequestException as e:
            print(f"请求发生错误: {e}")
            return None
        
        
if __name__ == '__main__':
    async def main():
        weather_api = WeatherApi()
        print(weather_api)
        result = await weather_api.execute(query_key="who")
        print(result)
    
    
    asyncio.run(main())