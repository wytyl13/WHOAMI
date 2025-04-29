#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/28 15:46
@Author  : weiyutao
@File    : ali.py
"""

import os
import sys

from typing import List

from alibabacloud_iqs20240712.client import Client as IQS20240712Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_iqs20240712 import models as iqs20240712_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient


class IPaaSOpenApiDemo:
    def __init__(self):
        # 设置环境变量
        # 更换为自己ak/sk
        os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'] = 'LTAI5tGceMyYFN26fwbSfxJH'
        os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'] = 'At2KTnOs9O29CwtpDgzUs1ozrRryhk'
        pass

    @staticmethod
    def create_client() -> IQS20240712Client:
        """
        初始化AK/SK
        """
        IPaaSOpenApiDemo.__init__(sys.argv[1:])

        """
        使用AK&SK初始化账号Client
        @return: Client
        @throws Exception
        """
        # 工程代码泄露可能会导致 AccessKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考。
        # 建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html。
        config = open_api_models.Config(
            # 必填，请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID。,
            access_key_id=os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'],
            # 必填，请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_SECRET。,
            access_key_secret=os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
        )
        # Endpoint 请参考 https://api.aliyun.com/product/IQS
        config.endpoint = 'iqs.cn-zhangjiakou.aliyuncs.com'
        return IQS20240712Client(config)


    @staticmethod
    def main(args: List[str]) -> None:
        client = IPaaSOpenApiDemo.create_client()
        runtime = util_models.RuntimeOptions()
        headers = {}

        """
        测试关键词搜索接口
        """
        request = iqs20240712_models.PlaceSearchNovaRequest(
            keywords="东淀湿地",
            # types="GAS_STATION|RESTAURANT|HOTEL|ATTRACTION",
            region="0136",
            page=1,
            size=20,
            city_limit=True
        )
        try:
            response = client.place_search_nova_with_options(request, headers, runtime)
            print(response.body)
        except Exception as error:
            print(error.message)
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)


        """
        测试周边搜索接口
        """
        request = iqs20240712_models.NearbySearchNovaRequest(
            keywords="湿地",
            longitude="116.635974",
            latitude="39.542041",
            types="GAS_STATION|RESTAURANT|HOTEL|ATTRACTION",
            page=1,
            size=10,
            radius=3000,
            city_limit=True
        )
        try:
            response = client.nearby_search_nova_with_options(request, headers, runtime)
            print(response.body)
        except Exception as error:
            print(error.message)
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)


        """
        测试地理编码接口
        """
        request = iqs20240712_models.GeoCodeRequest(
            address="北京大学",
            city="北京市",
        )
        try:
            response = client.geo_code_with_options(request, headers, runtime)
            print(response.body)
        except Exception as error:
            print(error.message)
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)


        """
        测试逆地理编码接口
        """
        request = iqs20240712_models.RgeoCodeRequest(
            longitude="116.310918",
            latitude="39.989027",
        )
        try:
            response = client.rgeo_code_with_options(request, headers, runtime)
            print(response.body)
        except Exception as error:
            print(error.message)
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)


        """
        测试驾车路线规划接口
        """
        try:
            request = iqs20240712_models.DrivingDirectionNovaRequest(
                origin_longitude="116.434307",
                origin_latitude="39.90909",
                destination_longitude="117.434446",
                destination_latitude="39.90816",
                plate="",
                car_type="",
            )
            response = client.driving_direction_nova_with_options(request, headers, runtime)
            print(response.body)
        except Exception as error:
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)


        """
        测试步行路线规划接口
        """
        try:
            request = iqs20240712_models.WalkingDirectionNovaRequest(
                origin_longitude="116.466485",
                origin_latitude="39.995197",
                destination_longitude="116.46424",
                destination_latitude="40.020642",
            )
            response = client.walking_direction_nova_with_options(request, headers, runtime)
            print(response.body)
        except Exception as error:
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)

        """
        测试骑行路线规划接口
        """
        try:
            request = iqs20240712_models.BicyclingDirectionNovaRequest(
                origin_longitude="116.434307",
                origin_latitude="39.90909",
                destination_longitude="117.434446",
                destination_latitude="39.90816",
            )
            response = client.bicycling_direction_nova_with_options(request, headers, runtime)
            print(response.body)
        except Exception as error:
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)


        """
        测试公交路线规划接口
        """
        try:
            # 公交路线查询。
            request = iqs20240712_models.TransitIntegratedDirectionRequest(
                destination_city="深圳",
                destination_latitude="22.539211",
                destination_longitude="113.950961",
                origin_city="深圳",
                origin_latitude="22.641356",
                origin_longitude="113.919828",
            )
            response = client.transit_integrated_direction_with_options(request, headers, runtime)
            print(response.body)
        except Exception as error:
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)


if __name__ == '__main__':
    IPaaSOpenApiDemo.main(sys.argv[1:])