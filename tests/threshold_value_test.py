import pytest
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
    Type,
    Any,
    List
)
import requests
import threading

from whoami.tool.disease_predict.sx_device_wavve_vital_sign_log import SxDeviceWavveVitalSignLog
from whoami.tool.disease_predict.threshold_value import ThresholdValue
from whoami.configs.sql_config import SqlConfig
from whoami.provider.sql_provider import SqlProvider
from whoami.provider.sql_provider import ModelType
from whoami.tool.health_report.sx_device_wavve_vital_sign_realtime import SxDeviceWavveVitalSignLogRealTime
from whoami.tool.health_report.standard_breath_heart import StandardBreathHeart
from whoami.tool.health_report.sx_device_wavve_vital_sign_config_info import DeviceWavveVitalSignConfigInfo


@pytest.mark.parametrize(
    "sql_config_path, sql_config, sql_provider, model, device_sn",
    [
        pytest.param(
            '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml', 
            None, 
            None, 
            SxDeviceWavveVitalSignLog,
            '13CFF349200080712111157C07'
        ),
    ]
)
def test_threshold(
    sql_config_path, 
    sql_config, 
    sql_provider,
    model,
    device_sn
):
    query_date = "2025-7-22"
    
    realtime_sql_provider = SqlProvider(sql_config_path=sql_config_path, sql_config=sql_config, model=SxDeviceWavveVitalSignLogRealTime)
    device_sn_ = realtime_sql_provider.get_record_by_condition({}, fields=["device_sn"])
    device_sn_list = [item['device_sn'] for item in device_sn_]
    
    # standard_sql_provider = SqlProvider(sql_config_path=sql_config_path, sql_config=sql_config, model=StandardBreathHeart)
    # for device_sn in device_sn_list:
    #     threshold_value = ThresholdValue(sql_config_path=sql_config_path, sql_config=sql_config, sql_provider=sql_provider, model=model, device_sn=device_sn)
    #     result = threshold_value._run('2025-2-23')
    #     result_ = standard_sql_provider.upsert_record_by_unique_field("device_sn", result, StandardBreathHeart)
    #     print(result)
    
    
    """
    # 后台线程执行请求，做睡眠报告分析并计算得到阈值详细信息
    """
    request_json = {
        "device_sn": device_sn_list,
        "query_date": query_date
    }
    
    def background_request(json_data):
        try:
            response_ = requests.post("http://localhost:8000/sleep_indices", json=json_data)
            print("后台请求完成")
        except Exception as e:
            print(f"请求发生错误: {e}")
    
    thread = threading.Thread(target=background_request, args=(request_json,))
    thread.start()  # 非阻塞执行
    
    """
    从info表查询到当天的呼吸率心率阈值数据并进行计算分析，然后存储到对应的阈值配置表
    """
    # 获取阈值信息
    sign_config_info_sql_provider = SqlProvider(sql_config_path=sql_config_path, sql_config=sql_config, model=DeviceWavveVitalSignConfigInfo)
    standard_sql_provider = SqlProvider(sql_config_path=sql_config_path, sql_config=sql_config, model=StandardBreathHeart)
    results = sign_config_info_sql_provider.get_record_by_condition(
        condition={"query_date": query_date}, 
        fields=["device_sn", "breath_bpm_low", "breath_bpm_high", "heart_bpm_low", "heart_bpm_high"],
    )
    
    # 更新阈值表
    for result in results:
        result["alarm_time_interval"] = 10
        result_ = standard_sql_provider.upsert_record_by_unique_field("device_sn", result, StandardBreathHeart)
        print(result_)
    
    # 更新redis存储的阈值
    try:
        response = requests.post("http://1.71.15.102:48080/admin-api/device/wavve-vital-sign-config/refresh") 
        response.raise_for_status()        
        print(response.json())
    except Exception as e:
        print(str(e))
