#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/31 09:00
@Author  : weiyutao
@File    : draw_line_img_test.py
"""

from whoami.configs.sql_config import SqlConfig
from whoami.provider.sql_provider import SqlProvider
from datetime import datetime, timedelta
from whoami.tool.health_report.sx_data_provider import SxDataProvider
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from datetime import datetime

font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))

def draw_line_image(data, data_name: str):
    """
    [{
        "id": 183988,
        "deviceSn": "13D2F349200080712111954D07", 
        "breathBpm": 12.1429, 呼吸频率
        "breathLine": -0.042671, 呼吸线
        "heartBpm": 100.781, 心率
        "heartLine": 0.371886, 心率线
        "distance": 0.8, 距离
        "signalIntensity": 33.8471, 信号强度
        "state": 2, 可信度
        "createTime": 1731468589000,
        "createDate": 1731427200000
    }]
    """
    if not data:
        return True, '数据为空！'
    
    int_tensor = data["int_tensor"]
    float_tensor = data["float_tensor"]
    state_zero_mask = (int_tensor[:, 2] != 0)
    
    int_tensor = int_tensor[state_zero_mask]
    float_tensor = float_tensor[state_zero_mask]
    
    try:
        breathBpm = int_tensor[:, 0].tolist()
        heartBpm = int_tensor[:, 1].tolist()
        breathLine = float_tensor[:, 0].tolist()
        heartLine = float_tensor[:, 1].tolist()
        create_time = int_tensor[:, 3].tolist()
        state = int_tensor[:, 2].tolist()
        
        createTime = [datetime.utcfromtimestamp(item).strftime('%Y-%m-%d %H:%M:%S') for item in create_time]
        colors_breath = ['black' if item == 0 else 'b' for item in state]
        colors_heart = ['black' if item != 2 else 'r' for item in state]
        
        # fig_1: 呼吸率和心率
        fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))
            
        # ax1.plot(createTime, breathBpm, color=colors, label='breathBpm')
        for i in range(len(breathBpm) - 1):
            ax1.plot(createTime[i:i+2], breathBpm[i:i+2], color=colors_breath[i], marker='.', markersize=0.1) 
        ax1.set_ylabel('呼吸率', fontproperties=font)
        
        ax2 = ax1.twinx()
        for i in range(len(heartBpm) - 1):
            ax2.plot(createTime[i:i+2], heartBpm[i:i+2], color=colors_heart[i], marker='.', markersize=0.1) 
        ax2.set_ylabel('心率', fontproperties=font)
        
        plt.xticks(rotation=45)
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))
        ax1.tick_params(axis='x', labelsize=5)
        
        plt.title('呼吸率和心率折线图-黑线表示状态不稳定', fontproperties=font)
        
        handles = [
            Line2D([0], [0], color='b', lw=2, label='呼吸率'),
            Line2D([0], [0], color='r', lw=2, label='心率'),
            Line2D([0], [0], color='black', lw=2, label='呼吸率或心率不稳定'),
        ]
        plt.legend(handles=handles, prop=font, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)

        # fig_2: 呼吸线和心线
        for i in range(len(breathBpm) - 1):
            ax3.plot(createTime[i:i+2], breathLine[i:i+2], color=colors_breath[i], marker='.', markersize=0.1) 
        ax3.set_ylabel('呼吸线', fontproperties=font)
        
        ax4 = ax3.twinx()
        for i in range(len(heartBpm) - 1):
            ax4.plot(createTime[i:i+2], heartLine[i:i+2], color=colors_heart[i], marker='.', markersize=0.1) 
        ax4.set_ylabel('心线', fontproperties=font)
        
        plt.xticks(rotation=45)
        # ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax3.xaxis.set_major_locator(MaxNLocator(nbins=20))
        ax3.tick_params(axis='x', labelsize=5)
        
        plt.title('呼吸线和心线折线图-黑线表示状态不稳定', fontproperties=font)
        
        handles = [
            Line2D([0], [0], color='b', lw=2, label='呼吸线'),
            Line2D([0], [0], color='r', lw=2, label='心线'),
            Line2D([0], [0], color='black', lw=2, label='呼吸率或心率不稳定'),
        ]
        plt.legend(handles=handles, prop=font, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
        save_file_name = f'{data_name}_{time.strftime("%Y%m%d%H%M%S")}.png'
        save_file_path = os.path.join(OUT_PATH, save_file_name)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        raise ValueError("fail to plot the image!") from e    
    return True

def draw_line_hear_breath(breath_bpm_1, heart_bpm_1, state_1, create_time_1, breath_bpm_2, heart_bpm_2, state_2, create_time_2, date_name):
    breath_bpm_1 = breath_bpm_1.tolist()
    breath_bpm_2 = breath_bpm_2.tolist()
    heart_bpm_1 = heart_bpm_1.tolist()
    heart_bpm_2 = heart_bpm_2.tolist()
    state_1 = state_1.tolist()
    state_2 = state_2.tolist()
    create_time_1 = create_time_1.tolist()
    create_time_2 = create_time_2.tolist()
    colors_breath_1 = ['black' if item == 0 else 'b' for item in state_1]
    colors_heart_1 = ['black' if item == 0 else 'r' for item in state_1]
    colors_breath_2 = ['black' if item == 0 else 'b' for item in state_2]
    colors_heart_2 = ['black' if item == 0 else 'r' for item in state_2]
    x_1 = [datetime.fromtimestamp(item).strftime('%Y-%m-%d %H:%M:%S') for item in create_time_1]
    x_2 = [datetime.fromtimestamp(item).strftime('%Y-%m-%d %H:%M:%S') for item in create_time_2]
    # fig_1: 呼吸率和心率
    fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))
    
    # ax1.plot(createTime, breathBpm, color=colors, label='breathBpm')
    for i in range(len(breath_bpm_1) - 1):
        ax1.plot(x_1[i:i+2], breath_bpm_1[i:i+2], color=colors_breath_1[i], marker='.', markersize=0.1) 
    ax1.set_ylabel('呼吸率', fontproperties=font)
    
    ax2 = ax1.twinx()
    for i in range(len(heart_bpm_1) - 1):
        ax2.plot(x_1[i:i+2], heart_bpm_1[i:i+2], color=colors_heart_1[i], marker='.', markersize=0.1) 
    ax2.set_ylabel('心率', fontproperties=font)
    
    
    plt.xticks(rotation=45)
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax1.tick_params(axis='x', labelsize=5)
    
    plt.title('呼吸率和心率折线图-黑线表示状态不稳定', fontproperties=font)
    
    handles = [
        Line2D([0], [0], color='b', lw=2, label='呼吸率'),
        Line2D([0], [0], color='r', lw=2, label='心率'),
        Line2D([0], [0], color='black', lw=2, label='呼吸率或心率不稳定'),
    ]
    plt.legend(handles=handles, prop=font, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    
    # fig_2: 呼吸线和心线
    for i in range(len(breath_bpm_2) - 1):
        ax3.plot(x_2[i:i+2], breath_bpm_2[i:i+2], color=colors_breath_2[i], marker='.', markersize=0.1) 
    ax3.set_ylabel('呼吸率', fontproperties=font)
    
    ax4 = ax3.twinx()
    for i in range(len(heart_bpm_2) - 1):
        ax4.plot(x_2[i:i+2], heart_bpm_2[i:i+2], color=colors_heart_2[i], marker='.', markersize=0.1) 
    ax4.set_ylabel('心率', fontproperties=font)
    
    plt.xticks(rotation=45)
    # ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax3.tick_params(axis='x', labelsize=5)
    
    plt.title('呼吸率和心率折线图-黑线表示状态不稳定', fontproperties=font)
    
    handles = [
        Line2D([0], [0], color='b', lw=2, label='呼吸率'),
        Line2D([0], [0], color='r', lw=2, label='心率'),
        Line2D([0], [0], color='black', lw=2, label='呼吸率或心率不稳定'),
    ]
    plt.legend(handles=handles, prop=font, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    save_file_name = f'{date_name}_{time.strftime("%Y%m%d%H%M%S")}.png'
    save_file_path = os.path.join(OUT_PATH, save_file_name)
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')

from whoami.tool.health_report.sleep_indices import SleepIndices
if __name__ == '__main__':
    
    sql_config_path = "/home/weiyutao/work/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
    sql_provider = SqlProvider(sql_config_path)
    query_text = "SELECT breath_bpm, heart_bpm, stable_num, UNIX_TIMESTAMP(create_time) as create_time FROM sx_device_sleep_vital_sign_log WHERE device_sn='C82E18AFA5B4' AND create_time>='2024-12-28 20:00:00' AND create_time<='2024-12-29 12:00:00'"
    sx_data_provider = SxDataProvider(sql_config_path=sql_config_path, sql_query=query_text, model=SleepIndices)
    dataloader = DataLoader(sx_data_provider, batch_size=60*60*20, shuffle=False)
    for batch in dataloader:
        breath_bpm = batch[:, 0]
        heart_bopm = batch[:, 1]
        state = batch[:, 2]
        create_time = batch[:, 3]
        data_1 = [breath_bpm, heart_bopm, state, create_time]
    
    query_text = "SELECT breath_bpm, heart_bpm, stable_num, UNIX_TIMESTAMP(create_time) as create_time FROM sx_device_sleep_vital_sign_log WHERE device_sn='C82E18AFA5B4' AND create_time>='2024-12-27 20:00:00' AND create_time<='2024-12-28 12:00:00'"
    sx_data_provider = SxDataProvider(sql_config_path=sql_config_path, sql_query=query_text, model=SleepIndices)
    dataloader = DataLoader(sx_data_provider, batch_size=60*60*20, shuffle=False)
    for batch in dataloader:
        breath_bpm = batch[:, 0]
        heart_bopm = batch[:, 1]
        state = batch[:, 2]
        create_time = batch[:, 3]
        data_2 = [breath_bpm, heart_bopm, state, create_time]
    
    draw_line_hear_breath(data_1[0], data_1[1], data_1[2], data_1[3], data_2[0], data_2[1], data_2[2], data_2[3], '2024_12_28_29')

    # device_sn = "13D7F349200080712111150807"
    # query_date = "2024-12-25"
    # current_date = datetime.strptime(query_date, '%Y-%m-%d')
    # current_date_str = current_date.strftime('%Y-%m-%d')
    # pre_date_str = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
    # start = pre_date_str + ' 20:00:00'
    # end = current_date_str + ' 09:00:00'
    # query = f"SELECT breath_line, heart_line, breath_bpm, heart_bpm, state, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM sx_device_wavve_vital_sign_log WHERE device_sn='{device_sn}' AND create_time >= '{start}' AND create_time < '{end}'"
    # sx_data_provider = SxDataProvider(sql_config_path=sql_config_path, sql_query=query)
    # dataloader = DataLoader(sx_data_provider, batch_size=60*60*14, shuffle=False)
    # # result = sql_provider.exec_sql(query=query)
    # for batch in dataloader:
    #     # print(batch["int_tensor"])
    #     # draw_line_image(batch, device_sn)
    #     print(batch)
        
    # sx_data_provider.set_sql_query(query)
    # dataloader = DataLoader(sx_data_provider, batch_size=60*60*14, shuffle=False)
    # # result = sql_provider.exec_sql(query=query)
    # for batch in dataloader:
    #     print(len(batch["int_tensor"]))


    # config = DetectorConfig.from_file("/work/ai/WHOAMI/whoami/tool/detect/default_config.yaml")
    # print(config)
    # config = SqlConfig.from_file("/home/weiyutao/work/WHOAMI/whoami/configs/yaml/sql_config_case.yaml")
    # print(config)
    
    # model_path = '/work/ai/object_detect_warning/models/yolov10m_20000_fire_epoch_250.pt'
    # yolo = UltraliticsDetector(model_path=model_path)
    # video_stream_detector = SxVideoStreamDetector(device_sn='BD3202818', topic_name='/fire/smoke/warning', url_str_flag='old')
    # print(video_stream_detector.process())
    # print(yolo.name)
    # print(yolo.model.info())
    # image_path = '/work/ai/object_detect_warning/data/12月17日(12).jpg'
    # result = yolo.predict(image_path)
    # print(result)
    
    