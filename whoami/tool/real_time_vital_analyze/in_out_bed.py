#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/05/23 14:30
@Author  : weiyutao
@File    : in_out_bed.py
"""

from typing import (
    overload,
    Optional,
    Tuple,
    Any
)
import numpy as np
import asyncio
from datetime import datetime, timedelta, timezone


from whoami.tool.agent.base_tool import tool
from whoami.tool.real_time_vital_analyze.str_enum import StrEnum


# 绘图导包
import os
import time  
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
# 方式1: 全局设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 方式2: 创建字体对象（如果您有特定字体文件）
font = fm.FontProperties(fname='path/to/chinese/font.ttf')
# 或使用系统字体
font = fm.FontProperties(family='SimHei')


ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PROGRAM_ROOT_DIRECTORY = os.path.abspath(os.path.join(ROOT_DIRECTORY, "../../"))
font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14)
from whoami.utils.utils import Utils
utils = Utils()


class DiagnosticModel(StrEnum):
    """Diagnostic model for in out bed."""
    fixed_threshold = "fixed_threshold"
    reconstruct = "reconstruct"
    classification = "classification"



@tool
class InOutBed:
    window_size: Optional[int] = None


    @overload
    def __init__(
        self,
        window_size: Optional[int] = None 
    ):
        ...



    @overload
    def __init__(self, *args, **kwargs):
        ...


    def __init__(self, *args, **kwargs):
        
        self.window_size = kwargs.pop('nonlinearity', None)
        if "xx" in kwargs:
            pass
    
    
    def process_ones_numpy(self, arr, threshold=30):
        """
        处理NumPy数组中的连续一元素：
        - 如果连续一的数量 >= threshold，保持为1
        - 否则，将其转换为0
        
        参数:
        arr -- 输入NumPy数组，包含0和1
        threshold -- 连续一的阈值，默认为30
        
        返回:
        处理后的NumPy数组
        """
        # 创建一个全0的结果数组
        result = np.zeros_like(arr)
        
        # 找出所有为1的位置
        ones_positions = np.where(arr == 1)[0]
        
        if len(ones_positions) > 0:
            # 计算连续1的起始位置
            # 通过比较相邻位置的差值，找出不连续的点
            breaks = np.where(np.diff(ones_positions) > 1)[0]
            
            # 所有连续1序列的起始索引
            starts = np.concatenate(([0], breaks + 1))
            
            # 所有连续1序列的结束索引
            ends = np.concatenate((breaks, [len(ones_positions) - 1]))
            
            # 处理每个连续1的序列
            for start_idx, end_idx in zip(starts, ends + 1):
                # 获取实际数组中连续1的起始和结束位置
                start_pos = ones_positions[start_idx]
                end_pos = ones_positions[end_idx - 1] + 1  # +1是因为切片是左闭右开
                
                # 计算连续1的长度
                length = end_pos - start_pos
                
                # 如果连续1的长度大于等于阈值，设置为1
                if length >= threshold:
                    result[start_pos:end_pos] = 1
        
        return result
    
    
    
    def draw_simplified_chart(
        self, 
        breath_bpm, 
        heart_bpm, 
        create_time, 
        in_out_bed,
        body_move_energy,
        query_date,
        device_sn,
        query_date_device_sn: str = None
    ):
        """
        简化版绘图函数，绘制呼吸率、心率、在离床状态和体动能量值
        
        参数:
        breath_bpm -- 呼吸率数组
        heart_bpm -- 心率数组  
        create_time -- 时间戳数组
        in_out_bed -- 在离床状态数组 (1=在床, 0=离床)
        body_move_energy -- 体动能量值数组
        query_date_device_sn -- 查询日期和设备序列号
        """
        
        try:
            # 将时间戳转换为可读格式
            x_time = [datetime.fromtimestamp(item).strftime('%Y-%m-%d %H:%M:%S') for item in create_time]
            
            # 创建图表 - 3个子图
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(25, 15))
            
            # 子图1: 呼吸率和心率
            # 呼吸率
            ax1.plot(x_time, breath_bpm, color='#0F52BA', marker='.', markersize=1, linewidth=1, label='呼吸率')
            ax1.set_ylabel('呼吸率 (次/分)', fontproperties=font, color='#0F52BA')
            ax1.tick_params(axis='y', labelcolor='#0F52BA')
            
            # 心率 (共享x轴)
            ax1_twin = ax1.twinx()
            ax1_twin.plot(x_time, heart_bpm, color='lightcoral', marker='.', markersize=1, linewidth=1, label='心率')
            ax1_twin.set_ylabel('心率 (次/分)', fontproperties=font, color='lightcoral')
            ax1_twin.tick_params(axis='y', labelcolor='lightcoral')
            
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=10))
            ax1.tick_params(axis='x', labelsize=10)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax1.set_title(f'【{query_date} & {device_sn}】呼吸率和心率', 
                        fontproperties=font, fontsize=16, fontweight='bold')
            
            # 添加图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, 
                    prop={'family': font.get_name(), 'size': 12}, loc='upper right')
            
            # 子图2: 在离床状态
            # 将在离床状态可视化为填充区域
            in_bed_mask = np.array(in_out_bed) == 1
            out_bed_mask = np.array(in_out_bed) == 0
            
            # 创建y轴数据（固定高度）
            y_bed = np.ones(len(in_out_bed))
            
            ax2.fill_between(x_time, 0, y_bed, where=in_bed_mask, 
                            color='lightgreen', alpha=0.7, label='在床')
            ax2.fill_between(x_time, 0, y_bed, where=out_bed_mask, 
                            color='#FF1493', alpha=0.7, label='离床')
            
            ax2.set_ylabel('在离床状态', fontproperties=font)
            ax2.set_ylim(0, 1.2)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['离床', '在床'])
            ax2.xaxis.set_major_locator(MaxNLocator(nbins=10))
            ax2.tick_params(axis='x', labelsize=10)
            ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax2.set_title(f'【{query_date} & {device_sn}】在离床状态', 
                        fontproperties=font, fontsize=16, fontweight='bold')
            ax2.legend(prop={'family': font.get_name(), 'size': 12}, loc='upper right')
            
            # 子图3: 体动能量值
            ax3.plot(x_time, body_move_energy, color='orange', marker='.', markersize=1, linewidth=1)
            ax3.set_ylabel('体动能量值', fontproperties=font)
            ax3.set_xlabel('时间', fontproperties=font)
            ax3.xaxis.set_major_locator(MaxNLocator(nbins=10))
            ax3.tick_params(axis='x', labelsize=10, rotation=45)
            ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax3.set_title(f'【{query_date} & {device_sn}】体动能量值', 
                        fontproperties=font, fontsize=16, fontweight='bold')
            
            # 调整子图间距
            plt.subplots_adjust(hspace=0.4)
            plt.tight_layout()
            
            # 保存图像
            save_dir = os.path.join(PROGRAM_ROOT_DIRECTORY, f"out/{query_date}")
            status, result = utils.init_directory(save_dir)
            if not status:
                raise ValueError(result)
            
            save_file_name = f'{query_date_device_sn}_{time.strftime("%Y%m%d%H%M%S")}.png'
            save_file_path = os.path.join(save_dir, save_file_name)
            plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
            
            self.logger.info(f"图表已保存至: {save_file_path}")
            
        except Exception as e:
            self.logger.error(f"绘图失败: {str(e)}")
            raise ValueError('绘图函数执行失败!') from e
    
    
        
    async def execute(
        self, 
        data: np.ndarray,
        window_size: Optional[int] = None,
        diagnostic_model: Optional[str] = None
    ):
        
        try:    
            # 1-离床   0-在床
            in_out_bed = np.where(data == 0, 0, 1)
            print(f"original: ---------------- {np.sum(in_out_bed)}")
            result = self.process_ones_numpy(in_out_bed, 10)

        except Exception as e:
            raise ValueError("Fail to exec in_out_bed status diagnostic function!")
        return result
    
    
    

if __name__ == '__main__':
    from whoami.tool.health_report.sx_device_wavve_vital_sign_log_20250522 import SxDeviceWavveVitalSignLog
    from whoami.provider.sql_provider import SqlProvider
    from whoami.configs.sql_config import SqlConfig
    
    in_out_bed = InOutBed()
    
    async def main(data):
        in_out_bed_status = await in_out_bed.execute(data)
        print(in_out_bed_status)
        print(f"post status: ------ {np.sum(in_out_bed_status)}")
        return in_out_bed_status
    
    SQL_CONFIG_PATH = "/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
    sql_provider = SqlProvider(model=SxDeviceWavveVitalSignLog, sql_config_path=SQL_CONFIG_PATH)
    
    table_name = "sx_device_wavve_vital_sign_log_20250522"
    device_sn = "13291C9D100040711117957007"
    query_date = "2025-5-22"
    current_date = datetime.strptime(query_date, '%Y-%m-%d')
    pre_date_str = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
    sql_query = f"SELECT signal_intensity, breath_bpm, heart_bpm, state, UNIX_TIMESTAMP(create_time) as create_time_timestamp FROM {table_name} WHERE device_sn='{device_sn}' AND create_time >= '{pre_date_str} 19:00' AND create_time < '{query_date} 7:00'"
    results = sql_provider.exec_sql(sql_query)
    data_with_npnan = np.where(results == None, np.nan, results).astype(np.float64)
    all_nan_rows = np.all(data_with_npnan[:, :-1] == 0, axis=1)
    print(len(results))
    print(np.sum(all_nan_rows))
    result = asyncio.run(main(data_with_npnan[:, 0]))
    print(len(result))
    timestamp = data_with_npnan[:, -1]
    breath_bpm = data_with_npnan[:, 1]
    heart_bpm = data_with_npnan[:, 2]
    in_out_bed_status = result
    body_move_energy = data_with_npnan[:, 3]

    in_out_bed.draw_simplified_chart(
        breath_bpm=breath_bpm,
        heart_bpm=heart_bpm,
        create_time=timestamp,
        in_out_bed=in_out_bed_status,
        body_move_energy=body_move_energy,
        query_date=query_date,
        device_sn=device_sn,
        query_date_device_sn=f'{query_date}-{device_sn}'
    )
    
    
    
    
    
    