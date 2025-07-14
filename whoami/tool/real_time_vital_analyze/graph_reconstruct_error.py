"""
@Time    : 2025/07/02
@Author  : Assistant
@File    : graph_reconstruct_error.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Optional
import pytz



from whoami.provider.sql_provider import SqlProvider
from whoami.tool.real_time_vital_analyze.sleep_data_state import SleepDataState


class GraphReconstructError:
    """
    重构误差图绘制类
    用于绘制指定设备在指定时间范围内的重构误差变化图
    """
    
    def __init__(self, device_sn: str, start_time: float, end_time: float, 
                 sql_config_path: str = "/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"):
        """
        初始化重构误差图绘制器
        
        Args:
            device_sn (str): 设备编号
            start_time (float): 开始时间戳
            end_time (float): 结束时间戳
            sql_config_path (str): SQL配置文件路径
        """
        self.device_sn = device_sn
        self.start_timestamp = self._date_to_timestamp(start_time)
        self.end_timestamp = self._date_to_timestamp(end_time)
        self.sql_config_path = sql_config_path
        self.sql_provider = SqlProvider(
            model=SleepDataState,
            sql_config_path=sql_config_path
        )
        self.data = None
    
    
    
    def _date_to_timestamp(self, date_str):
        """将日期字符串转换为Unix时间戳（上海时区）"""
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        
        # 明确指定为上海时区
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        dt_with_tz = shanghai_tz.localize(dt)
        
        return int(dt_with_tz.timestamp())
    
    
    def fetch_data(self) -> Optional[pd.DataFrame]:
        """
        从数据库获取重构误差数据
        
        Returns:
            Optional[pd.DataFrame]: 包含时间戳和重构误差的数据框，如果没有数据则返回None
        """
        try:
            sql_data = self.sql_provider.get_record_by_condition(
                condition={
                    "device_id": self.device_sn,
                    "timestamp": {"min": self.start_timestamp, "max": self.end_timestamp}
                },
                fields=["timestamp", "reconstruction_error"]
            )
            
            if sql_data:
                # 转换为DataFrame
                df = pd.DataFrame(sql_data)
                # 过滤掉重构误差为空的数据
                df = df.dropna(subset=['reconstruction_error'])
                # 按时间戳排序
                df = df.sort_values('timestamp')
                self.data = df
                return df
            else:
                print(f"设备 {self.device_sn} 在指定时间范围内没有找到数据")
                return None
                
        except Exception as e:
            print(f"获取数据时发生错误: {e}")
            return None
    
    
    def convert_timestamp_to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将时间戳转换为可读的日期时间格式（上海时区）
        
        Args:
            df (pd.DataFrame): 包含timestamp列的数据框
            
        Returns:
            pd.DataFrame: 添加了datetime列的数据框
        """
        df = df.copy()
        # 先转换为UTC时间，然后转换为上海时区
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(shanghai_tz)
        return df
    
    
    def plot_reconstruction_error(self, save_path: Optional[str] = None, 
                                show_plot: bool = True, 
                                figsize: tuple = (12, 6)) -> None:
        """
        绘制重构误差图
        
        Args:
            save_path (Optional[str]): 图片保存路径，如果为None则不保存
            show_plot (bool): 是否显示图表
            figsize (tuple): 图表尺寸
        """
        if self.data is None:
            self.fetch_data()
        
        if self.data is None or self.data.empty:
            print("没有可用的数据进行绘图")
            return
        
        # 转换时间戳
        plot_data = self.convert_timestamp_to_datetime(self.data)
        
        # 创建图表
        plt.figure(figsize=figsize)
        
        # 绘制重构误差线图
        plt.plot(plot_data['datetime'], plot_data['reconstruction_error'], 
                linewidth=1.5, color='red', alpha=0.8, label='重构误差')
        
        # 设置图表标题和标签
        plt.title(f'设备 {self.device_sn} 重构误差变化图', fontsize=14, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('重构误差值', fontsize=12)
        
        # 设置网格
        plt.grid(True, alpha=0.3)
        
        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45)
        
        # 添加图例
        plt.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        # 显示图表
        if show_plot:
            plt.show()
    
    
    def get_statistics(self) -> dict:
        """
        获取重构误差的统计信息
        
        Returns:
            dict: 包含统计信息的字典
        """
        if self.data is None:
            self.fetch_data()
        
        if self.data is None or self.data.empty:
            return {}
        
        reconstruction_errors = self.data['reconstruction_error']
        
        stats = {
            '数据点数量': len(reconstruction_errors),
            '最小值': reconstruction_errors.min(),
            '最大值': reconstruction_errors.max(),
            '平均值': reconstruction_errors.mean(),
            '中位数': reconstruction_errors.median(),
            '标准差': reconstruction_errors.std(),
            '25%分位数': reconstruction_errors.quantile(0.25),
            '75%分位数': reconstruction_errors.quantile(0.75)
        }
        
        return stats
    
    
    def print_statistics(self) -> None:
        """
        打印重构误差的统计信息
        """
        stats = self.get_statistics()
        
        if not stats:
            print("没有可用的统计数据")
            return
        
        print(f"\n设备 {self.device_sn} 重构误差统计信息:")
        print("-" * 40)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


# 使用示例
if __name__ == "__main__":
    # 示例用法
    device_sn = "13d7f349200080712111150807"
    start_time = "2025-7-4 12:30:00"  # 示例时间戳
    end_time = "2025-7-4 14:10:00"    # 示例时间戳
    
    # 创建重构误差图绘制器
    graph = GraphReconstructError(device_sn, start_time, end_time)
    
    # 获取数据
    data = graph.fetch_data()
    
    if data is not None:
        # 打印统计信息
        graph.print_statistics()
        
        # 绘制图表
        graph.plot_reconstruction_error(
            save_path="reconstruction_error_plot.png",
            show_plot=True
        )
    else:
        print("无法获取数据，请检查设备编号和时间范围")