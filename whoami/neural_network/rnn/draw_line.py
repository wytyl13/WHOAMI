#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/26 11:18
@Author  : weiyutao
@File    : draw_line.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import rcParams





#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/06/26
@Author  : weiyutao
@File    : plot_breath_heart_lines.py
@Description : 绘制呼吸线和心率线的折线图
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def plot_breath_heart_lines(data=None, csv_file=None, figsize=(15, 8), 
                           save_path=None, show_statistics=True):
    """
    绘制呼吸线和心率线的折线图
    
    Args:
        data: 数据，可以是以下格式之一：
            - pandas DataFrame，包含 'create_time', 'breath_line', 'heart_line' 列
            - 字符串，包含原始数据文本
            - None，则使用 csv_file 参数
        csv_file: CSV文件路径（当data=None时使用）
        figsize: 图片大小，默认(15, 8)
        save_path: 保存路径，如果为None则不保存
        show_statistics: 是否显示统计信息
    
    Returns:
        fig, axes: matplotlib图形对象
    """
    
    # 数据处理
    if data is None and csv_file is not None:
        # 从CSV文件读取
        df = pd.read_csv(csv_file)
    elif isinstance(data, str):
        # 从字符串解析数据
        lines = data.strip().split('\n')
        parsed_data = []
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                # 合并日期和时间
                date_time = f"{parts[0]} {parts[1]}"
                breath_val = float(parts[2])
                heart_val = float(parts[3])
                parsed_data.append([date_time, breath_val, heart_val])
        
        df = pd.DataFrame(parsed_data, columns=['create_time', 'breath_line', 'heart_line'])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        # 使用示例数据
        sample_data = """
2025/6/22 19:00 -0.00311127 0.061824
2025/6/22 19:00 0.00909772 -0.0287
2025/6/22 19:00 0.028574 0.03319
2025/6/22 19:00 -0.116531 0.144141
2025/6/22 19:00 0.0963904 0.224867
2025/6/22 19:00 0.187353 0.038389
2025/6/22 19:00 -0.215967 -0.0653
2025/6/22 19:00 0.00405151 -0.03353
2025/6/22 19:00 0.0144657 -0.06573
2025/6/22 19:00 -0.169248 -0.18478
2025/6/22 19:00 0.113074 0.137878
2025/6/22 19:00 0.0721232 -0.11095
2025/6/22 19:00 0.00502744 0.016141
        """.strip()
        return plot_breath_heart_lines(data=sample_data, figsize=figsize, 
                                     save_path=save_path, show_statistics=show_statistics)
    
    # 转换时间格式
    if 'create_time' in df.columns:
        try:
            df['create_time'] = pd.to_datetime(df['create_time'])
        except:
            # 如果时间格式有问题，创建序列号
            df['create_time'] = range(len(df))
    else:
        df['create_time'] = range(len(df))
    
    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle('呼吸线和心率线时序图', fontsize=16, fontweight='bold')
    
    # 颜色设置
    breath_color = '#2E86AB'  # 蓝色
    heart_color = '#A23B72'   # 紫红色
    
    # 绘制呼吸线
    axes[0].plot(df['create_time'], df['breath_line'], 
                color=breath_color, linewidth=2, marker='o', markersize=4,
                label='呼吸线', alpha=0.8)
    axes[0].set_ylabel('呼吸线数值', fontsize=12, fontweight='bold')
    axes[0].set_title('呼吸信号变化', fontsize=14, pad=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # 添加零线
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 绘制心率线
    axes[1].plot(df['create_time'], df['heart_line'], 
                color=heart_color, linewidth=2, marker='s', markersize=4,
                label='心率线', alpha=0.8)
    axes[1].set_ylabel('心率线数值', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('时间', fontsize=12, fontweight='bold')
    axes[1].set_title('心率信号变化', fontsize=14, pad=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # 添加零线
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 格式化x轴
    if isinstance(df['create_time'].iloc[0], pd.Timestamp):
        # 如果是时间数据，格式化时间轴
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axes[1].xaxis.set_major_locator(mdates.SecondLocator(interval=10))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[1].set_xlabel('数据点序号', fontsize=12, fontweight='bold')
    
    # 显示统计信息
    if show_statistics:
        breath_stats = f"呼吸线: 均值={df['breath_line'].mean():.4f}, 标准差={df['breath_line'].std():.4f}"
        heart_stats = f"心率线: 均值={df['heart_line'].mean():.4f}, 标准差={df['heart_line'].std():.4f}"
        
        # 在图上添加统计信息
        axes[0].text(0.02, 0.98, breath_stats, transform=axes[0].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[1].text(0.02, 0.98, heart_stats, transform=axes[1].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ 图片已保存到: {save_path}")
    
    return fig, axes


def plot_combined_lines(data, figsize=(15, 6), save_path=None, 
                       show_correlation=True, show_lines='both', x_tick_interval=500):
    """
    Plot breath and heart rate lines on the same graph (supports numpy array input)
    
    Args:
        data: numpy array with shape (n_samples, 3)
              Column 0: time (time series or index)
              Column 1: breath_line (breath signal data)
              Column 2: heart_line (heart rate data)
        figsize: figure size, default (15, 6)
        save_path: save path, None if not saving
        show_correlation: whether to show correlation info
        show_lines: which lines to show, options:
                   - 'both': show both breath and heart lines (dual y-axis)
                   - 'breath': show only breath line
                   - 'heart': show only heart rate line
        x_tick_interval: x-axis tick interval, default 500
    
    Returns:
        fig, ax1, ax2: matplotlib figure objects (ax2 is None for single line plots)
    """
    
    # Data processing
    if isinstance(data, np.ndarray):
        if data.shape[1] != 3:
            raise ValueError("numpy array must have 3 columns: [time, breath_line, heart_line]")
        
        df = pd.DataFrame(data, columns=['time', 'breath_line', 'heart_line'])
        
        # Handle time column - keep original datetime for real time display
        original_time = None
        if df['time'].dtype == 'object':  # String or datetime
            try:
                # Convert to datetime and keep original for display
                original_time = pd.to_datetime(df['time'])
                # Convert to numeric for plotting (seconds since first timestamp)
                df['time'] = (original_time - original_time.iloc[0]).dt.total_seconds()
                print("🕒 Converted datetime strings to numeric time series")
                print(f"🕒 Time range: {original_time.iloc[0]} to {original_time.iloc[-1]}")
            except:
                # If datetime conversion fails, use sequential indices
                df['time'] = np.arange(len(df))
                print("📋 Using sequential indices as time points")
        else:
            # Convert to numeric if not already
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df = df.dropna()  # Remove any NaN values
        
        # Convert breath and heart data to numeric
        df['breath_line'] = pd.to_numeric(df['breath_line'], errors='coerce')
        df['heart_line'] = pd.to_numeric(df['heart_line'], errors='coerce')
        df = df.dropna()  # Remove any NaN values
        
    else:
        raise ValueError("data must be numpy array with shape (n_samples, 3)")
    
    # Validate show_lines parameter
    if show_lines not in ['both', 'breath', 'heart']:
        raise ValueError("show_lines parameter must be 'both', 'breath', or 'heart'")
    
    print(f"📊 Data shape: {data.shape}")
    print(f"📈 Display mode: {show_lines}")
    print(f"🕒 Time range: [{df['time'].min():.1f}, {df['time'].max():.1f}] seconds")
    print(f"📋 Data range:")
    print(f"   Breath line: [{df['breath_line'].min():.4f}, {df['breath_line'].max():.4f}]")
    print(f"   Heart line: [{df['heart_line'].min():.4f}, {df['heart_line'].max():.4f}]")
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Color configuration
    color1 = '#2E86AB'  # Breath line color
    color2 = '#A23B72'  # Heart line color
    
    ax2 = None  # Initialize second y-axis as None
    lines = []
    
    # Plot different graphs based on display mode
    if show_lines == 'both':
        # Dual line mode (dual y-axis)
        ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Breath Signal', color=color1, fontsize=12, fontweight='bold')
        line1 = ax1.plot(df['time'], df['breath_line'], 
                         color=color1, linewidth=0.8, 
                         label='Breath', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Create second y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Heart Rate Signal', color=color2, fontsize=12, fontweight='bold')
        line2 = ax2.plot(df['time'], df['heart_line'], 
                         color=color2, linewidth=0.8,
                         label='Heart Rate', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.title('Breath Signal vs Heart Rate Signal', fontsize=16, fontweight='bold', pad=20)
        
        # Show correlation
        if show_correlation:
            correlation = df['breath_line'].corr(df['heart_line'])
            correlation_text = f"Correlation: {correlation:.4f}"
            ax1.text(0.02, 0.98, correlation_text, transform=ax1.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    elif show_lines == 'breath':
        # Show only breath line
        ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Breath Signal', fontsize=12, fontweight='bold')
        line1 = ax1.plot(df['time'], df['breath_line'], 
                         color=color1, linewidth=0.8,
                         label='Breath', alpha=0.8)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.title('Breath Signal Time Series', fontsize=16, fontweight='bold', pad=20)
        
        # Show statistics
        breath_stats = f"Mean: {df['breath_line'].mean():.4f}, Std: {df['breath_line'].std():.4f}"
        ax1.text(0.02, 0.98, breath_stats, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    elif show_lines == 'heart':
        # Show only heart rate line
        ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Heart Rate Signal', fontsize=12, fontweight='bold')
        line1 = ax1.plot(df['time'], df['heart_line'], 
                         color=color2, linewidth=0.8,
                         label='Heart Rate', alpha=0.8)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.title('Heart Rate Signal Time Series', fontsize=16, fontweight='bold', pad=20)
        
        # Show statistics
        heart_stats = f"Mean: {df['heart_line'].mean():.4f}, Std: {df['heart_line'].std():.4f}"
        ax1.text(0.02, 0.98, heart_stats, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))
    
    # Set x-axis ticks with real time display - very dense intervals
    max_time = df['time'].max()
    min_time = df['time'].min()
    time_range = max_time - min_time
    
    # Create very dense time ticks for maximum time resolution
    if original_time is not None:
        # Calculate time intervals based on data duration - very dense intervals
        duration_hours = time_range / 3600
        
        if duration_hours > 10:  # More than 10 hours, show every 30 minutes
            tick_interval_seconds = 1800  # 30 minutes
            time_format = '%H:%M'
        elif duration_hours > 6:  # 6-10 hours, show every 15 minutes
            tick_interval_seconds = 900   # 15 minutes
            time_format = '%H:%M'
        elif duration_hours > 3:  # 3-6 hours, show every 10 minutes
            tick_interval_seconds = 600   # 10 minutes
            time_format = '%H:%M'
        elif duration_hours > 1:  # 1-3 hours, show every 5 minutes
            tick_interval_seconds = 300   # 5 minutes
            time_format = '%H:%M'
        else:  # Less than 1 hour, show every 5 minutes
            tick_interval_seconds = 300   # 5 minutes
            time_format = '%H:%M'
        
        # Generate time ticks
        tick_times = np.arange(min_time, max_time + tick_interval_seconds, tick_interval_seconds)
        ax1.set_xticks(tick_times)
        
        # Convert tick positions back to datetime for labels
        tick_datetimes = [original_time.iloc[0] + pd.Timedelta(seconds=t) for t in tick_times]
        tick_labels = [dt.strftime(time_format) for dt in tick_datetimes]
        ax1.set_xticklabels(tick_labels, rotation=45)
        
    else:
        # Fallback for non-datetime data - many more ticks
        n_ticks = min(40, max(20, int(len(df) / 1000)))  # 20-40 ticks
        x_ticks = np.linspace(min_time, max_time, n_ticks)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([f"{x:.0f}" for x in x_ticks], rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ Figure saved to: {save_path}")
    
    return fig, ax1, ax2







if __name__ == '__main__':
    
    data = pd.read_csv("/work/ai/WHOAMI/device_info_13D2F34920008071211195A907_20250625_classifier.csv")
    np_data = np.array(data)
    sample_data = np_data[:, :-2]
    print(sample_data)
    # 方法2：合并绘制（双y轴）
    fig, ax1, ax2 = plot_combined_lines(
        data=sample_data,
        show_lines='breath',        # 或 'breath', 'heart'
        save_path="my_chart.png"
    )
    
    # 显示图表
    plt.show()
    
    print("✅ 图表绘制完成！")
    