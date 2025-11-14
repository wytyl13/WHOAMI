
import paho.mqtt.client as mqtt
import time
import threading
import numpy as np
import streamlit as st
from datetime import datetime
from collections import deque
import queue

# 配置
FIXED_SAMPLING_RATE = 128
SAMPLES_PER_BATCH = 128
SAMPLING_INTERVAL = 1.0 / FIXED_SAMPLING_RATE

MQTT_BROKER = "1.71.15.120"
MQTT_PORT = 1883
MQTT_TOPICS = ["/device/bcg", "10001/response", "/response/10001", "xiaozhi/10001", "10001"]

class MQTTDataManager:
    """MQTT数据管理器 - 线程安全的数据缓冲区"""
    
    def __init__(self):
        self.data_buffer = queue.Queue(maxsize=50)
        self.debug_messages = deque(maxlen=30)
        self.mqtt_client = None
        self.connected = False
        self.lock = threading.Lock()
        self._initialize_mqtt()
    
    def log_debug(self, message):
        """线程安全的日志记录"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        debug_message = f"[{timestamp}] {message}"
        with self.lock:
            self.debug_messages.append(debug_message)
        print(debug_message)
    
    def _initialize_mqtt(self):
        """初始化MQTT客户端"""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_connect
            self.mqtt_client.on_message = self._on_message
            self.mqtt_client.on_disconnect = self._on_disconnect
            
            self.log_debug(f"连接MQTT服务器 {MQTT_BROKER}:{MQTT_PORT}")
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            
        except Exception as e:
            self.log_debug(f"MQTT初始化失败: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        self.log_debug(f"MQTT连接结果: {rc}")
        if rc == 0:
            self.connected = True
            self.log_debug("MQTT连接成功")
            for topic in MQTT_TOPICS:
                result = client.subscribe(topic)
                self.log_debug(f"订阅主题 '{topic}': {result}")
        else:
            self.connected = False
            self.log_debug(f"MQTT连接失败: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT断开连接回调"""
        self.connected = False
        self.log_debug(f"MQTT连接断开: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT消息处理"""
        try:
            pythoncurrent_time_str = datetime.fromtimestamp(time.time()).strftime("%H:%M:%S")
            self.log_debug(f"收到MQTT消息! 主题: {msg.topic}, 长度: {len(msg.payload)}, {pythoncurrent_time_str}")
            
            # 处理原始数据
            raw_data = msg.payload
            cleaned_data = self._remove_tail_bytes(raw_data, 12)
            hex_spaced = self._bytes_to_hex_spaced(cleaned_data)
            
            # 构建数据包
            data_packet = {
                'hex_spaced': hex_spaced,
                'timestamp': time.time(),
                'topic': msg.topic,
                'data_length': len(cleaned_data)
            }
            
            # 添加到缓冲区
            try:
                self.data_buffer.put_nowait(data_packet)
                self.log_debug(f"数据已缓存，队列长度: {self.data_buffer.qsize()}")
            except queue.Full:
                # 如果队列满了，取出一个旧数据再放入新数据
                try:
                    self.data_buffer.get_nowait()
                    self.data_buffer.put_nowait(data_packet)
                    self.log_debug("队列已满，替换旧数据")
                except queue.Empty:
                    pass
            
        except Exception as e:
            self.log_debug(f"处理MQTT消息时出错: {e}")
    
    def _remove_tail_bytes(self, raw_data, tail_length=12):
        """去除尾部字节"""
        if len(raw_data) > tail_length:
            return raw_data[:-tail_length]
        return raw_data
    
    def _bytes_to_hex_spaced(self, data):
        """转换为带空格的hex字符串"""
        hex_continuous = data.hex().upper()
        return ' '.join([hex_continuous[i:i+2] for i in range(0, len(hex_continuous), 2)])
    
    def get_latest_data(self):
        """获取最新数据（非阻塞）"""
        try:
            return self.data_buffer.get_nowait()
        except queue.Empty:
            return None
    
    def get_debug_messages(self):
        """获取调试消息"""
        with self.lock:
            return list(self.debug_messages)
    
    def is_connected(self):
        """检查连接状态"""
        return self.connected
    
    def get_queue_size(self):
        """获取队列大小"""
        return self.data_buffer.qsize()


@st.cache_resource
def get_shared_mqtt_manager():
    """获取共享的MQTT管理器实例"""
    return MQTTDataManager()


def parse_bcg_data(hex_string, batch_timestamp):
    """解析BCG数据"""
    values = []
    markers = []
    high_byte_values = []  # 新增：第一个字节数据
    low_byte_values = []   # 新增：第二个字节数据
    
    status_mapping = {'00': 0, '0D': 7, '01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6}
    
    try:
        hex_values = hex_string.split()
        i = 1 if len(hex_values) > 0 and hex_values[0] == 'FF' else 0
        
        while i < len(hex_values):
            current_hex = hex_values[i]
            
            if current_hex in status_mapping:
                if i + 2 < len(hex_values):
                    try:
                        high_byte = int(hex_values[i + 1], 16)
                        low_byte = int(hex_values[i + 2], 16)
                        value = (high_byte << 8) | low_byte
                        
                        if value >= 32768:
                            value -= 65536
                        
                        if -32768 <= value <= 32767:
                            values.append(value)
                            markers.append(status_mapping[current_hex])
                            high_byte_values.append(high_byte)  # 存储第一个字节
                            low_byte_values.append(low_byte)   # 存储第二个字节
                        
                        i += 3
                    except (ValueError, IndexError):
                        i += 1
                else:
                    i += 1
            else:
                i += 1
                
    except Exception as e:
        st.error(f"解析数据时出错: {e}")
    
    # 调整数据长度到固定大小
    if len(values) > SAMPLES_PER_BATCH:
        values = values[:SAMPLES_PER_BATCH]
        markers = markers[:SAMPLES_PER_BATCH]
        high_byte_values = high_byte_values[:SAMPLES_PER_BATCH]
        low_byte_values = low_byte_values[:SAMPLES_PER_BATCH]
    elif len(values) < SAMPLES_PER_BATCH and len(values) > 0:
        last_value = values[-1]
        last_marker = markers[-1]
        last_high_byte = high_byte_values[-1]
        last_low_byte = low_byte_values[-1]
        while len(values) < SAMPLES_PER_BATCH:
            values.append(last_value)
            markers.append(last_marker)
            high_byte_values.append(last_high_byte)
            low_byte_values.append(last_low_byte)
    elif len(values) == 0:
        values = [0] * SAMPLES_PER_BATCH
        markers = [2] * SAMPLES_PER_BATCH
        high_byte_values = [0] * SAMPLES_PER_BATCH
        low_byte_values = [0] * SAMPLES_PER_BATCH
    
    # 生成时间戳
    times = []
    for i in range(len(values)):
        times.append(batch_timestamp + i * SAMPLING_INTERVAL)
    
    # 使用当前时间戳作为批次代表时间
    batch_center_time = batch_timestamp + (SAMPLES_PER_BATCH - 1) * SAMPLING_INTERVAL / 2
    
    return (np.array(values), np.array(times), np.array(markers), 
            np.array(high_byte_values), np.array(low_byte_values), batch_center_time)



def parse_bcg_data_bake(hex_string, batch_timestamp):
    """解析BCG数据"""
    values = []
    markers = []
    
    status_mapping = {'01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6}
    
    try:
        hex_values = hex_string.split()
        i = 1 if len(hex_values) > 0 and hex_values[0] == 'FF' else 0
        
        while i < len(hex_values):
            current_hex = hex_values[i]
            
            if current_hex in status_mapping:
                if i + 2 < len(hex_values):
                    try:
                        high_byte = int(hex_values[i + 1], 16)
                        low_byte = int(hex_values[i + 2], 16)
                        value = (high_byte << 8) | low_byte
                        
                        if value >= 32768:
                            value -= 65536
                        
                        if -32768 <= value <= 32767:
                            values.append(value)
                            markers.append(status_mapping[current_hex])
                        
                        i += 3
                    except (ValueError, IndexError):
                        i += 1
                else:
                    i += 1
            else:
                i += 1
                
    except Exception as e:
        st.error(f"解析数据时出错: {e}")
    
    # 调整数据长度到固定大小
    if len(values) > SAMPLES_PER_BATCH:
        values = values[:SAMPLES_PER_BATCH]
        markers = markers[:SAMPLES_PER_BATCH]
    elif len(values) < SAMPLES_PER_BATCH and len(values) > 0:
        last_value = values[-1]
        last_marker = markers[-1]
        while len(values) < SAMPLES_PER_BATCH:
            values.append(last_value)
            markers.append(last_marker)
    elif len(values) == 0:
        values = [0] * SAMPLES_PER_BATCH
        markers = [2] * SAMPLES_PER_BATCH
    
    # 生成时间戳
    times = []
    for i in range(len(values)):
        times.append(batch_timestamp + i * SAMPLING_INTERVAL)
    
    # 计算这个批次的代表时间（中间时间点）
    batch_center_time = batch_timestamp + (SAMPLES_PER_BATCH - 1) * SAMPLING_INTERVAL / 2
    # batch_center_time = time.time()
    
    return np.array(values), np.array(times), np.array(markers), batch_center_time


def process_mqtt_data(mqtt_manager):
    """处理MQTT数据"""
    processed_count = 0
    
    # 处理队列中的所有数据
    while processed_count < 5:  # 限制每次处理的数量
        data_packet = mqtt_manager.get_latest_data()
        if not data_packet:
            break
        
        try:
            # 初始化BCG数据存储
            if 'bcg_data' not in st.session_state:
                st.session_state.bcg_data = {
                    'values': deque(maxlen=10000),
                    'times': deque(maxlen=10000),
                    'markers': deque(maxlen=10000),
                    'high_byte_values': deque(maxlen=10000),  # 新增
                    'low_byte_values': deque(maxlen=10000),   # 新增
                    'raw_hex_data': deque(maxlen=100),
                    'batch_timestamps': deque(maxlen=100),
                    'batch_center_times': deque(maxlen=100)
                }
            
            # 解析数据
            hex_spaced = data_packet['hex_spaced']
            current_time = data_packet['timestamp']
            
            if hex_spaced:
                values, times, markers, high_byte_values, low_byte_values, batch_center_time = parse_bcg_data(hex_spaced, current_time)
                
                if len(values) == SAMPLES_PER_BATCH and np.all(np.isfinite(values)):
                    st.session_state.bcg_data['values'].extend(values.tolist())
                    st.session_state.bcg_data['times'].extend(times.tolist())
                    st.session_state.bcg_data['markers'].extend(markers.tolist())
                    st.session_state.bcg_data['high_byte_values'].extend(high_byte_values.tolist())  # 新增
                    st.session_state.bcg_data['low_byte_values'].extend(low_byte_values.tolist())   # 新增
                    st.session_state.bcg_data['raw_hex_data'].append(hex_spaced)
                    st.session_state.bcg_data['batch_timestamps'].append(current_time)
                    st.session_state.bcg_data['batch_center_times'].append(batch_center_time)
                    
                    processed_count += 1
            
        except Exception as e:
            st.error(f"处理数据时出错: {e}")
            break
    
    return processed_count


def create_bcg_chart(title, values, markers, times, batch_center_times, max_display_batches=5):
    """创建BCG波形图表 - 简化版带时间刻度"""
    if (values is None or markers is None or times is None or 
        len(values) == 0 or len(markers) == 0 or len(times) == 0):
        return f'''
        <div style="text-align: center; padding: 2rem; color: #666; background: #f8f9fa; border-radius: 8px;">
            <h4>{title}</h4>
            <p>等待数据中...</p>
        </div>'''
    
    try:
        values = np.array(values, dtype=np.float64)
        markers = np.array(markers, dtype=np.int32)
        times = np.array(times, dtype=np.float64)
        
        # 数据验证和清理
        valid_mask = (np.isfinite(values) & np.isfinite(times) & (np.abs(values) < 1e6))
        if not np.any(valid_mask):
            return f'<div style="text-align: center; padding: 2rem;"><h4>{title}</h4><p>无有效数据</p></div>'
        
        values = values[valid_mask]
        markers = markers[valid_mask]
        times = times[valid_mask]
        
        # 限制显示数据量
        max_display_points = max_display_batches * SAMPLES_PER_BATCH
        if len(values) > max_display_points:
            values = values[-max_display_points:]
            markers = markers[-max_display_points:]
            times = times[-max_display_points:]
        
        # 简化的SVG图表生成
        svg_width = 1200   # 增加宽度
        svg_height = 350   # 保持高度
        chart_height = 200 # 图表区域高度
        chart_top = 40     # 图表顶部位置
        
        y_min, y_max = float(np.min(values)), float(np.max(values))
        if abs(y_max - y_min) < 1e-6:
            y_center = (y_min + y_max) / 2
            y_min, y_max = y_center - 100, y_center + 100
        
        x_min, x_max = float(np.min(times)), float(np.max(times))
        if abs(x_max - x_min) < 1e-6:
            x_max = x_min + 1
        
        # 生成SVG路径
        # 颜色映射 - 根据markers状态分配颜色
        marker_colors = {
            0: "#808080",
            7: "#808080",
            1: "#FF5722",  # 橙红色
            2: "#2196F3",  # 蓝色  
            3: "#4CAF50",  # 绿色
            4: "#FF9800",  # 橙色
            5: "#9C27B0",  # 紫色
            6: "#F44336"   # 红色
        }

        # 按状态分组生成路径
        # path_segments = {}
        # step = max(1, len(values) // 1000)  # 减少点数

        # for i in range(0, len(values), step):
        #     marker = int(markers[i])
        #     x = 80 + ((times[i] - x_min) / (x_max - x_min)) * 1040  # 增加图表宽度
        #     y = chart_top + (1 - (values[i] - y_min) / (y_max - y_min)) * chart_height
            
        #     if marker not in path_segments:
        #         path_segments[marker] = []
        #     path_segments[marker].append((x, y))

        # 生成Y轴刻度
        y_ticks = []
        for i in range(6):  # 6个刻度
            tick_value = y_min + (y_max - y_min) * i / 5
            tick_y = chart_top + chart_height - (chart_height * i / 5)
            y_ticks.append((tick_y, tick_value))

        # 生成X轴时间刻度
        x_ticks = []
        if batch_center_times and len(batch_center_times) > 0:
            # 获取最后N个批次的中心时间
            display_batch_times = batch_center_times[-max_display_batches:]
            
            for i, batch_time in enumerate(display_batch_times):
                # 计算X轴位置
                time_progress = (batch_time - x_min) / (x_max - x_min) if x_max > x_min else 0
                tick_x = 80 + 1040 * time_progress
                
                try:
                    time_str = datetime.fromtimestamp(batch_time).strftime("%H:%M:%S")
                except:
                    time_str = f"批次{i+1}"
                
                x_ticks.append((tick_x, time_str))
        
        
        
        
        
        
        # 构建SVG内容
        svg_content = f'''
        <div style="margin: 1rem 0; text-align: center;">
            <h4 style="text-align: center; color: #2196F3;">{title}</h4>
            <svg width="{svg_width}" height="{svg_height}" style="border: 1px solid #ddd; background: white;">
                <!-- Y轴 -->
                <line x1="80" y1="{chart_top}" x2="80" y2="{chart_top + chart_height}" stroke="#666" stroke-width="2"/>
                <!-- X轴 -->
                <line x1="80" y1="{chart_top + chart_height}" x2="1120" y2="{chart_top + chart_height}" stroke="#666" stroke-width="2"/>'''

        # 添加Y轴刻度
        for tick_y, tick_value in y_ticks:
            svg_content += f'''
                <line x1="75" y1="{tick_y}" x2="85" y2="{tick_y}" stroke="#666" stroke-width="1"/>
                <text x="70" y="{tick_y + 5}" text-anchor="end" font-size="11" fill="#666">{tick_value:.0f}</text>'''

        # 添加X轴刻度
        for tick_x, time_str in x_ticks:
            svg_content += f'''
                <line x1="{tick_x}" y1="{chart_top + chart_height}" x2="{tick_x}" y2="{chart_top + chart_height + 5}" stroke="#666" stroke-width="1"/>
                <text x="{tick_x}" y="{chart_top + chart_height + 20}" text-anchor="middle" font-size="11" fill="#666">{time_str}</text>'''

        # 添加不同状态的数据线
        # 按状态段绘制不同颜色的连续线条
        # if len(values) > 1:
        #     step = max(1, len(values) // 1000)
            
        #     # 先绘制完整的连续数据线（用中性颜色）
        #     all_path_parts = []
        #     for i in range(0, len(values), step):
        #         x = 80 + ((times[i] - x_min) / (x_max - x_min)) * 1040
        #         y = chart_top + (1 - (values[i] - y_min) / (y_max - y_min)) * chart_height
                
        #         if i == 0:
        #             all_path_parts.append(f"M{x:.1f},{y:.1f}")
        #         else:
        #             all_path_parts.append(f"L{x:.1f},{y:.1f}")
            
        #     # 绘制底层连续线（细线，浅色）
        #     all_path_str = " ".join(all_path_parts)
        #     svg_content += f'''
        #         <path d="{all_path_str}" fill="none" stroke="#E0E0E0" stroke-width="1"/>'''
            
        #     # 再按状态绘制彩色段（覆盖在底层线上）
        #     current_segment = []
        #     current_marker = int(markers[0])
            
        #     for i in range(0, len(values), step):
        #         marker = int(markers[i])
        #         x = 80 + ((times[i] - x_min) / (x_max - x_min)) * 1040
        #         y = chart_top + (1 - (values[i] - y_min) / (y_max - y_min)) * chart_height
                
        #         if marker == current_marker:
        #             current_segment.append((x, y))
        #         else:
        #             # 绘制当前状态段
        #             if len(current_segment) > 1:
        #                 color = marker_colors.get(current_marker, "#666666")
        #                 path_parts = []
        #                 for j, (seg_x, seg_y) in enumerate(current_segment):
        #                     if j == 0:
        #                         path_parts.append(f"M{seg_x:.1f},{seg_y:.1f}")
        #                     else:
        #                         path_parts.append(f"L{seg_x:.1f},{seg_y:.1f}")
                        
        #                 path_str = " ".join(path_parts)
        #                 svg_content += f'''
        #         <path d="{path_str}" fill="none" stroke="{color}" stroke-width="3"/>'''
                    
        #             # 开始新段
        #             current_segment = [(x, y)]
        #             current_marker = marker
            
        #     # 绘制最后一段
        #     if len(current_segment) > 1:
        #         color = marker_colors.get(current_marker, "#666666")
        #         path_parts = []
        #         for j, (seg_x, seg_y) in enumerate(current_segment):
        #             if j == 0:
        #                 path_parts.append(f"M{seg_x:.1f},{seg_y:.1f}")
        #             else:
        #                 path_parts.append(f"L{seg_x:.1f},{seg_y:.1f}")
                
        #         path_str = " ".join(path_parts)
        #         svg_content += f'''
        #         <path d="{path_str}" fill="none" stroke="{color}" stroke-width="3"/>'''



        # 添加不同状态的数据线 - 简单连续绘制
        if len(values) > 1:
            step = max(1, len(values) // 1000)
            
            for i in range(0, len(values) - step, step):
                # 当前点和下一个点的坐标
                curr_x = 80 + ((times[i] - x_min) / (x_max - x_min)) * 1040
                curr_y = chart_top + (1 - (values[i] - y_min) / (y_max - y_min)) * chart_height
                next_x = 80 + ((times[i + step] - x_min) / (x_max - x_min)) * 1040
                next_y = chart_top + (1 - (values[i + step] - y_min) / (y_max - y_min)) * chart_height
                
                # 使用当前点的状态颜色
                current_marker = int(markers[i])
                color = marker_colors.get(current_marker, "#666666")
                
                # 绘制连接线段
                svg_content += f'''
                <line x1="{curr_x:.1f}" y1="{curr_y:.1f}" x2="{next_x:.1f}" y2="{next_y:.1f}" stroke="{color}" stroke-width="2"/>'''
        
        
        
        
        


        # 添加图例
        legend_y = chart_top - 15
        used_markers = list(set(markers))  # 获取数据中实际出现的状态
        used_markers.sort()  # 排序显示
        for i, marker in enumerate(used_markers):
            color = marker_colors.get(int(marker), "#666666")
            x_pos = 150 + i * 80
            svg_content += f'''
                <line x1="{x_pos}" y1="{legend_y}" x2="{x_pos + 20}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>
                <text x="{x_pos + 25}" y="{legend_y + 5}" font-size="12" fill="#333">状态{marker}</text>'''

        svg_content += f'''
                <!-- 数据信息 -->
                <text x="400" y="15" text-anchor="middle" fill="#333">
                    数据点: {len(values)} | 范围: {y_min:.0f} ~ {y_max:.0f}
                </text>
            </svg>
        </div>'''
        return svg_content
    except Exception as e:
        return f'<div style="text-align: center; padding: 2rem; color: red;"><h4>{title}</h4><p>图表生成错误: {e}</p></div>'
    
    
# Streamlit应用主体
st.set_page_config(
    page_title="BCG波形实时监控 - 缓存版",
    page_icon="📊",
    layout="wide"
)

# 获取共享的MQTT管理器
mqtt_manager = get_shared_mqtt_manager()

# 初始化session state
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

st.title("BCG波形实时监控系统 (缓存资源版)")

# 状态显示
col1, col2 = st.columns(2)
with col1:
    if mqtt_manager.is_connected():
        st.success("MQTT连接正常")
    else:
        st.error("MQTT连接断开")

with col2:
    queue_size = mqtt_manager.get_queue_size()
    if st.session_state.monitoring:
        st.success(f"监控中 - 队列: {queue_size}")
    else:
        st.info("监控已暂停")

# 控制面板
st.subheader("控制面板")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("开始监控", type="primary"):
        st.session_state.monitoring = True

with col2:
    if st.button("停止监控"):
        st.session_state.monitoring = False

with col3:
    if st.button("清除数据"):
        if 'bcg_data' in st.session_state:
            st.session_state.bcg_data = {
                'values': deque(maxlen=10000),
                'times': deque(maxlen=10000),
                'markers': deque(maxlen=10000),
                'high_byte_values': deque(maxlen=10000),  # 新增
                'low_byte_values': deque(maxlen=10000),   # 新增
                'raw_hex_data': deque(maxlen=100),
                'batch_timestamps': deque(maxlen=100),
                'batch_center_times': deque(maxlen=100)
            }
        st.success("数据已清除")

with col4:
    if st.button("重置MQTT"):
        st.cache_resource.clear()
        st.success("MQTT管理器已重置")

# 数据处理和显示
if st.session_state.monitoring:
    # 处理来自MQTT管理器的数据
    processed_count = process_mqtt_data(mqtt_manager)
    
    # 显示数据
    if 'bcg_data' in st.session_state and st.session_state.bcg_data.get('values'):
        bcg_data = st.session_state.bcg_data
        
        # 统计信息
        st.subheader("数据统计")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("总数据点", len(bcg_data['values']))
        with stat_col2:
            st.metric("数据批次", len(bcg_data['batch_timestamps']))
        with stat_col3:
            if bcg_data['values']:
                current_val = list(bcg_data['values'])[-1]
                st.metric("当前值", f"{current_val:.0f}")
        with stat_col4:
            st.metric("本轮处理", processed_count)
        
        
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("")  # 占位
        with col2:
            max_batches_options = [1, 3, 5, 7, 9, 20, 30, 50]
            selected_batches = st.selectbox(
                "显示批次数:", 
                options=max_batches_options, 
                index=0,  # 默认选择第一个(3)
                key="max_display_batches"
            )

        values_list = list(bcg_data['values'])
        markers_list = list(bcg_data['markers'])
        times_list = list(bcg_data['times'])

        
        
        # 实时波形图
        st.subheader("实时波形")
        values_list = list(bcg_data['values'])
        markers_list = list(bcg_data['markers'])
        times_list = list(bcg_data['times'])
        
        batch_center_times_list = list(bcg_data['batch_center_times'])
        chart_html = create_bcg_chart(
            "BCG实时波形", values_list, markers_list, times_list, batch_center_times_list, selected_batches
        )
        st.components.v1.html(chart_html, height=350, scrolling=False)
        
        # 新增：第一个字节波形图
        high_byte_values_list = list(bcg_data['high_byte_values'])
        chart_html_high = create_bcg_chart(
            "第一个字节波形", high_byte_values_list, markers_list, times_list, batch_center_times_list, selected_batches
        )
        st.components.v1.html(chart_html_high, height=350, scrolling=False)

        # 新增：第二个字节波形图
        low_byte_values_list = list(bcg_data['low_byte_values'])
        chart_html_low = create_bcg_chart(
            "第二个字节波形", low_byte_values_list, markers_list, times_list, batch_center_times_list, selected_batches
        )
        st.components.v1.html(chart_html_low, height=350, scrolling=False)
        
        # 原始数据预览
        if bcg_data.get('raw_hex_data'):
            with st.expander("原始数据预览"):
                latest_hex = list(bcg_data['raw_hex_data'])[-1]
                preview = latest_hex[:200] + "..." if len(latest_hex) > 200 else latest_hex
                st.code(preview)
    else:
        st.info("等待数据中...")

# 调试信息
with st.expander("系统信息"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**MQTT状态**")
        st.write(f"连接状态: {mqtt_manager.is_connected()}")
        st.write(f"数据队列长度: {mqtt_manager.get_queue_size()}")
    
    with col2:
        st.write("**调试日志**")
        debug_messages = mqtt_manager.get_debug_messages()
        if debug_messages:
            recent_messages = debug_messages[-10:]  # 显示最近10条
            for msg in recent_messages:
                st.text(msg)

# 自动刷新
if st.session_state.monitoring:
    time.sleep(1.28)
    st.rerun()