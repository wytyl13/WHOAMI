import paho.mqtt.client as mqtt
import json
import time
import traceback
import datetime
import os
import wave
import numpy as np
from scipy.io import wavfile
import requests
import json
import re



# 导入您提供的TextWakeSystem类
# 注意: 在使用前需要确保已安装pypinyin库: pip install pypinyin
from whoami.tool.xiaoxi.xiaoxi import TextWakeSystem  # 假设您已将唤醒系统代码保存为text_wake_system.py
from whoami.tool.xiaoxi.shuimian import SleepReportDetector  # 假设您已将唤醒系统代码保存为text_wake_system.py
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sleep_indices import SleepIndices
from whoami.utils.utils import Utils

url = "http://127.0.0.1:8888/chat_health_report"


utils = Utils()
STOPWORDS_1 = set()
with open("/work/ai/WHOAMI/whoami/tool/llm_application/stop_words.txt", "r", encoding="utf-8") as f:
    for line in f:
        STOPWORDS_1.add(line.strip())


topic_device_sn = {
    "/device/xiaozhi/10001": "13D4F349200080712111959C07",
    "/device/xiaozhi/10002": "13D4F349200080712111959C07"
}

wake_states = {
    "10001": {"last_wake_time": 0},
    "10002": {"last_wake_time": 0}
}

reponse_topics_list = list(topic_device_sn.keys())
topics_list = list(wake_states.keys())
SQL_CONFIG_PATH = "/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
# MQTT Configuration
BROKER_ADDRESS = "1.71.15.120"
PORT = 1883
TOPIC = "10001"
RESPONSE_TOPIC = "/device/xiaozhi/10001"

device_sn = topic_device_sn[RESPONSE_TOPIC]
# 添加全局变量来跟踪唤醒状态
last_wake_time = 0
wake_duration = 20  # 唤醒持续时间（秒）

# 创建唤醒系统实例
wake_system = TextWakeSystem()
wake_system.update_config({
    "extended_wake_phrases": ["智能小助手", "AI助理", "语音助手"]
})
sleep_report_system = SleepReportDetector()
sql_provider = SqlProvider(model=SleepIndices, sql_config_path=SQL_CONFIG_PATH)
# Callback when connection is established
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        # Subscribe to the health report topic
        for topic in topics_list:
            client.subscribe(topic)
            print(f"Subscribed to topic: {topic}")
    else:
        print(f"Failed to connect, return code {rc}")

# 消息发布后的回调
def on_publish(client, userdata, mid):
    print(f"消息已发布，消息ID: {mid}")


def save_with_scipy(pcm_data, filepath):
    """使用scipy保存音频"""
    # 假设这是16位有符号整数数据
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    
    # 尝试多种采样率
    for rate in [8000, 16000, 22050, 44100, 48000]:
        out_path = f"{filepath}_scipy_{rate}hz.wav"
        wavfile.write(out_path, rate, samples)
        print(f"使用scipy保存: {out_path}")


def save_specific_audio_format(byte_data, output_path):
    """
    保存16位有符号PCM数据为WAV文件
    
    参数:
    byte_data: 字节数据
    output_path: 输出WAV文件路径
    """
    # 直接将原始字节数据写入WAV文件
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)         # 单声道
        wav_file.setsampwidth(2)         # 16位
        wav_file.setframerate(8000)      # 尝试8kHz采样率
        wav_file.writeframes(byte_data)
    
    print(f"已将PCM数据保存到: {output_path}")
    print(f"数据长度: {len(byte_data)} 字节")
    print(f"时长: {len(byte_data) / (2 * 8000):.3f} 秒")  # 16位=2字节/样本



# Callback when a message is received
def on_message(client, userdata, msg):
    global last_wake_time
    print(f"Received message on topic {msg.topic}")
    
    try:
        topic_id = msg.topic
        if topic_id not in topics_list:
            print(f"未知的 topic: {topic_id}")
            return
        
        topic_index = topics_list.index(topic_id)
        response_topic = reponse_topics_list[topic_index]
        device_sn = topic_device_sn[response_topic]
        current_wake_state = wake_states[topic_id]
        
        message = msg.payload.decode()
        message = utils.remove_stopwords(message, stop_words=STOPWORDS_1)
        print(f"{topic_id} 收到消息: {message}")
        
        current_time = time.time()
        # 检查是否在唤醒状态（上次唤醒后的1分钟内）
        is_awake = (current_time - current_wake_state["last_wake_time"]) < wake_duration
        
        # 检测是否包含唤醒词
        huanxing_status = wake_system.detect(message)
        
        huanxing_status_ = huanxing_status
        if isinstance(huanxing_status, dict):
            huanxing_status_ = huanxing_status.get("matched", False)
        print(f"{topic_id} 收到消息: {message} \n- 唤醒状态: {huanxing_status}")
        # 如果检测到唤醒词或者在唤醒状态内
        if huanxing_status_ or is_awake:
            # 如果是新的唤醒词触发，更新唤醒时间
            if huanxing_status_:
                current_wake_state["last_wake_time"] = current_time
                print(f"检测到唤醒词: {huanxing_status.get('phrase', 'unknown')}")
                
                # 发送初始响应消息
                response = "嘿，您好！请问有什么可以帮您的吗？"
                print(f"检测到唤醒词: {huanxing_status.get('phrase', 'unknown')} 嘿，您好！请问有什么可以帮您的吗？")
                result = client.publish(response_topic, response)
            else:
                # 在唤醒状态中，但没有新的唤醒词
                if message != "" and message is not None and message != " ":
                    remaining_time = int(wake_duration - (current_time - last_wake_time))
                    print(f"处于唤醒状态中（还剩 {remaining_time} 秒）")
                    
                    # 处理用户请求
                    # 这里可以添加进一步的处理逻辑
                    
                    # 发送回应消息（可根据消息内容定制）
                    payload = {
                        "question": message,  # 用户问题
                        "user_id": device_sn,  # 用户ID
                    }

                    # 设置请求头
                    headers = {
                        "Content-Type": "application/json"
                    }
                    # 发送POST请求
                    print(f"payload: ------------------------{payload}")
                    response = requests.post(url, data=json.dumps(payload), headers=headers)
                    # 处理响应
                    if response.status_code == 200:
                        # 根据响应类型处理
                        if "application/json" in response.headers.get("Content-Type", ""):
                            # 处理JSON响应
                            data = response.json()
                            if data.get("success", False):
                                response_1 = data["data"]
                                # 步骤1：替换多个连续的换行为一个句号
                                processed_text = re.sub(r'\n{2,}', '。', response_1)

                                # 步骤2：替换单个换行为一个句号
                                processed_text = re.sub(r'\n', '。', processed_text)

                                # 步骤3：移除所有特殊符号（保留中文、英文、数字和基本标点）
                                processed_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,。，？?！!；;：:\'\"$%&\(\)\-\+\*/\\@#=<>《》【】\[\]]', '', processed_text)

                                # 步骤4：处理连续的多个句号
                                processed_text = re.sub(r'。{2,}', '。', processed_text)

                                # 步骤5：处理连续的多个空格
                                processed_text = re.sub(r'\s+', ' ', processed_text)


                                result = client.publish(response_topic, processed_text)
                            else:
                                result = client.publish(response_topic, "回答错误！")
                            print("响应数据:", data)
                        else:
                            # 处理文本或其他类型响应
                            result = client.publish(response_topic, "回答错误！")
                            print("响应内容:", response.text)
                    else:
                        print(f"请求失败，状态码: {response.status_code}")
                        print(f"错误信息: {response.text}")
                        result = client.publish(response_topic, "回答错误！")
                    
                    
                    """
                    # health_report_status = sleep_report_system.detect(message)
                    # print(f"health_report_status: {health_report_status}")
                    # if health_report_status["matched"]:
                    #     try:
                    #         sql_result = sql_provider.get_record_by_condition(
                    #             condition={"device_sn": device_sn, "query_date": '2025-3-20'}, 
                    #             fields=[
                    #                 'health_advice'
                    #             ]
                    #         )
                    #         if sql_result:
                                
                    #             health_advice = sql_result[0]["health_advice"]
                    #             health_advice = """
                    #             您的睡眠评分为55.56分（较差），睡眠效率低（72%）。总睡眠7小时57分钟中，深度睡眠仅1小时22分（17%，低于理想20-25%），浅睡眠占比过高（6小时35分钟）。夜间醒来9次，清醒时长达3小时5分钟，且有8次离床记录。生理指标显示：平均心率71次/分（正常），呼吸率13次/分（出现1次异常），体动频率9.93（总计298次，高于平均水平）。您的入睡时间为21分钟（正常范围），但睡眠连续性较差（连续清醒指数25）。改善建议：1、固定作息时间，避免在床上进行非睡眠活动。2、优化睡眠环境（温度18-22°C，安静无光）。3、睡前避免电子设备、咖啡因和酒精。4、尝试放松技巧（深呼吸、渐进式肌肉放松）。5、关注呼吸问题，可能需要调整睡姿。6、适当增加日间活动，但避免睡前剧烈运动。如问题持续，请考虑专业睡眠评估。
                    #             """

                    #             print(health_advice is None)
                    #             if health_advice == "":
                    #                 result = client.publish(RESPONSE_TOPIC, f"抱歉，没有查到您的睡眠报告数据呢！")
                    #             elif health_advice is None:
                    #                 result = client.publish(RESPONSE_TOPIC, f"抱歉，没有查到您的睡眠报告数据呢！")
                    #             else:
                    #                 result = client.publish(RESPONSE_TOPIC, health_advice)
                    #         else:
                    #             result = client.publish(RESPONSE_TOPIC, f"抱歉，没有查到您的睡眠报告数据呢！")
                    #     except Exception as e:
                    #         print(f"Fail to sql: {str(e)}")
                    #         result = client.publish(RESPONSE_TOPIC, f"哎呦，您的问题太深奥了！抱歉，我无法回答！")
                    # else:
                    #     result = client.publish(RESPONSE_TOPIC, f"我已收到您的消息: {message}，但我仅能回复您关于睡眠报告相关的内容哦！")

            if result.rc == 0:
                print(f"成功发送响应到主题 {response_topic}")
            else:
                print(f"发送响应失败，错误码: {result.rc}")
        else:
            print(f"现在处于非唤醒状态！{message}")
        
    except Exception as e:
        # 导入traceback模块获取详细错误信息
        import traceback
        # 处理异常
        print(f"处理消息时出错: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()  # 打印详细的堆栈跟踪信息，包括文件名、行号和代码
        print("Raw message:", msg.payload.decode())

# Create MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_publish = on_publish

# Connect to the broker
print(f"Connecting to broker at {BROKER_ADDRESS}:{PORT}...")
client.connect(BROKER_ADDRESS, PORT, 60)

# Start the loop
try:
    print("开始监听消息，按Ctrl+C退出...")
    client.loop_forever()
except KeyboardInterrupt:
    print("Disconnecting from broker")
    client.disconnect()
