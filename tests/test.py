import requests
import json

# API endpoint and authentication
api_url = "http://localhost:11434/api/chat"
api_key = "YOUR_API_KEY_HERE"

# Headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Payload data
payload = {
    "model": "qwen2.5:7b-instruct",
    "messages": [
        {
            "role": "system",
            "content": """
                你是一名专业的睡眠健康分析师，擅长解读复杂的睡眠监测数据。请根据提供的详细睡眠数据，全面、专业地回答用户的问题。
                分析指南：
                1. 如果用户询问单次睡眠报告，请详细解读以下方面：
                - 睡眠质量综合评估
                - 睡眠阶段分布合理性
                - 生理指标（心率、呼吸）正常性
                - 潜在健康风险
                - 具体改善建议

                2. 如果用户询问趋势或比较，请注意：
                - 根据当前数据记录进行趋势分析

                3. 分析要求：
                - 使用专业但易懂的语言
                - 数据解读要客观准确
                - 建议具体可执行
                - 语气友好，富有同理心

                4. 特别提示：
                - 如果数据显示异常，请谨慎但清晰地指出
                - 针对异常指标给出专业建议
                - 鼓励用户持续关注身体健康
                请根据上述指南和数据库信息（如果有），全面且专业地回答用户的具体问题。
                """
        },
        {
            "role": "system",
            "content": """[{'\n用户询问的关于畅丽娟的数据库检索信息': '[{\'device_sn\': \'13CFF34920008071211195E507\', \'device_type\': \'WAVVE_SLEEP_DETECTION\', \'device_name\': \'睡眠检测\', \'device_status\': 1, \'dept_id\': 186, \'room_id\': 60, \'bed_id\': 97, \'elderly_id\': 110, \'elderly_name\': \'畅丽娟\', \'institution_room_id\': 60, \'institution_bed_id\': 97, \'elderly_id_card\': \'142724198304153320\', \'elderly_sex\': \'2\', \'elderly_age\': 41, \'elderly_birthday\': \'1983-04-15\', \'elderly_address\': \'山西省运城市临猗县角杯镇\', \'contacts\': \'[{"name":"宋阳泽","phone":"13080342714"}]\'}]\n\n'}, {'\n数据字段说明': "{'id': '主键id', 'average_heart_bpm': '平均心率', 'average_breath_bpm': '平均呼吸率', 'total_num_hour': '总监测时长（小时）', 'total_num_hour_on_bed': '总在床时长（小时）', 'sleep_hour': '睡眠时长（小时）', 'deep_sleep_hour': '深度睡眠时长（小时）', 'waking_hour': '清醒时长（小时）', 'to_sleep_hour': '入睡时长（小时）', 'leave_bed_total_hour': '离床总时间（小时）', 'light_sleep_hour': '浅睡时长（小时）', 'total_num_second': '总监测时长（秒）', 'total_num_second_on_bed': '总在床时长（秒）', 'sleep_second': '睡眠时长（秒）', 'deep_sleep_second': '深度睡眠时长（秒）', 'waking_second': '清醒时长（秒）', 'to_sleep_second': '入睡时长（秒）', 'leave_bed_total_second': '离床总时间（秒）', 'light_sleep_second': '浅睡时长（秒）', 'leave_count': '离床次数', 'sleep_efficiency': '睡眠效率', 'deep_sleep_efficiency': '深睡效率', 'score': '评分', 'score_name': '评分归类', 'consist_count_waking': '连续?晚夜醒时长超过31分钟', 'consist_count_sleep_efficiency': '连续?晚睡眠效率小于80%', 'query_date': '查询日期', 'save_file_path': '睡眠阶段划分-心率折线图', 'creator': '创建者', 'create_time': '创建时间', 'updater': '更新者', 'update_time': '更新时间', 'deleted': '是否删除', 'tenant_id': '租户编号', 'waking_count': '夜醒次数（次）', 'on_bed_time': '上床时间（节点）', 'sleep_time': '入睡时间（节点）', 'waking_time': '醒来时间（节点）', 'sleep_stage_image_x_y': '睡眠分区绘图', 'max_breath_bpm': '最大呼吸率', 'min_breath_bpm': '最小呼吸率', 'max_heart_bpm': '最大心率', 'min_heart_bpm': '最小心率', 'body_move_count': '体动次数', 'body_move_exponent': '体动指数', 'average_body_move_count': '平均体动次数', 'max_body_move_count': '最大体动次数', 'min_body_move_count': '最小体动次数', 'breath_bpm_status': '呼吸率状态', 'heart_bpm_status': '心率状态', 'body_move_status': '体动状态', 'body_move_image_x_y': '体动绘图', 'breath_exception_image_sixty_x_y': '典型呼吸异常事件', 'breath_exception_count': '呼吸异常次数', 'breath_exception_exponent': '呼吸异常指数', 'breath_exception_image_x_y': '呼吸异常绘图', 'breath_bpm_image_x_y': '呼吸率绘图数据', 'heart_bpm_image_x_y': '心率绘图数据', 'leave_bed_time': '离床时间', 'device_sn': '设备编号', 'score_rank': '超越人数百分比', 'health_advice': '超越人数百分比'}\n\n当前数据记录：\n"}, {'报告姓名': '畅丽娟', '报告编号': '13CFF34920008071211195E507', '报告日期': '2025-03-17', '睡眠报告': {'id': 5576, 'average_heart_bpm': 69, 'average_breath_bpm': 16, 'total_num_hour': '12小时57分钟', 'total_num_hour_on_bed': '9小时11分钟', 'sleep_hour': '7小时23分钟', 'deep_sleep_hour': '1小时15分钟', 'waking_hour': '1小时48分钟', 'to_sleep_hour': '0小时27分钟', 'leave_bed_total_hour': '3小时45分钟', 'light_sleep_hour': '6小时8分钟', 'total_num_second': 46621, 'total_num_second_on_bed': 33101, 'sleep_second': 26600, 'deep_sleep_second': 4500, 'waking_second': 6501, 'to_sleep_second': 1625, 'leave_bed_total_second': 13503, 'light_sleep_second': 22098, 'leave_count': 8, 'sleep_efficiency': 0.8, 'deep_sleep_efficiency': 0.17, 'score': 58.34, 'score_name': '较差', 'consist_count_waking': 18, 'consist_count_sleep_efficiency': 17, 'query_date': '2025-03-17', 'save_file_path': 'none', 'creator': None, 'create_time': '2025-03-18T21:17:02', 'updater': None, 'update_time': '2025-03-19T09:18:46', 'deleted': False, 'tenant_id': 0, 'waking_count': 4, 'on_bed_time': '2025-03-16T20:01:54', 'sleep_time': '2025-03-16T20:01:54', 'waking_time': '2025-03-17T05:34:32', 'max_breath_bpm': 23, 'min_breath_bpm': 10, 'max_heart_bpm': 116, 'min_heart_bpm': 48, 'body_move_count': 69, 'body_move_exponent': 2.3, 'average_body_move_count': 3, 'max_body_move_count': 21, 'min_body_move_count': 0, 'breath_bpm_status': '正常', 'heart_bpm_status': '异常', 'body_move_status': '正常', 'breath_exception_count': 0, 'breath_exception_exponent': 0.0, 'leave_bed_time': '2025-03-17T08:40:57', 'device_sn': '13CFF34920008071211195E507', 'score_rank': 0.89}}, {'报告姓名': '畅丽娟', '报告编号': '13CFF34920008071211195E507', '报告日期': '2025-03-19', '睡眠报告': {'id': 5708, 'average_heart_bpm': 67, 'average_breath_bpm': 15, 'total_num_hour': '12小时58分钟', 'total_num_hour_on_bed': '10小时3分钟', 'sleep_hour': '8小时18分钟', 'deep_sleep_hour': '0小时21分钟', 'waking_hour': '0小时47分钟', 'to_sleep_hour': '0小时52分钟', 'leave_bed_total_hour': '2小时51分钟', 'light_sleep_hour': '7小时56分钟', 'total_num_second': 46715, 'total_num_second_on_bed': 36200, 'sleep_second': 29900, 'deep_sleep_second': 1299, 'waking_second': 2850, 'to_sleep_second': 3150, 'leave_bed_total_second': 10295, 'light_sleep_second': 28600, 'leave_count': 28, 'sleep_efficiency': 0.83, 'deep_sleep_efficiency': 0.04, 'score': 46.63, 'score_name': '较差', 'consist_count_waking': 19, 'consist_count_sleep_efficiency': 17, 'query_date': '2025-03-19', 'save_file_path': 'none', 'creator': None, 'create_time': '2025-03-19T09:43:06', 'updater': None, 'update_time': '2025-03-19T22:05:43', 'deleted': False, 'tenant_id': 0, 'waking_count': 2, 'on_bed_time': '2025-03-18T20:00:00', 'sleep_time': '2025-03-18T21:01:05', 'waking_time': '2025-03-18T21:58:32', 'max_breath_bpm': 30, 'min_breath_bpm': 10, 'max_heart_bpm': 117, 'min_heart_bpm': 47, 'body_move_count': 151, 'body_move_exponent': 5.03, 'average_body_move_count': 6, 'max_body_move_count': 36, 'min_body_move_count': 0, 'breath_bpm_status': '正常', 'heart_bpm_status': '异常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': '2025-03-19T08:58:22', 'device_sn': '13CFF34920008071211195E507', 'score_rank': 0.18}}, {'报告姓名': '畅丽娟', '报告编号': '13CFF34920008071211195E507', '报告日期': '2025-03-20', '睡眠报告': {'id': 5844, 'average_heart_bpm': 68, 'average_breath_bpm': 15, 'total_num_hour': '12小时58分钟', 'total_num_hour_on_bed': '9小时20分钟', 'sleep_hour': '7小时12分钟', 'deep_sleep_hour': '0小时47分钟', 'waking_hour': '1小时10分钟', 'to_sleep_hour': '0小时42分钟', 'leave_bed_total_hour': '3小时30分钟', 'light_sleep_hour': '6小时25分钟', 'total_num_second': 46715, 'total_num_second_on_bed': 33655, 'sleep_second': 25950, 'deep_sleep_second': 2850, 'waking_second': 4255, 'to_sleep_second': 2568, 'leave_bed_total_second': 12639, 'light_sleep_second': 23100, 'leave_count': 11, 'sleep_efficiency': 0.77, 'deep_sleep_efficiency': 0.11, 'score': 52.0, 'score_name': '较差', 'consist_count_waking': 20, 'consist_count_sleep_efficiency': 18, 'query_date': '2025-03-20', 'save_file_path': 'none', 'creator': None, 'create_time': '2025-03-20T09:28:39', 'updater': None, 'update_time': '2025-03-20T09:28:39', 'deleted': False, 'tenant_id': 0, 'waking_count': 3, 'on_bed_time': '2025-03-19T20:00:00', 'sleep_time': '2025-03-19T21:04:26', 'waking_time': '2025-03-20T04:44:51', 'max_breath_bpm': 25, 'min_breath_bpm': 9, 'max_heart_bpm': 114, 'min_heart_bpm': 48, 'body_move_count': 79, 'body_move_exponent': 2.63, 'average_body_move_count': 3, 'max_body_move_count': 24, 'min_body_move_count': 0, 'breath_bpm_status': '正常', 'heart_bpm_status': '异常', 'body_move_status': '正常', 'breath_exception_count': 0, 'breath_exception_exponent': 0.0, 'leave_bed_time': '2025-03-20T08:30:18', 'device_sn': '13CFF34920008071211195E507', 'score_rank': None}}]"""
        },
        {
            "role": "user",
            "content": "详细解释畅丽娟最近3天的呼吸有哪些异常情况，告诉我她的床位信息"
        }
    ],
    "options": {
        "temperature": 0.3
    },
    "stream": True
}

# # Function to handle streaming responses
# def handle_streaming_response(response):
#     # Check if response is successful
#     if response.status_code == 200:
#         # Process streaming response
#         for line in response.iter_lines():
#             if line:
#                 # Filter out keep-alive new lines
#                 if line.startswith(b'data: '):
#                     # Remove the 'data: ' prefix
#                     data = line[6:]
#                     if data.strip() == b'[DONE]':
#                         break
#                     try:
#                         json_data = json.loads(data)
#                         content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
#                         if content:
#                             print(content, end='', flush=True)
#                     except json.JSONDecodeError:
#                         print(f"Error parsing JSON: {data}")
#     else:
#         print(f"Error: {response.status_code}")
#         print(response.text)

# # Function to handle non-streaming responses
# def handle_regular_response(response):
#     if response.status_code == 200:
#         result = response.json()
#         message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
#         print(message)
#     else:
#         print(f"Error: {response.status_code}")
#         print(response.text)

# # Make the API request
# try:
#     # For streaming responses
#     if payload.get("stream", False):
#         with requests.post(api_url, headers=headers, json=payload, stream=True) as response:
#             handle_streaming_response(response)
#     # For regular responses
#     else:
#         response = requests.post(api_url, headers=headers, json=payload)
#         handle_regular_response(response)
# except Exception as e:
#     print(f"Exception occurred: {e}")
    
    
# 发送请求并打印结果
# 发送请求并获取结果

"""
try:
    # 关闭stream，获取完整响应
    if "stream" in payload:
        payload["stream"] = False
    
    # 发送请求
    response = requests.post(api_url, headers=headers, json=payload)
    
    # 输出状态信息
    print(f"状态码: {response.status_code}")
    
    # 处理响应
    if response.status_code == 200:
        try:
            result = response.json()
            
            # 从结构化数据中提取内容
            if "choices" in result and len(result["choices"]) > 0:
                if "message" in result["choices"][0]:
                    content = result["choices"][0]["message"]["content"]
                    print("\n=== 模型回复 ===")
                    print(content)
                    print("=== 回复结束 ===")
                else:
                    print("回复内容结构不完整")
            else:
                print("未找到回复内容")
                print("完整响应:", json.dumps(result, ensure_ascii=False, indent=2))
        except ValueError:
            print("无法解析JSON响应")
            print("原始响应:", response.text[:1000])  # 限制输出长度
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("错误详情:", response.text[:500])  # 限制输出长度

except Exception as e:
    print(f"发生异常: {e}")
"""
from whoami.llm_api.ollama_llm import OllamaLLM
from whoami.configs.llm_config import LLMConfig
from pathlib import Path
import asyncio


llm=OllamaLLM(config=LLMConfig.from_file(Path('/work/ai/WHOAMI/whoami/scripts/test/ollama_config.yaml')))

messages = [
    {
        "role": "system",
        "content": """
            你是一名专业的睡眠健康分析师，擅长解读复杂的睡眠监测数据。请根据提供的详细睡眠数据，全面、专业地回答用户的问题。
            分析指南：
            1. 如果用户询问单次睡眠报告，请详细解读以下方面：
            - 睡眠质量综合评估
            - 睡眠阶段分布合理性
            - 生理指标（心率、呼吸）正常性
            - 潜在健康风险
            - 具体改善建议

            2. 如果用户询问趋势或比较，请注意：
            - 根据当前数据记录进行趋势分析

            3. 分析要求：
            - 使用专业但易懂的语言
            - 数据解读要客观准确
            - 建议具体可执行
            - 语气友好，富有同理心

            4. 特别提示：
            - 如果数据显示异常，请谨慎但清晰地指出
            - 针对异常指标给出专业建议
            - 鼓励用户持续关注身体健康
            请根据上述指南和数据库信息（如果有），全面且专业地回答用户的具体问题。
            """
    },
    {
        "role": "system",
        "content": """[{'\n用户询问的关于畅丽娟的数据库检索信息': '[{\'device_sn\': \'13CFF34920008071211195E507\', \'device_type\': \'WAVVE_SLEEP_DETECTION\', \'device_name\': \'睡眠检测\', \'device_status\': 1, \'dept_id\': 186, \'room_id\': 60, \'bed_id\': 97, \'elderly_id\': 110, \'elderly_name\': \'畅丽娟\', \'institution_room_id\': 60, \'institution_bed_id\': 97, \'elderly_id_card\': \'142724198304153320\', \'elderly_sex\': \'2\', \'elderly_age\': 41, \'elderly_birthday\': \'1983-04-15\', \'elderly_address\': \'山西省运城市临猗县角杯镇\', \'contacts\': \'[{"name":"宋阳泽","phone":"13080342714"}]\'}]\n\n'}, {'\n数据字段说明': "{'id': '主键id', 'average_heart_bpm': '平均心率', 'average_breath_bpm': '平均呼吸率', 'total_num_hour': '总监测时长（小时）', 'total_num_hour_on_bed': '总在床时长（小时）', 'sleep_hour': '睡眠时长（小时）', 'deep_sleep_hour': '深度睡眠时长（小时）', 'waking_hour': '清醒时长（小时）', 'to_sleep_hour': '入睡时长（小时）', 'leave_bed_total_hour': '离床总时间（小时）', 'light_sleep_hour': '浅睡时长（小时）', 'total_num_second': '总监测时长（秒）', 'total_num_second_on_bed': '总在床时长（秒）', 'sleep_second': '睡眠时长（秒）', 'deep_sleep_second': '深度睡眠时长（秒）', 'waking_second': '清醒时长（秒）', 'to_sleep_second': '入睡时长（秒）', 'leave_bed_total_second': '离床总时间（秒）', 'light_sleep_second': '浅睡时长（秒）', 'leave_count': '离床次数', 'sleep_efficiency': '睡眠效率', 'deep_sleep_efficiency': '深睡效率', 'score': '评分', 'score_name': '评分归类', 'consist_count_waking': '连续?晚夜醒时长超过31分钟', 'consist_count_sleep_efficiency': '连续?晚睡眠效率小于80%', 'query_date': '查询日期', 'save_file_path': '睡眠阶段划分-心率折线图', 'creator': '创建者', 'create_time': '创建时间', 'updater': '更新者', 'update_time': '更新时间', 'deleted': '是否删除', 'tenant_id': '租户编号', 'waking_count': '夜醒次数（次）', 'on_bed_time': '上床时间（节点）', 'sleep_time': '入睡时间（节点）', 'waking_time': '醒来时间（节点）', 'sleep_stage_image_x_y': '睡眠分区绘图', 'max_breath_bpm': '最大呼吸率', 'min_breath_bpm': '最小呼吸率', 'max_heart_bpm': '最大心率', 'min_heart_bpm': '最小心率', 'body_move_count': '体动次数', 'body_move_exponent': '体动指数', 'average_body_move_count': '平均体动次数', 'max_body_move_count': '最大体动次数', 'min_body_move_count': '最小体动次数', 'breath_bpm_status': '呼吸率状态', 'heart_bpm_status': '心率状态', 'body_move_status': '体动状态', 'body_move_image_x_y': '体动绘图', 'breath_exception_image_sixty_x_y': '典型呼吸异常事件', 'breath_exception_count': '呼吸异常次数', 'breath_exception_exponent': '呼吸异常指数', 'breath_exception_image_x_y': '呼吸异常绘图', 'breath_bpm_image_x_y': '呼吸率绘图数据', 'heart_bpm_image_x_y': '心率绘图数据', 'leave_bed_time': '离床时间', 'device_sn': '设备编号', 'score_rank': '超越人数百分比', 'health_advice': '超越人数百分比'}\n\n当前数据记录：\n"}, {'报告姓名': '畅丽娟', '报告编号': '13CFF34920008071211195E507', '报告日期': '2025-03-17', '睡眠报告': {'id': 5576, 'average_heart_bpm': 69, 'average_breath_bpm': 16, 'total_num_hour': '12小时57分钟', 'total_num_hour_on_bed': '9小时11分钟', 'sleep_hour': '7小时23分钟', 'deep_sleep_hour': '1小时15分钟', 'waking_hour': '1小时48分钟', 'to_sleep_hour': '0小时27分钟', 'leave_bed_total_hour': '3小时45分钟', 'light_sleep_hour': '6小时8分钟', 'total_num_second': 46621, 'total_num_second_on_bed': 33101, 'sleep_second': 26600, 'deep_sleep_second': 4500, 'waking_second': 6501, 'to_sleep_second': 1625, 'leave_bed_total_second': 13503, 'light_sleep_second': 22098, 'leave_count': 8, 'sleep_efficiency': 0.8, 'deep_sleep_efficiency': 0.17, 'score': 58.34, 'score_name': '较差', 'consist_count_waking': 18, 'consist_count_sleep_efficiency': 17, 'query_date': '2025-03-17', 'save_file_path': 'none', 'creator': None, 'create_time': '2025-03-18T21:17:02', 'updater': None, 'update_time': '2025-03-19T09:18:46', 'deleted': False, 'tenant_id': 0, 'waking_count': 4, 'on_bed_time': '2025-03-16T20:01:54', 'sleep_time': '2025-03-16T20:01:54', 'waking_time': '2025-03-17T05:34:32', 'max_breath_bpm': 23, 'min_breath_bpm': 10, 'max_heart_bpm': 116, 'min_heart_bpm': 48, 'body_move_count': 69, 'body_move_exponent': 2.3, 'average_body_move_count': 3, 'max_body_move_count': 21, 'min_body_move_count': 0, 'breath_bpm_status': '正常', 'heart_bpm_status': '异常', 'body_move_status': '正常', 'breath_exception_count': 0, 'breath_exception_exponent': 0.0, 'leave_bed_time': '2025-03-17T08:40:57', 'device_sn': '13CFF34920008071211195E507', 'score_rank': 0.89}}, {'报告姓名': '畅丽娟', '报告编号': '13CFF34920008071211195E507', '报告日期': '2025-03-19', '睡眠报告': {'id': 5708, 'average_heart_bpm': 67, 'average_breath_bpm': 15, 'total_num_hour': '12小时58分钟', 'total_num_hour_on_bed': '10小时3分钟', 'sleep_hour': '8小时18分钟', 'deep_sleep_hour': '0小时21分钟', 'waking_hour': '0小时47分钟', 'to_sleep_hour': '0小时52分钟', 'leave_bed_total_hour': '2小时51分钟', 'light_sleep_hour': '7小时56分钟', 'total_num_second': 46715, 'total_num_second_on_bed': 36200, 'sleep_second': 29900, 'deep_sleep_second': 1299, 'waking_second': 2850, 'to_sleep_second': 3150, 'leave_bed_total_second': 10295, 'light_sleep_second': 28600, 'leave_count': 28, 'sleep_efficiency': 0.83, 'deep_sleep_efficiency': 0.04, 'score': 46.63, 'score_name': '较差', 'consist_count_waking': 19, 'consist_count_sleep_efficiency': 17, 'query_date': '2025-03-19', 'save_file_path': 'none', 'creator': None, 'create_time': '2025-03-19T09:43:06', 'updater': None, 'update_time': '2025-03-19T22:05:43', 'deleted': False, 'tenant_id': 0, 'waking_count': 2, 'on_bed_time': '2025-03-18T20:00:00', 'sleep_time': '2025-03-18T21:01:05', 'waking_time': '2025-03-18T21:58:32', 'max_breath_bpm': 30, 'min_breath_bpm': 10, 'max_heart_bpm': 117, 'min_heart_bpm': 47, 'body_move_count': 151, 'body_move_exponent': 5.03, 'average_body_move_count': 6, 'max_body_move_count': 36, 'min_body_move_count': 0, 'breath_bpm_status': '正常', 'heart_bpm_status': '异常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': '2025-03-19T08:58:22', 'device_sn': '13CFF34920008071211195E507', 'score_rank': 0.18}}, {'报告姓名': '畅丽娟', '报告编号': '13CFF34920008071211195E507', '报告日期': '2025-03-20', '睡眠报告': {'id': 5844, 'average_heart_bpm': 68, 'average_breath_bpm': 15, 'total_num_hour': '12小时58分钟', 'total_num_hour_on_bed': '9小时20分钟', 'sleep_hour': '7小时12分钟', 'deep_sleep_hour': '0小时47分钟', 'waking_hour': '1小时10分钟', 'to_sleep_hour': '0小时42分钟', 'leave_bed_total_hour': '3小时30分钟', 'light_sleep_hour': '6小时25分钟', 'total_num_second': 46715, 'total_num_second_on_bed': 33655, 'sleep_second': 25950, 'deep_sleep_second': 2850, 'waking_second': 4255, 'to_sleep_second': 2568, 'leave_bed_total_second': 12639, 'light_sleep_second': 23100, 'leave_count': 11, 'sleep_efficiency': 0.77, 'deep_sleep_efficiency': 0.11, 'score': 52.0, 'score_name': '较差', 'consist_count_waking': 20, 'consist_count_sleep_efficiency': 18, 'query_date': '2025-03-20', 'save_file_path': 'none', 'creator': None, 'create_time': '2025-03-20T09:28:39', 'updater': None, 'update_time': '2025-03-20T09:28:39', 'deleted': False, 'tenant_id': 0, 'waking_count': 3, 'on_bed_time': '2025-03-19T20:00:00', 'sleep_time': '2025-03-19T21:04:26', 'waking_time': '2025-03-20T04:44:51', 'max_breath_bpm': 25, 'min_breath_bpm': 9, 'max_heart_bpm': 114, 'min_heart_bpm': 48, 'body_move_count': 79, 'body_move_exponent': 2.63, 'average_body_move_count': 3, 'max_body_move_count': 24, 'min_body_move_count': 0, 'breath_bpm_status': '正常', 'heart_bpm_status': '异常', 'body_move_status': '正常', 'breath_exception_count': 0, 'breath_exception_exponent': 0.0, 'leave_bed_time': '2025-03-20T08:30:18', 'device_sn': '13CFF34920008071211195E507', 'score_rank': None}}]"""
    },
    {
        "role": "user",
        "content": "详细解释畅丽娟最近3天的呼吸有哪些异常情况，告诉我她的床位信息"
    }
]


async def main():
    content = await llm._whoami_text(messages=messages, timeout=30, user_stop_words=[])
    print(content)

if __name__ == '__main__':
    # 使用asyncio.run来运行异步主函数
    asyncio.run(main())