
from whoami.provider.sql_provider import SqlProvider
from whoami.tool.health_report.sleep_indices import SleepIndices


SQL_CONFIG_PATH = "/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml"
sql_provider = SqlProvider(model=SleepIndices, sql_config_path=SQL_CONFIG_PATH)


def test_sql_rag():
    sql_result = sql_provider.get_record_by_condition(
                    condition={"device_sn": "13D4F349200080712111959C07", "query_date": '2025-3-20'}, 
                    exclude_fields=[
                        'health_advice',
                        'sleep_stage_image_x_y',
                        'body_move_image_x_y',
                        'breath_exception_image_sixty_x_y',
                        'heart_bpm_image_x_y',
                        'breath_bpm_image_x_y',
                        'breath_exception_image_x_y',
                        'deep_sleep_second',
                        'total_num_second',
                        'total_num_second_on_bed',
                        'sleep_second',
                        'deep_sleep_second',
                        'waking_second',
                        'to_sleep_second',
                        'leave_bed_total_second',
                        'save_file_path',
                        'creator',
                        'create_time',
                        'updater',
                        'update_time',
                        'deleted',
                        'tenant_id',
                        'id'
                    ],
                    date_range={"date_field": "query_date", "start_date": "2025-3-10", "end_date": "2025-3-26"}
                )

    print(sql_result[0]["query_date"])
    
    
    
    
    """
    你是睡眠报告解读助手，请根据数据库检索结果，结合用户历史会话信息和当前提问，回答用户关于睡眠报告相关的问题：
    
    数据库检索结果：
    [
        
        "2025年3月20日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月20日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月21日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月21日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月22日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月22日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月23日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月23日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月24日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月24日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月25日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月25日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月26日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月26日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66}
    ]
    
    当前时间：2025-03-26
    
    历史会话消息：[
        {"role": "user", "content": "我昨天晚上睡得怎么样？"}, 
        {"role": "assistant", "content": "根据您提供的当前时间（2025-03-26）和数据库检索结果中的查询日期（2025-03-20），**您提问的“昨天晚上”（即2025-03-25）的睡眠数据并未在数据库中找到**。当前可用数据仅覆盖至2025-03-20，无法提供更近的睡眠报告。"}
        {"role": "user", "content": "那3月20日呢？"},
        {"role": "assistant", "content": "根据数据库中2025年3月20日的睡眠报告，以下是详细解读：

            根据数据库记录，2025年3月20日的睡眠报告显示：

            ### **核心数据**
            - **总卧床时间**：11小时7分钟
            - **实际睡眠时间**：7小时57分钟
            - **睡眠效率**：72%（低于理想值85%）
            - **睡眠得分**：55.56（评分“较差”，满分可能为100）

            ### **关键问题**
            1. **深睡不足**：仅1小时22分钟（占比约5%），效率仅17%，影响身体修复。
            2. **频繁觉醒**：夜间醒来9次，总觉醒时间3小时5分钟，睡眠连续性差。
            3. **呼吸异常**：呼吸频率波动大（最高21次/分钟，最低7次/分钟），标记为“异常”。
            4. **浅睡过长**：浅睡阶段占6小时35分钟，可能影响深度恢复。

            ### **其他指标**
            - 离床1次，总离床时间约59分钟；
            - 心率正常（50-117次/分钟），体动次数298次（高于平均12次/小时）。

            ### **建议**
            可关注睡前环境、压力管理或呼吸节奏，必要时咨询专业评估。是否需要进一步分析？
            ---

            ### **总结**
            3月20日的睡眠质量较差，主要问题集中在**深睡不足、频繁觉醒和呼吸异常**。建议优先排查夜间干扰因素，并关注呼吸健康。如持续存在问题，建议咨询医生或睡眠专科。

            需要更具体的分析或进一步建议吗？"
        },
        {"role": "user", "content": "如何解决频繁觉醒问题"},
        {"role": "assistant", "content": "为了解决频繁觉醒的问题，您可以尝试以下几个方法：

            1. **优化睡前环境**：确保睡眠环境安静、舒适且温度适宜。避免强光和噪音干扰。
            2. **建立规律作息**：尽量每天同一时间上床睡觉和起床，帮助调整身体生物钟。
            3. **减少晚间刺激活动**：睡前避免使用电子设备，减少咖啡因和酒精摄入，这些都可能影响深度睡眠。
            4. **放松身心**：通过冥想、深呼吸等方式减轻白天的压力，有助于提高夜间睡眠质量。
            5. **检查床上用品**：确保床垫和枕头适合您的身体需求，以减少不适引起的觉醒。

            如果这些问题持续存在，建议咨询医生或专业的睡眠专家进行进一步评估。"
        },
        {"role": "user", "content": "如何解决呼吸异常问题"},
        {"role": "assistant", "content": "为了解决呼吸异常问题，您可以尝试以下几个方法：

            1. **改善睡眠姿势**：保持正确的睡姿，避免过多翻身和侧卧，以减少对呼吸道的压力。
            2. **调节室内空气湿度**：确保室内空气湿润，可以使用加湿器，但注意定期清洁设备以防霉菌滋生。
            3. **检查过敏原**：如果您有呼吸系统疾病或过敏史，请排查并避开可能引起反应的过敏源。
            4. **适量运动**：适当的体育锻炼有助于增强肺功能和整体健康状况。但避免在睡前进行剧烈运动。
            5. **保持体重健康**：过重的身体会增加呼吸道负担，控制饮食、定期运动以维持理想体重。
            6. **咨询医生**：如果呼吸异常症状持续或伴有其他不适，请及时就医检查是否有潜在疾病。

            请根据自身情况调整上述建议，并在必要时寻求专业医疗帮助。"
        },
        {"role": "user", "content": "如何解决浅睡过长问题"},
        {"role": "assistant", "content": "为了解决浅睡过长的问题，您可以尝试以下几个方法：

            1. **优化睡眠环境**：确保卧室安静、舒适且温度适宜。减少噪音和光线干扰。
            2. **保持规律作息**：尽量每天同一时间上床睡觉和起床，帮助调整生物钟。
            3. **避免晚间刺激活动**：睡前避免使用电子设备，减少咖啡因和酒精摄入，这些都可能影响深度睡眠。
            4. **放松身心**：通过冥想、深呼吸等方式减轻白天的压力，有助于提高夜间睡眠质量。
            5. **改善枕头和床垫**：确保使用的枕头和床垫适合您的身体需求，减少不适引起的浅睡。

            如果这些问题持续存在，建议咨询医生或专业的睡眠专家进行进一步评估。"
        },
    ]
    
    当前提问：2025年3月26日睡的怎么样？
    
    注意事项：
    明确用户的问题时间和数据库检索结果中的时间是否可以对应，如果不对应，回答用户没有数据
    简要并严谨回答用户问题，不要含有特殊符号
    """
    
    
    
    
    
    
    """
    你是睡眠报告解读助手，请根据数据库检索结果，结合用户历史会话信息和当前提问，回答用户关于睡眠报告相关的问题：
    
    数据库检索结果：
    包含的报告日期为：2025年3月20日、2025年3月21日、2025年3月22日、2025年3月23日、2025年3月24日、2025年3月25日、2025年3月26日
    [
        
        "2025年3月20日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月20日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月21日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月21日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月22日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月22日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月23日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月23日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月24日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月24日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月25日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月25日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66},
        "2025年3月26日": {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月26日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712111959C07', 'score_rank': 0.66}
    ]
    
    当前时间：2025-03-26
    
    历史会话消息：[
        {"role": "user", "content": "我昨天晚上睡得怎么样？"},
        {"role": "assistant", "content": "根据2025年3月25日的睡眠数据，您的睡眠质量评分为55.56（较差）。具体表现如下：
            - 总睡眠时长7小时57分钟，深睡时长1小时22分钟。
            - 平均心率71次/分钟，呼吸频率13次/分钟，但呼吸频率标记为异常。
            - 夜间醒来9次，翻身次数较多（298次），可能影响睡眠连续性。
            - 睡眠效率72%，睡眠占床上时间比例偏低。

            建议关注呼吸状态，减少夜间翻身干扰，可尝试改善睡眠环境或调整睡前习惯。"
        },
    ]
    
    当前提问：前天呢？
    
    注意事项：
    明确用户的问题时间和数据库检索结果中的时间是否可以对应，如果不对应，回答用户没有数据
    简要并严谨回答用户问题，不要含有特殊符号
    """
    
    
    """
    你是专业的睡眠报告解读助手。请严格按照以下步骤处理用户问题：

    处理流程：
    1. 判断用户问题类型
    - 报告解读（如"昨晚睡得怎么样"）
    - 改善建议（如"如何提高睡眠质量"）
    - 特定睡眠问题（如"为什么频繁醒来"）
    - 其他睡眠相关咨询

    2. 对于报告解读问题：
    - 识别问题中的时间（将"昨天"转换为"2025-03-25"等）
    - 在数据库中检索对应日期的数据
    - 提取关键指标（总睡眠时长、深睡时长、心率、呼吸、睡眠效率等）
    - 结合历史数据，给出评价和针对性建议

    3. 对于改善建议问题：
    - 分析用户最近的睡眠数据找出主要问题（如深睡不足、频繁醒来等）
    - 提供针对性的改善建议
    - 提供通用的睡眠卫生建议

    4. 对于特定睡眠问题：
    - 提供专业解释和可能的原因
    - 给出缓解方法建议
    
    当前日期：2025-03-26

    可用数据日期：2025年3月20日、2025年3月21日、2025年3月22日、2025年3月23日、2025年3月24日、2025年3月25日、2025年3月26日
    
    数据库：
    {
        "2025年3月20日": 
            {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月20日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'score_rank': 0.66},
        "2025年3月21日": 
            {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月21日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'score_rank': 0.66},
        "2025年3月22日": 
            {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月22日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'score_rank': 0.66},
        "2025年3月23日": 
            {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月23日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'score_rank': 0.66},
        "2025年3月24日": 
            {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月24日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'score_rank': 0.66},
        "2025年3月25日": 
            {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月25日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'score_rank': 0.66},
        "2025年3月26日": 
            {'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hour': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sleep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': 2025年3月26日, 'waking_count': 9, 'on_bed_time': datetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart_bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'score_rank': 0.66}
    }

    [根据具体情况给出1-2条建议]
    用户问题历史：
    []
    当前问题：我昨天睡得怎么样？
    
    """
    
    
    
    """
    你是睡眠报告解读助手，请根据数据库检索结果，结合用户历史会话信息和当前提问，回答用户关于睡眠报告相关的问题：

    数据库检索结果：
    [{'average_heart_bpm': 71, 'average_breath_bpm': 13, 'total_num_hour': '12小时56分钟', 'total_num_hour_on_bed': '11小时7分钟', 'sleep_hour': '7小时57分钟', 'deep_sleep_hour': '1小时22分钟', 'waking_hou
    r': '3小时5分钟', 'to_sleep_hour': '0小时21分钟', 'leave_bed_total_hour': '1小时46分钟', 'light_sleep_hour': '6小时35分钟', 'light_sleep_second': 23700, 'leave_count': 8, 'sleep_efficiency': 0.72, 'deep_sl
    eep_efficiency': 0.17, 'score': 55.56, 'score_name': '较差', 'consist_count_waking': 25, 'consist_count_sleep_efficiency': 17, 'query_date': datetime.date(2025, 3, 20), 'waking_count': 9, 'on_bed_time': da
    tetime.datetime(2025, 3, 19, 20, 0), 'sleep_time': datetime.datetime(2025, 3, 19, 20, 4, 57), 'waking_time': datetime.datetime(2025, 3, 20, 2, 52, 14), 'max_breath_bpm': 21, 'min_breath_bpm': 7, 'max_heart
    _bpm': 117, 'min_heart_bpm': 50, 'body_move_count': 298, 'body_move_exponent': 9.93, 'average_body_move_count': 12, 'max_body_move_count': 22, 'min_body_move_count': 0, 'breath_bpm_status': '异常', 'heart_
    bpm_status': '正常', 'body_move_status': '正常', 'breath_exception_count': 1, 'breath_exception_exponent': 0.07, 'leave_bed_time': datetime.datetime(2025, 3, 20, 7, 43, 19), 'device_sn': '13D4F349200080712
    111959C07', 'score_rank': 0.66}]

    当前时间：2025-03-26
    
    历史会话消息：[{"role": "user", "content": "我昨天晚上睡得怎么样？"}, {"role": "assistant", "content": "根据您提供的当前时间（2025-03-26）和数据库检索结果中的查询日期（2025-03-20），**您提问的“昨天晚上
    ”（即2025-03-25）的睡眠数据并未在数据库中找到**。当前可用数据仅覆盖至2025-03-20，无法提供更近的睡眠报告。"}]

    当前提问：那3月20日呢？
    
    注意事项：
    明确用户的问题时间和数据库检索结果中的时间是否可以对应，如果不对应，回答用户没有数据
    简要并严谨回答用户问题，不要含有特殊符号
"""

